"""ハッチパッチによる符号付きのずれの測定（ring_plotter.deflection）。

合成スキャンで検証する: 既知のずれ δ = (a·(v·u) + b·sign ċ)·u を一次遅れ付きで加えた線を描き、
測定で D と係数が戻ること、スキャンを 180° 回しても同じ D になることを見る。
"""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ring_plotter.deflection import (DeflectionModel, Layout, Transform, fit_model, measure_scan, patch_polylines,
                                     predict_D, result_row, start_hooks, zigzag)
from ring_plotter.gcode import GcodeOptions, convert
from ring_plotter.kinematics import Kinematics

A_TRUE, B_TRUE, LAM = 0.35, 0.10, 3.0  # sin² が掛かるので D は最大 0.25 程度（間隔 1.2 で融合しない範囲）
MODEL = DeflectionModel(A_TRUE, B_TRUE, LAM)  # 既定の rod_axial_sin2
DPI = 400


@pytest.fixture(scope="module")
def kin():
    return Kinematics.from_params(dmin=89, dmax=179)


@pytest.fixture(scope="module")
def layout():
    # 小さめ: 2 か所 × 4 向き
    return Layout(grid_x=(-40.0, 40.0), grid_y=(60.0,))


def deflected(kin, poly, step=0.1):
    out, cur = [], np.zeros(2)
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(L / step)))
        vx, vy = (x1 - x0) / L, (y1 - y0) / L
        for i in range(1, n + 1):
            x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
            target = MODEL.target(kin, x, y, vx, vy)
            cur = cur + (target - cur) * (1 - math.exp(-(L / n) / LAM))
            out.append((x + cur[0], y + cur[1]))
    return out


def render(kin, layout, T, size_mm, dpi, ss=3):
    """合成スキャン（濃淡、ss 倍で描いて縮小してアンチエイリアス）。"""
    px = dpi / 25.4
    im = Image.new("L", (int(size_mm[0] * px * ss), int(size_mm[1] * px * ss)), 255)
    dr = ImageDraw.Draw(im)
    w = int(round(0.5 * px * ss))
    polys = [zigzag(p, layout) for p in layout.patches()] + [layout.mark_line()]
    for poly in polys:
        XY = T.apply(np.array(deflected(kin, poly))) * px * ss
        dr.line([tuple(p) for p in XY], fill=0, width=w, joint="curve")
    im = im.resize((int(size_mm[0] * px), int(size_mm[1] * px)), Image.BOX)
    return np.array(im), px


@pytest.fixture(scope="module")
def scan(kin, layout):
    """紙の一部（x −80〜80、y 20〜110）を 1° 傾けて描いた合成スキャン。"""
    return render(kin, layout, Transform(math.radians(1.0), 80.0, 110.0), (160, 100), DPI)


def test_separate_lines_register_and_measure(kin):
    """線ごとにペンを上げたパッチ（Layout.separate）も 1 つの塊として位置合わせでき、D が戻る。"""
    lay = Layout(grid_x=(-40.0, 40.0), grid_y=(60.0,), separate="alt")
    px = DPI / 25.4
    T = Transform(0.0, 80.0, 110.0)
    ss = 3
    im = Image.new("L", (int(160 * px * ss), int(100 * px * ss)), 255)
    dr = ImageDraw.Draw(im)
    w = int(round(0.5 * px * ss))
    for p in lay.patches():
        polys = patch_polylines(p, lay)
        assert len(polys) == lay.lines
        for poly in polys:
            XY = T.apply(np.array(deflected(kin, poly))) * px * ss
            dr.line([tuple(q) for q in XY], fill=0, width=w)
    XY = T.apply(np.array(lay.mark_line())) * px * ss
    dr.line([tuple(q) for q in XY], fill=0, width=w)
    gray = np.array(im.resize((int(160 * px), int(100 * px)), Image.BOX))
    reg, results = measure_scan(gray, px, lay)
    assert reg.mark_found and all(r.ok for r in results)
    for r in results:
        want = predict_D(kin, r.patch.x, r.patch.y, r.patch.theta, A_TRUE, B_TRUE)
        assert abs(r.D_mid - want) < 0.03
    # 始点の曲がりも線ごとに出る（deflected は線ごとに状態 0 から始めるので c ≈ −D_left になる）
    hooks = start_hooks(results[0], lay, lam=LAM)
    assert np.isfinite(hooks[1:]).all()


def test_gain_up_scales_travel_state(kin):
    """ペン上げ移動（travel）では目標値に gain_up が掛かり、ペン下げ（advance）では掛からない。"""
    from ring_plotter.deflection import Compensator

    full = Compensator(DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt"), kin)
    half = Compensator(DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", gain_up=0.5), kin)
    ab0, ab1 = kin.inverse(-60.0, 40.0), kin.inverse(-30.0, 40.0)  # 30 mm のペン上げ移動（λ の 20 倍）
    full.travel(ab0, ab1)
    half.travel(ab0, ab1)
    assert np.linalg.norm(full.state) > 0.1
    assert abs(np.linalg.norm(half.state) / np.linalg.norm(full.state) - 0.5) < 0.05
    # ペン下げで同じ距離を走れば両者は同じ目標値に収束する
    for comp in (full, half):
        for i in range(300):
            comp.advance(-30.0 + 0.1 * (i + 1), 40.0, 1.0, 0.0, 0.1)
    assert np.linalg.norm(full.state - half.state) < 1e-3


def test_speed_factor_scales_target_and_fits(kin, layout):
    """速度の係数 f(v) = 1 − c·max(0, 1 − v/v_ref) が目標値に掛かり、当てはめで c が戻る。"""
    m = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", c_speed=0.3)
    full = m.target(kin, -60.0, 40.0, 1.0, 0.0)
    assert np.allclose(m.target(kin, -60.0, 40.0, 1.0, 0.0, speed=30.0), full)
    assert np.allclose(m.target(kin, -60.0, 40.0, 1.0, 0.0, speed=0.0), 0.7 * full)
    assert np.allclose(m.target(kin, -60.0, 40.0, 1.0, 0.0, speed=15.0), 0.85 * full)
    assert np.allclose(m.target(kin, -60.0, 40.0, 1.0, 0.0, speed=60.0), full)  # 速いほうには効かない
    # 端の係数: 巡航に対する落ち込みをさらに掛ける。巡航中（speed = cruise）は効かない
    me = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", c_speed=0.3, c_end=0.4)
    assert np.allclose(me.target(kin, -60.0, 40.0, 1.0, 0.0, speed=15.0, cruise=15.0), 0.85 * full)
    assert np.allclose(me.target(kin, -60.0, 40.0, 1.0, 0.0, speed=0.0, cruise=15.0), 0.7 * 0.6 * full)
    assert np.allclose(me.target(kin, -60.0, 40.0, 1.0, 0.0, speed=7.5, cruise=15.0), 0.775 * 0.8 * full)
    assert "c-end=0.40" in me.describe() and DeflectionModel(**{k: v for k, v in me.to_json().items()}).c_end == 0.4
    a, b, c = 0.45, 0.09, 0.25
    rows = []
    for p in layout.patches():
        for th, v in ((p.theta, 30.0), (p.theta + 20.0, 20.0), (p.theta + 70.0, 10.0)):
            f = 1 - c * (1 - v / 30.0)
            D = predict_D(kin, p.x, p.y, th, a, b, "rod_axial_sin2_sqrt", f)
            rows.append(dict(x=p.x, y=p.y, theta=th, D=D, pitch=2.0, lam=1.5, ok=1, v=v))
    fit = fit_model(kin, rows, "rod_axial_sin2_sqrt", speed_factor=True, v_ref=30.0)
    assert abs(fit["a"] - a) < 1e-3 and abs(fit["b"] - b) < 1e-3 and abs(fit["c_speed"] - c) < 1e-2
    assert fit["rms"] < 1e-4
    assert fit_model(kin, rows, "rod_axial_sin2_sqrt")["rms"] > 0.005  # 係数なしでは合わない
    with pytest.raises(ValueError):
        fit_model(kin, [dict(r, v=math.nan) for r in rows], "rod_axial_sin2_sqrt", speed_factor=True)


def test_cruise_speed_is_capped_near_center(kin):
    from ring_plotter.deflection import cruise_speed

    assert cruise_speed(kin, -60.0, 40.0, 1.0, 0.0, 30.0, 2250.0) == pytest.approx(30.0)
    v = cruise_speed(kin, 3.0, 0.0, 0.0, 1.0, 30.0, 2250.0)  # 中心のそば: F の上限で遅い
    assert 1.0 < v < 20.0


def test_line_ends_measure_shortfall(kin):
    """線ごとにペンを上げたパッチで、終点を 0.3 mm 手前で止めた線の q_end が −0.3 になり、線幅も戻る。"""
    lay = Layout(grid_x=(-40.0, 40.0), grid_y=(60.0,), separate="alt")
    px = DPI / 25.4
    T = Transform(0.0, 80.0, 110.0)
    ss = 3
    r_pen = 0.35  # 実機の Prockey は線幅 0.70
    im = Image.new("L", (int(160 * px * ss), int(100 * px * ss)), 255)
    dr = ImageDraw.Draw(im)
    w = int(round(2 * r_pen * px * ss))

    def dot(p):
        X, Y = T.apply(np.array([p]))[0] * px * ss
        rr = r_pen * px * ss
        dr.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=0)

    for p in lay.patches():
        for poly in patch_polylines(p, lay):
            (x0, y0), (x1, y1) = poly
            L = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            end = (x1 - 0.3 * ux, y1 - 0.3 * uy)
            XY = T.apply(np.array([poly[0], end])) * px * ss
            dr.line([tuple(q) for q in XY], fill=0, width=w)
            dot(poly[0])
            dot(end)
    XY = T.apply(np.array(lay.mark_line())) * px * ss
    dr.line([tuple(q) for q in XY], fill=0, width=w)
    gray = np.array(im.resize((int(160 * px), int(100 * px)), Image.BOX))
    reg, results = measure_scan(gray, px, lay, margin=0.3)
    assert reg.mark_found and all(r.ok for r in results)
    qs = np.concatenate([r.q_start for r in results])
    qe = np.concatenate([r.q_end for r in results])
    wd = np.concatenate([r.width for r in results])
    assert np.isfinite(qs).all() and np.isfinite(qe).all()
    assert abs(np.mean(wd) - 2 * r_pen) < 0.05
    assert abs(np.mean(qs)) < 0.05 and np.std(qs) < 0.05
    assert abs(np.mean(qe) + 0.3) < 0.05 and np.std(qe) < 0.05
    # ペン半径を小さく指定すると、その分だけ端が外に出て見える
    _, r2 = measure_scan(gray, px, lay, margin=0.3, pen_radius=0.25)
    qe2 = np.concatenate([r.q_end for r in r2])
    assert abs((np.mean(qe2) - np.mean(qe)) - 0.1) < 0.03
    # ジグザグ（separate なし）では端点は出さない
    from ring_plotter.deflection import result_row

    assert math.isnan(result_row(measure_scan(gray, px, Layout(grid_x=(-40.0, 40.0), grid_y=(60.0,)))[1][0])["q_end"])


def test_patch_polylines_same_direction():
    lay = Layout(separate="same")
    polys = patch_polylines(lay.patches()[0], lay)
    d = [(b[0] - a[0], b[1] - a[1]) for a, b in polys]
    assert all(abs(dx - d[0][0]) < 1e-9 and abs(dy - d[0][1]) < 1e-9 for dx, dy in d)


def test_layout_patches_and_mark(layout):
    ps = layout.patches()
    assert len(ps) == 8
    assert {p.theta for p in ps} == {0.0, 45.0, 90.0, 135.0}
    assert layout.mark_line() is not None
    z = zigzag(ps[0], layout)
    assert len(z) == 2 * layout.lines
    # 線 0 は +θ 方向、線 1 は逆向き
    v0 = np.subtract(z[1], z[0])
    v1 = np.subtract(z[3], z[2])
    assert np.allclose(v0 / np.linalg.norm(v0), [math.cos(math.radians(ps[0].theta)), math.sin(math.radians(ps[0].theta))])
    assert np.allclose(v0, -v1)


def test_measure_recovers_signed_D_and_model(kin, layout, scan):
    gray, px = scan
    reg, results = measure_scan(gray, px, layout)
    assert reg.mark_found
    assert abs(math.degrees(reg.transform.phi) - 1.0) < 0.05
    assert reg.rms < 0.2
    assert all(r.ok for r in results)
    for r in results:
        true = predict_D(kin, r.center[0], r.center[1], r.theta, A_TRUE, B_TRUE)
        assert abs(r.D - true) < 0.02, (r.patch, r.D, true)
    fit = fit_model(kin, [result_row(r) for r in results])
    assert abs(fit["a"] - A_TRUE) < 0.03
    assert abs(fit["b"] - B_TRUE) < 0.02
    assert abs(fit["lam"] - LAM) < 0.7
    assert fit["rms"] < 0.01
    # 補正を掛けて描いた紙（残差だけが測れる）でも、掛けた係数を足し戻せば同じ係数に戻る
    rows = []
    for r in results:
        row = result_row(r)
        row["D"] -= predict_D(kin, r.center[0], r.center[1], r.theta, A_TRUE, B_TRUE)
        row["applied"] = MODEL
        rows.append(row)
    fit2 = fit_model(kin, rows)
    assert abs(fit2["a"] - fit["a"]) < 1e-6 and abs(fit2["b"] - fit["b"]) < 1e-6
    # 旧形（sin² なし）を指定すると当てはまりが悪くなる（2 か所だけで Δ の差が小さいので差はわずか）
    assert fit_model(kin, [result_row(r) for r in results], "rod_axial")["rms"] > fit["rms"]


def test_rotated_scan_gives_same_D(kin, layout, scan):
    """線の本数が偶数なので、スキャンを 180° 回しても（目印で向きを判別して）同じ D になる。"""
    gray, px = scan
    _, r0 = measure_scan(gray, px, layout)
    reg, r1 = measure_scan(gray[::-1, ::-1].copy(), px, layout)
    assert reg.mark_found
    assert abs(abs(math.degrees(reg.transform.phi)) - 179.0) < 0.05
    for a, b in zip(r0, r1):
        assert a.patch == b.patch
        assert abs(a.D - b.D) < 0.01


def test_sqrt_model_fits_and_differs_from_linear(kin, layout):
    """rod_axial_sin2_sqrt: 合成の D から係数が戻り、比例モデルでは当てはまりが悪い。"""
    a, b = 0.45, 0.09
    rows = []
    for p in layout.patches():
        for th in (p.theta, p.theta + 20.0, p.theta + 70.0):  # v·u の小さい向きも混ぜる
            D = predict_D(kin, p.x, p.y, th, a, b, "rod_axial_sin2_sqrt")
            rows.append(dict(x=p.x, y=p.y, theta=th, D=D, pitch=2.0, lam=1.5, ok=1))
    fit = fit_model(kin, rows, "rod_axial_sin2_sqrt")
    assert abs(fit["a"] - a) < 1e-9 and abs(fit["b"] - b) < 1e-9 and fit["rms"] < 1e-9
    assert fit_model(kin, rows, "rod_axial_sin2")["rms"] > 0.01
    # 補正側（target）と当てはめ側（predict_D）が同じ式を使っている
    m = DeflectionModel(a, b, 1.5, model="rod_axial_sin2_sqrt")
    for r in rows[:6]:
        v = (math.cos(math.radians(r["theta"])), math.sin(math.radians(r["theta"])))
        d = m.target(kin, r["x"], r["y"], *v)
        assert abs(float(d @ (-v[1], v[0])) - r["D"]) < 1e-9


@pytest.mark.parametrize("model, tol", [(MODEL, 0.03), (DeflectionModel(0.3, 0.1, LAM, model="rod_axial_sin2_sqrt"), 0.04)],
                         ids=["sin2", "sin2_sqrt"])
def test_compensation_cancels_deflection(kin, model, tol):
    """補正した G-code を、同じモデルの「物理」で歪ませると、元の折れ線に戻る。

    sqrt 形は v·u ≈ 0 で傾きが急なので、補正後の指令経路の向きと目標の向きのわずかな差が
    折り返し付近で残る（この向きでは最大 0.033 mm）。許容を少し広げてある。
    """
    from ring_plotter.kinematics import Unreachable
    from ring_plotter.simulate import simulate

    lay = Layout(grid_x=(30.0,), grid_y=(60.0,), angles=(45.0,), mark=False)
    poly = zigzag(lay.patches()[0], lay)
    opt = GcodeOptions(sort=False, keep_direction=True, step=0.5, return_home=False, deflection=model)
    lines, stats = convert(kin, [poly], opt)
    assert any("ずれの補正" in n for n in stats.notes)

    def physical(lines):
        """指令どおりに動いたペン先に、一次遅れ付きのずれを足す（ペン上げ移動も含めて歩く）。"""
        state = np.zeros(2)
        drawn, prev = [], None
        for st in simulate(lines, kin).strokes:
            for p in st.xy:
                if prev is not None:
                    d = np.subtract(p, prev)
                    ds = float(np.hypot(*d))
                    if ds > 1e-9:
                        try:
                            tgt = model.target(kin, p[0], p[1], d[0] / ds, d[1] / ds)
                            state = state + (tgt - state) * (1 - math.exp(-ds / LAM))
                        except Unreachable:  # 原点（Δ=88）付近は使う範囲の外
                            pass
                if st.pen_down:
                    drawn.append(np.array(p) + state)
                prev = p
        return drawn

    drawn = physical(lines)
    # 描かれた点が元の折れ線の上に乗っている
    segs = [(np.array(a), np.array(b)) for a, b in zip(poly, poly[1:])]

    def dist(q):
        best = math.inf
        for a, b in segs:
            ab = b - a
            t = float(np.clip((q - a) @ ab / (ab @ ab), 0, 1))
            best = min(best, float(np.linalg.norm(q - (a + t * ab))))
        return best

    errs = np.array([dist(q) for q in drawn])
    assert errs.max() < tol, errs.max()
    # 補正なしでは、ずれのぶん（0.1 mm 以上）外れる
    lines0, _ = convert(kin, [poly], GcodeOptions(sort=False, keep_direction=True, step=0.5, return_home=False))
    assert max(dist(q) for q in physical(lines0)) > 0.1


def test_static_check(kin, layout):
    """パッチ中心を半径方向に 0.5 mm 外へずらすと、その平均と原点 Δ 換算が出る。"""
    from ring_plotter.deflection import PatchResult, static_check

    results = []
    for p in layout.patches():
        r = math.hypot(p.x, p.y)
        results.append(PatchResult(p, (p.x * (1 + 0.5 / r), p.y * (1 + 0.5 / r)), p.theta, layout.pitch * 1.01,
                                   0.0, 0.0, 2.0, 0.01))
    chk = static_check(kin, layout, results)
    assert abs(chk.radial_mean - 0.5) < 1e-6 and abs(chk.tangential_mean) < 1e-6
    assert abs(chk.pitch_scale - 1.01) < 1e-9
    assert chk.delta_equiv < 0  # 半径が伸びる = Δ が小さい側
    assert 0.1 < abs(chk.delta_equiv) < 0.5


def test_keep_direction(kin):
    """keep_direction なら、並べ替えや巻き数の選択で折れ線を逆向きに描かない。"""
    # 2 本目は、直前の位置から見て終点のほうが近い（逆向きに描くほうが得）
    a = [(50.0, 50.0), (50.0, 70.0)]
    b = [(80.0, 90.0), (50.0, 72.0)]
    lines, _ = convert(kin, [a, b], GcodeOptions(sort=True, keep_direction=False, return_home=False))
    lines_k, _ = convert(kin, [a, b], GcodeOptions(sort=True, keep_direction=True, return_home=False))
    from ring_plotter.simulate import simulate

    def last_stroke_end(ls):
        st = [s for s in simulate(ls, kin).strokes if s.pen_down][-1]
        return st.xy[-1]

    x, y = last_stroke_end(lines)
    assert math.hypot(x - 80, y - 90) < 0.1  # 逆向き: 終点は b の始点
    x, y = last_stroke_end(lines_k)
    assert math.hypot(x - 50, y - 72) < 0.1  # 向きを保つ: 終点は b の終点
