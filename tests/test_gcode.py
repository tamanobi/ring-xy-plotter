import math

import numpy as np
import pytest

from ring_plotter.gcode import GcodeOptions, TwistError, choose_wrap, convert, subdivide
from ring_plotter.kinematics import Kinematics
from ring_plotter.simulate import simulate


@pytest.fixture
def kin():
    return Kinematics.from_params(dmin=89, dmax=179)


def square(size=100, cx=15, cy=10):
    h = size / 2
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]


def test_subdivide():
    pts = subdivide([(0, 0), (3.5, 0)], 1.0)
    assert len(pts) == 5
    assert pts[-1] == (3.5, 0)
    assert all(abs((b[0] - a[0]) - 0.875) < 1e-9 for a, b in zip(pts, pts[1:]))


def test_convert_square_roundtrip(kin):
    lines, stats = convert(kin, [square()], GcodeOptions())
    assert lines[0] == "G90" and lines[1] == "M5"
    assert stats.pen_downs == 1
    assert stats.segments == 400
    assert abs(stats.draw_mm - 400) < 1e-6
    assert lines[-1] == "G1 X0.000 Y88.000 F200"
    sim = simulate(lines, kin)
    assert not sim.problems
    # 描いた線が正方形の上に乗っている
    for st in sim.strokes:
        if not st.pen_down:
            continue
        for x, y in st.xy:
            dx, dy = abs(x - 15), abs(y - 10)
            assert abs(max(dx, dy) - 50) < 0.02


def test_lead_in_approaches_along_the_line(kin):
    """リードイン: 始点の 3 mm 手前からペンを上げたまま線の向きに入る。短い移動には入れない。"""
    a = [(-60.0, 40.0), (-40.0, 40.0)]  # +x 向き
    b = [(-40.0, 42.0), (-60.0, 42.0)]  # 隣の線、逆向き（2 mm の横移動）
    far = [(30.0, -50.0), (30.0, -30.0)]
    opt = GcodeOptions(lead_in=3.0, lead_in_min_travel=5.0, sort=False, keep_direction=True)
    lines, stats = convert(kin, [a, b, far], opt)
    assert stats.pen_downs == 3
    assert stats.lead_ins == 2  # a（原点から）と far。b は 2 mm の移動なので入れない
    sim = simulate(lines, kin)
    assert not sim.problems
    ups = [st for st in sim.strokes if not st.pen_down]
    # a の直前のペン上げ移動は始点 (-60, 40) で終わり、最後の区間は (-63, 40) → (-60, 40)
    before_a = ups[0]
    x, y = before_a.xy[-1]
    assert abs(x - -60) < 0.05 and abs(y - 40) < 0.05
    # 最後の 3 mm の向きが線の向き（+x）
    xs = [p[0] for p in before_a.xy if abs(p[1] - 40) < 0.1 and -63.1 < p[0] < -59.9]
    assert min(xs) < -62.9
    # 描いた線そのものはリードインの影響を受けない
    downs = [st for st in sim.strokes if st.pen_down]
    assert all(abs(p[1] - 40) < 0.02 for p in downs[0].xy)


def test_lead_in_skipped_when_swing_is_small(kin):
    """補正ありで、予測した始点の振りが小さければ入れない。"""
    from ring_plotter.deflection import DeflectionModel

    model = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt")
    a = [(-60.0, 40.0), (-40.0, 40.0)]
    opt_big = GcodeOptions(lead_in=3.0, lead_in_min_swing=10.0, deflection=model, sort=False, keep_direction=True)
    _, s_big = convert(kin, [a], opt_big)
    opt_small = GcodeOptions(lead_in=3.0, lead_in_min_swing=0.0, deflection=model, sort=False, keep_direction=True)
    _, s_small = convert(kin, [a], opt_small)
    assert s_big.lead_ins == 0 and s_small.lead_ins == 1


def test_lead_out_extends_the_stroke(kin):
    """リードアウト: 終点を最後の向きに MM だけペンを下ろしたまま延ばす。"""
    a = [(-60.0, 40.0), (-40.0, 40.0)]
    lines, stats = convert(kin, [a], GcodeOptions(lead_out=0.5, sort=False, keep_direction=True))
    assert stats.lead_outs == 1 and stats.lead_out_skipped == 0
    down = [st for st in simulate(lines, kin).strokes if st.pen_down][0]
    x, y = down.xy[-1]
    assert abs(x - -39.5) < 0.02 and abs(y - 40) < 0.02
    assert abs(stats.draw_mm - 20.5) < 1e-6
    # 届かない向きに延びる線（中心の穴へ）では入れない
    into_hole = [(-10.0, 0.0), (-1.6, 0.0)]
    _, s2 = convert(kin, [into_hole], GcodeOptions(lead_out=0.5, sort=False, keep_direction=True, step=0.2))
    assert s2.lead_outs == 0 and s2.lead_out_skipped == 1


def test_speed_factor_reduces_compensation_at_line_ends(kin):
    """速度の係数ありでは、加減速区間（線の両端）で補正量が中央より小さい。中央は f(v) どおりの倍率。"""
    from ring_plotter.deflection import DeflectionModel

    a = [(-60.0, 40.0), (-40.0, 40.0)]
    # 巡航 30 mm/s（= v_ref、fmax は効かない値）なら中央は f = 1 で、両端だけ落ちる
    from ring_plotter.timing import Motion

    kw = dict(sort=False, keep_direction=True, step=0.5, speed=30.0, fmax=3000.0, lead_in=3.0, lead_in_min_travel=0.0,
              motion=Motion(accel=100.0))
    m0 = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt")
    m1 = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", c_speed=0.3)
    off = []
    for m in (m0, m1):
        lines, stats = convert(kin, [a], GcodeOptions(deflection=m, **kw))
        assert stats.lead_ins == 1
        down = [st for st in simulate(lines, kin).strokes if st.pen_down][0]
        off.append(np.array([p[1] - 40.0 for p in down.xy]))  # 線に直交する補正量
    o0, o1 = off
    n = len(o0)
    assert np.abs(o0[n // 2]) > 0.05
    assert abs(o1[n // 2] / o0[n // 2] - 1.0) < 0.01
    # 終点（減速して止まる）と始点（止まった状態から加速）では小さい。速度 0 で (1 − c) = 0.7 に向かう
    assert 0.6 < abs(o1[-1] / o0[-1]) < 0.95
    assert 0.6 < abs(o1[1] / o0[1]) < 0.95
    # 速度 15 mm/s なら中央も f = 0.85
    kw15 = dict(kw, speed=15.0)
    l0, _ = convert(kin, [a], GcodeOptions(deflection=m0, **kw15))
    l1, _ = convert(kin, [a], GcodeOptions(deflection=m1, **kw15))
    mid = lambda ls: [st for st in simulate(ls, kin).strokes if st.pen_down][0].xy[n // 2][1] - 40.0
    assert abs(mid(l1) / mid(l0) - 0.85) < 0.02


def test_dry_run_keeps_pen_up_and_marks_crosses(kin):
    """脱調の試験: 折れ線はペン上げのまま、前後に (X, Y) の十字だけをペンを下ろして描く。"""
    pls = [[(-60.0, 40.0), (-40.0, 40.0)], [(30.0, -50.0), (30.0, -30.0)]]
    lines, stats = convert(kin, pls, GcodeOptions(dry_run=(0.0, -80.0), sort=False, keep_direction=True))
    assert lines.count("M3 S110") == 4  # 十字 2 本 × 前後
    sim = simulate(lines, kin)
    assert not sim.problems
    downs = [st for st in sim.strokes if st.pen_down]
    assert len(downs) == 4
    for st in downs:
        for x, y in st.xy:
            assert abs(x - 0.0) < 2.1 and abs(y - -80.0) < 2.1
    # 本体の移動は同じ経路をペン上げで通る（描画距離は十字 4 本ぶんだけ）
    assert abs(sim.draw_mm - 16.0) < 0.1
    assert any("脱調の試験" in n for n in stats.notes)


def test_line_through_center_region_has_no_branch_jump():
    """中心の近く（両方の枝が有効な領域）を通る線で α が跳ばない（紙の上の輪にならない）。"""
    kin = Kinematics.from_params({"R": 230.705, "d": 230.340, "e": 10.852}, dmin=89, dmax=179)
    poly = [(-30.0, 3.0), (30.0, 3.0)]  # 半径 3 mm で中心のそばを横切る
    lines, stats = convert(kin, [poly], GcodeOptions(step=1.0, return_home=False))
    assert stats.pen_downs == 1 and stats.branch_splits == 0
    sim = simulate(lines, kin)
    down = [st for st in sim.strokes if st.pen_down]
    assert len(down) == 1
    for x, y in down[0].xy:
        assert abs(y - 3.0) < 0.05 and -30.1 < x < 30.1
    # 1 セグメントあたりの α の変化が 180° 近くになる箇所がない
    prev = None
    for ln in lines:
        if ln.startswith("G1 X"):
            a = float(ln.split()[1][1:])
            if prev is not None:
                assert abs(a - prev) < 90, (prev, a)
            prev = a


def test_svg2ring_output_is_ascii(tmp_path):
    # GRBL は 0x80 以上のバイトをリアルタイムコマンド（0x84 = セーフティドアなど）として
    # 解釈するので、コメント行も含めて G-code は ASCII だけでなければならない
    from ring_plotter.svg2ring import main

    out = tmp_path / "t.gcode"
    assert main(["examples/test_pattern.svg", "-o", str(out), "--no-preview"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.isascii()
    assert text.startswith("; svg2ring ")


def test_feed_is_slower_near_center(kin):
    """中心に近いほど α の変化が速いので F は上限で頭打ち、遠くでは下がる。"""
    far = [(100, 0), (100, 10)]
    near = [(3, 0), (0, 3)]
    opt = GcodeOptions(fmax=100000, sort=False)
    _, s_far = convert(kin, [far], opt)
    lines_far, _ = convert(kin, [far], opt)
    lines_near, _ = convert(kin, [near], opt)
    f = lambda ls: [float(l.split("F")[1]) for l in ls if l.startswith("G1") and "F1000" not in l and "F200" not in l]
    assert max(f(lines_far)) < max(f(lines_near))


def test_unreachable_clip_splits(kin):
    # 中心を横切る線は穴で2本に分かれる
    lines, stats = convert(kin, [[(-20, 0), (20, 0)]], GcodeOptions(step=0.5))
    assert stats.paths_out == 2
    assert stats.points_dropped >= 1
    assert stats.pen_downs == 2


def test_unreachable_error(kin):
    from ring_plotter.kinematics import Unreachable

    with pytest.raises(Unreachable):
        convert(kin, [[(-20, 0), (20, 0)]], GcodeOptions(unreachable="error"))


def test_choose_wrap_prefers_near_start():
    ang = [(350.0, 450.0), (355.0, 455.0), (10.0 + 360, 470.0)]  # αは連続（350→370）
    k, rev = choose_wrap(ang, cur_alpha=0.0, twist=400, home=88)
    assert k == 1 and rev is False  # k=1 で α は −10〜10°。始点と終点が同じ距離なら正向き
    k, rev = choose_wrap(ang, cur_alpha=12.0, twist=400, home=88)
    assert k == 1 and rev is True
    k, rev = choose_wrap(ang, cur_alpha=355.0, twist=400, home=88)
    assert k == 0 and rev is False  # 直前が 355° なら巻かずにそのまま


def test_choose_wrap_limits_beta_too():
    ang = [(a, a + 170.0) for a in range(300, 360, 5)]  # α 300〜355（α だけなら可）、β 470〜525
    k, rev = choose_wrap(ang, cur_alpha=300.0, twist=400, home=88)
    assert k == 1  # β − 88 が 400 を超えるので、1周戻した側（α −60〜−5）を使う
    k, rev = choose_wrap(ang, cur_alpha=300.0, twist=400, home=88 + 40)
    assert k == 0  # B の原点が 40° 進めば収まる


def test_choose_wrap_twist_error():
    ang = [(a, a + 100) for a in range(0, 900, 10)]
    with pytest.raises(TwistError):
        choose_wrap(ang, 0.0, 400)


def test_twist_limit_respected(kin):
    """円を何本も描いても |α| が上限を超えない。"""
    circles = []
    for r in (30, 60, 90, 120):
        n = 90
        circles.append([(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)])
    lines, stats = convert(kin, circles, GcodeOptions(twist=400))
    assert -400 <= stats.alpha_min and stats.alpha_max <= 400
    assert -400 <= stats.beta_min - 88 and stats.beta_max - 88 <= 400
    sim = simulate(lines, kin, twist=400)
    assert not sim.problems


def test_sort_reduces_travel(kin):
    pls = [[(50 + 10 * i, 0), (50 + 10 * i, 20)] for i in range(10)][::-1]
    _, s1 = convert(kin, pls, GcodeOptions(sort=False))
    _, s2 = convert(kin, pls, GcodeOptions(sort=True))
    assert s2.travel_deg <= s1.travel_deg
