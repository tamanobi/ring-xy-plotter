#!/usr/bin/env python3
"""svg2ring で描いた絵のスキャンを設計の SVG と重ね、線ごとの横ずれ（法線方向）を測る。

1. 設計（SVG を svg2ring と同じ --scale/--rotate/--offset で置いたもの）にスキャンを相似変換で合わせる
   （墨の距離変換に対する chamfer の当てはめ → 測ったずれへの最小二乗で微調整）
2. 設計線上 0.25mm ごとに法線方向の断面を取り、0 に最も近い墨の塊の中心をずれとする。
   1.3mm 以内に別の線がある点（ハッチ、交差）はどの線の墨か決まらないので集計から外す
3. 集計: 線の端からの距離、線の長さ、曲率、機械中心からの半径ごと。線に沿った相関
4. --params と --dot-residuals を渡すと、第 1 層の幾何誤差（レールの楕円 c2/s2 と点打ちの残差。
   どちらも svg2ring は補正していない）から予測したずれを当てはめ、引いた残りも集計する
5. --gcode を渡すと、描いた向き（G-code のペン下ろし区間）に対する左右の偏り、始点・終点からの距離ごとに集計する

  python scripts/compare_drawing_scan.py scan.TIF images/image_2.svg --offset 2 -10
  python scripts/compare_drawing_scan.py scan.TIF images/image_2.svg --offset 2 -10 \\
      --params fit_params.json --dot-residuals fit_residuals.csv --gcode images/image_2.gcode --plot map.png

ずれの符号は設計線の左の法線（折れ線の向きに対して左）が正。--gcode の集計は描いた向きの左が正。
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.kinematics import Kinematics  # noqa: E402
from ring_plotter.svg_input import load_svg, transform  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
STEP = 0.25  # 設計線の標本間隔（mm）
SPAN = 1.2  # 法線方向の断面の片側（mm）
NEIGHBOR = 1.3  # この距離に別の線がある点は外す（mm）


def samples(polylines):
    """折れ線を STEP 間隔に並べ直し、点・弧長・線長・線番号・左法線を返す。"""
    P, T, L, I, N = [], [], [], [], []
    for k, p in enumerate(polylines):
        p = np.asarray(p, float)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        if len(p) < 2 or s[-1] < 1e-6:
            continue
        t = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1] / STEP)) + 1))
        q = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
        d = np.gradient(q, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        P.append(q); T.append(t); L.append(np.full(len(q), s[-1])); I.append(np.full(len(q), k))
        N.append(np.column_stack([-d[:, 1], d[:, 0]]))
    return map(np.concatenate, (P, T, L, I, N))


def to_px(xy, prm, pxmm):
    """機械座標（y 上向き）→ 画像の (列, 行)。prm = (回転, tx, ty, 倍率)、平行移動は y 上向きの mm。"""
    th, tx, ty, sc = prm
    c, s = math.cos(th), math.sin(th)
    x = sc * (c * xy[:, 0] - s * xy[:, 1]) + tx
    y = sc * (s * xy[:, 0] + c * xy[:, 1]) + ty
    return x * pxmm, -y * pxmm


def register(ink, P, pxmm):
    """墨の距離変換に対する chamfer の当てはめ。打ち切り距離を段階的に縮める。"""
    dist = ndimage.distance_transform_edt(~ink) / pxmm

    def cost(prm, pts, clip):
        cx, cy = to_px(pts, prm, pxmm)
        d = ndimage.map_coordinates(dist, [cy, cx], order=1, mode="constant", cval=5.0)
        return np.mean(np.minimum(d, clip) ** 2)

    iy, ix = np.nonzero(ink[::4, ::4])
    c_ink = np.array([ix.mean(), -iy.mean()]) * 4 / pxmm
    c_des = P.mean(axis=0)
    best = None
    for dth in np.radians([-1.0, 0.0, 1.0]):
        prm = np.array([dth, *(c_ink - c_des), 1.0])
        for clip in (3.0, 1.5, 0.8, 0.5):
            prm = optimize.minimize(cost, prm, args=(P[::8], clip), method="Powell",
                                    options=dict(xtol=1e-5, ftol=1e-9)).x
        c = cost(prm, P, 0.5)
        if best is None or c < best[0]:
            best = (c, prm)
    prm = best[1]
    for clip in (0.8, 0.5, 0.4):
        prm = optimize.minimize(cost, prm, args=(P, clip), method="Powell", options=dict(xtol=1e-6, ftol=1e-10)).x
    return prm


def measure(ink, P, N, prm, pxmm):
    """法線方向の断面で、0 に最も近い墨の塊の中心（ずれ）と幅を出す。画像系に回した法線も返す。"""
    th = prm[0]
    Nr = np.column_stack([math.cos(th) * N[:, 0] - math.sin(th) * N[:, 1],
                          math.sin(th) * N[:, 0] + math.cos(th) * N[:, 1]])
    cx, cy = to_px(P, prm, pxmm)
    offs = np.arange(-SPAN, SPAN + 1e-9, 0.02)
    sx = cx[:, None] + offs[None, :] * Nr[:, 0:1] * pxmm
    sy = cy[:, None] - offs[None, :] * Nr[:, 1:2] * pxmm
    prof = ndimage.map_coordinates(ink.astype(np.float32), [sy.ravel(), sx.ravel()], order=1).reshape(sx.shape) > 0.5
    dev = np.full(len(P), np.nan)
    width = np.full(len(P), np.nan)
    for i, row in enumerate(prof):
        if not row.any():
            continue
        lab, n = ndimage.label(row)
        best = None
        for j in range(1, n + 1):
            idx = np.nonzero(lab == j)[0]
            if idx[0] == 0 or idx[-1] == len(row) - 1:
                continue  # 断面の端にかかる塊は幅が分からない
            c = offs[idx].mean()
            if best is None or abs(c) < abs(best[0]):
                best = (c, offs[idx[-1]] - offs[idx[0]] + 0.02)
        if best is not None and abs(best[0]) < 1.0:
            dev[i], width[i] = best
    return dev, width, Nr, cx, cy


def similarity_design(P, Nr, prm, ok):
    """ずれを相似変換の微小量（tx, ty, 回転, 倍率）で表す計画行列。"""
    c, s = math.cos(prm[0]), math.sin(prm[0])
    Q = np.column_stack([c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1]])
    Q = Q - Q[ok].mean(axis=0)
    return np.column_stack([Nr[:, 0], Nr[:, 1], -Nr[:, 0] * Q[:, 1] + Nr[:, 1] * Q[:, 0], (Nr * Q).sum(1)])


def robust_lstsq(A, y, m):
    x = None
    for _ in range(3):
        x, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        r = y - A @ x
        m = m & (abs(r) < 3 * np.sqrt(np.mean(r[m] ** 2)))
    return x


def stats(d, label):
    if len(d) == 0:
        return
    print(f"  {label:34s} n={len(d):6d} 平均 {d.mean():+.3f} rms {np.sqrt(np.mean(d ** 2)):.3f}  "
          f">0.10 {np.mean(abs(d) > 0.10) * 100:5.1f}%  >0.15 {np.mean(abs(d) > 0.15) * 100:5.1f}%  "
          f">0.25 {np.mean(abs(d) > 0.25) * 100:5.1f}%")


def curvature(T, I, N):
    k = np.zeros(len(T))
    for pid in np.unique(I):
        j = np.nonzero(I == pid)[0]
        if len(j) >= 3:
            ang = np.unwrap(np.arctan2(N[j, 1], N[j, 0]))
            k[j] = np.gradient(ang, T[j] + 1e-9)
    return k  # 折れ線の向きに対して左に曲がるのが正（1/mm）


def along_corr(d, T, I, lags=(0.25, 1, 2, 4, 8, 16)):
    out = []
    for lag in lags:
        a, b = [], []
        for pid in np.unique(I):
            j = np.nonzero(I == pid)[0]
            mp = {round(t / STEP): v for t, v in zip(T[j], d[j])}
            for k, v in mp.items():
                w = mp.get(k + round(lag / STEP))
                if w is not None:
                    a.append(v); b.append(w)
        out.append((lag, np.corrcoef(a, b)[0, 1] if len(a) > 10 else math.nan))
    return out


def geometry_prediction(P, N, prm, residual_csv):
    """第 1 層の幾何誤差から、各点のずれ（設計の法線方向）を予測する。楕円の項と点打ち残差の補間の 2 列。"""
    kin = Kinematics.from_params(prm)
    R, d, e = prm["R"], prm["d"], prm["e"]
    c2, s2 = prm.get("c2", 0.0), prm.get("s2", 0.0)

    def fwd(al, be, ell):
        ra = R + (c2 * np.cos(2 * np.radians(al)) + s2 * np.sin(2 * np.radians(al)) if ell else 0)
        rb = R + (c2 * np.cos(2 * np.radians(be)) + s2 * np.sin(2 * np.radians(be)) if ell else 0)
        a, b = np.radians(al - 90), np.radians(be - 90)
        ax, ay = ra * np.cos(a), ra * np.sin(a)
        ux, uy = rb * np.cos(b) - ax, rb * np.sin(b) - ay
        c = np.hypot(ux, uy)
        ux, uy = ux / c, uy / c
        return np.column_stack([ax + d * ux - e * uy, ay + d * uy + e * ux])

    ab, prev = [], None
    for x, y in P:
        try:
            a, b = kin.inverse(x, y, near_alpha=prev)
            prev = a
        except ValueError:
            a, b = math.nan, math.nan
        ab.append((a, b))
    ab = np.array(ab)
    e_ell = fwd(ab[:, 0], ab[:, 1], True) - fwd(ab[:, 0], ab[:, 1], False)
    cols = [(N * e_ell).sum(1)]
    if residual_csv:
        # 点打ちのスキャンの座標は機械座標とほぼ同じ向き（fit_params の theta が 270° 前後）として扱う
        rows = [r for r in csv.DictReader(open(residual_csv, encoding="utf-8-sig")) if r["used"] == "1"]
        dab = np.array([[float(r["alpha"]), float(r["beta"])] for r in rows])
        dres = np.array([[float(r["dx"]), float(r["dy"])] for r in rows])
        dxy = fwd(dab[:, 0], dab[:, 1], False)
        e_res = np.column_stack([griddata(dxy, dres[:, k], P, method="linear") for k in (0, 1)])
        cols.append((N * e_res).sum(1))
    return np.column_stack(cols)


def gcode_runs(path, kin):
    """G-code のペン下ろし区間（X=α、Y=β）を機械座標の折れ線にする。"""
    runs, cur, pos, down = [], None, (0.0, 0.0), False
    for ln in open(path, encoding="ascii", errors="replace"):
        ln = ln.split(";")[0].strip()
        if ln.startswith("M3"):
            down, cur = True, [pos]
        elif ln.startswith("M5"):
            if down and cur and len(cur) > 1:
                runs.append(np.array([kin.forward(a, b) for a, b in cur]))
            down = False
        elif ln.startswith(("G0", "G1")):
            mx, my = re.search(r"X(-?[\d.]+)", ln), re.search(r"Y(-?[\d.]+)", ln)
            pos = (float(mx.group(1)) if mx else pos[0], float(my.group(1)) if my else pos[1])
            if down:
                cur.append(pos)
    return runs


def plot(path, scan, cx, cy, panels, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = 4
    small = np.array(scan.reduce(step))
    fig, axs = plt.subplots(1, len(panels), figsize=(8 * len(panels), 11.5), squeeze=False)
    sc = None
    for ax, (title, val) in zip(axs[0], panels):
        ax.imshow(small, cmap="gray", vmin=-300, vmax=255, extent=(0, small.shape[1] * step, small.shape[0] * step, 0))
        sc = ax.scatter(cx[m], cy[m], c=val[m], cmap="coolwarm", vmin=-0.25, vmax=0.25, s=1.5, lw=0)
        big = m & (abs(val) > 0.2)
        ax.scatter(cx[big], cy[big], facecolors="none", edgecolors="k", s=12, lw=0.4)
        ax.set_xlim(cx[m].min() - 100, cx[m].max() + 100)
        ax.set_ylim(cy[m].max() + 100, cy[m].min() - 100)
        ax.set_axis_off()
        ax.set_title(title)
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=0.5, label="normal deviation [mm] (+ = left of the design polyline; circled |d|>0.2)")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    print(f"図: {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan", help="スキャン画像（TIF/PNG。白地に黒い線）")
    ap.add_argument("svg", help="描いた SVG")
    ap.add_argument("--scale", type=float, default=1.0, help="svg2ring に渡した --scale")
    ap.add_argument("--rotate", type=float, default=0.0, help="svg2ring に渡した --rotate")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="svg2ring に渡した --offset")
    ap.add_argument("--mirror", action="store_true", help="svg2ring に渡した --mirror")
    ap.add_argument("--dpi", type=float, default=None, help="解像度。省略時は画像のメタデータ")
    ap.add_argument("--threshold", type=int, default=128, help="墨とみなす濃さ（0〜255）")
    ap.add_argument("--params", help="描いたときの fit_params.json（幾何誤差の予測と --gcode に使う）")
    ap.add_argument("--dot-residuals", help="同じ点打ちの fit_residuals.csv（幾何誤差の予測に加える）")
    ap.add_argument("--gcode", help="描いた G-code（描いた向きごとの集計。--params が要る）")
    ap.add_argument("-o", "--output", help="点ごとの CSV（既定: スキャン名 + _compare.csv）")
    ap.add_argument("--plot", help="ずれの地図 PNG")
    args = ap.parse_args(argv)
    if args.gcode and not args.params:
        ap.error("--gcode には --params が要ります")
    if args.dot_residuals and not args.params:
        ap.error("--dot-residuals には --params が要ります")

    scan = Image.open(args.scan)
    dpi = float(args.dpi or (scan.info.get("dpi") or (600, 600))[0])
    pxmm = dpi / 25.4
    scan = scan.convert("L")
    ink = np.array(scan) < args.threshold

    drawing = load_svg(args.svg, step=STEP)
    P, T, L, I, N = samples(transform(drawing.polylines, args.scale, args.rotate, *args.offset, args.mirror))
    print(f"{args.scan}: {ink.shape[1]}×{ink.shape[0]} px、{dpi:g} dpi。設計 {len(np.unique(I))} 本、{len(P)} 点")

    prm = register(ink, P, pxmm)
    dev, width, Nr, cx, cy = measure(ink, P, N, prm, pxmm)
    tree = cKDTree(P)
    amb = np.array([(I[np.array(nb)] != I[i]).any() for i, nb in enumerate(tree.query_ball_point(P, NEIGHBOR))])
    ok = ~amb & np.isfinite(dev) & ~(width > 1.1)
    A = similarity_design(P, Nr, prm, ok)
    x = robust_lstsq(A, dev, ok.copy())
    res = dev - A @ x
    print(f"位置合わせ: 回転 {math.degrees(prm[0] + x[2]):+.3f}°、倍率 {prm[3] * (1 + x[3]):.5f}")
    print(f"測れた点 {ok.sum()}（別の線が {NEIGHBOR:g}mm 以内で除外 {amb.sum()}）、墨の幅の中央値 {np.nanmedian(width[ok]):.2f}mm、"
          f"近くに墨がない設計点 {(~amb & np.isnan(dev)).sum()}")

    print("法線方向のずれ（相似変換を除いた後）")
    stats(res[ok], "全体")
    te = np.minimum(T, L - T)
    for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, math.inf)]:
        stats(res[ok & (te >= lo) & (te < hi)], f"近いほうの端から {lo}〜{hi} mm")
    for lo, hi in [(0, 3), (3, 10), (10, 30), (30, math.inf)]:
        stats(res[ok & (L >= lo) & (L < hi) & (te >= 1.5)], f"中央、線の長さ {lo}〜{hi} mm")
    kappa = curvature(T, I, N)
    for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 0.6), (0.6, math.inf)]:
        stats(res[ok & (abs(kappa) >= lo) & (abs(kappa) < hi) & (te >= 1.5)], f"中央、曲率 {lo}〜{hi} /mm")
    r = np.hypot(P[:, 0], P[:, 1])
    for lo, hi in [(0, 40), (40, 80), (80, 120), (120, math.inf)]:
        stats(res[ok & (r >= lo) & (r < hi)], f"機械中心から {lo}〜{hi} mm")
    print("  線に沿った相関: " + "、".join(f"{lag:g}mm {c:.2f}" for lag, c in along_corr(res[ok], T[ok], I[ok])))

    final, m = res, ok
    geo = None
    if args.params:
        fp = json.load(open(args.params, encoding="utf-8-sig"))
        pred = geometry_prediction(P, N, fp, args.dot_residuals)
        m = ok & np.isfinite(pred).all(axis=1)
        names = ["楕円"] + (["点打ち残差"] if args.dot_residuals else [])
        print(f"第 1 層の幾何誤差（{'＋'.join(names)}）による予測。予測できた点 {m.sum()}")
        base = res - A @ robust_lstsq(A, res, m.copy())
        g = robust_lstsq(np.column_stack([A, pred]), res, m.copy())[4:]
        print(f"  当てはめた係数（1 なら予測どおり）: " + "、".join(f"{n} {c:.2f}" for n, c in zip(names, g)))
        geo = np.nan_to_num(pred.sum(axis=1))
        final = res - geo
        final = final - A @ robust_lstsq(A, final, m.copy())
        stats(base[m], "予測を引く前")
        stats(final[m], "予測（係数 1）を引いた後")
        print("  引いた後の線に沿った相関: "
              + "、".join(f"{lag:g}mm {c:.2f}" for lag, c in along_corr(final[m], T[m], I[m])))

    if args.gcode:
        kin = Kinematics.from_params(json.load(open(args.params, encoding="utf-8-sig")))
        segs, tang = [], []
        for run in gcode_runs(args.gcode, kin):
            for a, b in zip(run[:-1], run[1:]):
                v = b - a
                n = np.hypot(*v)
                if n > 1e-6:
                    segs.append((a + b) / 2); tang.append(v / n)
        dist, j = cKDTree(np.array(segs)).query(P)
        tg = np.array(tang)[j]
        cos = (tg * np.column_stack([N[:, 1], -N[:, 0]])).sum(1)  # 設計の向き (Ny, -Nx) と描いた向き
        sgn = np.sign(cos)
        use = m & (dist < 1.5) & (abs(cos) > 0.8)
        left = final * sgn
        ts = np.where(sgn > 0, T, L - T)
        print(f"描いた向きに対する集計（G-code と対応がついた点 {use.sum()}。平均は描いた向きの左が正）")
        stats(left[use], "全体")
        for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6)]:
            stats(left[use & (ts >= lo) & (ts < hi)], f"始点から {lo}〜{hi} mm")
            stats(left[use & (L - ts >= lo) & (L - ts < hi)], f"終点まで {lo}〜{hi} mm")
        kt = kappa * sgn
        for lo, hi in [(-math.inf, -0.3), (-0.3, -0.05), (-0.05, 0.05), (0.05, 0.3), (0.3, math.inf)]:
            stats(left[use & (kt >= lo) & (kt < hi)], f"曲率（左に曲がるのが正）{lo:+g}〜{hi:+g}")
        head = np.degrees(np.arctan2(tg[:, 1], tg[:, 0])) % 360
        for b in range(0, 360, 45):
            stats(left[use & (head >= b) & (head < b + 45)], f"進む向き {b}〜{b + 45}°")

    out = args.output or args.scan.rsplit(".", 1)[0] + "_compare.csv"
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["line", "s", "length", "x", "y", "nx", "ny", "dev", "width", "used", "geometry", "remaining"])
        for k in range(len(P)):
            w.writerow([I[k], f"{T[k]:.2f}", f"{L[k]:.2f}", f"{P[k, 0]:.3f}", f"{P[k, 1]:.3f}", f"{N[k, 0]:.4f}",
                        f"{N[k, 1]:.4f}", f"{res[k]:.4f}", f"{width[k]:.3f}", int(m[k]),
                        f"{geo[k]:.4f}" if geo is not None else "", f"{final[k]:.4f}"])
    print(f"点ごとの値: {out}")
    if args.plot:
        panels = [("measured (similarity removed)", res)]
        if geo is not None:
            panels += [("predicted from layer 1", geo), ("remaining", final)]
        plot(args.plot, scan, cx, cy, panels, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
