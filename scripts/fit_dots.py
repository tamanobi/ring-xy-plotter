#!/usr/bin/env python3
"""手順3: スキャン画像の点を検出し、運動学のパラメータを最小二乗で合わせる。

  pip install numpy scipy pillow matplotlib
  python fit_dots.py scan.tif dots.csv
  python fit_dots.py scan.tif dots.csv --ellipse     # レールの楕円成分も合わせる

モデル（単位はmmと度）:
  Aのピン = R·(cos α', sin α')、Bのピン = R·(cos β', sin β')
  ペン先 = Aのピン + d·u + e·n   （u: A→Bの単位ベクトル、n: uに垂直）
  eには、Δの原点ずれと棒の横オフセットがまとめて入る（区別できないため）。
出力: fit_params.json、fit_residuals.csv、fit_residuals.png
"""
import argparse
import csv
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares

Image.MAX_IMAGE_PIXELS = None


def detect(path, dpi, thresh, max_dot, min_sep=3.0):
    """点の候補（紙の座標、mm、y 上向き）と紙の高さ。

    min_sep 未満の距離に別の候補がある点は落とす。点打ちの格子は 8 mm 以上離れているので、近い候補どうしは
    試し書きや汚れが点に割れたもの（10 月中旬: 紙の隅の 28 個の塊で粗い探索が外れた）。
    """
    im = Image.open(path)
    dpi = dpi or float(im.info.get("dpi", (600, 600))[0])
    a = np.array(im.convert("L")) < thresh
    lab, n = ndi.label(a, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    area = ndi.sum(a, lab, idx)
    com = np.array(ndi.center_of_mass(a, lab, idx))
    sl = ndi.find_objects(lab)
    size = np.array([max(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl])
    mm = 25.4 / dpi
    k = dpi / 600.0
    px = max_dot / mm  # 点の直径の上限（画素）
    ok = (area >= 8 * k * k) & (area <= 0.8 * px * px) & (size <= 1.3 * px)
    h = a.shape[0] * mm
    # 紙を上から見た向き（yを上向き）に直す
    pts = np.c_[com[ok, 1] * mm, h - com[ok, 0] * mm]
    if len(pts) > 1 and min_sep > 0:
        dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        np.fill_diagonal(dist, np.inf)
        isolated = dist.min(1) >= min_sep
        if (~isolated).any():
            print(f"近すぎる候補 {int((~isolated).sum())} 個を落とした（{min_sep:g} mm 以内に別の候補。試し書きや汚れ）")
        pts = pts[isolated]
    return pts, h


def forward(p, s, al, be, ellipse=False):
    R, d, e, th, tx, ty = p[:6]
    ra = rb = R
    if ellipse:
        c2, s2 = p[6:8]
        ra = R + c2 * np.cos(2 * al) + s2 * np.sin(2 * al)
        rb = R + c2 * np.cos(2 * be) + s2 * np.sin(2 * be)
    a, b = s * al + th, s * be + th
    ax, ay = ra * np.cos(a), ra * np.sin(a)
    ux, uy = rb * np.cos(b) - ax, rb * np.sin(b) - ay
    c = np.hypot(ux, uy)
    ux, uy = ux / c, uy / c
    return np.c_[ax + d * ux - e * s * uy + tx, ay + d * uy + e * s * ux + ty]


def match(q, pts, tol=None, percentile=None):
    """指令点 q と検出点 pts の対応づけ（相互最近傍）。

    指令点から見た最近傍と、検出点から見た最近傍が一致する組だけを使う。ゴミや試し書きが点の塊として
    検出されても、塊 1 つにつき指令点 1 つしか引き寄せられない（10 月中旬: 紙の隅の 28 個の塊で粗い探索が
    外れた）。tol なら距離がそれ未満、percentile なら相互最近傍の中で距離がその分位以下のものを使う。
    """
    dist = np.linalg.norm(q[:, None] - pts[None], axis=2)
    j = dist.argmin(1)
    i = dist.argmin(0)
    dm = dist.min(1)
    mutual = i[j] == np.arange(len(q))
    use = mutual.copy()
    if tol is not None:
        use &= dm < tol
    if percentile is not None and mutual.any():
        use &= dm <= np.percentile(dm[mutual], percentile)
    return j, use


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan")
    ap.add_argument("csv")
    ap.add_argument("--dpi", type=float, help="画像に解像度の情報がない場合に指定")
    ap.add_argument("--thresh", type=int, default=128, help="二値化の閾値")
    ap.add_argument("--max-dot", type=float, default=1.2,
                    help="点の直径の上限（mm）。にじむペンでは大きく、ゴミを拾うなら小さくする")
    ap.add_argument("--min-sep", type=float, default=3.0,
                    help="これより近くに別の候補がある点は落とす（mm、試し書きや汚れの塊。0 で無効）")
    ap.add_argument("--R0", type=float, default=230.0, help="ピン半径の初期値")
    ap.add_argument("--ellipse", action="store_true", help="レールの楕円成分も合わせる")
    ap.add_argument("--init", help="前回の fit_params.json。その解から始めて粗い探索を飛ばす（設置が大きく変わっていないとき）")
    ap.add_argument("-o", "--output", default="fit")
    args = ap.parse_args()

    pts, height = detect(args.scan, args.dpi, args.thresh, args.max_dot, args.min_sep)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    al = np.radians([float(r["alpha"]) for r in rows])
    be = np.radians([float(r["beta"]) for r in rows])
    print(f"点の候補 {len(pts)}個 / 指令 {len(rows)}点")

    if args.init:
        pj = json.load(open(args.init, encoding="utf-8"))
        s = int(pj.get("sign", 1))
        p = np.array([pj["R"], pj["d"], pj["e"], np.radians(pj["theta_deg"]), pj["tx"], pj["ty"]])
    else:
        # 1) 向きと位置の粗い探索（R, d は初期値に固定）
        c0 = np.median(pts, axis=0)
        best = None
        for s in (1, -1):
            for th0 in np.radians(np.arange(0, 360, 15)):
                for e0 in (-10, 0, 10):
                    p = np.array([args.R0, args.R0, e0, th0, c0[0], c0[1]])
                    for _ in range(5):
                        q = forward(p, s, al, be)
                        j, use = match(q, pts, percentile=80)
                        if use.sum() < 6:
                            break
                        f = lambda x: (forward(np.r_[p[:2], x], s, al, be)[use] - pts[j[use]]).ravel()
                        p[2:] = least_squares(f, p[2:]).x
                    q = forward(p, s, al, be)
                    j, use = match(q, pts, tol=1.0)
                    dm = np.linalg.norm(q - pts[j], axis=1)
                    # 1 mm 以内で対応した点が多いものを採り、同数なら残差で決める
                    score = (-int(use.sum()), float(np.median(dm)))
                    if best is None or score < best[0]:
                        best = (score, s, p.copy())
        _, s, p = best

    # 2) 全パラメータを合わせる（外れ値は対応づけから外す）
    if args.ellipse:
        p = np.r_[p, 0.0, 0.0]
    for _ in range(8):
        q = forward(p, s, al, be, args.ellipse)
        j, use = match(q, pts, tol=0.8)
        if use.sum() < len(p) + 2:
            print(f"対応づけが {use.sum()} 点しかありません（指令 {len(rows)} 点）。粗い探索が外れています。"
                  f"--init に前回の fit_params.json を渡すか、--max-dot と --thresh を見直してください")
            raise SystemExit(1)
        f = lambda x: (forward(x, s, al, be, args.ellipse)[use] - pts[j[use]]).ravel()
        r = least_squares(f, p, x_scale="jac")
        p = r.x
    q = forward(p, s, al, be, args.ellipse)
    j, use = match(q, pts, tol=0.8)
    res = pts[j] - q
    rn = np.linalg.norm(res, axis=1)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * (r.fun**2).sum() / max(len(r.fun) - len(p), 1)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))

    R, d, e = p[:3]
    L = float(np.hypot(d, e))
    print(f"対応づけ {use.sum()}/{len(rows)}点、残差RMS {np.sqrt((rn[use]**2).mean()):.3f}mm、最大 {rn[use].max():.3f}mm")
    if use.sum() < 0.6 * len(rows):
        print(f"  警告: 対応づけが {use.sum() / len(rows) * 100:.0f}% しかありません。結果は信用できません")
    print(f"  R = {R:.3f} ± {sd[0]:.3f} mm")
    print(f"  d = {d:.3f} ± {sd[1]:.3f} mm")
    print(f"  e = {e:.3f} ± {sd[2]:.3f} mm")
    print(f"  ペンまでの距離 = {L:.3f} mm、γ = {np.degrees(np.arctan2(e, d)):.3f}°、中心の穴の半径 = {abs(R - L):.3f} mm")
    if args.ellipse:
        print(f"  楕円成分（半径の振幅）= {np.hypot(p[6], p[7]):.3f} mm、長軸の向き α = {np.degrees(np.arctan2(p[7], p[6])) / 2 % 180:.1f}°")
    for k in np.where(~use)[0]:
        print(f"  見つからない点: α={rows[k]['alpha']} β={rows[k]['beta']}（予測位置 x={q[k,0]:.1f} y={height-q[k,1]:.1f}mm、画像左上から）")

    out = {"R": R, "d": d, "e": e, "sign": s, "theta_deg": float(np.degrees(p[3])),
           "tx": p[4], "ty": p[5], "rms_mm": float(np.sqrt((rn[use]**2).mean()))}
    if args.ellipse:
        out.update(c2=p[6], s2=p[7])
    json.dump(out, open(args.output + "_params.json", "w"), indent=2)
    with open(args.output + "_residuals.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["alpha", "beta", "pred_x", "pred_y", "dx", "dy", "used"])
        for k, row in enumerate(rows):
            w.writerow([row["alpha"], row["beta"], f"{q[k,0]:.3f}", f"{q[k,1]:.3f}",
                        f"{res[k,0]:.3f}", f"{res[k,1]:.3f}", int(use[k])])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlibがないので図は省略しました")
        return
    scale = 50  # 残差の拡大率
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.plot(q[use, 0], q[use, 1], "o", ms=3, color="#8a8f98", lw=0)
    for k in np.where(use)[0]:
        ax.plot([q[k, 0], q[k, 0] + scale * res[k, 0]], [q[k, 1], q[k, 1] + scale * res[k, 1]],
                color="#2a6fdb", lw=1.5, solid_capstyle="round")
    ax.plot(q[~use, 0], q[~use, 1], "x", ms=7, color="#444")
    x0, y0 = q[:, 0].min(), q[:, 1].min() - 12
    ax.plot([x0, x0 + scale * 0.2], [y0, y0], color="#2a6fdb", lw=1.5)
    ax.text(x0 + scale * 0.2 + 2, y0, "0.2 mm", va="center", fontsize=9, color="#444")
    ax.set_aspect("equal")
    ax.grid(color="#e6e6e6", lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(colors="#666", labelsize=8)
    ax.set_xlabel("x (mm)", color="#666", fontsize=9)
    ax.set_ylabel("y (mm)", color="#666", fontsize=9)
    ax.set_title(f"Fit residuals x{scale}  (RMS {out['rms_mm']:.3f} mm, x = dot not found)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(args.output + "_residuals.png")


if __name__ == "__main__":
    main()
