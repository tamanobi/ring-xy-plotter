#!/usr/bin/env python3
"""gen_hatch_patches.py で描いたパッチのスキャンから、パッチごとの符号付きのずれ D と距離定数 λ を測る。

隣り合う線の間隔が p − 2D、p + 2D の交互になることを使う（D は進行方向の左へのずれ）。
パッチは連結成分として見つけ、設計位置（レイアウト JSON）に回転と平行移動で合わせる。
向きの目印で 180° の回転も判別する。

  python scripts/measure_hatch_patches.py scan.tif --layout examples/hatch_patches_layout.json
  python scripts/measure_hatch_patches.py scan.tif --layout ... -o patches.csv --plot patches.png

出力: パッチごとの (場所, 向き, D, D_mid, λ, rms) を画面と CSV に。--plot で間隔の実測とフィットの図。
CSV は fit_deflection.py に渡す。
最後に紙 1 枚の静的な健康診断も出す: パッチ中心の設計位置からの半径方向のずれ（原点 Δ のずれに換算）と
間隔の倍率（紙の伸縮やスキャナの倍率）。紙ごとに追うと、原点合わせやペンの滑りの変化が分かる。
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.deflection import (CSV_FIELDS, LINE_WIDTH, DeflectionModel, Layout, gap_model, measure_scan,  # noqa: E402
                                     predict_D, predict_along, result_row, saturated, start_excursion, start_hooks,
                                     static_check)
from ring_plotter.kinematics import Kinematics  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def plot(results, lay, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.ok]
    if not ok:
        return
    ncol = len(lay.angles)
    nrow = math.ceil(len(ok) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.2 * nrow), squeeze=False, sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, ok):
        ax.set_visible(True)
        gaps = r.t[:, 1:] - r.t[:, :-1]
        for k in range(gaps.shape[1]):
            col = "#1f77b4" if k % 2 == 0 else "#d62728"
            ax.plot(r.s, gaps[:, k], color=col, lw=0.6, alpha=0.6)
        ss = np.linspace(r.s.min(), r.s.max(), 100)
        for k, col in ((0, "#1f77b4"), (1, "#d62728")):
            ax.plot(ss, gap_model(ss, np.full(100, k), lay.length, r.pitch, r.D, r.lam, r.start), color=col, lw=1.5)
        ax.set_title(f"({r.patch.site_x:g},{r.patch.site_y:g}) {r.patch.theta:g}deg  D={r.D:+.3f} lam={r.lam:.1f}",
                     fontsize=7)
        span = max(0.6, 2 * max(abs(r.D) for r in ok) + 0.2)
        ax.set_ylim(lay.pitch - span, lay.pitch + span)
        ax.tick_params(labelsize=6)
    fig.suptitle("gap between adjacent lines vs position along line (blue: pair 0,2,..  red: pair 1,3,..)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)


def drawn_lead_out(layout_path: str):
    """レイアウト JSON と同名の G-code（gen_hatch_patches.py の出力）のヘッダから、描いたときの
    --lead-out [mm] を読む。ヘッダに行が無ければ 0（リードアウトなし）、G-code が無ければ None。"""
    base = layout_path[:-len("_layout.json")] if layout_path.endswith("_layout.json") else layout_path.rsplit(".", 1)[0]
    path = base + ".gcode"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if not line.startswith(";"):
                break
            if line.startswith("; lead-out "):
                return float(line.split()[2].rstrip("m"))
    return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("scan", help="スキャン画像（TIF/PNG。白地に黒い線）")
    ap.add_argument("--layout", required=True, help="gen_hatch_patches.py が出したレイアウト JSON")
    ap.add_argument("--params", help="fit_dots.py の fit_params.json（R, d, e）。静的なずれの Δ 換算に使う")
    ap.add_argument("--dpi", type=float, default=None, help="解像度。省略時は画像のメタデータ")
    ap.add_argument("--threshold", type=int, default=128, help="二値化の閾値（これより暗い画素を墨とする）")
    ap.add_argument("-o", "--output", help="CSV（既定: スキャン名 + _patches.csv）")
    ap.add_argument("--plot", help="間隔の実測とフィットの図（PNG）")
    ap.add_argument("--column", type=float, default=0.25, help="線に沿った集計の刻み（mm）")
    ap.add_argument("--band", type=float, default=0.5, help="各線の横位置を探す帯の半幅（mm）")
    ap.add_argument("--margin", type=float, default=None,
                    help="両端の折り返しを除く長さ（mm）。既定 1.5、--starts のときは 0.3")
    ap.add_argument("--starts", action="store_true",
                    help="線ごとの始点の曲がり（始点でのずれの状態 − その線の定常値、mm）をパッチごとに出す")
    ap.add_argument("--starts-lam", type=float, default=1.4, help="--starts の当てはめに使う λ（mm）")
    ap.add_argument("--deflection", help="補正に使った係数 JSON。--starts で、補正なしなら出るはずの D の符号に対する"
                                         "始点の偏り（c·sign(D予測) の平均）を出すのに使う。--ends では δ·v（線に沿った"
                                         "ずれの予測）に対する終点の不足の当てはめに使う")
    ap.add_argument("--ends", action="store_true",
                    help="線ごとの端点の位置（q_start、q_end: 設計の端点からの差、進行方向が正、mm）を出す。"
                         "線ごとにペンを上げたパッチ（--separate）だけ")
    ap.add_argument("--pen-radius", type=float, default=None,
                    help="端点の測定で墨の範囲から引くペン先の半径（mm）。既定は線ごとに測った線幅の半分"
                         f"（設計の線幅 {LINE_WIDTH:g} に対して、10 月中旬の Prockey の紙は 0.72〜0.74）")
    ap.add_argument("--lead-out", type=float, default=None, metavar="MM",
                    help="この紙を描いたときの --lead-out。--ends で次の --lead-out の目安（これ − 終点の切片）を出すのに"
                         "使う。既定はレイアウトと同名の G-code のヘッダから読む")
    args = ap.parse_args(argv)
    if args.margin is None:
        args.margin = 0.3 if args.starts else 1.5

    lay = Layout.load(args.layout)
    im = Image.open(args.scan)
    dpi = args.dpi or (im.info.get("dpi") or (600, 600))[0]
    px = dpi / 25.4
    gray = np.array(im.convert("L"))
    reg, results = measure_scan(gray, px, lay, threshold=args.threshold, column=args.column, band=args.band,
                                margin=args.margin, pen_radius=args.pen_radius)

    n_ok = sum(r.ok for r in results)
    print(f"位置合わせ: 回転 {math.degrees(reg.transform.phi):.2f}°、残差 rms {reg.rms:.2f} mm、最大 {reg.max_err:.2f} mm、"
          f"目印 {'あり' if reg.mark_found else 'なし（向きは格子の対称性で決まらない。スキャンの向きを確認）'}")
    print(f"パッチ {n_ok}/{len(results)} 個を測定")
    print(f"{'site':>12s} {'theta':>6s} {'D':>7s} {'D_mid':>7s} {'lam':>5s} {'pitch':>6s} {'rms':>6s}  note")
    for r in results:
        if r.ok:
            sat = "  融合（|D| が大きすぎて線が重なる。--pitch を大きくする）" if saturated(r.D, r.pitch) else ""
            print(f"({r.patch.site_x:+4.0f},{r.patch.site_y:+4.0f}) {r.patch.theta:6.0f} {r.D:+7.3f} {r.D_mid:+7.3f} "
                  f"{r.lam:5.1f} {r.pitch:6.3f} {r.rms:6.3f}{sat}")
        else:
            print(f"({r.patch.site_x:+4.0f},{r.patch.site_y:+4.0f}) {r.patch.theta:6.0f} {'-':>7s} {'-':>7s} "
                  f"{'-':>5s} {'-':>6s} {'-':>6s}  {r.note}")
    oks = [r for r in results if r.ok]
    if oks:
        print(f"|D| 最大 {max(abs(r.D) for r in oks):.3f} mm、λ 中央値 {np.median([r.lam for r in oks]):.1f} mm、"
              f"間隔の平均 {np.mean([r.pitch for r in oks]):.3f} mm（設計 {lay.pitch:g}）")
        prm = json.load(open(args.params, encoding="utf-8")) if args.params else None
        chk = static_check(Kinematics.from_params(prm, dmin=89, dmax=179), lay, results)
        if chk:
            print(chk.describe())
    if args.starts and oks:
        print(f"\n始点の曲がり（c: λ = {args.starts_lam:g} mm で当てはめた始点でのずれ。線 0 はペン下ろし直後、他は折り返しか直前の線から。"
              f"exc: 始点 0.3〜3.5 mm での最大の横ずれ）")
        dm = DeflectionModel.load(args.deflection) if args.deflection else None
        kin_s = Kinematics.from_params(json.load(open(args.params, encoding="utf-8")) if args.params else None,
                                       dmin=89, dmax=179)
        print(f"{'site':>12s} {'theta':>6s} {'D_mid':>7s} {'D_pred':>7s} | {'c line0':>7s} | c lines 1..: rms   mean | exc: mean  >0.15")
        all_c, all_e, bias = [], [], []
        for r in oks:
            c = start_hooks(r, lay, lam=args.starts_lam)
            e = start_excursion(r, lay)
            rest = c[1:][~np.isnan(c[1:])]
            er = e[~np.isnan(e)]
            all_c += list(rest)
            all_e += list(er)
            dp = predict_D(kin_s, r.center[0], r.center[1], r.theta, dm.a, dm.b, dm.model) if dm else math.nan
            if dm and abs(dp) >= 0.1:
                bias += [(abs(dp), v) for v in rest * np.sign(dp)]
            print(f"({r.patch.site_x:+4.0f},{r.patch.site_y:+4.0f}) {r.patch.theta:6.0f} {r.D_mid:+7.3f} {dp:+7.3f} | {c[0]:+7.3f} | "
                  f"{np.sqrt(np.mean(rest ** 2)) if len(rest) else math.nan:8.3f} {np.mean(rest) if len(rest) else math.nan:+6.3f} | "
                  f"{np.mean(er) if len(er) else math.nan:6.3f} {np.mean(er > 0.15) * 100 if len(er) else math.nan:5.0f}%")
        if all_c:
            a, e = np.array(all_c), np.array(all_e)
            print(f"全線（線 0 を除く）: c rms {np.sqrt(np.mean(a ** 2)):.3f}、最大 {np.abs(a).max():.3f} mm")
            print(f"始点 0.3〜3.5 mm の最大横ずれ: 平均 {np.mean(e):.3f}、0.15 mm 超 {np.mean(e > 0.15) * 100:.0f}%、"
                  f"0.2 mm 超 {np.mean(e > 0.2) * 100:.0f}%（{len(e)} 本）")
            if bias:
                b = np.array(bias)
                slope, icpt = np.polyfit(b[:, 0], b[:, 1], 1)
                print(f"偏り（|D予測| ≥ 0.1 の {len(b)} 本、c·sign(D予測)）: 平均 {np.mean(b[:, 1]):+.3f} mm、"
                      f"|D予測| に対する傾き {slope:+.2f}（切片 {icpt:+.3f}）。負なら線は定常値の手前から始まる。"
                      f"リードインありの紙なら、ペン上げの倍率 gain_up ≈ {1 + slope:.2f}")

    if args.ends and oks:
        if not lay.separate:
            print("\n--ends: ジグザグのパッチでは端が折り返しでつながっているので端点は測れません（--separate alt で描く）")
        else:
            widths = np.concatenate([r.width for r in oks])
            widths = widths[~np.isnan(widths)]
            rad = (f"ペン半径 {args.pen_radius:g}（指定）" if args.pen_radius is not None
                   else f"ペン半径は線幅の半分（線幅の実測 平均 {np.mean(widths):.3f}、sd {np.std(widths):.3f} mm）")
            print(f"\n端点の位置（設計の端点からの差、進行方向が正、mm。{rad} を墨の範囲から引いた値。"
                  f"q_start: 始点、q_end: 終点。dv: 補正なしのずれの線に沿った成分 δ·v の予測）")
            dm = DeflectionModel.load(args.deflection) if args.deflection else None
            kin_e = Kinematics.from_params(json.load(open(args.params, encoding="utf-8")) if args.params else None,
                                           dmin=89, dmax=179)
            print(f"{'site':>12s} {'theta':>6s} {'dv':>6s} | {'q_start':>7s} {'sd':>5s} | {'q_end':>7s} {'sd':>5s}  n")
            qs_all, qe_all, pairs = [], [], []
            for r in oks:
                qs, qe = r.q_start[~np.isnan(r.q_start)], r.q_end[~np.isnan(r.q_end)]
                dv = predict_along(kin_e, r.center[0], r.center[1], r.theta, dm.a, dm.b, dm.model) if dm else math.nan
                qs_all += list(qs)
                qe_all += list(qe)
                if dm:
                    pairs += [(dv, v) for v in qe]
                print(f"({r.patch.site_x:+4.0f},{r.patch.site_y:+4.0f}) {r.patch.theta:6.0f} {dv:+6.2f} | "
                      f"{np.mean(qs) if len(qs) else math.nan:+7.3f} {np.std(qs) if len(qs) else math.nan:5.3f} | "
                      f"{np.mean(qe) if len(qe) else math.nan:+7.3f} {np.std(qe) if len(qe) else math.nan:5.3f}  {len(qe)}")
            if qs_all:
                qs_a, qe_a = np.array(qs_all), np.array(qe_all)
                print(f"全線: 始点 {np.mean(qs_a):+.3f}（sd {np.std(qs_a):.3f}）、終点 {np.mean(qe_a):+.3f}"
                      f"（sd {np.std(qe_a):.3f}）、{len(qe_a)} 本")
            if pairs:
                b = np.array(pairs)
                slope, icpt = np.polyfit(b[:, 0], b[:, 1], 1)
                near = b[np.abs(b[:, 0]) < 0.05, 1]
                print(f"終点の当てはめ q_end = {icpt:+.3f} {slope:+.2f}·(δ·v)"
                      + (f"、|δ·v| < 0.05 の {len(near)} 本の平均 {np.mean(near):+.3f}" if len(near) else ""))
                lo = args.lead_out if args.lead_out is not None else drawn_lead_out(args.layout)
                if lo is None:
                    print("  次の --lead-out の目安 = この紙を描いたときの --lead-out − 切片"
                          f"（{-icpt:+.3f}）。G-code が見つからないので --lead-out MM で描いたときの値を渡すと計算する")
                else:
                    print(f"  次の --lead-out の目安 = 描いたときの {lo:g} − 切片 {icpt:+.3f} = {max(0.0, lo - icpt):.2f}"
                          "（切片 + は行き過ぎ、− は不足。±0.03 程度なら変えなくてよい）")

    out = args.output or args.scan.rsplit(".", 1)[0] + "_patches.csv"
    with open(out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(result_row(r))
    print(f"CSV: {out}")
    if args.plot:
        plot(results, lay, args.plot)
        print(f"図: {args.plot}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
