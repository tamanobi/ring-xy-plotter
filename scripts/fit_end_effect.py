#!/usr/bin/env python3
"""線の両端の落ち込み（加減速区間）の係数を、描いた紙のスキャンと G-code から当てはめる。

描いたときの G-code の指令経路を GRBL の台形の速度つきで歩き、力モデルでずれを順方向に再現して
「描かれる位置 = 指令 + ずれ」を出し、測った線の形（線ごとに中央の平均を引いたもの）の始点側・終点側
0.3〜3.5 mm と比べる。その残差 rms が最小になる係数を探す。

  python scripts/fit_end_effect.py sep_alt_li3.TIF --params fit_params.json \
      --deflection deflect_params_sqrt_v.json -o deflect_params_sqrt_v_end.json
  python scripts/fit_end_effect.py a.TIF b.TIF --fit c_end scale ...   # 複数の紙、全体の倍率も

各スキャンのレイアウトと G-code は、同じ名前の *_layout.json と *.gcode を使う（--layout / --gcode で上書き）。
--deflection は描いたときに掛けた係数の JSON（補正なしの紙でも、モデルの a, b, λ として渡す。
補正なしなら --applied-none）。出力 JSON は入力の係数に当てはめた値を入れたもの（scale は a, b に畳み込む）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.deflection import DeflectionModel, Layout, measure_scan  # noqa: E402
from ring_plotter.endfit import (Sheet, end_residuals, fit_coefficients, match_lines, predicted_tracks,  # noqa: E402
                                 run_state, sheet_summary, speed_factors, trace_gcode)
from ring_plotter.kinematics import Kinematics  # noqa: E402
from ring_plotter.preview import load_motion  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def plot(sheets_data, model, p, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, sh, results, lay, p0 in sheets_data:
        ok = [r for r in results if r.ok]
        ncol = len(lay.angles)
        nrow = math.ceil(len(ok) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.4 * nrow), squeeze=False, sharex=True, sharey=True)
        preds = {}
        for tag, pp in (("before", p0), ("after", p)):
            f = speed_factors(sh.trace, pp["c_speed"], model.v_ref, pp["c_end"])
            st = run_state(sh.trace, model.lam, f, pp["scale"], model.gain_up)
            preds[tag] = predicted_tracks(sh.matches, sh.trace, st)
        by_patch = {}
        for i, lm in enumerate(sh.matches):
            by_patch.setdefault(lm.patch_index, []).append(i)
        ok_idx = [i for i, r in enumerate(results) if r.ok]
        for ax, pi in zip(axes.flat, ok_idx):
            r = results[pi]
            for i in by_patch.get(pi, []):
                lm = sh.matches[i]
                col = "#1f77b4" if lm.sgn > 0 else "#d62728"
                m = ~np.isnan(lm.t) & (np.abs(lm.s) < 4)
                if m.sum() < 5:
                    continue
                ax.plot(lm.s, lm.t - np.mean(lm.t[m]), color=col, lw=0.6, alpha=0.6)
                for tag, ls in (("before", ":"), ("after", "--")):
                    pr = preds[tag][i]
                    ax.plot(lm.s, pr - np.mean(pr[m]), color=col, lw=0.9, ls=ls, alpha=0.9)
            ax.set_title(f"({r.patch.site_x:g},{r.patch.site_y:g}) {r.patch.theta:g}deg  Dmid={r.D_mid:+.3f}", fontsize=7)
            ax.set_ylim(-0.4, 0.4)
            ax.axhline(0, color="k", lw=0.3)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"{name}: solid = measured, dotted = model as applied, dashed = fitted "
                     f"(c_end={p['c_end']:.2f} c_speed={p['c_speed']:.2f} scale={p['scale']:.2f})", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out if len(sheets_data) == 1 else out.rsplit(".", 1)[0] + f"_{name}.png"
        fig.savefig(path, dpi=110)
        print(f"図: {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("scan", nargs="+", help="スキャン画像（同名の *_layout.json と *.gcode を使う）")
    ap.add_argument("--layout", nargs="*", default=None, help="レイアウト JSON（スキャンと同じ順）")
    ap.add_argument("--gcode", nargs="*", default=None, help="G-code（スキャンと同じ順）")
    ap.add_argument("--params", required=True, help="fit_dots.py の fit_params.json")
    ap.add_argument("--deflection", required=True, help="描いたときに掛けた係数の JSON（モデルの a, b, λ もこれ）")
    ap.add_argument("--applied-none", action="store_true", help="補正なしで描いた紙（係数 JSON はモデルとしてだけ使う）")
    ap.add_argument("--fit", nargs="+", default=["c_end"], choices=["c_end", "c_speed", "scale"],
                    help="当てはめる係数（既定 c_end）。scale は a, b に掛かる全体の倍率")
    ap.add_argument("--dmin", type=float, default=0.3, help="端から見る区間の内側（mm）")
    ap.add_argument("--dmax", type=float, default=3.5, help="端から見る区間の外側（mm）")
    ap.add_argument("--dpi", type=float, default=None)
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--max-rate", type=float, default=2000, help="GRBL の $110/$111（度/分、軸ごと）")
    ap.add_argument("--accel", type=float, default=200, help="GRBL の $120/$121（描いたときの値）")
    ap.add_argument("--junction", type=float, default=0.01, help="GRBL の $11")
    ap.add_argument("--plot", help="線の形（測定・当てはめ前・当てはめ後）の図（PNG）")
    ap.add_argument("-o", "--output", help="係数 JSON の出力（既定: 入力の JSON 名 + _end.json）")
    args = ap.parse_args(argv)

    kin = Kinematics.from_params(json.load(open(args.params, encoding="utf-8-sig")), dmin=89, dmax=179)
    model = DeflectionModel.load(args.deflection)
    motion = load_motion(args)
    sheets, sheets_data = [], []
    for i, scan in enumerate(args.scan):
        base = scan.rsplit(".", 1)[0]
        layout = args.layout[i] if args.layout else base + "_layout.json"
        gcode = args.gcode[i] if args.gcode else base + ".gcode"
        lay = Layout.load(layout)
        im = Image.open(scan)
        dpi = args.dpi or (im.info.get("dpi") or (600, 600))[0]
        gray = np.array(im.convert("L"))
        reg, results = measure_scan(gray, dpi / 25.4, lay, threshold=args.threshold, margin=0.3)
        lines = open(gcode, encoding="utf-8").read().splitlines()
        tr = trace_gcode(lines, kin, model, motion)
        matches = match_lines(results, lay, tr)
        name = os.path.basename(base)
        print(f"{name}: パッチ {sum(r.ok for r in results)}/{len(results)}、線の対応 {len(matches)} 本、"
              f"刻み {len(tr.ds)}（{'補正なし' if args.applied_none else '補正あり'}の紙として扱う）")
        sh = Sheet(name, tr, matches, lay.length)
        sheets.append(sh)
        sheets_data.append((name, sh, results, lay))

    p0 = dict(c_end=model.c_end, c_speed=model.c_speed, scale=1.0)
    fit = fit_coefficients(sheets, model, tuple(args.fit), args.dmin, args.dmax)
    p = fit["params"]
    print(f"\n当てはめ（{', '.join(args.fit)}）: c_end {p0['c_end']:.3f} → {p['c_end']:.3f}、"
          f"c_speed {p0['c_speed']:.3f} → {p['c_speed']:.3f}、scale {p['scale']:.3f}")
    print(f"両端 {args.dmin:g}〜{args.dmax:g} mm の残差 rms: {fit['rms_before']:.4f} → {fit['rms']:.4f} mm（{fit['n']} 点）")
    print(f"\n{'sheet':>20s} | 始点側 {'meas':>6s} {'before':>6s} {'after':>6s} | 終点側 {'meas':>6s} {'before':>6s} {'after':>6s}"
          f"   （|D予測|≥0.1 の線の、中央からの落ち込みの平均。D予測の向きが正）")
    for name, sh, results, lay in sheets_data:
        b = sheet_summary(sh, model, p0, kin, lay, results, args.dmin, args.dmax)
        a = sheet_summary(sh, model, p, kin, lay, results, args.dmin, args.dmax)
        print(f"{name:>20s} | {b['meas_start']:+13.3f} {b['pred_start']:+6.3f} {a['pred_start']:+6.3f} | "
              f"{b['meas_end']:+13.3f} {b['pred_end']:+6.3f} {a['pred_end']:+6.3f}")

    out = args.output or args.deflection.rsplit(".", 1)[0] + "_end.json"
    d = json.load(open(args.deflection, encoding="utf-8-sig"))
    d.update(c_end=p["c_end"], c_speed=p["c_speed"], a=model.a * p["scale"], b=model.b * p["scale"],
             end_fit=dict(fit=args.fit, scale=p["scale"], rms=fit["rms"], rms_before=fit["rms_before"], n=fit["n"],
                          dmin=args.dmin, dmax=args.dmax, accel=args.accel,
                          source=" ".join(os.path.basename(s) for s in args.scan)))
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(d, fp, indent=1)
    print(f"JSON: {out}")
    if args.plot:
        plot([(n, sh, res, lay, p0) for n, sh, res, lay in sheets_data], model, p, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
