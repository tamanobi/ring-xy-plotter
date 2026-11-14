#!/usr/bin/env python3
"""手順3: フィッティング用の点の格子を打つG-codeと、指令値の一覧CSVを作る。

前提: ストッパーで原点合わせ済み（Bをカラーに当てて G92 X0 Y88）。X=α、Y=β、単位は度。
αごとに、BをΔの小さい側から大きい側へ動かしながら点を打つ。
各点には必ず同じ向き（A・Bとも＋方向）から近づく。

  python gen_dots.py                     # dots.gcode と dots.csv を出力（A4に収まる範囲）
  python gen_dots.py --reverse -o dots_rev   # 逆向きから近づく版（バックラッシュ測定用）
"""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha-step", type=float, default=30, help="αの刻み（度）")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[130, 138, 146, 154, 162, 170], help="点を打つΔ（度）")
    ap.add_argument("--markers", type=float, nargs="*", default=[134, 142],
                    help="向きの目印としてα=0だけに追加で打つΔ（度）")
    ap.add_argument("--lead", type=float, default=5,
                    help="最初の点の手前に取る助走（度）")
    ap.add_argument("--reverse", action="store_true",
                    help="A・Bとも−方向から近づく（バックラッシュ測定用）")
    ap.add_argument("--home", type=float, default=88, help="原点（ストッパー位置）のΔ。終了時にここへ戻る")
    ap.add_argument("--feed", type=float, default=800, help="送り速度F")
    ap.add_argument("--down", default="M3 S110", help="ペンを下ろすコマンド")
    ap.add_argument("--up", default="M5", help="ペンを上げるコマンド")
    ap.add_argument("--dwell-down", type=float, default=0.3)
    ap.add_argument("--dwell-up", type=float, default=0.2)
    ap.add_argument("--settle", type=float, default=0.3,
                    help="移動後、ペンを下ろす前に待つ時間（秒）")
    ap.add_argument("-o", "--output", default="dots",
                    help="出力ファイル名（拡張子なし）")
    args = ap.parse_args()

    n = int(round(360 / args.alpha_step))
    alphas = [args.alpha_step * i for i in range(n)]
    if args.reverse:
        alphas = alphas[::-1]

    f = f"F{args.feed:.0f}"
    out = ["G90", args.up, f"G4 P{args.dwell_up}"]
    rows = []

    if args.reverse:  # 最初のαへ上側から近づくための助走
        out.append(f"G1 X{alphas[0] + args.lead:.3f} Y{alphas[0] + args.lead + 150:.3f} {f}")

    for a in alphas:
        ds = [(d, 0) for d in args.deltas]
        if abs(a) < 1e-9:
            ds += [(d, 1) for d in args.markers]
        ds.sort(reverse=args.reverse)
        lead = -args.lead if not args.reverse else args.lead
        # αを決め、Bは最初の点の手前（助走ぶん）へ
        out.append(f"G1 X{a:.3f} Y{a + ds[0][0] + lead:.3f} {f}")
        for d, marker in ds:
            out.append(f"G1 Y{a + d:.3f} {f}")
            out.append(f"G4 P{args.settle}")
            out += [args.down, f"G4 P{args.dwell_down}", args.up, f"G4 P{args.dwell_up}"]
            rows.append((a, a + d, marker))
    out.append(f"G1 X0 Y{args.home + 2:.3f} F{args.feed:.0f}")  # ストッパーの2°手前まで
    out.append(f"G1 X0 Y{args.home:.3f} F200")  # ゆっくり当たり位置へ（当たりを確認する）

    with open(args.output + ".gcode", "w", encoding="utf-8") as fp:
        fp.write("\n".join(out) + "\n")
    with open(args.output + ".csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["alpha", "beta", "marker"])
        w.writerows(rows)
    print(f"{args.output}.gcode: {len(out)}行 / {args.output}.csv: {len(rows)}点")


if __name__ == "__main__":
    main()
