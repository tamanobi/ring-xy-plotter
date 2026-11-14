#!/usr/bin/env python3
"""手順2: Aを止めてBだけ動かし、Aのピンを中心とする円弧を描くG-codeを作る。

前提: ストッパーで原点合わせ済み（Bをカラーに当てて G92 X0 Y88）。X=α、Y=β、単位は度。
円弧の半径 = Aのピンからペン先までの距離。
全部の円弧が1点で交われば、その距離 = R（設計則どおり）。

  python gen_arcs.py -o arcs.gcode
  python gen_arcs.py --dmin 100 -o arcs_long.gcode   # 長い円弧（大きい紙が必要）
  python gen_arcs.py --count 4 -o arcs4.gcode
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=8, help="円弧の本数（360°を等分）")
    ap.add_argument("--dmin", type=float, default=130, help="描き始めのΔ（度）")
    ap.add_argument("--dmax", type=float, default=180, help="描き終わりのΔ（度）")
    ap.add_argument("--home", type=float, default=88, help="原点（ストッパー位置）のΔ。終了時にここへ戻る")
    ap.add_argument("--feed", type=float, default=500, help="描画の送り速度F")
    ap.add_argument("--travel", type=float, default=1000, help="移動の送り速度F")
    ap.add_argument("--down", default="M3 S55", help="ペンを下ろすコマンド")
    ap.add_argument("--up", default="M5", help="ペンを上げるコマンド")
    ap.add_argument("--dwell", type=float, default=0.2, help="上げ下げ後の待ち時間（秒）")
    ap.add_argument("-o", "--output", default="arcs.gcode")
    args = ap.parse_args()

    up = [args.up, f"G4 P{args.dwell}"]
    down = [args.down, f"G4 P{args.dwell}"]

    out = ["G90", *up]
    for i in range(args.count):
        a = 360.0 * i / args.count
        # 次の開始位置へ（Δは dmax→dmin へ直線的に変わるので範囲内）
        out.append(f"G1 X{a:.3f} Y{a + args.dmin:.3f} F{args.travel:.0f}")
        out += down
        out.append(f"G1 Y{a + args.dmax:.3f} F{args.feed:.0f}")  # Bだけ動かす
        out += up
    out.append(f"G1 X0 Y{args.home + 2:.3f} F{args.travel:.0f}")  # ストッパーの2°手前まで
    out.append(f"G1 X0 Y{args.home:.3f} F200")  # ゆっくり当たり位置へ（当たりを確認する）

    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out) + "\n")
    print(f"{args.output}: {len(out)}行、{args.count}本、Δ = {args.dmin}〜{args.dmax}°")


if __name__ == "__main__":
    main()
