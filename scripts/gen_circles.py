#!/usr/bin/env python3
"""手順1: Δを固定して2台を一緒に回し、同心円を描くG-codeを作る。

前提: ストッパーで原点合わせ済み（Bをカラーに当てて G92 X0 Y88）。X=α、Y=β、単位は度。

  python gen_circles.py -o circles.gcode
  python gen_circles.py --deltas 160 140 -o small.gcode
  python gen_circles.py --deltas 120 --redraw -o backlash.gcode
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deltas", type=float, nargs="+", default=[160, 140, 120, 100],
                    help="描く円のΔ（度）。大きいほど小さい円")
    ap.add_argument("--home", type=float, default=88, help="原点（ストッパー位置）のΔ。終了時にここへ戻る")
    ap.add_argument("--feed", type=float, default=1000, help="送り速度F")
    ap.add_argument("--overlap", type=float, default=10,
                    help="1周を超えて重ね描きする角度（閉じ具合の確認用）")
    ap.add_argument("--redraw", action="store_true",
                    help="復路もペンを下ろしたまま戻る（バックラッシュの確認用）")
    ap.add_argument("--down", default="M3 S55", help="ペンを下ろすコマンド")
    ap.add_argument("--up", default="M5", help="ペンを上げるコマンド")
    ap.add_argument("--dwell", type=float, default=0.2, help="上げ下げ後の待ち時間（秒）")
    ap.add_argument("-o", "--output", default="circles.gcode")
    args = ap.parse_args()

    f = args.feed
    up = [args.up, f"G4 P{args.dwell}"]
    down = [args.down, f"G4 P{args.dwell}"]
    end = 360 + args.overlap

    out = ["G90", *up]
    for d in args.deltas:
        out.append(f"G1 X0 Y{d:.3f} F{f:.0f}")           # Bだけ動かしてΔを設定
        out += down
        out.append(f"G1 X{end:.3f} Y{d + end:.3f} F{f:.0f}")  # 一緒に1周＋重ね
        if not args.redraw:
            out += up
        out.append(f"G1 X0 Y{d:.3f} F{f:.0f}")           # 戻る（ケーブルのねじれを解く）
        if args.redraw:
            out += up
    out.append(f"G1 X0 Y{args.home + 2:.3f} F{f:.0f}")  # ストッパーの2°手前まで
    out.append(f"G1 X0 Y{args.home:.3f} F200")  # ゆっくり当たり位置へ（当たりを確認する）

    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out) + "\n")
    print(f"{args.output}: {len(out)}行、Δ = {args.deltas}")


if __name__ == "__main__":
    main()
