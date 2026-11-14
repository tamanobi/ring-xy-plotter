#!/usr/bin/env python3
"""進行方向で決まるペン先のずれを符号付きで測るための、ジグザグのハッチパッチを描く G-code。

紙の各所（既定: x = −70, 0, 70、y = −105, −35, 35, 105 の 12 か所）に、向きの違う 4 つのパッチ
（線 12 本、長さ 20 mm、間隔 1.2 mm のジグザグ）を 2×2 に並べる。左上に向きの目印の短い線を 1 本描く。
描いたらスキャン（600 dpi）して measure_hatch_patches.py で測る。レイアウトは <出力>_layout.json に書く。

  python scripts/gen_hatch_patches.py --params scripts/fit_params.json -o examples/hatch_patches.gcode
  python scripts/gen_hatch_patches.py --params scripts/fit_params.json --angles 0 30 60 90 120 150 -o patches6.gcode
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.deflection import Layout, add_deflection_args, load_deflection, patch_polylines  # noqa: E402
from ring_plotter.gcode import (GcodeOptions, TwistError, add_dry_run_args, add_lead_in_args, convert,  # noqa: E402
                                dry_run_option, lead_in_options)
from ring_plotter.kinematics import Unreachable  # noqa: E402
from ring_plotter.preview import add_common_args, load_kin, load_motion, parse_paper, render  # noqa: E402
from ring_plotter.simulate import simulate  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    d = Layout()
    ap.add_argument("-o", "--output", default="hatch_patches.gcode")
    ap.add_argument("--pitch", type=float, default=d.pitch, help="線の間隔（mm）")
    ap.add_argument("--length", type=float, default=d.length, help="線の長さ（mm）")
    ap.add_argument("--lines", type=int, default=d.lines, help="線の本数（偶数）")
    ap.add_argument("--angles", type=float, nargs="+", default=list(d.angles), help="1 か所あたりの線の向き（度）")
    ap.add_argument("--grid-x", type=float, nargs="+", default=list(d.grid_x), help="場所の x（mm）")
    ap.add_argument("--grid-y", type=float, nargs="+", default=list(d.grid_y), help="場所の y（mm）")
    ap.add_argument("--block", type=float, default=d.block, help="ブロック内のパッチ中心のオフセット（mm）")
    ap.add_argument("--no-mark", action="store_true", help="向きの目印を描かない")
    ap.add_argument("--separate", choices=["alt", "same"], default="",
                    help="線ごとにペンを上げる（始点の曲がりの試験用）。alt: 向きは交互（ジグザグと同じ）、"
                         "same: 全部同じ向き（進入が線と逆向きになる最悪条件）")
    ap.add_argument("--speed", type=float, default=30, help="ペン先の速度（mm/秒）")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=2250, help="描画中の F の上限（度/分）")
    ap.add_argument("--travel", type=float, default=1000)
    ap.add_argument("--dwell", type=float, default=0.3)
    ap.add_argument("--dwell-up", type=float, default=0.1)
    ap.add_argument("--no-preview", action="store_true")
    add_deflection_args(ap)
    add_lead_in_args(ap)
    add_dry_run_args(ap)
    add_common_args(ap)
    args = ap.parse_args(argv)
    if args.lines % 2:
        print("--lines は偶数にしてください（180° 対称にするため）")
        return 1

    lay = Layout(pitch=args.pitch, length=args.length, lines=args.lines, angles=tuple(args.angles),
                 grid_x=tuple(args.grid_x), grid_y=tuple(args.grid_y), block=args.block, mark=not args.no_mark,
                 separate=args.separate)
    kin = load_kin(args, 89, 179)
    polylines = [pl for p in lay.patches() for pl in patch_polylines(p, lay)]
    if lay.mark_line():
        polylines.append(lay.mark_line())

    # 向きを保つ（keep_direction）。逆向きに描くと全線の向きが反転して D の符号が変わる
    opt = GcodeOptions(speed=args.speed, step=args.step, fmax=args.fmax, travel=args.travel,
                       dwell=args.dwell, dwell_up=args.dwell_up, twist=args.twist, home=args.home,
                       unreachable="error", sort=True, keep_direction=True, motion=load_motion(args),
                       deflection=load_deflection(args), **lead_in_options(args), **dry_run_option(args))
    header = [
        f"; gen_hatch_patches {datetime.now():%Y-%m-%d %H:%M}",
        f"; {kin.describe(ascii_only=True)}",
        *([f"; DRY RUN: pen up for the patches, cross marks at ({opt.dry_run[0]:g},{opt.dry_run[1]:g}) before and after"]
          if opt.dry_run else []),
        *([f"; {opt.deflection.describe(ascii_only=True)}"] if opt.deflection else []),
        *([f"; separate={lay.separate}"] if lay.separate else []),
        *([f"; lead-in {opt.lead_in:g}mm (min travel {opt.lead_in_min_travel:g}mm, min swing {opt.lead_in_min_swing:g}mm)"]
          if opt.lead_in > 0 else []),
        *([f"; lead-out {opt.lead_out:g}mm"] if opt.lead_out > 0 else []),
        f"; pitch={lay.pitch:g}mm length={lay.length:g}mm lines={lay.lines} angles={','.join(f'{a:g}' for a in lay.angles)}",
        f"; grid x={','.join(f'{v:g}' for v in lay.grid_x)} y={','.join(f'{v:g}' for v in lay.grid_y)} block={lay.block:g}"
        f" speed={args.speed:g}mm/s fmax={args.fmax:g} accel={args.accel:g} step={args.step:g}mm",
        f"; send G92 X0 Y{args.home:g} (stopper origin) before this file",
    ]
    assert all(h.isascii() for h in header)
    try:
        lines, stats = convert(kin, polylines, opt, header)
    except (Unreachable, TwistError) as e:
        print(f"中止: {e}")
        return 1

    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    layout_path = args.output.rsplit(".", 1)[0] + "_layout.json"
    lay.save(layout_path)
    print(stats.summary())
    print(f"パッチ {len(lay.patches())} 個（{len(lay.grid_x) * len(lay.grid_y)} か所 × {len(lay.angles)} 向き）")
    print(f"G-code: {args.output}\nレイアウト: {layout_path}")

    sim = simulate(lines, kin, home=args.home, guard=tuple(args.guard), twist=args.twist, motion=load_motion(args))
    if sim.problems:
        print("検査で問題:")
        for p in sim.problems[:20]:
            print("  " + p)
    if not args.no_preview:
        png = args.output.rsplit(".", 1)[0] + "_preview.png"
        render(sim, kin, png, polylines, parse_paper(args.paper or "210x297"), args.show, not args.no_travel,
               title=args.output)
        print(f"プレビュー: {png}")
    return 1 if sim.problems else 0


if __name__ == "__main__":
    sys.exit(main())
