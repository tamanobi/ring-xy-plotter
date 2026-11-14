#!/usr/bin/env python3
"""線の曲がりの切り分け用: 同じ直線を往路・復路・往路の順に、ペンを上げて始点に戻りながら重ね描きする。

方向によってペン先がずれるなら、重ねた3本のうち復路だけが別の位置に出る。
あわせて、片方のユニットだけを往復させる円弧（Aだけ、Bだけ）も描く。
往路と復路の円弧が棒の方向にずれていれば、そのユニットがレール上で半径方向に動いている。

  python scripts/gen_overdraw.py --params scripts/fit_params.json -o examples/overdraw.gcode
  python scripts/gen_overdraw.py --params scripts/fit_params.json --fmax 1200 -o examples/overdraw_f1200.gcode
"""
import argparse
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.gcode import GcodeOptions, TwistError, convert  # noqa: E402
from ring_plotter.kinematics import Unreachable  # noqa: E402
from ring_plotter.preview import add_common_args, load_kin, load_motion, parse_paper, render  # noqa: E402
from ring_plotter.simulate import simulate  # noqa: E402

# 線の中心 (x, y) と名前。image_2 で曲がりが出た髪の付近、出なかった襟と上側
SITES = [
    ("hair-L", -30.0, 15.0),
    ("hair-R", 30.0, 15.0),
    ("collar", 60.0, -50.0),
    ("top", -30.0, 100.0),
    ("bottom", 0.0, -100.0),
]


def line(cx, cy, angle_deg, length):
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    h = length / 2
    return [(cx - dx * h, cy - dy * h), (cx + dx * h, cy + dy * h)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-o", "--output", default="overdraw.gcode")
    ap.add_argument("--length", type=float, default=60.0, help="線の長さ（mm）")
    ap.add_argument("--angles", type=float, nargs="+", default=[-45.0, 45.0],
                    help="線の向き（度、反時計回り。-45 は左上から右下）")
    ap.add_argument("--passes", type=int, default=3, help="重ね描きの回数（往・復・往…）")
    ap.add_argument("--sites", nargs="+", default=[s[0] for s in SITES],
                    help="描く場所の名前: " + " ".join(s[0] for s in SITES))
    ap.add_argument("--arc", type=float, default=5.0, help="片軸だけ往復させる角度（度）。0 で省略")
    ap.add_argument("--speed", type=float, default=30, help="ペン先の速度（mm/秒）")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=2250, help="描画中の F の上限（度/分）")
    ap.add_argument("--travel", type=float, default=1000)
    ap.add_argument("--dwell", type=float, default=0.3)
    ap.add_argument("--dwell-up", type=float, default=0.1)
    ap.add_argument("--no-preview", action="store_true")
    add_common_args(ap)
    args = ap.parse_args(argv)

    kin = load_kin(args, 89, 179)
    sites = [s for s in SITES if s[0] in args.sites]

    # 1. 重ね描きの直線。同じ線を passes 回、向きを交互にして並べる。
    #    convert は sort=False なら順番を守り、向きは「直前の位置に近い始点」を選ぶので、
    #    前の線の終点から始まるこの並びでは意図した向きが保たれる。
    polylines = []
    for _, cx, cy in sites:
        for ang in args.angles:
            seg = line(cx, cy, ang, args.length)
            for i in range(args.passes):
                polylines.append(seg if i % 2 == 0 else seg[::-1])

    opt = GcodeOptions(speed=args.speed, step=args.step, fmax=args.fmax, travel=args.travel,
                       dwell=args.dwell, dwell_up=args.dwell_up, twist=args.twist, home=args.home,
                       unreachable="error", sort=False, return_home=False, motion=load_motion(args))
    header = [
        f"; gen_overdraw {datetime.now():%Y-%m-%d %H:%M}",
        f"; {kin.describe(ascii_only=True)}",
        f"; sites={','.join(s[0] for s in sites)} angles={','.join(f'{a:g}' for a in args.angles)}"
        f" length={args.length:g}mm passes={args.passes} arc={args.arc:g}deg"
        f" speed={args.speed:g}mm/s fmax={args.fmax:g} step={args.step:g}mm",
        f"; send G92 X0 Y{args.home:g} (stopper origin) before this file",
    ]
    assert all(h.isascii() for h in header)
    try:
        lines, stats = convert(kin, polylines, opt, header)
    except (Unreachable, TwistError) as e:
        print(f"中止: {e}")
        return 1

    # 2. 片軸だけの往復。各場所の線の中心で、Aだけ +arc/-arc/+arc、次に Bだけ。
    #    convert の出力は直前の (α, β) で終わっているので、その巻き数に合わせてつなぐ。
    if args.arc > 0:
        last = [ln for ln in lines if ln.startswith("G1 ")][-1]
        cur_a = float(last.split()[1][1:])
        f_axis = f"{min(args.fmax, args.max_rate):.0f}"
        for name, cx, cy in sites:
            a, b = kin.inverse(cx, cy)
            k = round((cur_a - a) / 360)
            a, b = a + 360 * k, b + 360 * k
            if abs(a) > args.twist or abs(b - args.home) > args.twist:
                print(f"{name}: ねじれ上限のため片軸往復を省略")
                continue
            lines += [f"; axis sweep at {name}", f"G1 X{a:.3f} Y{b:.3f} F{args.travel:.0f}",
                      "M3 S110", f"G4 P{args.dwell:g}"]
            for axis in ("X", "Y"):
                base = a if axis == "X" else b
                for s in (+1, -1, +1):
                    v = base + s * args.arc
                    lines.append(f"G1 {axis}{v:.3f} F{f_axis}")
                lines.append(f"G1 {axis}{base:.3f} F{f_axis}")
            lines += ["M5", f"G4 P{args.dwell_up:g}"]
            cur_a = a
        lines += [f"G1 X0.000 Y{args.home + 2:.3f} F{args.travel:.0f}", f"G1 X0.000 Y{args.home:.3f} F200"]

    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(stats.summary())
    print(f"G-code: {args.output}")

    sim = simulate(lines, kin, home=args.home, guard=tuple(args.guard), twist=args.twist, motion=load_motion(args))
    if sim.problems:
        print("検査で問題:")
        for p in sim.problems[:20]:
            print("  " + p)
    if not args.no_preview:
        png = args.output.rsplit(".", 1)[0] + "_preview.png"
        render(sim, kin, png, polylines, parse_paper(args.paper), args.show, not args.no_travel, title=args.output)
        print(f"プレビュー: {png}")
    return 1 if sim.problems else 0


if __name__ == "__main__":
    sys.exit(main())
