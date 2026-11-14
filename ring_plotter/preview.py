"""G-code を実機に流す前に、PC 上で描画結果を確かめる。

  ring-preview out.gcode                    # out_preview.png に描く
  ring-preview out.gcode --svg in.svg       # 元の SVG を重ねる（配置は svg2ring と同じ指定を付ける）
  ring-preview out.gcode --paper 210x297    # A4 の枠を描く
  ring-preview out.gcode --show             # ウィンドウで表示

順運動学で紙面に戻した線、ペン上げ移動、届く範囲（外側の円と中心の穴）を描き、
Δのガード、ねじれ上限、F の上限（軸ごと）の違反を報告する。所要時間は GRBL の加減速
（--accel、--junction）を入れて推定する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .kinematics import Kinematics
from .simulate import Simulation, simulate
from .svg_input import Polyline
from .timing import Motion

# 色（白地で見やすいもの）
C_RING = "#9aa0a6"
C_REACH = "#c9ccd1"
C_PAPER = "#b8c4d6"
C_DRAW = "#1f2937"
C_TRAVEL = "#e07b39"
C_SVG = "#2a9d8f"
C_START = "#d62828"


def render(sim: Simulation, kin: Kinematics, output: str | None, svg_polylines: list[Polyline] | None = None,
           paper: tuple[float, float] | None = None, show: bool = False, travel: bool = True,
           title: str = "") -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    reach = kin.reach()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)
    ax.add_patch(Circle((0, 0), kin.R, fill=False, ec=C_RING, lw=1.0, ls="--"))
    ax.add_patch(Circle((0, 0), reach.r_max, fill=False, ec=C_REACH, lw=1.0))
    ax.add_patch(Circle((0, 0), reach.r_min, fill=True, fc="#f3d2d2", ec="none"))
    if paper:
        w, h = paper
        ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=False, ec=C_PAPER, lw=1.0))
    if svg_polylines:
        for pl in svg_polylines:
            ax.plot([p[0] for p in pl], [p[1] for p in pl], color=C_SVG, lw=2.5, alpha=0.35,
                    solid_capstyle="round", solid_joinstyle="round")
    first = True
    for st in sim.strokes:
        xs, ys = [p[0] for p in st.xy], [p[1] for p in st.xy]
        if st.pen_down:
            ax.plot(xs, ys, color=C_DRAW, lw=0.9, solid_capstyle="round", solid_joinstyle="round")
        elif travel:
            ax.plot(xs, ys, color=C_TRAVEL, lw=0.5, ls=(0, (3, 3)), alpha=0.7)
        if first and st.xy:
            ax.plot(xs[0], ys[0], "o", ms=5, color=C_START)
            first = False
    lim = kin.R * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(color="#ececec", lw=0.5)
    ax.tick_params(colors="#666", labelsize=8)
    ax.set_xlabel("x (mm)  ← α=0° is at the bottom", color="#666", fontsize=9)
    ax.set_ylabel("y (mm)", color="#666", fontsize=9)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    m, s = divmod(int(round(sim.est_time_s)), 60)
    title = os.path.basename(title)
    ax.set_title(
        f"{title}  draw {sim.draw_mm:.0f} mm, {sim.pen_downs} pen-downs, ~{m}m{s:02d}s, "
        f"α {sim.alpha_min:.0f}..{sim.alpha_max:.0f}°, Δ {sim.delta_min:.0f}..{sim.delta_max:.0f}°"
        + (f", {len(sim.problems)} problems" if sim.problems else ""),
        fontsize=9, loc="left")
    fig.tight_layout()
    if output:
        fig.savefig(output)
    if show:
        plt.show()
    plt.close(fig)


def parse_paper(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    w, h = s.lower().replace("×", "x").split("x")
    return float(w), float(h)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--params", help="fit_dots.py が出力した fit_params.json（R, d, e）")
    ap.add_argument("--home", type=float, default=88, help="原点（ストッパー位置）のΔ（度）")
    ap.add_argument("--twist", type=float, default=400, help="|α| の上限（ケーブルのねじれ、度）")
    ap.add_argument("--guard", type=float, nargs=2, default=(88, 180), metavar=("DMIN", "DMAX"),
                    help="grbl_send.py のガード範囲（度）")
    ap.add_argument("--max-rate", type=float, default=2000, help="GRBL の $110/$111（度/分、軸ごと）")
    ap.add_argument("--accel", type=float, default=200,
                    help="GRBL の $120/$121（度/秒²、軸ごと）。所要時間の推定と、ずれの補正の速度係数（加減速区間の長さ）に使う。"
                         "実機の設定と合わせること")
    ap.add_argument("--junction", type=float, default=0.01, help="GRBL の $11 junction deviation。所要時間の推定に使う")
    ap.add_argument("--paper", help="プレビューに描く紙の枠。例 210x297")
    ap.add_argument("--no-travel", action="store_true", help="ペン上げ移動を描かない")
    ap.add_argument("--show", action="store_true", help="ウィンドウで表示する")


def load_motion(args) -> Motion:
    return Motion(max_rate=args.max_rate, accel=args.accel, junction_deviation=args.junction)


def load_kin(args, dmin: float | None = None, dmax: float | None = None) -> Kinematics:
    prm = json.load(open(args.params, encoding="utf-8-sig")) if args.params else None  # BOM 付きも読む
    kw = {}
    if dmin is not None:
        kw["dmin"] = dmin
    if dmax is not None:
        kw["dmax"] = dmax
    return Kinematics.from_params(prm, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gcode")
    ap.add_argument("-o", "--output", help="出力 PNG（既定: G-code の名前 + _preview.png）")
    ap.add_argument("--svg", help="重ねて描く元の SVG")
    ap.add_argument("--scale", type=float, default=1.0, help="--svg の配置（svg2ring と同じ）")
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"))
    ap.add_argument("--mirror", action="store_true")
    add_common_args(ap)
    args = ap.parse_args(argv)

    kin = load_kin(args)
    lines = open(args.gcode, encoding="utf-8").read().splitlines()
    sim = simulate(lines, kin, home=args.home, guard=tuple(args.guard), twist=args.twist, motion=load_motion(args))
    svg_pl = None
    if args.svg:
        from .svg_input import load_svg, transform

        svg_pl = transform(load_svg(args.svg).polylines, args.scale, args.rotate, *args.offset, args.mirror)
    out = args.output or (args.gcode.rsplit(".", 1)[0] + "_preview.png")
    render(sim, kin, out, svg_pl, parse_paper(args.paper), args.show, not args.no_travel, title=args.gcode)
    print(kin.describe())
    print(sim.summary())
    print(f"プレビュー: {out}")
    return 1 if sim.problems else 0


if __name__ == "__main__":
    sys.exit(main())
