"""SVG をリング式プロッタの G-code にする。

  svg2ring drawing.svg                        # drawing.gcode と drawing_preview.png を出す
  svg2ring drawing.svg -o out.gcode --scale 0.8 --offset 0 10
  svg2ring drawing.svg --params fit_params.json
  svg2ring drawing.svg --unreachable error    # 届かない点があれば止める
  svg2ring drawing.svg --dwell 0.2 --dwell-up 0.1 --accel 100   # 待ち時間と、推定に使う加速度

配置: SVG の中心をリングの中心に合わせ、SVG の下が α=0° の方向（手前）。
      --scale、--rotate（度、反時計回り）、--offset DX DY（mm）、--mirror で調整する。
出力: X=α、Y=β（度）。ストッパー原点（G92 X0 Y88）で始め、終了時に戻る。
      そのまま `python scripts/grbl_send.py COMx out.gcode` で送れる。
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime

from .deflection import add_deflection_args, load_deflection
from .gcode import GcodeOptions, TwistError, add_dry_run_args, add_lead_in_args, convert, dry_run_option, lead_in_options
from .kinematics import Unreachable
from .preview import add_common_args, load_kin, load_motion, parse_paper, render
from .simulate import simulate
from .svg_input import load_svg, transform


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("svg")
    ap.add_argument("-o", "--output", help="出力 G-code（既定: SVG と同名の .gcode）")
    g = ap.add_argument_group("配置")
    g.add_argument("--scale", type=float, default=1.0, help="倍率")
    g.add_argument("--rotate", type=float, default=0.0, help="回転（度、反時計回り）")
    g.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                   help="リング中心からのずらし（mm、y は上向き）")
    g.add_argument("--mirror", action="store_true", help="左右反転")
    g.add_argument("--fit", action="store_true", help="届く範囲に収まるよう縮小する（拡大はしない）")
    g = ap.add_argument_group("動き")
    g.add_argument("--speed", type=float, default=15, help="ペン先の速度（mm/秒）")
    g.add_argument("--step", type=float, default=1.0, help="経路の細分化（mm）")
    g.add_argument("--fmax", type=float, default=1500, help="描画中の F の上限（度/分）")
    g.add_argument("--travel", type=float, default=1000, help="ペン上げ移動の F（度/分）")
    g.add_argument("--dmin", type=float, default=89, help="使う Δ の下限（度）。ガードより少し内側にする")
    g.add_argument("--dmax", type=float, default=179, help="使う Δ の上限（度）")
    g.add_argument("--unreachable", choices=["clip", "error"], default="clip",
                   help="届かない点の扱い。clip: 捨てて分割 / error: 止める")
    g.add_argument("--no-sort", action="store_true", help="折れ線を並べ替えない（SVG の順に描く）")
    g.add_argument("--keep-direction", action="store_true",
                   help="折れ線を SVG の向きのまま描く（並べ替えで逆向きにしない。塗りの隣の線を同じ向きに描き、"
                        "向きで決まる横ずれで白い筋が出るのを防ぐ。戻りの移動が増える）")
    g.add_argument("--no-return", action="store_true", help="終了時に原点へ戻らない")
    add_deflection_args(g)
    add_lead_in_args(g)
    add_dry_run_args(g)
    g = ap.add_argument_group("ペン")
    g.add_argument("--down", default="M3 S110", help="ペンを下ろすコマンド")
    g.add_argument("--up", default="M5", help="ペンを上げるコマンド")
    g.add_argument("--dwell", "--dwell-down", type=float, default=0.3, help="ペンを下げたあとの待ち（秒）")
    g.add_argument("--dwell-up", type=float, default=None,
                   help="ペンを上げたあとの待ち（秒）。紙から離れれば十分なので短くできる。既定は --dwell と同じ")
    g = ap.add_argument_group("プレビューと検査")
    g.add_argument("--no-preview", action="store_true", help="PNG を出さない")
    g.add_argument("--preview", help="プレビュー PNG の名前（既定: G-code の名前 + _preview.png）")
    add_common_args(g)
    args = ap.parse_args(argv)

    kin = load_kin(args, args.dmin, args.dmax)
    out_path = args.output or (args.svg.rsplit(".", 1)[0] + ".gcode")

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        drawing = load_svg(args.svg, step=args.step)
    for w in ws:
        print(f"警告: {w.message}")
    print(f"{args.svg}: {drawing.width_mm:.1f}×{drawing.height_mm:.1f} mm、折れ線 {len(drawing.polylines)}本、"
          f"長さ {drawing.total_length:.0f} mm")
    if not drawing.polylines:
        print("描く線がありません")
        return 2

    scale = args.scale
    polylines = transform(drawing.polylines, scale, args.rotate, *args.offset, args.mirror)
    if args.fit:
        reach = kin.reach()
        rmax = max((p[0] ** 2 + p[1] ** 2) ** 0.5 for pl in polylines for p in pl)
        if rmax > reach.r_max:
            scale *= reach.r_max / rmax * 0.999
            polylines = transform(drawing.polylines, scale, args.rotate, *args.offset, args.mirror)
            print(f"--fit: 倍率を {scale:.4f} にして半径 {reach.r_max:.1f} mm に収めました")

    opt = GcodeOptions(speed=args.speed, step=args.step, fmax=args.fmax, travel=args.travel,
                       down=args.down, up=args.up, dwell=args.dwell, dwell_up=args.dwell_up,
                       twist=args.twist, home=args.home, unreachable=args.unreachable,
                       sort=not args.no_sort, keep_direction=args.keep_direction, return_home=not args.no_return,
                       motion=load_motion(args),
                       deflection=load_deflection(args), **lead_in_options(args), **dry_run_option(args))
    # ヘッダは ASCII だけにする。GRBL は 0x80 以上のバイトをリアルタイムコマンドとして解釈するので、
    # コメント行でも日本語や「°」を送るとセーフティドアやオーバーライドが誤って発動する
    header = [
        f"; svg2ring {datetime.now():%Y-%m-%d %H:%M} {args.svg.encode('ascii', 'replace').decode()}",
        f"; {kin.describe(ascii_only=True)}",
        f"; scale={scale:g} rotate={args.rotate:g} offset=({args.offset[0]:g},{args.offset[1]:g})"
        f"{' mirror' if args.mirror else ''} speed={args.speed:g}mm/s fmax={args.fmax:g} accel={args.accel:g}"
        f" step={args.step:g}mm twist={args.twist:g} dwell={opt.dwell:g}/{opt.dwell_up_s:g}s",
        f"; send G92 X0 Y{args.home:g} (stopper origin) before this file",
    ]
    if opt.deflection:
        header.insert(3, f"; {opt.deflection.describe(ascii_only=True)}")
    if opt.lead_in > 0:
        header.insert(-1, f"; lead-in {opt.lead_in:g}mm (min travel {opt.lead_in_min_travel:g}mm,"
                          f" min swing {opt.lead_in_min_swing:g}mm)")
    if opt.lead_out > 0:
        header.insert(-1, f"; lead-out {opt.lead_out:g}mm")
    if opt.keep_direction:
        header.insert(-1, "; keep-direction (polylines drawn in SVG order of points)")
    if opt.dry_run:
        header.insert(1, f"; DRY RUN: pen up for the drawing, cross marks at ({opt.dry_run[0]:g},{opt.dry_run[1]:g})"
                         " before and after")
    assert all(h.isascii() for h in header)
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            lines, stats = convert(kin, polylines, opt, header)
        for w in ws:
            print(f"警告: {w.message}")
    except (Unreachable, TwistError) as e:
        print(f"中止: {e}")
        return 1

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(kin.describe())
    print(stats.summary())
    print(f"G-code: {out_path}")

    # 出した G-code をそのまま読み直して検査し、プレビューを描く
    sim = simulate(lines, kin, home=args.home, guard=tuple(args.guard), twist=args.twist, motion=load_motion(args))
    if sim.problems:
        print("検査で問題:")
        for p in sim.problems[:20]:
            print("  " + p)
    if not args.no_preview:
        png = args.preview or (out_path.rsplit(".", 1)[0] + "_preview.png")
        render(sim, kin, png, polylines, parse_paper(args.paper), args.show, not args.no_travel, title=out_path)
        print(f"プレビュー: {png}")
    return 1 if sim.problems else 0


if __name__ == "__main__":
    sys.exit(main())
