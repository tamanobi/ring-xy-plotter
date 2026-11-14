"""線画の画像を、ペンで描ける SVG にする（Inkscape の中心線トレースの代わり）。

  img2svg drawing.png                       # drawing.svg と drawing_trace.png（確認用）
  img2svg drawing.png --width 150           # 幅 150mm にする（既定: 半径 165mm の円に収まる大きさ）
  img2svg drawing.png --fill outline        # 塗りは輪郭だけ（hatch / outline / none / skeleton）
  img2svg drawing.png --fill-width 1.2 --hatch-spacing 0.6 --hatch-angle 30
  img2svg drawing.png --hatch-join 0            # ハッチングをジグザグにつながず、1本ずつ描く
  img2svg drawing.png --no-join --smooth 1.0 --spur 1.0
  img2svg drawing.png --pen-width 0.4 --fill-width 4 --render   # 太い線は平行パスで（入り抜きのあるペン画向け）

線は中心線に、塗り（--fill-width より太い部分）は輪郭＋ハッチングにする。
--pen-width を与えると、ペン幅より太い線は、局所幅から決めた本数の平行パスで描く
（1 本目は strokes レイヤー、2 本目以降は passes レイヤー）。
出力 SVG は Inkscape で開いて直せる（線・平行パス・輪郭・ハッチングは別レイヤー）。
そのあと `svg2ring drawing.svg` で G-code にする。
"""
from __future__ import annotations

import argparse
import math
import sys

from .raster import TraceOptions, load_ink, trace, write_overlay, write_render, write_svg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("image")
    ap.add_argument("-o", "--output", help="出力 SVG（既定: 画像と同名の .svg）")
    g = ap.add_argument_group("大きさ（mm。どれか1つ。既定は --fit-radius 165）")
    g.add_argument("--width", type=float, help="出力の幅")
    g.add_argument("--height", type=float, help="出力の高さ")
    g.add_argument("--dpi", type=float, help="画像の解像度から決める")
    g.add_argument("--fit-radius", type=float, default=165, help="画像の対角がこの半径の円に収まるようにする")
    g = ap.add_argument_group("二値化")
    g.add_argument("--threshold", type=int, help="これより暗い画素をインクにする（既定: Otsu で自動）")
    g.add_argument("--invert", action="store_true", help="白い線を描く画像")
    g.add_argument("--gap-contrast", type=float, default=48,
                   help="二重線の間の狭い隙間を残す。両側よりこの値以上明るい画素は、閾値より暗くても紙とみなす（0 で無効）")
    g = ap.add_argument_group("線（単位は mm。内部で画素に換算）")
    g.add_argument("--fill-width", type=float, default=1.5, help="これより太い部分を塗りとして扱う")
    g.add_argument("--fill-taper", type=float, default=15,
                   help="塗りから出る細くなる先端を、この長さ（mm）まで塗りに含める。これより長く続くものは線とみなす")
    g.add_argument("--fill-taper-width", type=float,
                   help="先端を塗りに含める下限の幅（mm）。既定は線の典型的な太さの 1.5 倍を自動で使う")
    g.add_argument("--close", type=float, default=0.3,
                   help="塗り判定の前に、この半径で隙間を閉じる。細かいハッチングの密集部が塗りにまとまる（0 で無効）")
    g.add_argument("--keep-gap", type=float, default=10,
                   help="--close で埋まる隙間のうち、この長さ（mm）以上続く細い隙間（二重線の間）は埋めずに二重線のまま残す（0 で無効）")
    g.add_argument("--spur", type=float, default=0.8, help="交点から出る、この長さ未満のひげを落とす")
    g.add_argument("--min-length", type=float, default=0.6, help="この長さ未満の孤立した線を落とす")
    g.add_argument("--join", type=float, default=60, help="交点で線をつなぐときに許す曲がり（度）")
    g.add_argument("--no-join", action="store_true", help="交点で線をつながない")
    g.add_argument("--smooth", type=float, default=1.5, help="平滑化の強さ（画素、ガウスのσ）")
    g.add_argument("--simplify", type=float, default=0.05, help="点の間引きの許容誤差")
    g = ap.add_argument_group("平行パス（--pen-width を与えると、太い線をペン幅ぶんずつずらした平行パスで描く）")
    g.add_argument("--pen-width", type=float,
                   help="ペンの線幅（mm）。線の局所幅から本数を決める。太さの上限は --fill-width（それより太い部分は塗り）")
    g.add_argument("--pass-pitch", type=float, default=0.8, help="平行パスの間隔（ペン幅の倍数。1 未満で隙間が出ない）")
    g.add_argument("--pass-min-run", type=float, default=1.5,
                   help="本数がこの長さ（mm）より短くしか続かない区間は隣の本数に揃える（交差の膨らみ対策）")
    g.add_argument("--width-smooth", type=float, default=1.0, help="幅の線に沿った平滑化（mm、ガウスのσ）")
    g.add_argument("--pass-blend", type=float, default=0.6,
                   help="本数が変わるところで、パスの位置をなめらかに移す長さ（mm、ガウスのσ）")
    g = ap.add_argument_group("塗り")
    g.add_argument("--fill", choices=["hatch", "outline", "none", "skeleton"], default="hatch",
                   help="hatch: 輪郭＋ハッチング / outline: 輪郭だけ / none: 無視 / skeleton: 線と同じく中心線")
    g.add_argument("--hatch-spacing", type=float, default=0.5, help="ハッチングの間隔（mm）")
    g.add_argument("--hatch-angle", type=float, default=45, help="ハッチングの角度（度）")
    g.add_argument("--hatch-join", type=float, default=2.0,
                   help="隣のハッチングの端がこの距離（間隔の倍数）以内なら、ジグザグに1本につなぐ。"
                        "ペン上げ下げが減るが、つなぎの線で濃く見える。1.2 で隣の行だけ、0 でつながない")
    g = ap.add_argument_group("出力")
    g.add_argument("--stroke-width", type=float, help="SVG の線の太さ（表示用、mm。既定は --pen-width、なければ 0.3）")
    g.add_argument("--no-overlay", action="store_true", help="確認用 PNG を出さない")
    g.add_argument("--render", action="store_true",
                   help="全部の線をペン幅（--pen-width、なければ --stroke-width）で白地に描いた PNG（_render.png）も出す")
    args = ap.parse_args(argv)

    ink = load_ink(args.image, args.threshold, args.invert, args.gap_contrast)
    H, W = ink.shape
    if args.width:
        mm_per_px = args.width / W
    elif args.height:
        mm_per_px = args.height / H
    elif args.dpi:
        mm_per_px = 25.4 / args.dpi
    else:
        mm_per_px = 2 * args.fit_radius / math.hypot(W, H)
    px = lambda mm: mm / mm_per_px  # noqa: E731

    opt = TraceOptions(
        threshold=args.threshold, invert=args.invert,
        fill_half_width_px=px(args.fill_width) / 2, close_px=px(args.close), keep_gap_px=px(args.keep_gap),
        taper_half_width_px=px(args.fill_taper_width) / 2 if args.fill_taper_width else None,
        taper_len_px=px(args.fill_taper),
        spur_px=px(args.spur), min_px=px(args.min_length),
        join_turn_deg=0 if args.no_join else args.join, smooth_sigma_px=args.smooth,
        simplify_px=px(args.simplify), fill_mode=args.fill,
        hatch_spacing_px=px(args.hatch_spacing), hatch_angle_deg=args.hatch_angle, hatch_min_px=px(0.3),
        hatch_join=args.hatch_join,
        pen_px=px(args.pen_width) if args.pen_width else 0.0, pass_pitch=args.pass_pitch,
        pass_min_run_px=px(args.pass_min_run), width_sigma_px=px(args.width_smooth),
        pass_blend_px=px(args.pass_blend),
    )
    res = trace(ink, opt)
    base = args.image.rsplit(".", 1)[0]
    out = args.output or base + ".svg"
    stroke_mm = args.stroke_width or args.pen_width or 0.3
    write_svg(res, out, mm_per_px, stroke_mm)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for pl in res.all for a, b in zip(pl, pl[1:])) * mm_per_px
    print(f"{args.image}: {W}×{H} px → {W * mm_per_px:.1f}×{H * mm_per_px:.1f} mm（{mm_per_px:.4f} mm/px）")
    for n in res.notes:
        print("  " + n)
    print(f"  折れ線 {len(res.all)}本、合計 {total:.0f} mm")
    print(f"SVG: {out}")
    if not args.no_overlay:
        ov = base + "_trace.png"
        write_overlay(res, args.image, ov)
        print(f"確認用: {ov}")
    if args.render:
        rn = base + "_render.png"
        write_render(res, rn, px(stroke_mm))
        print(f"ペン幅で描いた見え方: {rn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
