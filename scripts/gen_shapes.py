#!/usr/bin/env python3
"""手順4: XY座標の図形を逆運動学で (α, β) に変換し、検証用のG-codeを作る。

前提: ストッパーで原点合わせ済み（Bをカラーに当てて G92 X0 Y88）。X=α、Y=β、単位は度。
図形の座標系: リングの中心が原点、α=0°のときのユニットAの方向が下（−y）、単位はmm。
つまり「0°」と書いた側を手前にして紙を見たときの、普通のXY座標。

  python gen_shapes.py                         # 100mmの正方形と内接円（中心を少しずらす）
  python gen_shapes.py --params fit_params.json
  python gen_shapes.py --shape square --dots      # 正方形の線＋外側110mmの点（引きずりの確認）
  python gen_shapes.py --shape square --size 60 --cx 0 --cy 40 -o sq60.gcode
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.kinematics import DEFAULT_PARAMS as DEFAULT, Kinematics  # noqa: E402


def subdivide(path, step):
    out = [path[0]]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
    return out


def make_paths(shape, size, cx, cy):
    h = size / 2
    paths = []
    if shape in ("square", "both"):
        paths.append([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)])
    if shape in ("circle", "both"):
        n = 180
        paths.append([(cx + h * math.cos(2 * math.pi * i / n), cy + h * math.sin(2 * math.pi * i / n))
                      for i in range(n + 1)])
    return paths


def to_gcode(kin, paths, speed, step, fmax, travel, down, up, dwell, twist=400.0, dots=(), lead=2.0,
             approach="together", sweep=5.0, home=88.0, dot_dwell=0.5):
    out = ["G90", up, f"G4 P{dwell}"]
    cur = (0.0, home)  # 原点（ストッパー位置）
    lo = hi = 0.0
    for path in paths:
        pts = subdivide(path, step)
        # 経路全体を (α, β) に変換し、αが連続するようにつなぐ
        abs_ = []
        for p in pts:
            al, be = kin.inverse(*p)
            if abs_:
                k = round((abs_[-1][0] - al) / 360)
                al, be = al + 360 * k, be + 360 * k
            abs_.append((al, be))
        # 回り方（360°の整数倍）と描く向きの候補から、ケーブルのねじれが
        # 制限内で、直前の位置からの移動がいちばん短いものを選ぶ
        mid = (min(a for a, _ in abs_) + max(a for a, _ in abs_)) / 2
        k0 = round(mid / 360)
        cands = []
        for k in (k0 - 1, k0, k0 + 1):
            sh = [(a - 360 * k, b - 360 * k) for a, b in abs_]
            worst = max(abs(a) for a, _ in sh)
            for rev in (False, True):
                start = sh[-1] if rev else sh[0]
                cands.append((worst > twist, abs(start[0] - cur[0]) if worst <= twist else worst, k, rev))
        _, _, k, rev = min(cands)
        abs_ = [(a - 360 * k, b - 360 * k) for a, b in abs_]
        if rev:
            abs_, pts = abs_[::-1], pts[::-1]
        out.append(f"G1 X{abs_[0][0]:.3f} Y{abs_[0][1]:.3f} F{travel:.0f}")
        out += [down, f"G4 P{dwell}"]
        cur = abs_[0]
        for p0, p1, ab in zip(pts, pts[1:], abs_[1:]):
            ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            da = math.hypot(ab[0] - cur[0], ab[1] - cur[1])
            f = min(fmax, da / (ds / speed) * 60) if ds > 0 else fmax
            out.append(f"G1 X{ab[0]:.3f} Y{ab[1]:.3f} F{max(f, 30):.0f}")
            cur = ab
        lo = min(lo, min(a for a, _ in abs_))
        hi = max(hi, max(a for a, _ in abs_))
        out += [up, f"G4 P{dwell}"]
    # 点: 線を引かずにペンを下ろすだけ。どの点にもA・Bとも＋方向から近づく
    for x, y in dots:
        al, be = kin.inverse(x, y)
        k = round((cur[0] - al) / 360)
        if abs(al + 360 * k) > twist:
            k = round(-al / 360)
        al, be = al + 360 * k, be + 360 * k
        if approach == "bsweep":
            # 点の格子と同じ近づき方: Aを先に止め、Bだけを＋方向に動かして合わせる
            out.append(f"G1 X{al - lead:.3f} Y{be - sweep - lead:.3f} F{travel:.0f}")
            out.append(f"G1 X{al:.3f} Y{be - sweep:.3f} F{travel:.0f}")
            out.append(f"G1 Y{be:.3f} F{travel:.0f}")
        else:
            # 図形の線と同じ近づき方: A・Bを一緒に＋方向へ動かして合わせる
            out.append(f"G1 X{al - lead:.3f} Y{be - lead:.3f} F{travel:.0f}")
            out.append(f"G1 X{al:.3f} Y{be:.3f} F{travel:.0f}")
        out += ["G4 P0.3", down, f"G4 P{dot_dwell}", up, f"G4 P{dwell}"]
        cur = (al, be)
        lo, hi = min(lo, al - lead), max(hi, al)
    out.append(f"G1 X0 Y{home + 2:.3f} F{travel:.0f}")  # ストッパーの2°手前まで
    out.append(f"G1 X0 Y{home:.3f} F200")  # ゆっくり当たり位置へ（当たりを確認する）
    return out, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shape", choices=["square", "circle", "both"], default="both")
    ap.add_argument("--size", type=float, default=100, help="正方形の一辺＝円の直径（mm）")
    ap.add_argument("--cx", type=float, default=15, help="図形の中心x（mm）")
    ap.add_argument("--cy", type=float, default=10, help="図形の中心y（mm）")
    ap.add_argument("--params", help="fit_dots.py が出力した fit_params.json")
    ap.add_argument("--speed", type=float, default=15, help="ペン先の速度（mm/秒）")
    ap.add_argument("--step", type=float, default=1.0, help="経路の細分化（mm）")
    ap.add_argument("--fmax", type=float, default=1500, help="描画中のFの上限")
    ap.add_argument("--travel", type=float, default=1000, help="移動のF")
    ap.add_argument("--dots", action="store_true",
                    help="図形の外側の正方形（一辺 size+2×margin）の角と辺の中点に点を打つ")
    ap.add_argument("--dots-margin", type=float, default=5, help="点の正方形を外側へ広げる量（mm）")
    ap.add_argument("--approach", choices=["together", "bsweep"], default="together",
                    help="点への近づき方。together: A・B一緒に / bsweep: Aを止めてBだけ動かす")
    ap.add_argument("--dots-only", action="store_true", help="線を描かず、点だけ打つ")
    ap.add_argument("--home", type=float, default=88, help="原点（ストッパー位置）のΔ。終了時にここへ戻る")
    ap.add_argument("--dot-dwell", type=float, default=0.5, help="点を打つときにペンを下ろしている時間（秒）")
    ap.add_argument("--twist", type=float, default=400, help="αの絶対値の上限（ケーブルのねじれ制限、度）")
    ap.add_argument("--dmin", type=float, default=90)
    ap.add_argument("--dmax", type=float, default=180)
    ap.add_argument("--down", default="M3 S110")
    ap.add_argument("--up", default="M5")
    ap.add_argument("--dwell", type=float, default=0.3)
    ap.add_argument("-o", "--output", default="shapes.gcode")
    args = ap.parse_args()

    prm = dict(DEFAULT)
    if args.params:
        prm.update({k: v for k, v in json.load(open(args.params)).items() if k in prm})
    kin = Kinematics(prm["R"], prm["d"], prm["e"], args.dmin, args.dmax)
    paths = make_paths(args.shape, args.size, args.cx, args.cy)
    dots = []
    if args.dots:
        # 図形より一回り大きい正方形の、角と辺の中点（反時計回り）
        g = args.size / 2 + args.dots_margin
        dots = [(args.cx + g * ux, args.cy + g * uy) for ux, uy in
                [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]]
    if args.dots_only:
        paths = []
    out, lo, hi = to_gcode(kin, paths, args.speed, args.step, args.fmax, args.travel,
                           args.down, args.up, args.dwell, args.twist, dots,
                           approach=args.approach, home=args.home, dot_dwell=args.dot_dwell)
    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out) + "\n")
    print(f"{args.output}: {len(out)}行、R={kin.R:.3f} d={kin.d:.3f} e={kin.e:.3f}")
    print(f"αの範囲 {lo:.0f}°〜{hi:.0f}°（ケーブルのねじれの目安）")


if __name__ == "__main__":
    main()
