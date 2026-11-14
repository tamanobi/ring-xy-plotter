#!/usr/bin/env python3
"""gen_overdraw.py で描いた重ね描きのスキャンから、各線の位置ごとのインクの帯幅を測る。

帯幅がペン1本の線幅（約0.5mm）なら往復は重なっており、太いぶんが往路と復路のずれ。
十字（±45°の2本の線）の中心は自動で見つける。線の角度は画素からフィットするので紙の傾きは吸収する。
中心±14mmは片軸往復の円弧が重なるので集計から外す。

  python scripts/measure_overdraw.py scan.TIF
  python scripts/measure_overdraw.py scan.tif --dpi 600 --sites 5 --strips strips.png

出力: 十字ごとに2本の線について、線に沿った位置（mm、2mm刻み）と帯幅（mm）。
--strips を付けると、各線を水平に回転して並べた画像（縦4倍）も出す。
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def find_crosses(binary, px_per_mm, n):
    """粗い画像で連結成分を取り、大きい順に n 個の重心を返す（十字は1成分になる）。"""
    from scipy import ndimage

    small = binary[::8, ::8]
    small = ndimage.binary_dilation(small, iterations=2)
    lab, k = ndimage.label(small)
    sizes = ndimage.sum(small, lab, range(1, k + 1))
    order = np.argsort(sizes)[::-1][:n]
    cents = ndimage.center_of_mass(small, lab, [i + 1 for i in order])
    return [(cx * 8, cy * 8) for cy, cx in cents]


def fit_line(x, y):
    mx, my = x.mean(), y.mean()
    w, v = np.linalg.eigh(np.cov(np.vstack([x - mx, y - my])))
    return (mx, my), (v[0, 1], v[1, 1])


def refine_center(binary, cx, cy, px):
    """十字の2本の線をフィットし、交点と2本の方向を返す。"""
    R = int(35 * px)
    sub = binary[max(0, int(cy) - R):int(cy) + R, max(0, int(cx) - R):int(cx) + R]
    ys, xs = np.nonzero(sub)
    x = (xs - min(R, int(cx))) / px
    y = (ys - min(R, int(cy))) / px
    c = np.array([0.0, 0.0])
    for _ in range(2):
        rx, ry = x - c[0], y - c[1]
        r = np.hypot(rx, ry)
        ann = (r > 14) & (r < 30)
        th = np.degrees(np.arctan2(ry[ann], rx[ann])) % 180
        h, _ = np.histogram(th, bins=np.arange(0, 181, 1.0))
        lines = []
        for lo in (0, 90):
            pk = lo + np.argmax(h[lo:lo + 90]) + 0.5
            dth = (np.degrees(np.arctan2(ry, rx)) % 180 - pk + 90) % 180 - 90
            m = ann & (np.abs(dth) < 4)
            lines.append(fit_line(x[m], y[m]))
        (p1, d1), (p2, d2) = lines
        t = np.linalg.solve(np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]]), np.array([p2[0] - p1[0], p2[1] - p1[1]]))
        c = np.array([p1[0] + d1[0] * t[0], p1[1] + d1[1] * t[0]])
    return (cx + c[0] * px - 0, cy + c[1] * px), [d for _, d in lines], (x - c[0], y - c[1])


def widths_along(x, y, d, exclude=14.0, span=29.5, step=2.0):
    """線の方向 d に沿って、位置ごとの帯幅と細い部分からのずれを返す。"""
    dx, dy = d
    if dx < 0:
        dx, dy = -dx, -dy
    along = x * dx + y * dy
    perp = x * (-dy) + y * dx
    sel = (np.abs(perp) < 2.5) & (np.abs(along) > exclude) & (np.abs(along) < span)
    a, p = along[sel], perp[sel]
    bins = np.arange(-30, 31, step)
    idx = np.digitize(a, bins)
    slope, icpt = 0.0, 0.0
    for _ in range(3):
        res = p - (slope * a + icpt)
        width = {}
        for b in np.unique(idx):
            rr = res[idx == b]
            if len(rr) > 150:
                width[b] = (np.percentile(rr, 2), np.percentile(rr, 98))
        if not width:
            return [], dy > 0
        wmin = min(hi - lo for lo, hi in width.values())
        thin = np.zeros_like(a, bool)
        for b, (lo, hi) in width.items():
            if hi - lo < wmin + 0.08:
                thin |= idx == b
        slope, icpt = np.polyfit(a[thin], p[thin], 1)
    res = p - (slope * a + icpt)
    rows = []
    for b in sorted(width):
        rr = res[idx == b]
        rows.append((bins[b - 1] + step / 2, np.percentile(rr, 98) - np.percentile(rr, 2)))
    return rows, dy > 0  # 画像で y 下向きなので dy>0 は左上→右下（−45°の線）


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("scan", help="スキャン画像（TIF/PNG。白地に黒い線）")
    ap.add_argument("--dpi", type=float, default=None, help="解像度。省略時は画像のメタデータ")
    ap.add_argument("--sites", type=int, default=5, help="十字の数")
    ap.add_argument("--strips", help="各線を水平に回転して並べた画像を出す")
    args = ap.parse_args()

    im = Image.open(args.scan)
    dpi = args.dpi or (im.info.get("dpi") or (600, 600))[0]
    px = dpi / 25.4
    gray = im.convert("L")
    binary = np.array(gray) < 128
    crosses = find_crosses(binary, px, args.sites)
    crosses.sort(key=lambda c: (round(c[1] / (40 * px)), c[0]))  # 上から、左から

    strips = []
    W, H = int(64 * px), int(6 * px)
    for i, (cx, cy) in enumerate(crosses):
        (cx, cy), dirs, (x, y) = refine_center(binary, cx, cy, px)
        print(f"\n=== cross {i + 1}: center ({cx / px:.1f}, {cy / px:.1f}) mm from top-left ===")
        for d in dirs:
            rows, is_m45 = widths_along(x, y, d)
            label = "-45deg line (UL-LR)" if is_m45 else "+45deg line (LL-UR)"
            if not rows:
                print(f"  {label}: not enough pixels")
                continue
            wmin = min(w for _, w in rows)
            print(f"  {label}: thinnest {wmin:.2f} mm")
            print("    along(mm): " + " ".join(f"{a:+5.0f}" for a, _ in rows))
            print("    width(mm): " + " ".join(f"{w:5.2f}" for _, w in rows))
            if args.strips:
                ang = math.degrees(math.atan2(d[1], d[0]))
                ang = ang if abs(ang) <= 90 else ang - 180 if ang > 0 else ang + 180
                Rc = int(40 * px)
                crop = gray.crop((int(cx) - Rc, int(cy) - Rc, int(cx) + Rc, int(cy) + Rc))
                rot = crop.rotate(ang, resample=Image.BICUBIC, center=(Rc, Rc), fillcolor=255)
                band = rot.crop((Rc - W // 2, Rc - H // 2, Rc + W // 2, Rc + H // 2)).resize((W // 2, H * 2), Image.LANCZOS)
                ImageDraw.Draw(band).text((5, 2), f"cross {i + 1} {label} (left = -30mm, vertical x4)", fill=0)
                strips.append(band)
    if args.strips and strips:
        total = Image.new("L", (W // 2, sum(s.height + 4 for s in strips)), 200)
        yy = 0
        for s in strips:
            total.paste(s, (0, yy))
            yy += s.height + 4
        total.save(args.strips)
        print(f"\nstrips: {args.strips}")


if __name__ == "__main__":
    main()
