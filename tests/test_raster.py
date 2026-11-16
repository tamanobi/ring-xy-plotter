import math

import numpy as np
import pytest

from ring_plotter.raster import (TraceOptions, bright_gaps, close_gaps, hatch, join_at_junctions, load_ink, prune,
                                 rdp, smooth, split_fills, trace, trace_skeleton)


def canvas(h=120, w=160):
    return np.zeros((h, w), dtype=bool)


def draw_line(img, p0, p1, width=3):
    """(x, y) の2点を結ぶ太さ width の線を描く。"""
    n = int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * 2) + 1
    r = width / 2
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    for i in range(n + 1):
        x = p0[0] + (p1[0] - p0[0]) * i / n
        y = p0[1] + (p1[1] - p0[1]) * i / n
        img |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return img


def test_split_fills_separates_blob_from_line():
    img = canvas()
    draw_line(img, (10, 60), (150, 60), 3)
    yy, xx = np.mgrid[:120, :160]
    blob = (xx - 120) ** 2 + (yy - 30) ** 2 <= 15 ** 2
    img |= blob
    strokes, fills = split_fills(img, fill_half_width_px=4)
    assert fills[30, 120] and not fills[60, 40]
    assert strokes[60, 40] and not strokes[30, 120]
    assert (fills & blob).sum() > 0.9 * blob.sum()


def test_close_merges_dense_hatching():
    img = canvas()
    for x in range(40, 100, 4):  # 間隔 4px の縦線の束
        draw_line(img, (x, 20), (x, 80), 2)
    _, fills0 = split_fills(img, 4, close_px=0)
    _, fills1 = split_fills(img, 4, close_px=2.5)
    assert fills0.sum() == 0
    assert fills1.sum() > 0.8 * img.sum()


def test_bright_gap_between_double_lines_stays_paper(tmp_path):
    # 太さ 3px の線2本の間に、閾値より暗い（120）1px の隙間。隙間は紙として残り、線は2本になる
    gray = np.full((60, 160), 255, dtype=np.uint8)
    gray[28:31, 10:150] = 10
    gray[31, 10:150] = 120
    gray[32:35, 10:150] = 10
    gray[40:43, 10:150] = 120  # 1本だけの薄い線は尾根ではないのでインクのまま
    ridge = bright_gaps(gray, contrast=48)
    assert ridge[31, 80] and not ridge[29, 80] and not ridge[41, 80]
    from PIL import Image

    p = tmp_path / "g.png"
    Image.fromarray(gray).save(p)
    ink = load_ink(str(p), threshold=143)
    assert ink[29, 80] and ink[33, 80] and not ink[31, 80] and ink[41, 80]
    assert load_ink(str(p), threshold=143, gap_contrast=0)[31, 80]
    res = trace(ink, TraceOptions(fill_half_width_px=4, smooth_sigma_px=0))
    assert len(res.strokes) == 3 and not res.outlines


def test_keep_gap_leaves_long_double_line_open():
    img = canvas()
    draw_line(img, (10, 40), (150, 40), 3)
    draw_line(img, (10, 44), (150, 44), 3)  # 隙間 1px の二重線（長さ 140px）
    for x in range(40, 100, 4):  # 間隔 4px、長さ 30px の縦線の束（短い隙間は埋まる）
        draw_line(img, (x, 70), (x, 100), 2)
    c0 = close_gaps(img, 2.5)
    c1 = close_gaps(img, 2.5, keep_len_px=50)
    assert c0[42, 80] and not c1[42, 80]
    assert c0[85, 42] and c1[85, 42]
    _, fills = split_fills(img, 4, close_px=2.5, keep_gap_px=50)
    assert not fills[42, 80] and fills[85, 42]


def test_trace_skeleton_single_line():
    img = canvas()
    draw_line(img, (10, 60), (150, 60), 3)
    from skimage.morphology import skeletonize

    paths, junctions = trace_skeleton(skeletonize(img))
    assert len(paths) == 1 and not junctions
    p = paths[0]
    assert p.j0 is None and p.j1 is None
    assert abs(p.length() - 140) < 6


def test_cross_is_joined_through_junction():
    img = canvas()
    draw_line(img, (20, 60), (140, 60), 3)
    draw_line(img, (80, 10), (80, 110), 3)
    from skimage.morphology import skeletonize

    paths, junctions = trace_skeleton(skeletonize(img))
    assert len(junctions) == 1 and len(paths) == 4
    assert all((p.j0 is None) != (p.j1 is None) for p in paths)
    joined = join_at_junctions(paths, 60)
    assert len(joined) == 2
    lengths = sorted(p.length() for p in joined)
    assert abs(lengths[0] - 100) < 8 and abs(lengths[1] - 120) < 8


def test_prune_spur():
    img = canvas()
    draw_line(img, (20, 60), (140, 60), 3)
    draw_line(img, (80, 60), (84, 50), 3)  # 短いひげ
    from skimage.morphology import skeletonize

    paths, _ = trace_skeleton(skeletonize(img))
    kept = prune(paths, spur_px=12, min_px=4)
    assert len(kept) < len(paths)
    assert all(not ((p.j0 is None) != (p.j1 is None) and p.length() < 12) for p in kept)


def test_smooth_and_rdp():
    zig = [(i, (i % 2) * 1.0) for i in range(40)]  # 1px の階段
    sm = smooth(zig, 1.5)
    assert sm[0] == zig[0] and sm[-1] == zig[-1]
    assert max(abs(y - 0.5) for _, y in sm[3:-3]) < 0.2
    pts = rdp(sm, 0.3)
    assert len(pts) < 8
    assert rdp([(0, 0), (1, 0.01), (2, 0)], 0.1) == [(0, 0), (2, 0)]


def test_hatch_covers_blob():
    img = canvas()
    yy, xx = np.mgrid[:120, :160]
    img |= (xx - 80) ** 2 + (yy - 60) ** 2 <= 30 ** 2
    segs = hatch(img, spacing_px=4, angle_deg=45, min_len_px=2)
    # 円はジグザグで数本につながる。線分（偶数番目の区間）は 15〜25 本
    assert len(segs) <= 6
    n = 0
    for pl in segs:
        for (r0, c0), (r1, c1) in zip(pl[0::2], pl[1::2]):
            assert img[int(round(r0)), int(round(c0))] and img[int(round(r1)), int(round(c1))]
            assert abs(abs(r1 - r0) - abs(c1 - c0)) < 1.5  # 45°
            n += 1
    assert 15 <= n <= 25


def test_hatch_join_factor():
    img = canvas()
    yy, xx = np.mgrid[:120, :160]
    img |= (xx - 80) ** 2 + (yy - 60) ** 2 <= 30 ** 2
    joined = hatch(img, spacing_px=4, angle_deg=45, min_len_px=2, join_factor=2.0)
    single = hatch(img, spacing_px=4, angle_deg=45, min_len_px=2, join_factor=0)
    # つながないと全部が2点の線分で、線分の数は同じ
    assert all(len(pl) == 2 for pl in single)
    assert len(single) == sum(len(pl) // 2 for pl in joined)
    assert len(single) > len(joined)
    # 倍数を小さくすると、つながる本数が減る（単調）
    n = [len(hatch(img, 4, 45, 2, join_factor=f)) for f in (0, 1.2, 2.0)]
    assert n[0] >= n[1] >= n[2]


def test_trace_end_to_end():
    img = canvas()
    draw_line(img, (10, 20), (150, 20), 3)
    yy, xx = np.mgrid[:120, :160]
    img |= (xx - 80) ** 2 + (yy - 80) ** 2 <= 20 ** 2
    res = trace(img, TraceOptions(fill_half_width_px=4, hatch_spacing_px=4))
    assert len(res.strokes) == 1
    assert len(res.outlines) == 1 and sum(len(pl) for pl in res.hatches) > 10
    (x0, y0), (x1, y1) = res.strokes[0][0], res.strokes[0][-1]
    assert abs(y0 - 20) < 1.5 and abs(y1 - 20) < 1.5 and abs(abs(x1 - x0) - 140) < 6


def test_pass_counts_drops_short_runs():
    from ring_plotter.raster import pass_counts

    w = np.full(60, 5.2)  # pen 2, pitch 1.6 → (5.2−2)/1.6 = 2 → 3 本
    w[20:24] = 8.0  # 交差の膨らみのような短い区間
    w[40:60] = 2.0  # 端は細い → 1 本
    arclen = np.arange(60, dtype=float)
    n = pass_counts(w, arclen, pen_px=2.0, pitch_px=1.6, min_run_px=6)
    assert n[:40].tolist() == [3] * 40  # 短い膨らみは隣に揃う
    assert n[40:].tolist() == [1] * 20  # 端の細い区間は残る


def test_pass_counts_closed_with_same_end_runs_terminates():
    """閉じた線で、短い最初の区間がつなぎ目の向こうの最後の区間と同じ本数でも、止まらずに終わる。"""
    import threading

    from ring_plotter.raster import pass_counts

    w = np.full(26, 5.2)  # pen 2, pitch 1.6 → 3 本
    w[:3] = w[-3:] = 2.0  # 最初と最後は 1 本。最初の区間は短く、つなぎ目の向こうの最後の区間と同じ本数
    arclen = np.arange(26, dtype=float)
    out = []
    t = threading.Thread(target=lambda: out.append(
        pass_counts(w, arclen, pen_px=2.0, pitch_px=1.6, min_run_px=6, closed=True)), daemon=True)
    t.start()
    t.join(timeout=5)
    assert out, "pass_counts が 5 秒で終わらない"
    assert out[0].tolist() == [1] * 3 + [3] * 20 + [1] * 3  # 端の細い区間はそのまま残る


def test_multipass_straight_thick_line():
    img = canvas()
    draw_line(img, (20, 60), (140, 60), 9)
    res = trace(img, TraceOptions(fill_half_width_px=20, pen_px=3.0, pass_pitch=0.8, pass_min_run_px=6,
                                  width_sigma_px=2, pass_blend_px=2, simplify_px=0.2))
    # (9−3)/2.4 = 2.5 → 3 本: 基準パス 1 本 + 追加 2 本
    assert len(res.strokes) == 1 and len(res.passes) == 2
    ys = sorted(np.mean([y for _, y in pl]) for pl in res.strokes + res.passes)
    assert abs(ys[1] - 60) < 0.7
    assert abs((ys[2] - ys[0]) - 2 * 2.4) < 0.7
    for pl in res.strokes + res.passes:  # 直線のまま（端の丸みで多少ぶれる）
        ys_pl = [y for _, y in pl]
        assert max(ys_pl) - min(ys_pl) < 0.6


def test_multipass_thin_line_is_single():
    img = canvas()
    draw_line(img, (20, 60), (140, 60), 3)
    res = trace(img, TraceOptions(fill_half_width_px=20, pen_px=3.0))
    assert len(res.strokes) == 1 and res.passes == []


def test_prune_drops_short_junction_to_junction_stub():
    from ring_plotter.raster import Path

    stub = Path([(0, 0), (0, 1)], j0=1, j1=2)  # 割れた交点の間の 1 画素
    arm = Path([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)], j0=1, j1=2)
    out = prune([stub, arm], spur_px=6, min_px=4)
    assert out == [arm]


def test_prune_drops_one_pixel_loop_at_a_junction():
    from ring_plotter.raster import Path

    loop = Path([(20.5, 493.0), (20.0, 490.0), (20.5, 493.0)], j0=2, j1=2)  # 交点 → 1 画素 → 同じ交点
    ring = Path([(0.0, 0.0)] + [(3 * math.sin(t), 3 - 3 * math.cos(t)) for t in np.linspace(0.3, 6.0, 20)] + [(0.0, 0.0)],
                j0=3, j1=3)  # 半径 3px の本物の輪
    out = prune([loop, ring], spur_px=6, min_px=4)
    assert out == [ring]
