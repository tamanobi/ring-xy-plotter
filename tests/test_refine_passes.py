import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ring_plotter.refine_passes import (  # noqa: E402
    LossWeights, SoftRenderer, Strokes, add_passes, contour_passes, join_lines, optimise, remove_passes, resample,
    split_corners,
)

DEV = torch.device("cpu")


def test_resample_keeps_vertices_and_limits_segment_length():
    pl = resample([(0.0, 0.0), (10.0, 0.0), (10.0, 3.0)], 4.0)
    assert pl[0] == (0.0, 0.0) and pl[-1] == (10.0, 3.0) and (10.0, 0.0) in pl
    assert max(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pl, pl[1:])) <= 4.0 + 1e-9


def test_render_line_has_pen_width_and_union_saturates():
    rend = SoftRenderer(40, 60, pen_px=6.0, tau=0.3, max_seg=6.0, device=DEV)
    s = Strokes.from_polylines([resample([(10.0, 20.0), (50.0, 20.0)], 6.0)], DEV)
    img = rend.render(s).detach().numpy()
    col = img[:, 30]
    assert abs(col.sum() - 6.0) < 0.5  # 線の断面の積分 ≈ ペン幅
    assert img[20, 30] > 0.99 and img[5, 30] < 0.01
    # 同じ線を 2 本重ねても 1 を超えない（和集合）。縁のぼかしの分（〜2τ）だけ少し太くなる
    s2 = Strokes.from_polylines([resample([(10.0, 20.0), (50.0, 20.0)], 6.0)] * 2, DEV)
    img2 = rend.render(s2).detach().numpy()
    assert img2.max() <= 1.0 + 1e-6 and abs(img2[:, 30].sum() - 6.0) < 1.0


def test_optimise_moves_line_onto_target():
    rend = SoftRenderer(40, 60, pen_px=6.0, tau=0.3, max_seg=6.0, device=DEV)
    target = rend.render(Strokes.from_polylines([resample([(10.0, 22.0), (50.0, 22.0)], 6.0)], DEV)).detach()
    pls = [resample([(10.0, 18.0), (50.0, 18.0)], 6.0)]
    out, img, terms = optimise(pls, rend, target, LossWeights(), iters=100, lr=0.2, device=DEV)
    ys = [y for _, y in out[0]]
    assert abs(np.mean(ys) - 22.0) < 0.3 and float(terms["l2"]) < 1e-3


def test_remove_drops_redundant_and_add_fills_missing():
    rend = SoftRenderer(40, 60, pen_px=6.0, tau=0.3, max_seg=6.0, device=DEV)
    line = resample([(10.0, 20.0), (50.0, 20.0)], 6.0)
    target = rend.render(Strokes.from_polylines([line], DEV)).detach()
    # 同じ線が 2 本: 1 本は消しても損失が増えない
    pls, origin = remove_passes([line, list(line)], [0, 0], rend, target, 0.3, 0.0, DEV)
    assert len(pls) == 1
    # 何もない状態から追加すると、線のあるところに折れ線ができる
    empty = torch.zeros(40, 60)
    new = add_passes(empty.numpy(), target.numpy(), 6.0, 0.5, 6.0, 1.8, 6.0)
    assert len(new) == 1
    assert all(abs(y - 20.0) < 1.5 for _, y in new[0])
    xs = [x for x, _ in new[0]]
    assert min(xs) < 15 and max(xs) > 45


def test_add_passes_makes_dots_for_isolated_blobs():
    target = np.zeros((40, 60), dtype=np.float32)
    target[18:23, 28:33] = 1.0  # 5 px の点
    new = add_passes(np.zeros_like(target), target, 6.0, 0.5, 6.0, 1.8, 6.0, dot_min_px=3.0)
    assert len(new) == 1 and len(new[0]) == 2
    (x0, y0), (x1, y1) = new[0]
    assert abs((x0 + x1) / 2 - 30.0) < 1 and abs((y0 + y1) / 2 - 20.0) < 1 and np.hypot(x1 - x0, y1 - y0) < 2


def test_contour_passes_follow_the_edge_and_fill_wide_regions():
    wide = np.zeros((60, 80), dtype=bool)
    wide[10:50, 10:70] = True  # 40×60 px の長方形、ペン 6 px、ピッチ 4.8 px
    pls = contour_passes(wide, 6.0, 4.8, 6.0, 6.0)
    assert len(pls) >= 3
    # 一番外の線は縁からペン半径（3 px）の位置の閉じた輪
    outer = max(pls, key=lambda pl: max(x for x, _ in pl) - min(x for x, _ in pl))
    xs, ys = [x for x, _ in outer], [y for _, y in outer]
    assert abs(min(xs) - 12.5) < 1.5 and abs(max(xs) - 66.5) < 1.5 and abs(min(ys) - 12.5) < 1.5
    assert np.hypot(outer[0][0] - outer[-1][0], outer[0][1] - outer[-1][1]) < 1.5
    # 等高線だけで領域の大半が埋まる（幅 wide_px 以上の欠けは中心線でなく等高線になる）
    rend = SoftRenderer(60, 80, 6.0, 0.3, 6.0, DEV)
    img = rend.render(Strokes.from_polylines(pls, DEV)).detach().numpy()
    assert img[wide].mean() > 0.85
    new = add_passes(np.zeros_like(img), wide.astype(np.float32), 6.0, 0.5, 6.0, 1.8, 6.0, wide_px=12.0, pitch_px=4.8)
    assert len(new) >= 3 and all(len(pl) > 2 for pl in new[:3])


def _max_deviation(pl):
    """折れ線の、両端を結ぶ弦からの最大距離。"""
    a = np.asarray(pl)
    p, q = a[0], a[-1]
    d = q - p
    n = np.hypot(*d)
    return float(np.abs((a[:, 0] - p[0]) * d[1] - (a[:, 1] - p[1]) * d[0]).max() / n) if n > 0 else 0.0


def test_direction_field_follows_nearby_lines_and_falls_back_to_region_axis():
    from ring_plotter.hatch import direction_field

    H, W = 100, 160
    line = [(10.0, 30.0), (150.0, 30.0)]  # 水平線
    mask = np.zeros((H, W), dtype=bool)
    mask[20:40, 10:150] = True  # 線のまわり
    mask[70:80, 20:140] = True  # 線から遠い横長の領域（主軸は水平）
    c2, s2 = direction_field([line], (H, W), 6.0, fallback_mask=mask)
    assert c2.shape == (H, W)
    assert c2[25, 80] > 0.95 and abs(s2[25, 80]) < 0.2  # 線のそばは θ ≈ 0
    assert c2[75, 80] > 0.9  # 遠くは領域の主軸（水平）
    # 斜め 45° の線なら (cos 90°, sin 90°) = (0, 1)
    c2d, s2d = direction_field([[(10.0, 10.0), (90.0, 90.0)]], (H, W), 6.0)
    assert abs(c2d[50, 50]) < 0.2 and s2d[50, 50] > 0.95


def test_flow_passes_draws_straight_parallel_lines_along_the_field():
    from ring_plotter.hatch import flow_passes

    H, W = 60, 120
    wide = np.zeros((H, W), dtype=bool)
    wide[10:50, 10:110] = True  # 40×100 px、ペン 6 px、ピッチ 4.8 px
    c2, s2 = np.ones((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)  # 全部水平
    pls = flow_passes(wide, c2, s2, 6.0, 4.8, 6.0, 6.0, simplify_px=1.5)
    # 線は水平でまっすぐ、縁から 3 px 内側まで
    assert len(pls) >= 6
    for pl in pls:
        assert _max_deviation(pl) < 0.5
        ys = [y for _, y in pl]
        assert max(ys) - min(ys) < 1.0 and min(ys) >= 12.5 and max(ys) <= 47.5
        xs = [x for x, _ in pl]
        assert min(xs) < 15 and max(xs) > 105
    # 間隔はほぼピッチ、全体で領域が埋まる
    ys = sorted(np.mean([y for _, y in pl]) for pl in pls)
    gaps = np.diff(ys)
    assert gaps.min() > 0.5 * 4.8 and gaps.max() < 1.6 * 4.8
    rend = SoftRenderer(H, W, 6.0, 0.3, 6.0, DEV)
    img = rend.render(Strokes.from_polylines(pls, DEV)).detach().numpy()
    assert img[wide].mean() > 0.85
    # 斜めの場なら斜めの直線になる
    th = np.radians(30)
    c2r = np.full((H, W), np.cos(2 * th), dtype=np.float32)
    s2r = np.full((H, W), np.sin(2 * th), dtype=np.float32)
    plr = flow_passes(wide, c2r, s2r, 6.0, 4.8, 6.0, 6.0, simplify_px=1.5)
    assert len(plr) >= 6
    for pl in plr:
        (x0, y0), (x1, y1) = pl[0], pl[-1]
        ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 180
        assert abs(ang - 30) < 3 and _max_deviation(pl) < 0.5


def test_add_passes_flow_uses_existing_lines_for_the_direction():
    H, W = 80, 120
    target = np.zeros((H, W), dtype=np.float32)
    target[20:60, 20:100] = 1.0
    hint = resample([(20.0, 70.0), (100.0, 10.0)], 6.0)  # 塗りの上を斜めに横切るヒント線
    new = add_passes(np.zeros_like(target), target, 6.0, 0.5, 6.0, 1.8, 6.0, wide_px=12.0, pitch_px=4.8,
                     fill="flow", field_pls=[hint], field_sigma_px=12.0, simplify_px=1.5)
    long = [pl for pl in new if np.hypot(pl[-1][0] - pl[0][0], pl[-1][1] - pl[0][1]) > 20]
    assert len(long) >= 5
    want = np.degrees(np.arctan2(-60, 80)) % 180
    for pl in long:
        (x0, y0), (x1, y1) = pl[0], pl[-1]
        assert abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180 - want) < 5


def test_direction_field_edges_make_a_narrow_bar_follow_its_own_axis():
    from ring_plotter.hatch import direction_field, mask_edges

    H, W = 100, 160
    bar = np.zeros((H, W), dtype=bool)
    bar[48:60, 20:140] = True  # 幅 12 px の横長の塗り（眉）
    vertical = [[(float(x), 0.0), (float(x), 42.0)] for x in range(10, 150, 8)]  # 上は縦線（髪）、塗りの 6 px 手前まで
    vertical += [[(float(x), 66.0), (float(x), 100.0)] for x in range(10, 150, 8)]
    c2, s2 = direction_field(vertical, (H, W), 12.0, fallback_mask=bar)
    assert c2[54, 80] < -0.9  # 縁を入れないと縦（θ = 90°、cos 2θ = −1）
    c2e, s2e = direction_field(vertical, (H, W), 12.0, fallback_mask=bar, edge_pls=mask_edges(bar), edge_weight=1.0)
    assert c2e[54, 80] > 0.5  # 縁を入れると自分の長手方向（横）


def test_outline_passes_only_where_the_edge_faces_paper():
    from ring_plotter.hatch import outline_passes

    H, W = 80, 120
    wide = np.zeros((H, W), dtype=bool)
    wide[20:60, 20:100] = True
    target = wide.astype(np.float32)
    # 四方が紙: 閉じた輪郭が 1 本、縁からペン半径（3 px）内側
    lines, centre = outline_passes(wide, target, 6.0, 6.0, 6.0, simplify_px=0.5)
    assert len(lines) == 1
    xs, ys = [x for x, _ in lines[0]], [y for _, y in lines[0]]
    assert abs(min(xs) - 22.5) < 1.5 and abs(max(xs) - 96.5) < 1.5 and abs(min(ys) - 22.5) < 1.5 and abs(max(ys) - 56.5) < 1.5
    assert np.hypot(lines[0][0][0] - lines[0][-1][0], lines[0][0][1] - lines[0][-1][1]) < 1.5
    assert centre.sum() > 100
    # 上側が既存のインクに接していれば、その辺には引かず、開いた線になる
    target2 = target.copy()
    target2[:20, :] = 1.0
    lines2, _ = outline_passes(wide, target2, 6.0, 6.0, 6.0, simplify_px=0.5)
    assert len(lines2) == 1
    top = [x for x, y in lines2[0] if y < 24]  # 上辺に残るのは角の点だけ
    assert all(x < 26 or x > 93 for x in top)
    assert np.hypot(lines2[0][0][0] - lines2[0][-1][0], lines2[0][0][1] - lines2[0][-1][1]) > 50


def test_add_passes_flow_outline_then_hatch_inside():
    H, W = 80, 120
    target = np.zeros((H, W), dtype=np.float32)
    target[20:60, 20:100] = 1.0
    c2 = np.ones((H, W), dtype=np.float32)
    new = add_passes(np.zeros_like(target), target, 6.0, 0.5, 6.0, 1.8, 6.0, wide_px=12.0, pitch_px=4.8,
                     fill="flow", field_pls=[[(0.0, 40.0), (120.0, 40.0)]], field_sigma_px=12.0, simplify_px=1.5,
                     edge_weight=1.0, outline=True)
    closed = [pl for pl in new if np.hypot(pl[0][0] - pl[-1][0], pl[0][1] - pl[-1][1]) < 1.5 and len(pl) > 4]
    assert len(closed) == 1  # 輪郭
    inner = [pl for pl in new if pl not in closed]
    assert len(inner) >= 4
    for pl in inner:  # 流線は輪郭の内側（y 22.5 の輪郭から 4.8 以上離れる）
        ys = [y for _, y in pl]
        assert min(ys) > 22.5 + 4.0 and max(ys) < 56.5 - 4.0 and max(ys) - min(ys) < 1.0
    rend = SoftRenderer(H, W, 6.0, 0.3, 6.0, DEV)
    img = rend.render(Strokes.from_polylines(new, DEV)).detach().numpy()
    assert img[target > 0.5].mean() > 0.85


def test_cover_keeps_fill_lines_at_pitch_and_flags_wide_gaps():
    from ring_plotter.refine_passes import Cover

    H, W = 60, 100
    rend = SoftRenderer(H, W, 6.0, 0.3, 6.0, DEV)
    target = torch.zeros(H, W)
    target[12:48, 10:90] = 1.0  # 幅 36 px の塗り、ペン 6 px、ピッチ 4 px
    interior = torch.zeros(H, W)
    interior[16:44, 14:86] = 1.0  # ペン半径だけ内側
    cover = Cover(SoftRenderer(H, W, 4.0, 0.3, 6.0, DEV), interior, 1.0)
    # 間隔 6 px（ペン幅）の線: 主の像は塗りを埋めるが、cover の項は欠けを見つける
    sparse = [resample([(10.0, float(y)), (90.0, float(y))], 6.0) for y in range(15, 46, 6)]
    s = Strokes.from_polylines(sparse, DEV)
    assert rend.render(s).detach()[20:40, 20:80].mean() > 0.95
    assert float(cover.loss(s)) > 0.01
    assert cover.missing(s)[16:44, 14:86].mean() > 0.2
    # 間隔 4 px の線は cover も満たす。削除は cover があると間引かない
    dense = [resample([(10.0, float(y)), (90.0, float(y))], 6.0) for y in range(14, 47, 4)]
    sd = Strokes.from_polylines(dense, DEV)
    assert float(cover.loss(sd)) < 3e-3  # ちょうどピッチの線でも縁のぼかしの分だけ継ぎ目が薄く残る
    # （内部の外にある両端の線は cover の対象外なので、主の判定だけで消えてよい）
    kept, _ = remove_passes(dense, [0] * len(dense), rend, target, 0.3, 0.0, DEV, cover=cover)
    assert {pl[0][1] for pl in kept} >= {18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0}
    kept_nc, _ = remove_passes(dense, [0] * len(dense), rend, target, 0.3, 0.0, DEV)
    assert len(kept_nc) <= 5  # cover がないと 1 本おきに間引かれる


def test_join_lines_connects_continuing_ends_in_either_orientation():
    target = np.ones((40, 80), dtype=np.float32)
    a = [(10.0, 20.0), (20.0, 20.0), (30.0, 20.0)]
    b = [(50.0, 20.0), (42.0, 20.0), (34.0, 20.0)]  # 逆向きに並んでいる
    c = [(10.0, 5.0), (30.0, 5.0)]  # 離れている線
    pls, origin, n = join_lines([a, b, c], [0, 1, 0], target, 6.0, 45, 10.0)
    assert n == 1 and len(pls) == 2
    joined = max(pls, key=len)
    xs = [x for x, _ in joined]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)  # 一方向に並ぶ
    assert len(joined) == 6
    # 間に白紙があればつながない
    target[:, 30:34] = 0.0
    _, _, n2 = join_lines([a, b], [0, 0], target, 6.0, 45, 10.0)
    assert n2 == 0
    # 向きが続いていなければつながない
    d = [(32.0, 22.0), (32.0, 40.0)]
    _, _, n3 = join_lines([a, d], [0, 0], np.ones_like(target), 6.0, 45, 10.0)
    assert n3 == 0


def test_split_corners_splits_at_sharp_corner_only():
    pl = resample([(0.0, 0.0), (30.0, 0.0), (30.0, 30.0)], 3.0)  # 直角
    pls, origin, n = split_corners([pl], [0], 75, 6.0)
    assert n == 1 and len(pls) == 2 and pls[0][-1] == pls[1][0] == (30.0, 0.0)
    arc = resample([(0.0, 0.0), (30.0, 5.0), (60.0, 0.0)], 3.0)  # ゆるい曲がり
    _, _, n2 = split_corners([arc], [0], 75, 6.0)
    assert n2 == 0
