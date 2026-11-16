"""線画の画像（PNG など）を、ペンで描ける折れ線にする。

Inkscape の中心線トレースの代わり。やること:
  1. 二値化（Otsu、または閾値指定）
  2. 距離変換で「線」と「塗り」を分ける。幅が fill_width より太い部分は塗り
  3. 線: 細線化（スケルトン）→ 画素の連なりを折れ線にたどる → 短いひげを落とす
     → 交点で、まっすぐ続く線どうしをつなぐ → ガウス平滑化で階段をならす → 点を間引く
  3'. ペン幅が与えられたら、線の局所幅（距離変換）をペン幅で割って本数を決め、
     中心線に沿って法線方向にずらした平行パスで太さを出す（multipass）
  4. 塗り: 輪郭線と、指定の角度・間隔のハッチング
  5. SVG（mm 単位）に書く。svg2ring でそのまま変換できる

座標は画像の画素（x 右、y 下）で処理し、最後に mm にする。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

Polyline = list[tuple[float, float]]

_K8 = np.ones((3, 3), dtype=np.uint8)
_NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGH8 = _NEIGH4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


# --- 1. 二値化 -------------------------------------------------------------
def bright_gaps(gray: np.ndarray, contrast: float = 48, floor: float = 64) -> np.ndarray:
    """二重線の間の 1〜2px の隙間のように、両側（2px 以内）より contrast 以上明るい画素。

    こういう画素は閾値で暗い側に入ってしまっても紙の白なので、インクから除く。
    4方向（縦・横・斜め2つ）のどれかで尾根になっていればよい。
    """
    g = gray.astype(np.int16)
    H, W = g.shape
    P = np.pad(g, 2, mode="edge")
    ridge = np.zeros(g.shape, dtype=bool)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        def sh(k):
            return P[2 + k * dr:2 + k * dr + H, 2 + k * dc:2 + k * dc + W]

        side_a = np.minimum(sh(-1), sh(-2))
        side_b = np.minimum(sh(1), sh(2))
        ridge |= g - np.maximum(side_a, side_b) >= contrast
    return ridge & (g >= floor)


def load_ink(path: str, threshold: int | None = None, invert: bool = False,
             gap_contrast: float = 48) -> np.ndarray:
    """画像を読み、インク（描く部分）が True の配列を返す。

    gap_contrast > 0 なら、暗い線に挟まれた明るい尾根（二重線の狭い隙間）を
    閾値より暗くてもインクにしない（bright_gaps）。
    """
    from PIL import Image
    from skimage.filters import threshold_otsu

    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    g = np.asarray(im.convert("L"))
    if invert:
        g = 255 - g
        threshold = None if threshold is None else 255 - threshold
    if threshold is None:
        threshold = int(threshold_otsu(g))
    ink = g < threshold
    if gap_contrast > 0:
        ink &= ~bright_gaps(g, gap_contrast)
    return ink


# --- 2. 線と塗りの分離 ---------------------------------------------------------
def _disk(radius: float) -> np.ndarray:
    r = int(math.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= radius * radius + 1e-9


def typical_half_width(ink: np.ndarray) -> float:
    """線の典型的な半幅（スケルトン上の距離の中央値）。"""
    from skimage.morphology import skeletonize

    dist = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink)
    return float(np.median(dist[sk])) if sk.any() else 1.0


def close_gaps(ink: np.ndarray, close_px: float, keep_len_px: float = 0.0) -> np.ndarray:
    """半径 close_px の閉じ処理。ただし、埋まる隙間のうち細長くて keep_len_px 以上続くもの
    （二重線の間の隙間）は埋めない。ハッチングの密集部の隙間は短いので埋まる。
    """
    from skimage.morphology import skeletonize

    if close_px <= 0:
        return ink
    closed = ndi.binary_closing(ink, structure=_disk(close_px), border_value=0) | ink
    if keep_len_px <= 0:
        return closed
    gaps = closed & ~ink
    lab, n = ndi.label(gaps, structure=_K8)
    if not n:
        return closed
    # 隙間ごとに中心線の長さと枝分かれの数を見る。二重線の隙間は長くて枝がほぼない。
    # ハッチングの密集部の隙間は、線の端どうしでつながって網目になるので枝が多い
    sk = skeletonize(gaps)
    deg = ndi.convolve(sk.astype(np.uint8), _K8, mode="constant") - sk
    length = np.bincount(lab[sk], minlength=n + 1)
    branches = np.bincount(lab[sk & (deg >= 3)], minlength=n + 1)
    simple = (length >= keep_len_px) & (branches * 16 <= length)
    simple[0] = False
    if simple.any():
        closed &= ~np.isin(lab, np.nonzero(simple)[0])
    return closed


def split_fills(ink: np.ndarray, fill_half_width_px: float, close_px: float = 0.0,
                taper_half_width_px: float | None = None, taper_len_px: float = 80.0,
                keep_gap_px: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """太い部分を塗りとして分ける。(線, 塗り) を返す。

    半幅が fill_half_width_px を超える部分を「芯」とする。芯から出ていて半幅が
    taper_half_width_px を超える「腕」のうち、長さが taper_len_px 以内で終わるものは
    細くなっていく先端（髪の先など）とみなして塗りに含める。それより長く続く腕は線
    （塗りにつながる太い輪郭線など）なので塗りに含めない。先端はその先で線の太さまで
    細くなった部分が「線」として残る（まつげの付け根から出る線など）。
    taper_half_width_px が None なら、線の典型的な半幅の 1.5 倍（少なくとも 1px）にする。

    close_px > 0 なら、その半径で閉じ処理をしてから太さを見る。細かいハッチングや網点の
    密集部（線の隙間が 2·close_px 以下）が、ひとつの塗りにまとまる。二重線の間のような
    keep_gap_px 以上続く細い隙間は埋めない（close_gaps）。
    """
    if fill_half_width_px <= 0:
        return ink, np.zeros_like(ink)
    closed = close_gaps(ink, close_px, keep_gap_px)
    dist = ndi.distance_transform_edt(closed)
    hi = fill_half_width_px
    # 芯: 半幅 hi 超の部分。太めの線の中にできる1〜2px の筋や、線の途中の小さな膨らみは芯にしない
    core = ndi.binary_opening(dist > hi, structure=_K8)
    cl, nc = ndi.label(core, structure=_K8)
    if nc:
        area = np.bincount(cl.ravel())
        core = np.isin(cl, np.nonzero(area >= 2 * hi * hi)[0][1:])
    if not core.any():
        return ink, np.zeros_like(ink)
    if taper_half_width_px is None:
        taper_half_width_px = max(1.0, 1.5 * typical_half_width(ink))
    lo = min(taper_half_width_px, hi)
    # 芯からの距離で、芯のまわりの殻（幅 hi−lo 程度）と、そこから出る腕を分ける
    dcore = ndi.distance_transform_edt(~core)
    margin = hi - lo + 1.5
    thick = dist > lo
    shell = thick & (dcore <= margin)
    arms = thick & (dcore > margin)
    axis = shell
    al, n = ndi.label(arms, structure=_K8)
    if n:
        touch = np.unique(al[ndi.binary_dilation(shell, structure=_K8) & arms])
        touch = touch[touch > 0]
        if touch.size:
            reach = np.asarray(ndi.maximum(dcore, al, touch)) - margin
            axis = axis | np.isin(al, touch[reach <= taper_len_px])
    fills = ndi.binary_dilation(axis, structure=_disk(lo + 1)) & closed
    # 線: 塗りに触れている画素を1周ぶん除き、塗りの縁の帯の中だけにある切れ端を落とす
    strokes = ink & ~ndi.binary_dilation(fills, structure=_K8)
    band = ndi.binary_dilation(fills, structure=_K8, iterations=3)
    lab, _ = ndi.label(strokes, structure=_K8)
    keep = np.unique(lab[strokes & ~band])
    strokes = np.isin(lab, keep[keep > 0])
    return strokes, fills


# --- 3. スケルトンをたどる -------------------------------------------------------
@dataclass
class Path:
    pts: list[tuple[int, int]]  # (row, col)。端が交点なら交点の重心（float）を含む
    j0: int | None = None  # 始点側の交点 id
    j1: int | None = None  # 終点側の交点 id
    closed: bool = False

    def length(self) -> float:
        return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(self.pts, self.pts[1:]))


def trace_skeleton(skel: np.ndarray) -> tuple[list[Path], dict[int, tuple[float, float]]]:
    """スケルトン画像を、交点で区切った画素の列にする。"""
    sk = skel.astype(np.uint8)
    deg = ndi.convolve(sk, _K8, mode="constant") - sk  # 8近傍の数
    junc = (deg >= 3) & skel
    jl, nj = ndi.label(junc, structure=_K8)
    jcent = {i + 1: tuple(c) for i, c in enumerate(ndi.center_of_mass(junc, jl, range(1, nj + 1)))}
    rest = skel & ~junc
    rl, nr = ndi.label(rest, structure=_K8)
    H, W = skel.shape

    def junction_at(r, c):
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, H), max(c - 1, 0), min(c + 2, W)
        ids = jl[r0:r1, c0:c1]
        ids = ids[ids > 0]
        return int(ids[0]) if ids.size else None

    paths: list[Path] = []
    objs = ndi.find_objects(rl)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        sub = rl[sl] == i
        rr, cc = np.nonzero(sub)
        remaining = {(int(r) + sl[0].start, int(c) + sl[1].start) for r, c in zip(rr, cc)}
        # 端点（成分内の近傍が1つ以下）から歩き始める
        def ncount(p):
            return sum((p[0] + dr, p[1] + dc) in remaining for dr, dc in _NEIGH8)

        while remaining:
            ends = [p for p in remaining if ncount(p) <= 1]
            start = min(ends) if ends else min(remaining)
            cur = start
            remaining.discard(cur)
            pts = [cur]
            while True:
                nxt = None
                for dr, dc in _NEIGH8:  # 4近傍を先に見る（斜めで飛ばさない）
                    q = (cur[0] + dr, cur[1] + dc)
                    if q in remaining:
                        nxt = q
                        break
                if nxt is None:
                    break
                remaining.discard(nxt)
                pts.append(nxt)
                cur = nxt
            p = Path(pts)
            p.j0, p.j1 = junction_at(*pts[0]), junction_at(*pts[-1])
            if p.j0 is not None:
                p.pts.insert(0, jcent[p.j0])
            if p.j1 is not None:
                p.pts.append(jcent[p.j1])
            if p.j0 is None and p.j1 is None and len(pts) > 2:
                a, b = pts[0], pts[-1]
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:  # 閉じた輪
                    p.closed = True
                    p.pts.append(p.pts[0])
            paths.append(p)
    return paths, jcent


def prune(paths: list[Path], spur_px: float, min_px: float) -> list[Path]:
    """交点から出る短いひげと、短い孤立した線と、交点どうしをつなぐ短い線を落とす。

    骨格の交点は 1 画素ずれて 2 つに割れることが多く、その間の 1〜2 画素が両端とも交点の
    「線」として残る。描くとペン下ろし 1 回ぶんの点になるので、min_px 未満なら落とす
    （つないでいた線どうしに min_px 以下の隙間が残るが、ペンの線幅の下では見えない）。
    交点の塊に両側で接する画素は「交点から出て同じ交点に戻る」線になる。その長さは交点の
    重心までの往復で膨らむので、交点からの広がり（重心から最も遠い画素までの距離）で判定する。
    """
    out = []
    for p in paths:
        L = p.length()
        free = (p.j0 is None) + (p.j1 is None)
        if free == 1 and L < spur_px:
            continue
        if free == 2 and L < min_px and not p.closed:
            continue
        if free == 0:
            if p.j0 == p.j1:
                c = p.pts[0]
                L = max(math.hypot(q[0] - c[0], q[1] - c[1]) for q in p.pts[1:-1]) if len(p.pts) > 2 else 0.0
            if L < min_px:
                continue
        out.append(p)
    return out


def _tangent(pts, from_start: bool, n: int = 6):
    """端から n 点ぶん内側へ向かう方向（単位ベクトル）。"""
    seq = pts[:n + 1] if from_start else pts[::-1][:n + 1]
    a, b = seq[0], seq[-1]
    v = (b[0] - a[0], b[1] - a[1])
    h = math.hypot(*v)
    return (v[0] / h, v[1] / h) if h > 0 else (0.0, 0.0)


def join_at_junctions(paths: list[Path], max_turn_deg: float = 60.0) -> list[Path]:
    """交点で、向きがまっすぐ続く線どうしをつなぐ（髪の毛が交差を越えて1本になる）。"""
    paths = [Path(list(p.pts), p.j0, p.j1, p.closed) for p in paths]
    alive = {id(p): p for p in paths}
    incident: dict[int, list[tuple[int, bool]]] = {}  # 交点 id → [(path id, at_start)]
    for p in paths:
        if p.j0 is not None:
            incident.setdefault(p.j0, []).append((id(p), True))
        if p.j1 is not None:
            incident.setdefault(p.j1, []).append((id(p), False))
    cos_lim = math.cos(math.radians(180 - max_turn_deg))  # 2方向の内積がこれ以下なら「まっすぐ」

    for j, ends in incident.items():
        ends = [(pid, s) for pid, s in ends if pid in alive]
        while len(ends) >= 2:
            best = None
            for i in range(len(ends)):
                for k in range(i + 1, len(ends)):
                    (pa, sa), (pb, sb) = ends[i], ends[k]
                    if pa == pb:
                        continue
                    ta = _tangent(alive[pa].pts, sa)
                    tb = _tangent(alive[pb].pts, sb)
                    dot = ta[0] * tb[0] + ta[1] * tb[1]
                    if dot <= cos_lim and (best is None or dot < best[0]):
                        best = (dot, i, k)
            if best is None:
                break
            _, i, k = best
            (pa, sa), (pb, sb) = ends[i], ends[k]
            A, B = alive[pa], alive[pb]
            # A を「j で終わる向き」、B を「j で始まる向き」にそろえて連結
            a_pts, a_far = (A.pts[::-1], A.j1) if sa else (A.pts, A.j0)
            b_pts, b_far = (B.pts, B.j1) if sb else (B.pts[::-1], B.j0)
            merged = Path(a_pts + b_pts[1:], a_far, b_far)
            del alive[pa], alive[pb]
            alive[id(merged)] = merged
            # 他の交点に残っている参照を付け替える
            for jj, lst in incident.items():
                for n, (pid, s) in enumerate(lst):
                    if pid == pa:
                        lst[n] = (id(merged), True)
                    elif pid == pb:
                        lst[n] = (id(merged), False)
            ends = [e for n, e in enumerate(ends) if n not in (i, k)]
            ends = [(pid, s) for pid, s in ends if pid in alive]
            if a_far == j or b_far == j:  # 自分自身に戻る輪
                break
    return list(alive.values())


# --- 平滑化と間引き ---------------------------------------------------------
def smooth(pts: list[tuple[float, float]], sigma: float, closed: bool = False) -> list[tuple[float, float]]:
    if sigma <= 0 or len(pts) < 3:
        return list(pts)
    a = np.asarray(pts, dtype=float)
    mode = "wrap" if closed else "nearest"
    if closed:
        a = a[:-1]
    b = ndi.gaussian_filter1d(a, sigma, axis=0, mode=mode)
    if closed:
        b = np.vstack([b, b[:1]])
    else:
        b[0], b[-1] = a[0], a[-1]  # 端点（交点）は動かさない
    return [tuple(p) for p in b]


def rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Douglas–Peucker で点を間引く。"""
    if len(pts) < 3 or eps <= 0:
        return list(pts)
    a = np.asarray(pts, dtype=float)
    keep = np.zeros(len(a), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(a) - 1)]
    while stack:
        i, k = stack.pop()
        if k - i < 2:
            continue
        p, q = a[i], a[k]
        d = q - p
        L = math.hypot(*d)
        seg = a[i + 1:k]
        if L < 1e-12:
            dist = np.linalg.norm(seg - p, axis=1)
        else:
            dist = np.abs((seg[:, 0] - p[0]) * d[1] - (seg[:, 1] - p[1]) * d[0]) / L
        m = int(dist.argmax())
        if dist[m] > eps:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, k)]
    return [tuple(p) for p in a[keep]]


# --- 3'. 平行パスで太さを出す -----------------------------------------------------
def path_widths(pts, dist: np.ndarray) -> np.ndarray:
    """スケルトンの画素列に沿った局所の線幅（px）。

    距離変換は中心の画素から最寄りの背景画素までの距離なので、幅 w（画素数）の線の中心では
    (w+1)/2 になる。幅は 2·dist − 1。pts は (row, col)。端の交点の重心は float なので最寄りの画素を見る。
    """
    H, W = dist.shape
    rr = np.clip(np.rint([p[0] for p in pts]).astype(int), 0, H - 1)
    cc = np.clip(np.rint([p[1] for p in pts]).astype(int), 0, W - 1)
    return np.maximum(1.0, 2.0 * dist[rr, cc] - 1.0)


def _runs(values: np.ndarray):
    """等しい値が続く区間 [(start, end_exclusive, value), ...]。"""
    out = []
    i = 0
    n = len(values)
    while i < n:
        k = i
        while k < n and values[k] == values[i]:
            k += 1
        out.append((i, k, values[i]))
        i = k
    return out


def pass_counts(width_px: np.ndarray, arclen_px: np.ndarray, pen_px: float, pitch_px: float,
                min_run_px: float, closed: bool = False) -> np.ndarray:
    """各点で何本のパスを描くか。n 本で覆える幅は pen + (n−1)·pitch なので、それに近い n。

    長さ min_run_px 未満しか続かない n の区間は、両隣の小さい方に揃える（交差で膨らむ
    骨格や、幅のぶれで 1 本と 2 本が短く行き来するのを抑える）。
    """
    n = np.maximum(1, np.rint((width_px - pen_px) / pitch_px).astype(int) + 1)
    if len(n) < 2:
        return n
    changed = True
    while changed:
        changed = False
        runs = _runs(n)
        if len(runs) < 2:
            break
        # 短い区間のうち、両隣との差が最大のものから潰す（隣に揃えると区間が併合されて長くなる）
        for idx, (a, b, v) in enumerate(runs):
            L = arclen_px[b - 1] - arclen_px[a]
            if L >= min_run_px:
                continue
            neigh = []
            if idx > 0:
                neigh.append(runs[idx - 1][2])
            elif closed:
                neigh.append(runs[-1][2])
            if idx < len(runs) - 1:
                neigh.append(runs[idx + 1][2])
            elif closed:
                neigh.append(runs[0][2])
            if not neigh:
                continue
            target = min(neigh)
            if target == v:
                # 閉じた線で最初と最後の区間が同じ本数だと、つなぎ目の向こうの隣が自分と同じ値になる。
                # 揃えても何も変わらないので飛ばす（飛ばさないと changed = True のまま止まらない）
                continue
            if idx in (0, len(runs) - 1) and not closed:
                # 端の短い区間は、本数が多ければ隣に揃える（入り抜きの細い端はそのまま残す）
                if v <= target:
                    continue
            n[a:b] = target
            changed = True
            break
    return n


def _normals(a: np.ndarray, closed: bool) -> np.ndarray:
    """折れ線の各点での単位法線（左手側）。接線は中央差分。"""
    if closed and len(a) > 2:
        prv = np.vstack([a[-2:-1], a[:-1]])
        nxt = np.vstack([a[1:], a[1:2]])
        t = nxt - prv
    else:
        t = np.gradient(a, axis=0)
    h = np.hypot(t[:, 0], t[:, 1])
    h[h == 0] = 1.0
    t /= h[:, None]
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def _drop_backtrack(pts: np.ndarray, base: np.ndarray) -> np.ndarray:
    """オフセットした折れ線で、進む向きが元の線と逆になる点（内側の角で折り返す点）を落とす。"""
    if len(pts) < 3:
        return pts
    d_base = np.diff(base, axis=0)
    d_off = np.diff(pts, axis=0)
    ok = np.ones(len(pts), dtype=bool)
    bad = (d_base * d_off).sum(axis=1) < 0
    ok[1:][bad] = False
    ok[0] = ok[-1] = True
    return pts[ok]


def multipass(pts, width_px: np.ndarray, pen_px: float, pitch_px: float, min_run_px: float,
              width_sigma_px: float, blend_sigma_px: float, closed: bool = False,
              simplify_px: float = 0.0) -> tuple[list[Polyline], int]:
    """平滑化した 1 本の中心線 pts（(row, col)）と、各点の幅から、n 本の平行パスを作る。

    幅は弧長方向にガウス平滑化（width_sigma_px）してから本数 n(s) にする。パス i のオフセットは
    (i − (n−1)/2)·pitch で、n が変わるところでは blend_sigma_px で滑らかにつなぐ。n が i+1 未満の
    区間ではパス i は存在しないので、そこで切れる（入りでは扇状に分かれて見える）。
    返り値は (パスの一覧（先頭が基準パス）, 最大本数)。パスは (row, col)。
    """
    a = np.asarray(pts, dtype=float)

    def single():
        return [rdp([tuple(p) for p in a], simplify_px) if simplify_px > 0 else [tuple(p) for p in a]], 1

    if len(a) < 2:
        return single()
    seg = np.hypot(*np.diff(a, axis=0).T)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    step = max(float(np.median(seg)), 1e-6)
    mode = "wrap" if closed else "nearest"
    w = np.asarray(width_px, dtype=float)
    if closed:
        w = w.copy()
        w[-1] = w[0]
    if width_sigma_px > 0:
        w = ndi.gaussian_filter1d(w, width_sigma_px / step, mode=mode)
    n = pass_counts(w, arclen, pen_px, pitch_px, min_run_px, closed)
    nmax = int(n.max())
    if nmax == 1:
        return single()
    normals = _normals(a, closed)
    out: list[Polyline] = []
    for i in range(nmax):
        active = n >= i + 1
        offset = (i - (n - 1) / 2.0) * pitch_px
        if blend_sigma_px > 0:
            # 存在しない区間は隣の値で埋めてから平滑化する（端で値が引っ張られないように）
            idx = np.arange(len(a))
            if active.any() and not active.all():
                offset = np.interp(idx, idx[active], offset[active])
            offset = ndi.gaussian_filter1d(offset, blend_sigma_px / step, mode=mode)
        off_pts = a + offset[:, None] * normals
        # active な区間ごとに 1 本
        for s, e, v in _runs(active.astype(np.int8)):
            if not v:
                continue
            if e - s < 2:
                continue
            piece = _drop_backtrack(off_pts[s:e], a[s:e])
            if simplify_px > 0:
                piece = np.asarray(rdp([tuple(p) for p in piece], simplify_px))
            if len(piece) >= 2:
                out.append([tuple(p) for p in piece])
    return out, nmax


# --- 4. 塗り -------------------------------------------------------------------
def fill_outlines(fills: np.ndarray) -> list[list[tuple[float, float]]]:
    from skimage.measure import find_contours

    pad = np.pad(fills, 1)
    out = []
    for c in find_contours(pad.astype(float), 0.5):
        out.append([(float(r) - 1, float(col) - 1) for r, col in c])  # (row, col)
    return out


def hatch(fills: np.ndarray, spacing_px: float, angle_deg: float, min_len_px: float,
          sample_px: float = 0.5, join_factor: float = 2.0) -> list[list[tuple[float, float]]]:
    """角度 angle_deg（画像の x 軸から、y 下向き）の平行線で塗りを埋める。(row, col) の点列。

    線分は1本おきに向きを変え、隣の線分の端が spacing_px × join_factor 以内で、つなぎの線が
    塗りの中を通るときはジグザグに1本の折れ線にする（ペン上げ下げが減る）。join_factor が 0 なら
    つながず、すべて2点の線分にする。つなぎの線のぶん密度が上がって濃く見えることがある。
    """
    if not fills.any() or spacing_px <= 0:
        return []
    H, W = fills.shape
    th = math.radians(angle_deg)
    u = np.array([math.sin(th), math.cos(th)])  # (row, col) 方向の単位ベクトル
    n = np.array([-u[1], u[0]])
    c = np.array([H / 2, W / 2])
    half = math.hypot(H, W) / 2
    s = np.arange(-half, half, spacing_px)
    t = np.arange(-half, half, sample_px)
    P = c[None, None, :] + s[:, None, None] * n[None, None, :] + t[None, :, None] * u[None, None, :]
    r = np.rint(P[..., 0]).astype(int)
    col = np.rint(P[..., 1]).astype(int)
    inside = (r >= 0) & (r < H) & (col >= 0) & (col < W)
    hit = np.zeros(r.shape, dtype=bool)
    hit[inside] = fills[r[inside], col[inside]]
    # 各線分を、1本おきに向きを変えてジグザグにつなぐ。つなぎの線が塗りの中を通るときだけ
    join_px = spacing_px * join_factor

    def inside(p, q):
        m = (p + q) / 2
        r0, c0 = int(round(m[0])), int(round(m[1]))
        return 0 <= r0 < H and 0 <= c0 < W and fills[r0, c0]

    out: list[list[tuple[float, float]]] = []
    prev_end = None  # 直前の線分の終点（np.array）と、それが属する折れ線
    for i in range(len(s)):
        row = hit[i]
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        segs = [(P[i, a], P[i, b]) for a, b in zip(starts, ends) if (b - a) * sample_px >= min_len_px]
        if i % 2:
            segs = [(q, p) for p, q in segs[::-1]]
        for p, q in segs:
            if (join_px > 0 and prev_end is not None and np.hypot(*(p - prev_end)) <= join_px
                    and inside(prev_end, p)):
                out[-1].append((float(p[0]), float(p[1])))
                out[-1].append((float(q[0]), float(q[1])))
            else:
                out.append([(float(p[0]), float(p[1])), (float(q[0]), float(q[1]))])
            prev_end = q
    return out


# --- まとめ -------------------------------------------------------------------
@dataclass
class TraceOptions:
    threshold: int | None = None
    invert: bool = False
    fill_half_width_px: float = 4.0  # これより太い部分は塗り
    close_px: float = 0.0  # 塗り判定の前の閉じ処理の半径（細かいハッチングを塗りにまとめる）
    keep_gap_px: float = 0.0  # これ以上続く細い隙間（二重線の間）は閉じ処理で埋めない（0 で無効）
    taper_half_width_px: float | None = None  # 塗りを細い先端まで伸ばす下限の半幅（None で自動）
    taper_len_px: float = 80.0  # これより長く続く腕は先端ではなく線とみなす
    outline_smooth_px: float = 0.7  # 塗りの輪郭の平滑化（線より弱くして先端を残す）
    spur_px: float = 6.0  # 交点から出るひげの長さの上限
    min_px: float = 4.0  # 孤立した線の長さの下限
    join_turn_deg: float = 60.0  # 交点でつなぐときに許す曲がり（0 でつながない）
    smooth_sigma_px: float = 1.5
    simplify_px: float = 0.4
    fill_mode: str = "hatch"  # hatch: 輪郭＋ハッチング / outline: 輪郭だけ / none: 塗りを無視
    hatch_spacing_px: float = 3.0
    hatch_angle_deg: float = 45.0
    hatch_min_px: float = 2.0
    hatch_join: float = 2.0  # 隣のハッチングをジグザグにつなぐ端の距離（間隔の倍数。0 でつながない）
    pen_px: float = 0.0  # ペンの線幅。0 より大きければ、太い線を平行パスで描く（multipass）
    pass_pitch: float = 0.8  # 平行パスの間隔（ペン幅の倍数）
    pass_min_run_px: float = 8.0  # 本数がこれより短くしか続かない区間は隣に揃える
    width_sigma_px: float = 5.0  # 幅の弧長方向の平滑化（ガウスのσ）
    pass_blend_px: float = 3.0  # 本数が変わるところでオフセットをなめらかにつなぐ長さ（ガウスのσ）


@dataclass
class TraceResult:
    strokes: list[Polyline]  # (x, y) 画素
    outlines: list[Polyline]
    hatches: list[Polyline]
    width_px: int
    height_px: int
    ink: np.ndarray | None = None
    fills: np.ndarray | None = None
    skeleton: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    passes: list[Polyline] = field(default_factory=list)  # 平行パスの 2 本目以降（1 本目は strokes）

    @property
    def all(self) -> list[Polyline]:
        return self.strokes + self.passes + self.outlines + self.hatches


def _xy(pts):
    return [(float(c), float(r)) for r, c in pts]


def trace(ink: np.ndarray, opt: TraceOptions | None = None) -> TraceResult:
    from skimage.morphology import skeletonize

    opt = opt or TraceOptions()
    H, W = ink.shape
    strokes_mask, fills = split_fills(ink, opt.fill_half_width_px if opt.fill_mode != "skeleton" else 0,
                                      opt.close_px, opt.taper_half_width_px, opt.taper_len_px, opt.keep_gap_px)
    skel = skeletonize(strokes_mask)
    paths, _ = trace_skeleton(skel)
    n0 = len(paths)
    paths = prune(paths, opt.spur_px, opt.min_px)
    n1 = len(paths)
    if opt.join_turn_deg > 0:
        paths = join_at_junctions(paths, opt.join_turn_deg)
    n2 = len(paths)
    strokes, passes = [], []
    dist = ndi.distance_transform_edt(ink) if opt.pen_px > 0 else None
    hist: dict[int, int] = {}
    for p in paths:
        pts = smooth(p.pts, opt.smooth_sigma_px, p.closed)
        if dist is None:
            pts = rdp(pts, opt.simplify_px)
            if len(pts) >= 2:
                strokes.append(_xy(pts))
            continue
        w = path_widths(p.pts, dist)
        pls, nmax = multipass(pts, w, opt.pen_px, opt.pen_px * opt.pass_pitch, opt.pass_min_run_px,
                              opt.width_sigma_px, opt.pass_blend_px, p.closed, opt.simplify_px)
        hist[nmax] = hist.get(nmax, 0) + 1
        if pls and len(pls[0]) >= 2:
            strokes.append(_xy(pls[0]))
        passes += [_xy(pl) for pl in pls[1:] if len(pl) >= 2]
    outlines, hatches = [], []
    if opt.fill_mode in ("hatch", "outline") and fills.any():
        for c in fill_outlines(fills):
            c = rdp(smooth(c, opt.outline_smooth_px, closed=True), opt.simplify_px)
            if len(c) >= 3:
                outlines.append(_xy(c))
        if opt.fill_mode == "hatch":
            hatches = [_xy(h) for h in hatch(fills, opt.hatch_spacing_px, opt.hatch_angle_deg, opt.hatch_min_px,
                                             join_factor=opt.hatch_join)]
    res = TraceResult(strokes, outlines, hatches, W, H, ink, fills, skel, passes=passes)
    res.notes.append(f"スケルトン {int(skel.sum())} px → 線 {n0}本 → ひげ除去 {n1}本 → 交点で連結 {n2}本")
    if dist is not None:
        by_n = "、".join(f"{k}本: {hist[k]}" for k in sorted(hist))
        res.notes.append(f"平行パス（最大本数ごとの線の数）{by_n} → 追加のパス {len(passes)}本")
    res.notes.append(f"塗り {int(fills.sum())} px → 輪郭 {len(outlines)}本、ハッチング {len(hatches)}本")
    return res


# --- 出力 -----------------------------------------------------------------------
def write_svg(res: TraceResult, path: str, mm_per_px: float, stroke_mm: float = 0.3) -> None:
    w, h = res.width_px * mm_per_px, res.height_px * mm_per_px

    def poly(pl):
        return " ".join(f"{x * mm_per_px:.3f},{y * mm_per_px:.3f}" for x, y in pl)

    groups = [("strokes", "#000000", res.strokes), ("passes", "#008040", res.passes),
              ("fill-outlines", "#c00000", res.outlines), ("hatch", "#0060c0", res.hatches)]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
             f'width="{w:.3f}mm" height="{h:.3f}mm" viewBox="0 0 {w:.3f} {h:.3f}">']
    for gid, color, pls in groups:
        if not pls:
            continue
        lines.append(f'<g id="{gid}" inkscape:groupmode="layer" inkscape:label="{gid}" fill="none" '
                     f'stroke="{color}" stroke-width="{stroke_mm}" stroke-linecap="round" stroke-linejoin="round">')
        lines += [f'<polyline points="{poly(pl)}"/>' for pl in pls]
        lines.append("</g>")
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def write_overlay(res: TraceResult, image_path: str, out_path: str, scale: int = 2) -> None:
    """元の画像を薄くして、その上にトレース結果を描く確認用 PNG。"""
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("L").convert("RGB")
    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.7)
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for pls, color in ((res.hatches, (60, 140, 220)), (res.outlines, (220, 40, 40)), (res.passes, (0, 140, 70)),
                       (res.strokes, (0, 0, 0))):
        for pl in pls:
            dr.line([(x * scale, y * scale) for x, y in pl], fill=color, width=1)
            dr.ellipse([pl[0][0] * scale - 1.5, pl[0][1] * scale - 1.5, pl[0][0] * scale + 1.5, pl[0][1] * scale + 1.5],
                       fill=(0, 160, 60))
    im.save(out_path)


def write_render(res: TraceResult, out_path: str, pen_px: float, scale: int = 2) -> None:
    """全部の線をペン幅で白地に描いた、紙の上の見え方の確認用 PNG。"""
    from PIL import Image, ImageDraw

    im = Image.new("L", (res.width_px * scale, res.height_px * scale), 255)
    dr = ImageDraw.Draw(im)
    wpx = max(1, int(round(pen_px * scale)))
    r = wpx / 2
    for pl in res.all:
        xy = [(x * scale, y * scale) for x, y in pl]
        dr.line(xy, fill=0, width=wpx, joint="curve")
        for x, y in (xy[0], xy[-1]):  # 丸い端
            dr.ellipse([x - r, y - r, x + r, y + r], fill=0)
    im.save(out_path)
