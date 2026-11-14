"""方向場に沿ったハッチング: 太い欠けを、まわりの線の向きに揃えた等間隔の流線で塗る。

人が塗るときは、輪郭や近くの線の向きに合わせて、まっすぐか緩く曲がった線を並べる。
距離変換の等高線（refine_passes.contour_passes）は欠け領域の縁の形をなぞるので、既存の線の
間に残った不規則な隙間では玉ねぎ状・渦巻きになる。ここでは
  1. 既存の折れ線の接線を倍角 (cos 2θ, sin 2θ) で画像に書き込み、ガウスぼかしで広げて方向場にする
     （倍角なので線の向きの正負は区別しない）。近くに線がない所は σ を倍々に広げて拾い、
     それでも無ければ欠け領域の主軸の向きにする
  2. 欠け領域の中で、方向場の流線を間隔 d_sep で等間隔に引く（Jobard–Lefer: 引いた流線の
     両脇 d_sep の位置から次の流線を出し、他の流線に d_sep/2 まで近づいたら止める）
座標は画素（x 右、y 下）。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import ndimage as ndi

from .raster import _K8, rdp

Polyline = list[tuple[float, float]]

# 向きの既定: 追加する線は、全体としてこの向き（紙の下向き、少し右）に進むように並べる。
# 隣の線を逆向きに描くと、向きで決まる横ずれが逆になり、片側が開いて白い筋になる。
# 流線は親の線と同じ向きにし、成分の最初の線やそれ以外の追加線はこの向きに揃える
ORIENT = (0.287, 0.958)  # (0.3, 1) を正規化


def orient(pl: Polyline, ref=ORIENT) -> Polyline:
    """折れ線全体の進む向きが ref と反対なら逆順にする。"""
    vx, vy = pl[-1][0] - pl[0][0], pl[-1][1] - pl[0][1]
    return pl[::-1] if vx * ref[0] + vy * ref[1] < 0 else pl


# --- 方向場 -----------------------------------------------------------------------
def _sample_segments(pls: list[Polyline], step: float = 1.0):
    """全線分を step 間隔で標本化した点 (S, 2) と、接線の倍角 (cos 2θ, sin 2θ) (S, 2)。"""
    a = np.array([p for pl in pls for p in pl[:-1]], dtype=np.float64).reshape(-1, 2)
    b = np.array([p for pl in pls for p in pl[1:]], dtype=np.float64).reshape(-1, 2)
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-9
    a, d, L = a[keep], d[keep], L[keep]
    n = np.maximum(1, np.ceil(L / step)).astype(int)
    idx = np.repeat(np.arange(len(a)), n)
    j = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)
    t = (j + 0.5) / n[idx]
    pts = a[idx] + d[idx] * t[:, None]
    ux, uy = d[idx, 0] / L[idx], d[idx, 1] / L[idx]
    return pts, np.stack([ux * ux - uy * uy, 2 * ux * uy], axis=1)


def _block_sum(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    H, W = a.shape
    Hp, Wp = -(-H // f) * f, -(-W // f) * f
    b = np.zeros((Hp, Wp), dtype=a.dtype)
    b[:H, :W] = a
    return b.reshape(Hp // f, f, Wp // f, f).sum(axis=(1, 3))


def _upsample(a: np.ndarray, f: int, shape) -> np.ndarray:
    if f == 1:
        return a[:shape[0], :shape[1]]
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)[:shape[0], :shape[1]]


def mask_edges(mask: np.ndarray) -> list[Polyline]:
    """二値領域の縁（画素の境界）を折れ線 (x, y) にする。"""
    from skimage.measure import find_contours

    return [[(float(c), float(r)) for r, c in cont] for cont in find_contours(mask.astype(np.float32), 0.5) if len(cont) >= 2]


def direction_field(pls: list[Polyline], shape: tuple[int, int], sigma_px: float, fallback_mask: np.ndarray | None = None,
                    levels: int = 5, min_coherence: float = 0.15, edge_pls: list[Polyline] | None = None,
                    edge_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """折れ線の接線から方向場 (cos 2θ, sin 2θ)（各 shape、float32）を作る。

    線の標本点（1 px ごと）の倍角ベクトルを画像に足し込み、σ = sigma_px のガウスで重み付き平均する。
    edge_pls（塗る領域の縁など）も重み edge_weight で同じように足す。
    線が近くにない（重みが小さい）か、向きがばらばら（平均の長さが重みの min_coherence 未満）の画素は
    σ を 2 倍、4 倍… と広げて（levels 段階）埋める。fallback_mask の中でそれでも決まらない画素は
    その連結成分の主軸の向き、残りは最も近い決まった画素の値にする。
    計算は σ が 4 px 程度になる縮小画像で行い、最後に元の大きさに戻す。
    """
    from skimage.measure import regionprops

    H, W = shape
    ds = max(1, int(round(sigma_px / 4)))
    sig = sigma_px / ds
    Hs, Ws = -(-H // ds), -(-W // ds)
    A = np.zeros((H, W), dtype=np.float64)
    B = np.zeros_like(A)
    N = np.zeros_like(A)
    sources = [(pls, 1.0)] + ([(edge_pls, edge_weight)] if edge_pls and edge_weight > 0 else [])
    sources = [(p, w) for p, w in sources if p]
    for src, w in sources:
        pts, d2 = _sample_segments(src)
        ix = np.clip(np.floor(pts[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.floor(pts[:, 1]).astype(int), 0, H - 1)
        flat = iy * W + ix
        A += w * np.bincount(flat, weights=d2[:, 0], minlength=H * W).reshape(H, W)
        B += w * np.bincount(flat, weights=d2[:, 1], minlength=H * W).reshape(H, W)
        N += w * np.bincount(flat, minlength=H * W).reshape(H, W)
    A, B, N = _block_sum(A, ds), _block_sum(B, ds), _block_sum(N, ds)
    c2 = np.zeros((Hs, Ws), dtype=np.float64)
    s2 = np.zeros_like(c2)
    unset = np.ones((Hs, Ws), dtype=bool)
    mask_s = None if fallback_mask is None else _block_sum(fallback_mask.astype(np.int32), ds) > 0
    if sources:
        for k in range(levels):
            f = 2 ** k
            Ak, Bk, Nk = (ndi.gaussian_filter(_block_sum(x, f), sig) for x in (A, B, N))
            coh = np.hypot(Ak, Bk)
            # 線に沿って粗い画素あたり ds·f 個の標本がある。ぼかした重みが線から 2.5σ 以内に相当する値以上なら採用
            ok = (Nk > ds * f * 0.02 / sig) & (coh > min_coherence * np.maximum(Nk, 1e-12))
            with np.errstate(invalid="ignore", divide="ignore"):
                ck, sk = np.where(ok, Ak / coh, 0.0), np.where(ok, Bk / coh, 0.0)
            ok, ck, sk = (_upsample(x, f, (Hs, Ws)) for x in (ok, ck, sk))
            take = unset & ok
            c2[take], s2[take] = ck[take], sk[take]
            unset &= ~take
            if mask_s is not None and not (unset & mask_s).any():
                break
    if mask_s is not None and (unset & mask_s).any():
        lab, _ = ndi.label(mask_s, structure=_K8)
        for rp in regionprops(lab):
            sub = unset[rp.slice] & rp.image
            if not sub.any():
                continue
            ux, uy = -math.sin(rp.orientation), -math.cos(rp.orientation)  # 主軸の向き (x, y)
            c2[rp.slice][sub], s2[rp.slice][sub] = ux * ux - uy * uy, 2 * ux * uy
            unset[rp.slice][sub] = False
    if unset.any() and not unset.all():
        iy, ix = ndi.distance_transform_edt(unset, return_distances=False, return_indices=True)
        c2, s2 = c2[iy, ix], s2[iy, ix]
    return _upsample(c2, ds, (H, W)).astype(np.float32), _upsample(s2, ds, (H, W)).astype(np.float32)


# --- 流線 -------------------------------------------------------------------------
def _bilinear(img: np.ndarray, x: float, y: float) -> float:
    """画素中心 (i + 0.5) を格子点とする双一次補間。範囲外は最も近い画素。"""
    H, W = img.shape
    fx, fy = min(max(x - 0.5, 0.0), W - 1.0), min(max(y - 0.5, 0.0), H - 1.0)
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    tx, ty = fx - x0, fy - y0
    return float((img[y0, x0] * (1 - tx) + img[y0, x1] * tx) * (1 - ty) + (img[y1, x0] * (1 - tx) + img[y1, x1] * tx) * ty)


class _Field:
    def __init__(self, c2: np.ndarray, s2: np.ndarray):
        self.c2, self.s2 = c2, s2

    def direction(self, x: float, y: float, prev):
        """(x, y) の単位方向。prev と同じ側（内積 ≥ 0）の向きを返す。向きが決まらなければ None。"""
        c, s = _bilinear(self.c2, x, y), _bilinear(self.s2, x, y)
        if c * c + s * s < 1e-4:
            return None
        th = 0.5 * math.atan2(s, c)
        d = (math.cos(th), math.sin(th))
        if prev is not None and d[0] * prev[0] + d[1] * prev[1] < 0:
            d = (-d[0], -d[1])
        return d


class _PointGrid:
    """セル幅 cell の格子に点を入れ、半径 r ≤ cell 以内に点があるかを 3×3 セルで調べる。"""

    def __init__(self, cell: float):
        self.cell = cell
        self.cells: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def add(self, x: float, y: float) -> None:
        self.cells.setdefault((int(x // self.cell), int(y // self.cell)), []).append((x, y))

    def near(self, x: float, y: float, r: float) -> bool:
        cx, cy = int(x // self.cell), int(y // self.cell)
        r2 = r * r
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for px, py in self.cells.get((i, j), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < r2:
                        return True
        return False


class _Tracer:
    """1 つの連結成分の中で流線を引く。"""

    def __init__(self, edt: np.ndarray, half_pen: float, field: _Field, d_sep: float, d_test: float, step: float,
                 max_turn_deg: float = 45.0, max_steps: int = 20000):
        self.edt, self.half, self.field = edt, half_pen, field
        self.d_sep, self.d_test, self.h = d_sep, d_test, step
        self.cos_turn = math.cos(math.radians(max_turn_deg))
        self.max_steps = max_steps
        self.grid = _PointGrid(d_sep)  # 採用した流線の点
        self.lag = int(math.ceil(2 * d_sep / step))  # 自分自身との接近は、これより前の点とだけ調べる

    def inside(self, x: float, y: float) -> bool:
        H, W = self.edt.shape
        return 0 <= x < W and 0 <= y < H and _bilinear(self.edt, x, y) >= self.half

    def _march(self, p0, d0, own: _PointGrid, pending: list) -> list:
        """p0 から向き d0 に、止まるまで進む。p0 は含まない。own には lag 個遅れて自分の点を入れる。"""
        out = []
        p, d = p0, d0
        for _ in range(self.max_steps):
            d1 = self.field.direction(p[0], p[1], d)
            if d1 is None:
                break
            pm = (p[0] + 0.5 * self.h * d1[0], p[1] + 0.5 * self.h * d1[1])
            d2 = self.field.direction(pm[0], pm[1], d1)
            if d2 is None or d2[0] * d[0] + d2[1] * d[1] < self.cos_turn:
                break
            q = (p[0] + self.h * d2[0], p[1] + self.h * d2[1])
            if not self.inside(q[0], q[1]):
                lo, hi = p, q  # 境界まで二分して端を伸ばす
                for _ in range(4):
                    m = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
                    if self.inside(m[0], m[1]):
                        lo = m
                    else:
                        hi = m
                if math.hypot(lo[0] - p[0], lo[1] - p[1]) > 0.25 * self.h and not self.grid.near(lo[0], lo[1], self.d_test):
                    out.append(lo)
                break
            if self.grid.near(q[0], q[1], self.d_test) or own.near(q[0], q[1], self.d_test):
                break
            out.append(q)
            pending.append(q)
            if len(pending) > self.lag:
                own.add(*pending.pop(0))
            p, d = q, d2
        return out

    def trace(self, seed, ref=None) -> list:
        """seed から両側へ引く。ref があれば、出来た折れ線の向きを ref と同じ側にする（隣の線と同じ向きに描く）。"""
        d0 = self.field.direction(seed[0], seed[1], ref)
        if d0 is None:
            return []
        own = _PointGrid(self.d_sep)
        pending = [seed]
        fwd = self._march(seed, d0, own, pending)
        for q in pending:  # 前向きの点を全部入れて、後ろ向きがぶつかれるようにする（種のそばは除く）
            own.add(*q)
        own2 = _PointGrid(self.d_sep)
        for q in fwd[self.lag:]:
            own2.add(*q)
        back = self._march(seed, (-d0[0], -d0[1]), own2, [seed])
        return back[::-1] + [seed] + fwd

    def accept(self, pts: list) -> None:
        for x, y in pts:
            self.grid.add(x, y)


def _polyline_length(pts) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _hatch_component(edt: np.ndarray, half_pen: float, field: _Field, d_sep: float, step: float, min_len: float) -> list[Polyline]:
    inner = edt >= half_pen
    if not inner.any():
        return []
    tr = _Tracer(edt, half_pen, field, d_sep, 0.5 * d_sep, step)
    occ = np.zeros(edt.shape, dtype=bool)  # 採用した流線の点と、失敗した種のまわり（再出発の候補から外す）
    r = int(math.ceil(d_sep))
    stride = max(1, int(d_sep))
    queue: deque = deque()  # (x, y, 再出発の種か, 向きの手本)
    lines: list[Polyline] = []

    def block(x, y):
        iy, ix = int(round(y)), int(round(x))
        occ[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1] = True

    while True:
        if not queue:
            # 再出発: まだ流線から d_sep 以上離れている点を、奥から順に全部積む（EDT は一度で済ませる）。
            # 再出発の種は、使えなくても必ず occ に印を付けて、同じ場所を選び直さないようにする
            if occ.any():
                dist = ndi.distance_transform_edt(~occ)
                dist[~inner] = 0
                sub = dist[::stride, ::stride]
                iy, ix = np.nonzero(sub >= 0.95 * d_sep)
                if not len(iy):
                    break
                order = np.argsort(-sub[iy, ix], kind="stable")
                queue.extend((float(ix[i] * stride + 0.5), float(iy[i] * stride + 0.5), True, None) for i in order)
            else:
                iy, ix = np.unravel_index(int(np.argmax(edt)), edt.shape)
                queue.append((ix + 0.5, iy + 0.5, True, None))
        x, y, restart, ref = queue.popleft()
        if not tr.inside(x, y) or tr.grid.near(x, y, 0.95 * d_sep):
            if restart:
                block(x, y)
            continue
        pts = tr.trace((x, y), ref)
        if len(pts) < 2 or _polyline_length(pts) < min_len:
            block(x, y)
            continue
        if ref is None:  # 親がない線は既定の向きに。親がある線は親と同じ向き（trace が揃える）
            pts = orient(pts)
        tr.accept(pts)
        lines.append(pts)
        for i, (px, py) in enumerate(pts):
            occ[min(max(int(py), 0), occ.shape[0] - 1), min(max(int(px), 0), occ.shape[1] - 1)] = True
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
            tx, ty = b[0] - a[0], b[1] - a[1]
            n = math.hypot(tx, ty)
            if n < 1e-9:
                continue
            t = (tx / n, ty / n)
            nx, ny = -t[1] * d_sep, t[0] * d_sep
            queue.append((px + nx, py + ny, False, t))
            queue.append((px - nx, py - ny, False, t))
    return lines


def outline_passes(wide: np.ndarray, target: np.ndarray, pen_px: float, min_len_px: float, seg_px: float,
                   simplify_px: float = 0.5, smooth_px: float = 1.5, probe_px: float | None = None
                   ) -> tuple[list[Polyline], np.ndarray]:
    """塗る前に、領域の縁のうち外側が紙（target < 0.5）の部分へ輪郭線を引く。

    人が塗るときのように、見える縁を先に 1 本の線でなぞる。縁からペン半径内側の等高線を、
    外向きに probe_px（既定 ペン幅）進んだ先の目標が白い区間だけ残す（既存の線や、まだ塗っていない
    インクに接する縁には引かない）。輪郭線の折れ線と、輪郭線の中心を描いた二値画像を返す。
    """
    from PIL import Image, ImageDraw
    from skimage.measure import find_contours

    from .raster import smooth
    from .refine_passes import resample

    H, W = wide.shape
    probe = probe_px or pen_px
    edt, (near_r, near_c) = ndi.distance_transform_edt(wide, return_indices=True)
    out: list[Polyline] = []
    im = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im)
    for cont in find_contours(edt, pen_px / 2):
        r = np.clip(np.round(cont[:, 0]).astype(int), 0, H - 1)
        c = np.clip(np.round(cont[:, 1]).astype(int), 0, W - 1)
        # 外向き: 最も近い領域外の画素へ向かう向き。そこからさらに probe 進んだ先を見る
        vr, vc = near_r[r, c] - cont[:, 0], near_c[r, c] - cont[:, 1]
        n = np.hypot(vr, vc)
        n[n < 1e-9] = 1.0
        qr = np.clip(np.round(cont[:, 0] + vr / n * (n + probe)).astype(int), 0, H - 1)
        qc = np.clip(np.round(cont[:, 1] + vc / n * (n + probe)).astype(int), 0, W - 1)
        paper = target[qr, qc] < 0.5
        closed = bool(np.hypot(*(cont[0] - cont[-1])) < 1.0)
        if paper.all():
            runs = [np.arange(len(cont))]
        else:
            # 紙に面した区間に分ける。閉じた輪では区切り目を紙でない点に回してから分ける
            if closed:
                k = int(np.argmin(paper))
                cont, paper = np.roll(cont[:-1], -k, axis=0), np.roll(paper[:-1], -k)
            idx = np.nonzero(paper)[0]
            if not len(idx):
                continue
            cuts = np.nonzero(np.diff(idx) > 1)[0] + 1
            runs = np.split(idx, cuts)
        for run in runs:
            pts = [(float(cc), float(rr)) for rr, cc in cont[run]]
            if len(pts) < 2:
                continue
            pts = rdp(smooth(pts, smooth_px, closed and len(run) == len(cont)), simplify_px)
            if len(pts) >= 2 and _polyline_length(pts) >= min_len_px:
                out.append(resample(pts, seg_px))
                draw.line([(x, y) for x, y in pts], fill=255, width=1)
    return out, np.asarray(im) > 0


def flow_passes(wide: np.ndarray, c2: np.ndarray, s2: np.ndarray, pen_px: float, pitch_px: float, min_len_px: float,
                seg_px: float, simplify_px: float = 0.5, step_px: float | None = None) -> list[Polyline]:
    """太い領域 wide を、方向場 (c2, s2) の流線で間隔 pitch_px に塗る。

    線の中心は縁から pen/2 以上内側（距離変換で判定）。連結成分ごとに切り出して処理する。
    流線は simplify_px で間引き（ほぼまっすぐな所は 1 本の線分になる）、seg_px 以下に分け直す。
    """
    from .refine_passes import resample

    H, W = wide.shape
    lab, n = ndi.label(wide, structure=_K8)
    half = pen_px / 2
    step = step_px or max(1.0, pen_px / 2)
    pad = int(math.ceil(pitch_px)) + 2
    out: list[Polyline] = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        rows = slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, H))
        cols = slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, W))
        m = lab[rows, cols] == k
        edt = ndi.distance_transform_edt(m)
        if edt.max() < half:
            continue
        field = _Field(c2[rows, cols], s2[rows, cols])
        for pts in _hatch_component(edt, half, field, pitch_px, step, min_len_px):
            pts = rdp([(x + cols.start, y + rows.start) for x, y in pts], simplify_px)
            if len(pts) >= 2 and _polyline_length(pts) >= min_len_px:
                out.append(resample(pts, seg_px))
    return out
