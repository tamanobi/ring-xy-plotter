"""img2svg の折れ線を、元画像に合うように PyTorch で動かす（ソフトラスタライザ）。

入力は元の画像（PNG など）と img2svg が出した SVG。ペン幅は固定で、折れ線の頂点の位置だけを変数にする。
  1. 画像をグレースケールで読み、インクの濃さ t ∈ [0,1]（黒 = 1）にする。--scale 倍に bicubic で拡大
  2. SVG の折れ線を画素座標にし、線分の長さが --seg-len 以下になるよう点を足す
  3. 描画: 各線分について、まわりの K×K 画素への距離 d から
       c = sigmoid((w/2 − d)/τ)       （w: ペン幅、τ: 縁のぼかし）
     を求め、全線分の和集合 1 − Π(1 − c) を像にする。線分ごとに小さな窓だけ計算するので、
     全画素×全線分にはならない
  4. 損失 = 平均二乗誤差（像 − t）＋ 形の正則化（2 階差分が初期値から変わる量）＋ 長さ
  5. Adam で頂点を動かし、SVG（mm）と、像の PNG、残差の PNG を出す

座標は拡大後の画素（x 右、y 下）で処理し、最後に mm にする。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from .svg_input import load_svg

Polyline = list[tuple[float, float]]


# --- 入力 -----------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    """画像をグレースケール（0..255、uint8）で読む。透明は白に合成する。"""
    from PIL import Image

    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    return np.asarray(im.convert("L"))


def upscale(gray: np.ndarray, scale: int) -> np.ndarray:
    """グレースケールを scale 倍に bicubic で拡大する（float32、0..255）。"""
    from PIL import Image

    im = Image.fromarray(gray)
    if scale != 1:
        im = im.resize((gray.shape[1] * scale, gray.shape[0] * scale), Image.BICUBIC)
    return np.asarray(im).astype(np.float32)


def ink_target(g: np.ndarray) -> np.ndarray:
    """インクの濃さ t（黒 1、白 0）。画素値をそのまま被覆率とみなす（縁のアンチエイリアスがそのまま部分的なインクになる）。"""
    return np.clip((255.0 - g) / 255.0, 0.0, 1.0)


def pen_floor(target: np.ndarray, ink: np.ndarray, pen_px: float) -> np.ndarray:
    """目標に「ペンの下限幅」を入れる。ink は閾値で二値化したインク。

    ペン幅より細い線は、どう描いてもペン幅になる。そのまま L2 にすると、細い線は
    描かない方が損失が小さくなり、最適化が線を縮めて消してしまう。そこで、インクを
    細線化した中心線をペン半径で太らせた像と元の目標の max を目標にする。
    太い部分は中心線を太らせてもインクの中に収まるので変わらない。薄い線や点も、
    閾値に入っていればペン幅の濃いインクになる。
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    skel = skeletonize(ink)
    dist = ndi.distance_transform_edt(~skel)
    floor = np.clip(pen_px / 2 + 0.5 - dist, 0, 1).astype(np.float32)
    return np.maximum(target, floor)


def svg_to_px(svg_path: str, width_px: int, height_px: int, step_mm: float = 0.5) -> tuple[list[Polyline], float]:
    """SVG の折れ線を画像の画素座標（x 右、y 下、拡大前）にする。mm/px も返す。"""
    d = load_svg(svg_path, step=step_mm)
    mm_per_px = d.width_mm / width_px
    if abs(d.height_mm / height_px - mm_per_px) > 1e-3 * mm_per_px:
        raise ValueError(f"SVG の縦横比 {d.width_mm:.3f}×{d.height_mm:.3f} mm が画像 {width_px}×{height_px} px と合いません")
    cx, cy = d.width_mm / 2, d.height_mm / 2
    pls = [[((x + cx) / mm_per_px, (cy - y) / mm_per_px) for x, y in pl] for pl in d.polylines]
    return pls, mm_per_px


def resample(pl: Polyline, max_len: float) -> Polyline:
    """線分が max_len より長ければ等分して点を足す。元の頂点は残す。"""
    out = [pl[0]]
    for (x0, y0), (x1, y1) in zip(pl, pl[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / max_len))
        for i in range(1, n + 1):
            t = i / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


# --- 折れ線の集合をテンソルに -------------------------------------------------------
@dataclass
class Strokes:
    """全折れ線の頂点を 1 つの配列にまとめたもの。"""
    verts: "torch.Tensor"  # (M, 2) 拡大後の画素
    offsets: list[int]  # 折れ線 k の頂点は verts[offsets[k]:offsets[k+1]]
    seg_a: "torch.Tensor"  # (N,) 線分の始点の添字
    seg_b: "torch.Tensor"  # (N,) 終点の添字
    mid: "torch.Tensor"  # (M2,) 2 階差分を取る内点の添字（前後の点は mid±1）
    poly_of_seg: "torch.Tensor"  # (N,) 線分が属する折れ線の番号

    @classmethod
    def from_polylines(cls, pls: list[Polyline], device) -> "Strokes":
        import torch

        offsets = [0]
        for pl in pls:
            offsets.append(offsets[-1] + len(pl))
        verts = torch.tensor([p for pl in pls for p in pl], dtype=torch.float32, device=device)
        a = np.concatenate([np.arange(o, o2 - 1) for o, o2 in zip(offsets, offsets[1:])])
        mid = np.concatenate([np.arange(o + 1, o2 - 1) for o, o2 in zip(offsets, offsets[1:])])
        pid = np.concatenate([np.full(o2 - o - 1, k) for k, (o, o2) in enumerate(zip(offsets, offsets[1:]))])
        return cls(verts, offsets,
                   torch.as_tensor(a, device=device), torch.as_tensor(a + 1, device=device),
                   torch.as_tensor(mid, device=device), torch.as_tensor(pid, device=device))

    def polylines(self) -> list[Polyline]:
        v = self.verts.detach().cpu().numpy()
        return [[(float(x), float(y)) for x, y in v[o:o2]] for o, o2 in zip(self.offsets, self.offsets[1:])]

    def second_diff(self):
        v = self.verts
        return v[self.mid - 1] - 2 * v[self.mid] + v[self.mid + 1]

    def first_diff(self):
        return self.verts[self.seg_b] - self.verts[self.seg_a]

    def seg_lengths(self):
        return (self.verts[self.seg_b] - self.verts[self.seg_a]).norm(dim=1)


# --- 描画 -----------------------------------------------------------------------
class SoftRenderer:
    """線分の集合をペン幅 w のインクの像にする。微分できる。"""

    def __init__(self, height: int, width: int, pen_px: float, tau: float, max_seg: float, device):
        import torch

        self.H, self.W = height, width
        self.pen = pen_px
        self.tau = tau
        # 線分の中点を中心にした K×K の窓。線分の半分＋ペン半径＋ぼかしの裾（4τ）が収まる大きさ
        self.K = int(math.ceil(max_seg / 2 + pen_px / 2 + 4 * tau)) * 2 + 2
        k = torch.arange(self.K, device=device, dtype=torch.float32)
        self.gy, self.gx = torch.meshgrid(k, k, indexing="ij")  # (K, K)
        self.device = device

    def coverage(self, s: Strokes):
        """各線分の窓の中の c = sigmoid((w/2 − d)/τ) と、窓の画素の平面での添字。

        返り値: softplus の値 (N, K, K)、添字 (N, K, K)、有効マスク (N, K, K)
        """
        import torch
        import torch.nn.functional as F

        a = s.verts[s.seg_a]  # (N, 2)
        b = s.verts[s.seg_b]
        mid = (a + b) / 2
        origin = torch.floor(mid).detach() - self.K // 2  # (N, 2) 窓の左上の画素
        px = origin[:, None, None, 0] + self.gx + 0.5  # 画素の中心 (N, K, K)
        py = origin[:, None, None, 1] + self.gy + 0.5
        ab = b - a
        ab2 = (ab * ab).sum(1).clamp_min(1e-8)  # (N,)
        apx = px - a[:, None, None, 0]
        apy = py - a[:, None, None, 1]
        t = ((apx * ab[:, None, None, 0] + apy * ab[:, None, None, 1]) / ab2[:, None, None]).clamp(0, 1)
        dx = apx - t * ab[:, None, None, 0]
        dy = apy - t * ab[:, None, None, 1]
        d = torch.sqrt(dx * dx + dy * dy + 1e-12)
        # 1 − c = sigmoid((d − w/2)/τ) なので −log(1 − c) = softplus((w/2 − d)/τ)
        sp = F.softplus((self.pen / 2 - d) / self.tau)
        ix = origin[:, None, None, 0].long() + self.gx.long()
        iy = origin[:, None, None, 1].long() + self.gy.long()
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        idx = (iy.clamp(0, self.H - 1) * self.W + ix.clamp(0, self.W - 1))
        return sp, idx, ok

    def render(self, s: Strokes):
        """像 (H, W)、インク 1、紙 0。和集合 1 − Π(1 − c_i) = 1 − exp(−Σ softplus)。"""
        import torch

        sp, idx, ok = self.coverage(s)
        acc = torch.zeros(self.H * self.W, device=self.device, dtype=torch.float32)
        acc = acc.index_add(0, idx.flatten(), (sp * ok).flatten())
        return 1 - torch.exp(-acc.view(self.H, self.W))


# --- 損失 -----------------------------------------------------------------------
@dataclass
class LossWeights:
    shape: float = 1.0  # 2 階差分が初期値から変わる量（うねり・角の変化を抑える）
    stretch: float = 1.0  # 線分の長さが初期値から変わる量（頂点が線に沿って滑って団子になるのを抑える）
    length: float = 0.0  # 線分の長さの和（画素あたり）


@dataclass
class Cover:
    """塗りの内部で線の間隔を保つ項。幅 pitch の細いペンの描画器で描き、interior（0/1）を埋めていない分が欠け。"""
    rend: SoftRenderer
    interior: "torch.Tensor"  # (H, W)
    weight: float

    def loss(self, s: Strokes):
        return self.weight * (((1 - self.rend.render(s)) * self.interior) ** 2).mean()

    def missing(self, s: Strokes) -> np.ndarray:
        """内部で細いペンの像が 0.5 未満の画素（追加の対象）。"""
        import torch

        with torch.no_grad():
            return ((self.rend.render(s) < 0.5) & (self.interior > 0.5)).cpu().numpy()


def losses(img, target, s: Strokes, d2_init, len_init, w: LossWeights, seg_norm: float, cover: Cover | None = None):
    """各項を dict で返す。合計は total。"""
    l2 = ((img - target) ** 2).mean()
    terms = {"l2": l2}
    if cover is not None:
        terms["cover"] = cover.loss(s)
    if w.shape > 0 and s.mid.numel():
        terms["shape"] = w.shape * ((s.second_diff() - d2_init) ** 2).sum(1).mean() / seg_norm ** 2
    if w.stretch > 0:
        terms["stretch"] = w.stretch * ((s.seg_lengths() - len_init) ** 2).mean() / seg_norm ** 2
    if w.length > 0:
        terms["length"] = w.length * s.seg_lengths().sum() / (target.numel())
    terms["total"] = sum(terms.values())
    return terms


# --- 出力 -----------------------------------------------------------------------
def write_svg(groups: list[tuple[str, str, list[Polyline]]], path: str, width_mm: float, height_mm: float,
              stroke_mm: float) -> None:
    """折れ線（mm、x 右、y 下）を img2svg と同じ形の SVG に書く。groups は (レイヤ名, 色, 折れ線)。"""
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
             f'width="{width_mm:.3f}mm" height="{height_mm:.3f}mm" viewBox="0 0 {width_mm:.3f} {height_mm:.3f}">']
    for gid, color, pls in groups:
        if not pls:
            continue
        lines.append(f'<g id="{gid}" inkscape:groupmode="layer" inkscape:label="{gid}" fill="none" '
                     f'stroke="{color}" stroke-width="{stroke_mm}" stroke-linecap="round" stroke-linejoin="round">')
        lines += ['<polyline points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in pl) + '"/>' for pl in pls]
        lines.append("</g>")
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def _save(im, path: str) -> str:
    """画像を保存する。ビューアが開いていて書けないときは少し待って再試行し、だめなら別名にする。"""
    for i in range(3):
        try:
            im.save(path)
            return path
        except OSError:
            time.sleep(0.5)
    alt = path.rsplit(".", 1)[0] + time.strftime("_%H%M%S.png")
    im.save(alt)
    print(f"  {path} に書けなかったので {alt} に保存した")
    return alt


def save_gray(arr: np.ndarray, path: str) -> None:
    """インク 1 = 黒 で保存。"""
    from PIL import Image

    _save(Image.fromarray((255 * (1 - np.clip(arr, 0, 1))).astype(np.uint8)), path)


def save_residual(img: np.ndarray, target: np.ndarray, path: str) -> None:
    """残差。足りないインク（元にあって像にない）を赤、余分（像にあって元にない）を青で、灰色の元画像の上に。"""
    from PIL import Image

    r = np.clip(target - img, -1, 1)
    base = 255 - 60 * np.clip(target, 0, 1)  # 元画像を薄く
    rgb = np.stack([base, base, base], axis=-1)
    miss = np.clip(r, 0, 1)[..., None]
    extra = np.clip(-r, 0, 1)[..., None]
    rgb = rgb * (1 - miss) + np.array([220, 30, 30]) * miss
    rgb = rgb * (1 - extra) + np.array([30, 80, 220]) * extra
    _save(Image.fromarray(rgb.astype(np.uint8)), path)


# --- 最適化・削除・追加 ------------------------------------------------------------
def optimise(pls: list[Polyline], rend: SoftRenderer, target, weights: LossWeights, iters: int, lr: float,
             device, log=None, cover: Cover | None = None) -> tuple[list[Polyline], "torch.Tensor", dict]:
    """頂点を Adam で動かす。正則化は呼び出し時の折れ線を基準にする。像と損失の項も返す。"""
    import torch

    s = Strokes.from_polylines(pls, device)
    s.verts.requires_grad_(True)
    with torch.no_grad():
        d2_init = s.second_diff().clone()
        len_init = s.seg_lengths().clone()
    if iters > 0:
        opt = torch.optim.Adam([s.verts], lr=lr)
        for it in range(1, iters + 1):
            opt.zero_grad(set_to_none=True)
            img = rend.render(s)
            terms = losses(img, target, s, d2_init, len_init, weights, rend.pen, cover)
            terms["total"].backward()
            opt.step()
            if log and (it % max(1, iters // 5) == 0):
                log(f"{it:4d}", terms)
    with torch.no_grad():
        img = rend.render(s)
        terms = losses(img, target, s, d2_init, len_init, weights, rend.pen, cover)
    return s.polylines(), img, terms


def removal_gains(rend: SoftRenderer, s: Strokes, target, weight=None) -> "torch.Tensor":
    """各折れ線を消したときの二乗誤差の和の増加量 (P,)。負なら消した方がよい。weight は画素ごとの重み。

    和集合は 1 − exp(−Σ softplus) なので、折れ線 k を消した像は、k の線分の分だけ
    累積を引いて exp し直せば出る。k の線分の窓に入る画素だけ見ればよい。
    """
    import torch

    sp, idx, ok = rend.coverage(s)
    sp = sp * ok
    HW = rend.H * rend.W
    acc = torch.zeros(HW, device=rend.device).index_add(0, idx.flatten(), sp.flatten())
    tflat = target.flatten()
    wflat = None if weight is None else weight.flatten()
    base_err = (1 - torch.exp(-acc) - tflat) ** 2
    if wflat is not None:
        base_err = base_err * wflat
    pid = s.poly_of_seg[:, None, None].expand_as(idx)
    key = pid * HW + idx  # (折れ線, 画素) の組
    key_u, inv = torch.unique(key.flatten(), return_inverse=True)
    acc_k = torch.zeros(len(key_u), device=rend.device).index_add(0, inv, sp.flatten())
    pix, pk = key_u % HW, key_u // HW
    err_wo = (1 - torch.exp(-(acc[pix] - acc_k)) - tflat[pix]) ** 2
    if wflat is not None:
        err_wo = err_wo * wflat[pix]
    return torch.zeros(len(s.offsets) - 1, device=rend.device).index_add(0, pk, err_wo - base_err[pix])


def _bbox(pl: Polyline, pad: float):
    xs, ys = [p[0] for p in pl], [p[1] for p in pl]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def remove_passes(pls: list[Polyline], origin: list[int], rend: SoftRenderer, target, below: float,
                  lift_px2: float, device, log=None, cover: Cover | None = None) -> tuple[list[Polyline], list[int]]:
    """消しても損失がほとんど増えない折れ線を落とす。

    増加量が below × 折れ線のインクの面積（長さ×ペン幅）+ lift_px2（ペン上げ下げ 1 回のコスト）
    未満なら候補。候補どうしが近い（窓が重なる）と、互いに「相手があるから要らない」に
    なっているかもしれないので、一度に消すのは離れたものだけにして、再評価を繰り返す。
    cover があれば、塗りの内部の間隔の項の増加量（重み付き）も足す。
    """
    import torch

    while pls:
        s = Strokes.from_polylines(pls, device)
        with torch.no_grad():
            gain = removal_gains(rend, s, target)
            if cover is not None:
                gain = gain + cover.weight * removal_gains(cover.rend, s, cover.interior, cover.interior)
            area = torch.zeros(len(pls), device=device).index_add(0, s.poly_of_seg, s.seg_lengths()) * rend.pen
            margin = (gain - below * area - lift_px2).cpu().numpy()
        cand = [int(i) for i in np.argsort(margin) if margin[i] < 0]
        if not cand:
            break
        taken, boxes = [], []
        for i in cand:
            b = _bbox(pls[i], rend.K / 2)
            if any(b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3] for o in boxes):
                continue
            taken.append(i)
            boxes.append(b)
        drop = set(taken)
        if log:
            n_added = sum(origin[i] > 0 for i in taken)
            log(f"    削除 {len(taken)}本（候補 {len(cand)}、うち追加由来 {n_added}）→ 残り {len(pls) - len(taken)}本")
        pls = [pl for i, pl in enumerate(pls) if i not in drop]
        origin = [o for i, o in enumerate(origin) if i not in drop]
    return pls, origin


def contour_passes(wide: np.ndarray, pen_px: float, pitch_px: float, min_len_px: float, seg_px: float,
                   smooth_px: float = 1.5, simplify_px: float = 0.5) -> list[Polyline]:
    """太い領域を、縁に平行な線（距離変換の等高線）で塗る。縁から pen/2、以後 pitch ごとに内側へ。

    人が塗るときのように輪郭に沿った長い線になる。一番内側に残る芯は、次のラウンドで
    細い欠けとして中心線で埋まる。
    """
    from scipy import ndimage as ndi
    from skimage.measure import find_contours

    from .raster import rdp, smooth

    dist = ndi.distance_transform_edt(wide)
    out = []
    level = pen_px / 2
    while level < dist.max():
        for c in find_contours(dist, level):
            closed = bool(np.hypot(*(c[0] - c[-1])) < 1.0)
            pts = rdp(smooth([tuple(p) for p in c], smooth_px, closed), simplify_px)
            if len(pts) >= 2 and sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:])) >= min_len_px:
                out.append(resample([(float(cc), float(r)) for r, cc in pts], seg_px))
        level += pitch_px
    return out


def add_passes(img_np: np.ndarray, target_np: np.ndarray, pen_px: float, thresh: float, min_len_px: float,
               min_width_px: float, seg_px: float, dot_min_px: float = 0.0, wide_px: float = 0.0,
               pitch_px: float = 0.0, smooth_px: float = 1.5, fill: str = "contour",
               field_pls: list[Polyline] | None = None, field_sigma_px: float = 0.0,
               simplify_px: float = 0.5, edge_weight: float = 0.0, outline: bool = False,
               cover_miss: np.ndarray | None = None) -> list[Polyline]:
    """足りないインク（目標 − 像 > thresh、または cover_miss）から新しい折れ線を作る。

    幅 wide_px 以上の太い欠けは、fill="flow" なら既存の線 field_pls の向きに揃えた流線
    （hatch.flow_passes、方向場のぼかし σ = field_sigma_px）、"contour" なら縁に平行な等高線
    （contour_passes）で塗る。それ以外は細線化した中心線。
    flow では、欠け領域の縁の向きも重み edge_weight で方向場に入れる（眉のような幅の狭い横長の
    塗りが、まわりの線ではなく自分の長手方向に塗られる）。outline なら、紙に面した縁に先に
    輪郭線を引き（hatch.outline_passes）、その内側を流線で塗る。
    幅が min_width_px 未満の細い隙間は（削除の判定で消されるだけなので）拾わない。
    線にならなかった孤立した塊（直径 dot_min_px 以上で、描いたインクに接していない）は
    「点」として、長軸方向の短い線にする。
    できた折れ線は simplify_px で間引く（ほぼまっすぐな線は少数の線分になり、形の正則化が
    その直線を保つ）。
    """
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize

    from .raster import _disk, prune, rdp, smooth, trace_skeleton

    miss = (target_np - img_np) > thresh
    if cover_miss is not None:
        miss = miss | cover_miss
    out = []
    thin = miss
    if wide_px > 0:
        wide = ndi.binary_opening(miss, _disk(wide_px / 2))  # 幅 wide_px 以上の部分だけ残る
        if wide.any():
            pitch = pitch_px or 0.8 * pen_px
            if fill == "flow":
                from .hatch import direction_field, flow_passes, mask_edges, outline_passes

                hatch_mask = wide
                if outline:
                    lines, centre = outline_passes(wide, target_np, pen_px, min_len_px, seg_px, simplify_px, smooth_px)
                    if lines:
                        out += lines
                        # 流線の中心は輪郭線の中心から pitch 以上離す（縁から pen/2 の判定と合わせて）
                        hatch_mask = wide & ~ndi.binary_dilation(centre, _disk(max(0.0, pitch - pen_px / 2)))
                edges = mask_edges(wide) if edge_weight > 0 else None
                c2, s2 = direction_field(field_pls or [], miss.shape, field_sigma_px or 2 * pen_px, fallback_mask=wide,
                                         edge_pls=edges, edge_weight=edge_weight)
                out += flow_passes(hatch_mask, c2, s2, pen_px, pitch, min_len_px, seg_px, simplify_px)
            else:
                out += contour_passes(wide, pen_px, pitch, min_len_px, seg_px, smooth_px, simplify_px)
            thin = miss & ~wide
    skel = skeletonize(thin)
    if min_width_px > 0:
        width = 2 * ndi.distance_transform_edt(thin) - 1  # path_widths と同じ見積もり
        skel &= width >= min_width_px
    paths, _ = trace_skeleton(skel)
    paths = prune(paths, spur_px=pen_px / 2, min_px=min_len_px)
    for p in paths:
        pts = rdp(smooth(p.pts, smooth_px, p.closed), simplify_px)
        if len(pts) >= 2 and sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, pts[1:])) >= min_len_px:
            out.append(resample([(float(c), float(r)) for r, c in pts], seg_px))
    from .hatch import orient

    out = [orient(pl) for pl in out]  # 追加する線は向きを揃える（流線は親と同じ向き、それ以外は既定の向き）
    if dot_min_px <= 0:
        return out
    lab = label(miss, connectivity=2)
    H, W = miss.shape
    covered = set()
    for pl in out:
        for x, y in pl:
            r, c = int(round(y)), int(round(x))
            if 0 <= r < H and 0 <= c < W:
                covered.add(lab[r, c])
    inked = ndi.binary_dilation(img_np > 0.5, iterations=1)
    for rp in regionprops(lab):
        if rp.label in covered or 2 * math.sqrt(rp.area / math.pi) < dot_min_px:
            continue
        if inked[rp.coords[:, 0], rp.coords[:, 1]].any():
            continue  # 描いた線の端に接する欠けで、点ではない
        cy, cx = rp.centroid
        ux, uy = -math.sin(rp.orientation), -math.cos(rp.orientation)  # 長軸の向き（x, y）
        half = max(0.5, (rp.axis_major_length - pen_px) / 2)
        out.append([(cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)])
    return out


# --- 後処理 -----------------------------------------------------------------------
def _end_tangent(pts: np.ndarray, at_start: bool, tail_px: float) -> np.ndarray:
    """端から外へ向かう単位ベクトル。端から tail_px ほど内側の点との差で取る。"""
    seq = pts if at_start else pts[::-1]
    d = np.cumsum(np.r_[0, np.hypot(*(np.diff(seq, axis=0).T))])
    k = int(np.searchsorted(d, tail_px))
    k = min(max(k, 1), len(seq) - 1)
    v = seq[0] - seq[k]
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.zeros(2)


def join_lines(pls: list[Polyline], origin: list[int], target_np: np.ndarray, gap_px: float, turn_deg: float,
               tail_px: float) -> tuple[list[Polyline], list[int], int]:
    """端どうしが近く、向きが続いていて、間にインクがある折れ線をつなぐ。つないだ数も返す。"""
    from scipy.spatial import cKDTree

    arrs = [np.asarray(pl, dtype=float) for pl in pls]
    ends = [(k, s) for k in range(len(arrs)) for s in (True, False)]
    pts = np.array([arrs[k][0] if s else arrs[k][-1] for k, s in ends])
    tang = [_end_tangent(arrs[k], s, tail_px) for k, s in ends]
    H, W = target_np.shape
    cands = []
    for i, j in cKDTree(pts).query_pairs(gap_px):
        if ends[i][0] == ends[j][0]:
            continue
        d = pts[j] - pts[i]
        L = float(np.hypot(*d))
        if L > 0.5:
            u = d / L
            turn = max(math.degrees(math.acos(np.clip(np.dot(tang[i], u), -1, 1))),
                       math.degrees(math.acos(np.clip(np.dot(tang[j], -u), -1, 1))))
        else:
            turn = math.degrees(math.acos(np.clip(-np.dot(tang[i], tang[j]), -1, 1)))
        if turn > turn_deg:
            continue
        if L > 1.0:  # 間にインクがあるか（元の絵で離れている線はつながない）
            n = int(math.ceil(L))
            sx = np.clip(np.round(pts[i][0] + d[0] * (np.arange(1, n) / n)), 0, W - 1).astype(int)
            sy = np.clip(np.round(pts[i][1] + d[1] * (np.arange(1, n) / n)), 0, H - 1).astype(int)
            if target_np[sy, sx].mean() < 0.5:
                continue
        cands.append((turn / max(turn_deg, 1e-9) + L / gap_px, i, j))
    link: dict[tuple[int, bool], tuple[int, bool]] = {}
    for _, i, j in sorted(cands):
        if ends[i] in link or ends[j] in link:
            continue
        link[ends[i]] = ends[j]
        link[ends[j]] = ends[i]
    visited: set[int] = set()
    out, out_origin = [], []
    for k in range(len(pls)):
        if k in visited:
            continue
        cur, fwd = k, True  # 後ろへたどって先頭を探す
        seen = {k}
        while (cur, fwd) in link:
            nxt, nxt_is_start = link[(cur, fwd)]
            if nxt in seen:
                break
            seen.add(nxt)
            cur, fwd = nxt, not nxt_is_start
        chain: list[tuple[float, float]] = []
        og = []
        while cur not in visited:
            visited.add(cur)
            seq = pls[cur] if fwd else pls[cur][::-1]
            if chain and math.hypot(seq[0][0] - chain[-1][0], seq[0][1] - chain[-1][1]) < 0.5:
                seq = seq[1:]
            chain += seq
            og.append(origin[cur])
            if (cur, not fwd) not in link:
                break
            nxt, nxt_is_start = link[(cur, not fwd)]
            cur, fwd = nxt, nxt_is_start
        out.append(chain)
        out_origin.append(0 if 0 in og else max(og))
    return out, out_origin, len(pls) - len(out)


def split_corners(pls: list[Polyline], origin: list[int], turn_deg: float, min_len_px: float,
                  k: int = 2) -> tuple[list[Polyline], list[int], int]:
    """鋭い角（前後 k 点で測った向きの変化が turn_deg を超える極大）で折れ線を分ける。分けた数も返す。"""
    out, out_origin, n_split = [], [], 0
    for pl, o in zip(pls, origin):
        pts = np.asarray(pl, dtype=float)
        n = len(pts)
        if n < 2 * k + 1 or turn_deg <= 0:
            out.append(pl)
            out_origin.append(o)
            continue
        a = pts[k:n - k] - pts[:n - 2 * k]
        b = pts[2 * k:] - pts[k:n - k]
        ang = np.abs(np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))))
        d = np.cumsum(np.r_[0, np.hypot(*(np.diff(pts, axis=0).T))])
        cuts, last = [], 0
        for m in range(len(ang)):
            i = m + k
            if ang[m] <= turn_deg or (m > 0 and ang[m] < ang[m - 1]) or (m + 1 < len(ang) and ang[m] < ang[m + 1]):
                continue
            if d[i] - d[last] >= min_len_px and d[-1] - d[i] >= min_len_px:
                cuts.append(i)
                last = i
        bounds = [0] + cuts + [n - 1]
        for s, e in zip(bounds, bounds[1:]):
            out.append(pl[s:e + 1])
            out_origin.append(o)
        n_split += len(cuts)
    return out, out_origin, n_split


# --- 本体 -----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="img2svg の折れ線を元画像に合わせて動かす（PyTorch）")
    ap.add_argument("image", help="元の画像")
    ap.add_argument("svg", help="初期値の SVG（img2svg の出力）")
    ap.add_argument("-o", "--output", help="出力 SVG（既定: SVG と同名 _refined.svg）")
    ap.add_argument("--pen-width", type=float, required=True, help="ペンの線幅（mm）")
    ap.add_argument("--scale", type=int, default=2, help="画像の拡大率（整数）")
    ap.add_argument("--tau", type=float, default=0.35, help="線の縁のぼかし（拡大後の画素）")
    ap.add_argument("--cover", type=float, default=1.0,
                    help="塗りの内部（目標のインクを ペン半径 だけ縮めた所）で、線の間隔を --fill-pitch 以下に保つ項の重み"
                         "（0 で無効）。幅 fill-pitch の細いペンで描いた像が内部を埋めていない分を欠けとみなす。"
                         "これがないと、削除と最適化が塗りの線をペン幅の間隔まで間引き、描いたときのずれで白い筋が出る")
    ap.add_argument("--seg-len", type=float, help="線分の最大長（mm、既定: ペン幅）")
    ap.add_argument("--iters", type=int, default=200, help="ラウンドごとの Adam の反復回数（0 で描画と損失だけ）")
    ap.add_argument("--rounds", type=int, default=3, help="最適化 → 削除 → 追加 を繰り返す回数（0 で最適化だけ）")
    ap.add_argument("--lr", type=float, default=0.03, help="Adam の学習率（mm）")
    ap.add_argument("--remove-below", type=float, default=0.3,
                    help="消したときの損失の増加 ÷ 折れ線のインクの面積 がこれ未満なら消す（0 で削除しない）")
    ap.add_argument("--outer", type=int, default=2, help="ラウンド → 後処理 を繰り返す回数（後処理の結果をまた最適化する）")
    ap.add_argument("--lift-cost", type=float, default=0.1,
                    help="ペン上げ下げ 1 回のコスト（mm²。損失の増加がこれ未満の折れ線も消す。0 で無効）")
    ap.add_argument("--fill-wide", type=float, default=2.0,
                    help="幅がこれ（ペン幅の倍数）以上の欠けは、縁に平行な等高線で塗る（0 で常に中心線）")
    ap.add_argument("--fill-pitch", type=float, default=0.8, help="太い欠けを塗る線の間隔（ペン幅の倍数）")
    ap.add_argument("--fill-mode", choices=("flow", "contour"), default="flow",
                    help="太い欠けの塗り方。flow: まわりの線の向きに揃えた流線（既定）、contour: 縁に平行な等高線")
    ap.add_argument("--flow-smooth", type=float, default=2.0,
                    help="flow の方向場のぼかし σ（ペン幅の倍数）。大きいほど向きが広く揃い、線がまっすぐになる")
    ap.add_argument("--flow-edge", type=float, default=1.0,
                    help="flow の方向場に入れる、欠け領域の縁の向きの重み（線の接線が 1。0 で入れない）")
    ap.add_argument("--flow-outline", action=argparse.BooleanOptionalAction, default=True,
                    help="flow で、紙に面した縁に先に輪郭線を引いてから内側を塗る（--no-flow-outline で切る）")
    ap.add_argument("--add-simplify", type=float, default=0.25,
                    help="追加する線の点の間引きの許容誤差（ペン幅の倍数）。ほぼまっすぐな線は直線にする")
    ap.add_argument("--add-thresh", type=float, default=0.5, help="足りないインクとみなす 目標 − 像 の下限")
    ap.add_argument("--add-min-len", type=float, help="追加する折れ線の最小の長さ（mm、既定: ペン幅）")
    ap.add_argument("--dot-min", type=float, default=0.5,
                    help="孤立したインクの塊をこの直径（ペン幅の倍数）以上なら点として打つ（0 で打たない）")
    ap.add_argument("--join-gap", type=float, help="後処理: この距離以内の端どうしをつなぐ（mm、既定: ペン幅。0 でつながない）")
    ap.add_argument("--join-turn", type=float, default=45, help="後処理: つなぐときに許す向きの変化（度）")
    ap.add_argument("--split-turn", type=float, default=75, help="後処理: この角度より鋭い角で折れ線を分ける（度、0 で分けない）")
    ap.add_argument("--simplify", type=float, default=0.02, help="後処理: 点の間引きの許容誤差（mm）")
    ap.add_argument("--shape", type=float, default=1.0, help="形の正則化（2 階差分の変化）の重み")
    ap.add_argument("--stretch", type=float, default=1.0, help="伸縮の正則化（線分の長さの変化）の重み")
    ap.add_argument("--length", type=float, default=0.0, help="長さの重み")
    ap.add_argument("--threshold", type=int,
                    help="ペンの下限幅を入れる線（薄い線や点を含む）の二値化の閾値（既定: Otsu）。img2svg の --threshold と揃える")
    ap.add_argument("--no-pen-floor", action="store_true", help="目標にペンの下限幅を入れない（細い線が消えやすい）")
    ap.add_argument("--device", default=None, help="cuda / cpu（既定: cuda があれば cuda）")
    ap.add_argument("--no-images", action="store_true", help="像と残差の PNG を出さない")
    args = ap.parse_args(argv)

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gray = load_gray(args.image)
    H0, W0 = gray.shape
    S = args.scale
    thr = args.threshold
    if thr is None:
        from skimage.filters import threshold_otsu

        thr = int(threshold_otsu(gray))
    g_up = upscale(gray, S)
    target_np = ink_target(g_up)
    H, W = target_np.shape
    pls0, mm_per_px = svg_to_px(args.svg, W0, H0)
    px = lambda mm: mm / mm_per_px * S  # noqa: E731  mm → 拡大後の画素
    pen_px = px(args.pen_width)
    print(f"  インク {target_np.sum() / 1e3:.0f} k px（被覆率の和）", end="")
    if not args.no_pen_floor:
        target_np = pen_floor(target_np, g_up <= thr, pen_px)
        print(f" → 閾値 {thr} 以下の線にペンの下限幅を入れて {target_np.sum() / 1e3:.0f} k px", end="")
    print()
    target = torch.as_tensor(target_np, device=device)
    seg_px = px(args.seg_len) if args.seg_len else pen_px
    pls = [resample([(x * S, y * S) for x, y in pl], seg_px) for pl in pls0]
    s = Strokes.from_polylines(pls, device)
    n_seg = int(s.seg_a.numel())
    print(f"{args.image}: {W0}×{H0} px → ×{S} = {W}×{H}（{mm_per_px / S:.4f} mm/px）、ペン幅 {pen_px:.2f} px、{device}")
    print(f"{args.svg}: 折れ線 {len(pls)}本、頂点 {len(s.verts)}、線分 {n_seg}（最大 {seg_px:.2f} px）")

    rend = SoftRenderer(H, W, pen_px, args.tau, seg_px, device)
    cover = None
    if args.cover > 0 and args.fill_pitch < 1.0:
        from scipy import ndimage as ndi

        from .raster import _disk

        interior = ndi.binary_erosion(target_np >= 0.75, _disk(pen_px / 2 + 0.5))
        if interior.any():
            cover = Cover(SoftRenderer(H, W, args.fill_pitch * pen_px, args.tau, seg_px, device),
                          torch.as_tensor(interior.astype(np.float32), device=device), args.cover)
            print(f"  塗りの内部 {interior.sum() / 1e3:.0f} k px に、間隔 {args.fill_pitch * pen_px:.2f} px 以下の線を要求（重み {args.cover}）")
    print(f"  窓 {rend.K}×{rend.K} px × {n_seg} 線分 = {rend.K * rend.K * n_seg / 1e6:.1f} M 要素")
    weights = LossWeights(shape=args.shape, stretch=args.stretch, length=args.length)

    def report(tag, terms):
        print(f"  {tag}: " + "  ".join(f"{k} {float(v.detach()):.5f}" for k, v in terms.items()))

    def length_mm(pls_px):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for pl in pls_px for a, b in zip(pl, pl[1:])) * mm_per_px / S

    t0 = time.perf_counter()
    with torch.no_grad():
        img = rend.render(s)
        terms = losses(img, target, s, s.second_diff(), s.seg_lengths(), weights, pen_px, cover)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  描画 {time.perf_counter() - t0:.2f} s")
    report("初期", terms)
    min_len_px = px(args.add_min_len) if args.add_min_len else pen_px
    lift_px2 = args.lift_cost / (mm_per_px / S) ** 2
    pls_init = pls
    origin = [0] * len(pls)  # 折れ線の由来: 0 初期値、k ラウンド k で追加

    def removal(pls, origin):
        pls, origin = remove_passes(pls, origin, rend, target, args.remove_below, lift_px2, device, print, cover)
        _, img, terms = optimise(pls, rend, target, weights, 0, 0, device, cover=cover)
        return pls, origin, img, terms

    def rounds(pls, origin):
        for rnd in range(1, args.rounds + 1):
            print(f"ラウンド {rnd}/{args.rounds}")
            pls, img, terms = optimise(pls, rend, target, weights, args.iters, px(args.lr), device, report, cover)
            report("  最適化後", terms)
            if args.remove_below > 0 or lift_px2 > 0:
                pls, origin, img, terms = removal(pls, origin)
                report("  削除後", terms)
            t_add = time.perf_counter()
            new = add_passes(img.cpu().numpy(), target_np, pen_px, args.add_thresh, min_len_px,
                             args.remove_below * pen_px, seg_px, args.dot_min * pen_px,
                             args.fill_wide * pen_px, args.fill_pitch * pen_px, fill=args.fill_mode,
                             field_pls=pls, field_sigma_px=args.flow_smooth * pen_px,
                             simplify_px=max(0.5, args.add_simplify * pen_px), edge_weight=args.flow_edge,
                             outline=args.flow_outline,
                             cover_miss=cover.missing(Strokes.from_polylines(pls, device)) if cover else None)
            t_add = time.perf_counter() - t_add
            pls, origin = pls + new, origin + [rnd] * len(new)
            _, img, terms = optimise(pls, rend, target, weights, 0, 0, device, cover=cover)
            print(f"    追加 {len(new)}本（{length_mm(new):.0f} mm、{t_add:.1f} s）→ {len(pls)}本、合計 {length_mm(pls):.0f} mm")
            report("  追加後", terms)
        print("仕上げ")
        pls, img, terms = optimise(pls, rend, target, weights, args.iters, px(args.lr), device, report, cover)
        if args.rounds > 0 and (args.remove_below > 0 or lift_px2 > 0):
            pls, origin, img, terms = removal(pls, origin)
            report("  削除後", terms)
            pls, img, terms = optimise(pls, rend, target, weights, args.iters // 2, px(args.lr), device, report, cover)
        report("  最適化の結果", terms)
        return pls, origin

    def postprocess(pls, origin):
        print("後処理")
        join_gap = px(args.join_gap) if args.join_gap is not None else pen_px
        if join_gap > 0:
            n0 = len(pls)
            pls, origin, n_join = join_lines(pls, origin, target_np, join_gap, args.join_turn, 2 * pen_px)
            from .hatch import orient

            pls = [orient(pl) if o > 0 else pl for pl, o in zip(pls, origin)]  # つないだ追加線の向きを揃え直す
            print(f"  結合 {n_join} か所: {n0} → {len(pls)}本")
        if args.split_turn > 0:
            pls, origin, n_split = split_corners(pls, origin, args.split_turn, pen_px)
            print(f"  鋭角で分割 {n_split} か所 → {len(pls)}本")
        if args.simplify > 0:
            from .raster import rdp

            nv0 = sum(len(pl) for pl in pls)
            pls = [rdp(pl, px(args.simplify)) for pl in pls]
            print(f"  間引き: 頂点 {nv0} → {sum(len(pl) for pl in pls)}")
        return pls, origin

    t0 = time.perf_counter()
    for outer in range(1, args.outer + 1):
        if args.outer > 1:
            print(f"=== 外側 {outer}/{args.outer} ===")
        if outer > 1:
            pls = [resample(pl, seg_px) for pl in pls]  # 間引いた線分を描画器の窓に合わせて分け直す
        pls, origin = rounds(pls, origin)
        pls, origin = postprocess(pls, origin)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  最適化と後処理 {time.perf_counter() - t0:.1f} s")
    if len(pls) == len(pls_init) and all(len(a) == len(b) for a, b in zip(pls, pls_init)):
        moved = torch.tensor([p for pl in pls for p in pl]) - torch.tensor([p for pl in pls_init for p in pl])
        q = torch.quantile(moved.norm(dim=1), torch.tensor([0.5, 0.9, 1.0])) * mm_per_px / S
        print(f"  頂点の移動 中央値 {q[0]:.3f}  90% {q[1]:.3f}  最大 {q[2]:.3f} mm")
    # 描画器の窓は線分長 ≤ seg_px を前提にしているので、確認用の描画は再分割して行う（形は同じ）
    _, img, terms = optimise([resample(pl, seg_px) for pl in pls], rend, target, weights, 0, 0, device, cover=cover)
    report("最終", terms)

    base = args.svg.rsplit(".", 1)[0]
    out = args.output or base + "_refined.svg"
    out_base = out.rsplit(".", 1)[0]
    mm = mm_per_px / S
    pls_mm = [[(x * mm, y * mm) for x, y in pl] for pl in pls]
    kept = [pl for pl, o in zip(pls_mm, origin) if o == 0]
    added = [pl for pl, o in zip(pls_mm, origin) if o > 0]
    write_svg([("strokes", "#000000", kept), ("added", "#008040", added)], out,
              W0 * mm_per_px, H0 * mm_per_px, args.pen_width)
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for pl in pls_mm for a, b in zip(pl, pl[1:]))
    print(f"SVG: {out}（折れ線 {len(pls_mm)}本、うち追加 {len(added)}本、合計 {total:.0f} mm）")
    if not args.no_images:
        img_np = img.detach().cpu().numpy()
        save_gray(img_np, out_base + "_render.png")
        save_residual(img_np, target_np, out_base + "_residual.png")
        save_gray(target_np, out_base + "_target.png")
        print(f"像: {out_base}_render.png  残差: {out_base}_residual.png（赤: 足りない、青: 余分）"
              f"  目標: {out_base}_target.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
