"""進行方向で決まるペン先のずれ（ハッチングの曲がり）の測定と力モデル。

測り方（scripts/gen_hatch_patches.py と scripts/measure_hatch_patches.py）:
  ジグザグのハッチパッチ（1本おきに向きが反転する N 本の線、間隔 p）を紙の各所に描く。
  ペン先が進行方向の左へ D だけずれるなら、隣り合う線の間隔は p − 2D と p + 2D の交互になる。
  線の本数 N を偶数にすると、パッチを 180° 回して読んでも同じ D になる（スキャンの向きに依存しない）。
  折り返しではずれが連続なので隣の線と間隔 p のまま接し、走るにつれて p ± 2D に開く。
  その立ち上がりを exp(−s/λ) で当てはめて距離定数 λ を出す。

座標の約束:
  紙の座標は kinematics.py と同じ（リング中心が原点、y 上向き、mm）。
  線 k（k = 0..N−1）の向きは (−1)^k·(cos θ, sin θ)、位置は中心から (k − (N−1)/2)·p だけ m 方向。
  m = (−sin θ, cos θ) は線 0 の進行方向の左。D は「進行方向の左へのずれ」で、線 0 では +m 方向。

力モデル（重ね描きの検算で得た形）:
  δ = (a·sin²(Δ/2)·(v·u) + b·sign(ċ))·u   [mm]      （model = "rod_axial_sin2"）
  u: 棒の方向（A→B の単位ベクトル）、v: ペン先の進行方向、ċ: 弦の伸縮、Δ: 2 台のなす角。
  線に直交する成分は D = δ·m = −(…)·(v·n)（n は u の左）。
  a の項は A 本体が半径方向に動く（溝の遊び）ぶんが棒に沿って出るもので、幾何的に sin²(Δ/2) = (弦/2R)²
  が掛かる。2 台が向かい合うほど（中心に近いほど）大きい。sin² を掛けない旧形は model = "rod_axial"。
  10 月中旬の実測（3 枚 113 パッチ）: a ≈ +0.53、b ≈ +0.08、λ ≈ 1.9 mm（a > 0 はペン先が棒方向に「進む」向き）。
  model = "rod_axial_sin2_sqrt": a の項の (v·u) を sign(v·u)·√|v·u| に替えた形。
  10 月中旬の補正入りの紙で、ずれが v·u に比例せず「小さい v·u で早く立ち上がって飽和する」ことが分かった
  （|v·u| ≈ 0.3〜0.5 で比例モデルより 25〜50% 大きく、0.6〜0.7 で小さい）。係数の数は同じで残差 rms が
  0.038 → 0.015 mm に下がる。v·u = 0 で傾きが無限になるので、実装は v·u / √(|v·u| + VU_EPS) で丸める
  （当てはめと補正で同じ式を使う）。

速度の係数:
  線の両端 2〜3 mm（GRBL の加減速区間）ではずれが線の中央より 2〜3 割小さい。巡航速度 v [mm/s] に対して
  目標値に f(v) = 1 − c_speed·max(0, 1 − v/v_ref) を掛ける（v_ref = 30、c_speed ≈ 0.2〜0.3）。
  各点の v は補正前の経路で GRBL の台形（timing.block_profile）を回して出す。F の上限で遅い中心付近の線は
  定常でも f < 1 になるので、係数の当てはめにも各パッチの巡航速度（cruise_speed）を入れる。

端点:
  線ごとにペンを上げたパッチ（Layout.separate）では、線に沿った端点の位置も測る（q_start、q_end。
  墨の範囲からペン半径を引いた値と設計の端点の差、進行方向が正）。終点は一律 0.2 mm ほど手前で終わる。

補正（DeflectionModel, Compensator。gcode.convert が使う）:
  経路を細分化点ごとに歩き、ずれの状態を δ += (δ目標 − δ)(1 − exp(−ds/λ)) で更新して、
  目標点 − δ を逆運動学に渡す。ペン上げ移動（角度空間の直線）も xy に戻して歩く。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .kinematics import Kinematics, Unreachable

Polyline = list[tuple[float, float]]


# --- レイアウト ------------------------------------------------------------
@dataclass(frozen=True)
class Patch:
    x: float  # パッチの中心（紙の座標、mm）
    y: float
    theta: float  # 線 0 の向き（度、反時計回り、x 軸から）
    site_x: float  # 所属する場所（2×2 ブロックの中心）
    site_y: float


@dataclass
class Layout:
    pitch: float = 1.2  # 線の間隔（mm）。狭い側（p − 2D）でも線幅 0.5 mm の中心線を分離できる値
    length: float = 20.0  # 線の長さ（mm）
    lines: int = 12  # 線の本数。偶数にする（180° 対称になり、スキャンの向きに依存しない）
    angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)  # 1 か所あたりの線の向き
    grid_x: tuple[float, ...] = (-70.0, 0.0, 70.0)  # 場所の格子（A4 縦を中心に置く前提）
    grid_y: tuple[float, ...] = (-105.0, -35.0, 35.0, 105.0)
    block: float = 14.0  # ブロック内でのパッチ中心のオフセット（2 列に並べる）
    mark: bool = True  # 左上に向きの目印（短い線）を描く
    # "" はジグザグ（線を折り返しでつなぐ）。"alt" は線ごとにペンを上げ、向きはジグザグと同じく交互。
    # "same" は線ごとにペンを上げ、全部同じ向き（進入が線と逆向きになる、始点の曲がりの最悪条件）
    separate: str = ""

    @property
    def width(self) -> float:
        return (self.lines - 1) * self.pitch

    def patches(self) -> list[Patch]:
        rows = math.ceil(len(self.angles) / 2)
        out = []
        for sy in self.grid_y:
            for sx in self.grid_x:
                for i, th in enumerate(self.angles):
                    ox = (2 * (i % 2) - 1) * self.block if len(self.angles) > 1 else 0.0
                    oy = ((rows - 1) / 2 - i // 2) * 2 * self.block
                    out.append(Patch(sx + ox, sy + oy, float(th), sx, sy))
        return out

    def mark_line(self) -> Polyline | None:
        """向きの目印。格子の左上の外側に 10 mm の水平線。"""
        if not self.mark:
            return None
        x = min(self.grid_x) - 2 * self.block
        y = max(self.grid_y) + 2 * self.block
        return [(x - 5.0, y), (x + 5.0, y)]

    def to_json(self) -> dict:
        d = asdict(self)
        d["patches"] = [asdict(p) for p in self.patches()]
        d["mark_line"] = self.mark_line()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        kw = {k: d[k] for k in ("pitch", "length", "lines", "angles", "grid_x", "grid_y", "block", "mark", "separate")
              if k in d}
        for k in ("angles", "grid_x", "grid_y"):
            if k in kw:
                kw[k] = tuple(float(v) for v in kw[k])
        return cls(**kw)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_json(), fp, indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Layout":
        return cls.from_json(json.load(open(path, encoding="utf-8-sig")))


def line_frame(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """線 0 の進行方向 v と、その左 m。"""
    th = math.radians(theta)
    v = np.array([math.cos(th), math.sin(th)])
    return v, np.array([-v[1], v[0]])


def zigzag(patch: Patch, lay: Layout) -> Polyline:
    """パッチ 1 つぶんのジグザグ折れ線。線 k の向きは (−1)^k v。"""
    v, m = line_frame(patch.theta)
    c = np.array([patch.x, patch.y])
    pts = []
    for k in range(lay.lines):
        base = c + (k - (lay.lines - 1) / 2) * lay.pitch * m
        sgn = 1 if k % 2 == 0 else -1
        pts.append(tuple(base - sgn * lay.length / 2 * v))
        pts.append(tuple(base + sgn * lay.length / 2 * v))
    return [(float(x), float(y)) for x, y in pts]


def patch_polylines(patch: Patch, lay: Layout) -> list[Polyline]:
    """パッチを描く折れ線。lay.separate が空ならジグザグ 1 本、"alt"/"same" なら線ごとに 1 本。"""
    zz = zigzag(patch, lay)
    if not lay.separate:
        return [zz]
    out = []
    for k in range(lay.lines):
        a, b = zz[2 * k], zz[2 * k + 1]
        if lay.separate == "same" and k % 2 == 1:
            a, b = b, a  # 奇数線も線 0 と同じ向きにする
        out.append([a, b])
    return out


# --- スキャンの読み取り --------------------------------------------------------
@dataclass
class Transform:
    """紙の座標 (x, y)[mm, y 上向き] → スキャン画像の座標 (X, Y)[mm, 左上原点, Y 下向き]。
    [X, Y] = Rot(phi) @ [x, −y] + [tx, ty]"""
    phi: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy).astype(float)
        p = np.column_stack([xy[:, 0], -xy[:, 1]])
        c, s = math.cos(self.phi), math.sin(self.phi)
        return p @ np.array([[c, s], [-s, c]]) + np.array([self.tx, self.ty])

    def invert(self, XY: np.ndarray) -> np.ndarray:
        XY = np.atleast_2d(XY).astype(float) - np.array([self.tx, self.ty])
        c, s = math.cos(self.phi), math.sin(self.phi)
        p = XY @ np.array([[c, -s], [s, c]])
        return np.column_stack([p[:, 0], -p[:, 1]])


def _kabsch(src: np.ndarray, dst: np.ndarray) -> Transform:
    """src（[x, −y]）を dst（画像 mm）に重ねる回転と平行移動（最小二乗）。"""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:  # 鏡映は許さない
        Vt[-1] *= -1
        Rm = (U @ Vt).T
    phi = math.atan2(Rm[1, 0], Rm[0, 0])
    t = md - ms @ Rm.T
    return Transform(phi, float(t[0]), float(t[1]))


def find_blobs(binary: np.ndarray, px: float, min_area_mm2: float = 1.0,
               dilate_mm: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """粗い画像で連結成分を取り、重心（画像 mm）と面積（mm²）を返す。パッチは折り返しでつながって 1 成分になる。

    線ごとにペンを上げたパッチ（Layout.separate）は線が離れているので、dilate_mm を間隔の半分より
    大きくして 1 成分にまとめる。
    """
    from scipy import ndimage

    ds = max(1, int(round(px * 0.17)))  # 約 0.17 mm/px に落とす
    small = binary[::ds, ::ds]
    small = ndimage.binary_dilation(small, iterations=max(1, int(round(dilate_mm / (ds / px)))))
    lab, k = ndimage.label(small)
    if k == 0:
        return np.zeros((0, 2)), np.zeros(0)
    idx = np.arange(1, k + 1)
    areas = ndimage.sum(small, lab, idx) * (ds / px) ** 2
    keep = areas >= min_area_mm2
    cents = np.array(ndimage.center_of_mass(small, lab, idx[keep])).reshape(-1, 2)
    return cents[:, ::-1] * ds / px, areas[keep]  # (X, Y)


@dataclass
class Registration:
    transform: Transform
    matched: list[int]  # パッチごとの blob の添字（−1 は未検出）
    mark_found: bool
    rms: float
    max_err: float


def register(blobs: np.ndarray, areas: np.ndarray, lay: Layout, tol: float = 10.0) -> Registration:
    """blob の重心を設計位置に合わせる。0° と 180° の両方を試し、目印も含めて残差が小さいほうを採る。"""
    patches = lay.patches()
    nominal = np.array([[p.x, p.y] for p in patches])
    mark = lay.mark_line()
    targets = nominal if mark is None else np.vstack([nominal, [np.mean(mark, axis=0)]])
    src = np.column_stack([targets[:, 0], -targets[:, 1]])
    big = blobs[areas > 0.3 * np.median(areas)] if len(blobs) else blobs
    best = None
    for phi0 in (0.0, math.pi):
        T = Transform(phi0, 0.0, 0.0)
        img = T.apply(nominal)
        T.tx, T.ty = (big.mean(axis=0) - img.mean(axis=0)) if len(big) else (0.0, 0.0)
        match = None
        for _ in range(8):
            img = T.apply(targets)
            d = np.linalg.norm(img[:, None, :] - blobs[None, :, :], axis=2)
            match = d.argmin(axis=1)
            ok = d[np.arange(len(targets)), match] < tol
            if ok.sum() < 3:
                break
            T = _kabsch(src[ok], blobs[match[ok]])
        img = T.apply(targets)
        err = np.linalg.norm(img - blobs[match], axis=1)
        ok = err < tol
        score = math.sqrt(np.mean(np.minimum(err, tol) ** 2))
        if best is None or score < best[0]:
            best = (score, T, match, ok, err)
    score, T, match, ok, err = best
    n = len(patches)
    matched = [int(match[i]) if ok[i] else -1 for i in range(n)]
    mark_found = bool(mark is not None and ok[n])
    pe = err[:n][ok[:n]]
    return Registration(T, matched, mark_found, float(np.sqrt(np.mean(pe ** 2))) if len(pe) else math.nan,
                        float(pe.max()) if len(pe) else math.nan)


@dataclass
class PatchResult:
    patch: Patch
    center: tuple[float, float]  # 実測の中心（紙の座標）
    theta: float  # 実測の線の向き（紙の座標、度）
    pitch: float  # 実測の間隔
    D: float  # 進行方向の左へのずれ（mm）。モデルフィット
    D_mid: float  # 線の中央部だけから出した D（モデルなし）
    lam: float  # 距離定数（mm）
    rms: float  # フィット残差
    s: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 列の位置（線に沿って、中心が 0）
    t: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # [列, 線] 各線の横位置（nan は欠測）
    ok: bool = True
    note: str = ""
    start: float = 0.0  # 線の始点でのずれの状態（gap_model の start）
    lam_free: float = math.nan  # λ を固定する前（1 段目）の値
    # 線ごとの端点の、線に沿った位置の設計値からの差（mm、進行方向が正）。線ごとにペンを上げたパッチだけ
    q_start: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_end: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 線ごとの線幅（mm）


def gap_model(s: np.ndarray, k: np.ndarray, length: float, pitch: float, D: float, lam: float,
              start: float = 0.0) -> np.ndarray:
    """ペア k（線 k と k+1）の間隔。中央で p ∓ 2D、両端の折り返しに向かって exp(−s/λ) で戻る。

    start は線の始点でのずれの状態（折り返し前の線の −D から +D へ向かう道のりの割合）。
    折り返しの区間（長さ p）でずれが折り返し自身の目標値へ動くので 0 とは限らない。
    """
    g = 1 - (1 - start) * (np.exp(-(length / 2 - s) / lam) + np.exp(-(s + length / 2) / lam))
    return pitch - 2 * np.where(k % 2 == 0, 1.0, -1.0) * D * g


END_LEVEL = 0.2  # 端と線幅を決める、墨の集計の高さの割合（中央の高さに対して）
END_SMOOTH = 0.12  # 集計をならす箱の幅（mm）。画素の間隔（600 dpi で 0.042）より広くして集計の穴を埋める。
# 箱でならすと段差の 20% 点は外へ 0.3 × 箱幅だけ動くが、線幅（半径）と端の両方に同じだけ掛かるので差し引きで消える


def _profile(x: np.ndarray, w: np.ndarray, edges: np.ndarray, bin_w: float) -> np.ndarray:
    prof, _ = np.histogram(x, bins=edges, weights=w)
    n = max(1, int(round(END_SMOOTH / bin_w)))
    return np.convolve(prof, np.ones(n) / n, mode="same")


def _crossings(prof: np.ndarray, cen: np.ndarray, i0: int, thr: float, bin_w: float) -> tuple[float, float] | None:
    """集計 prof（bin の中心 cen）を i0 から外へ歩き、thr を切る位置を bin の間で線形補間して返す。"""
    if prof[i0] <= thr:
        return None
    lo = i0
    while lo - 1 >= 0 and prof[lo - 1] > thr:
        lo -= 1
    hi = i0
    while hi + 1 < len(prof) and prof[hi + 1] > thr:
        hi += 1
    if lo == 0 or hi == len(prof) - 1:
        return None  # 集計の範囲まで墨が続く
    s_min = cen[lo] - bin_w * (prof[lo] - thr) / max(prof[lo] - prof[lo - 1], 1e-9)
    s_max = cen[hi] + bin_w * (prof[hi] - thr) / max(prof[hi] - prof[hi + 1], 1e-9)
    return float(s_min), float(s_max)


def line_ends(s: np.ndarray, t: np.ndarray, w: np.ndarray, tt: np.ndarray, lay: Layout,
              pen_radius: float | None = None, bin_w: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """線ごとの端点の、線に沿った位置の設計値からの差（q_start, q_end。mm、進行方向が正）と線幅。

    s, t は墨の画素の線に沿った位置と横位置（パッチ中心が原点）、w はその濃さの重み、tt は track の結果
    （線の横位置の列）。線 k の帯の墨を線に沿って bin_w で集計し、中央（|s| < L/2 − 2）の高さの END_LEVEL を
    切る位置を中心から外へ歩いて探す。墨の端からペン半径を引いた（戻した）位置を端点の中心とし、
    設計の端点（±L/2）との差を出す。
    ペン半径は、省略時は線幅（線の中央の横方向の集計の、同じ END_LEVEL を切る幅。戻り値の 3 つ目）の半分。
    10 月中旬の Prockey の紙では線幅 0.72〜0.74 mm で、設計の線幅 0.5（半径 0.25）より太い。
    端の形（半円）は段差ではないので、合成スキャンでは線幅が 0.04 太く出て q が 0.02 ほど内側に寄る。
    中心 s = 0 はパッチの墨の広がりの中点なので、全線に共通の線に沿ったずれは見えない（始点と終点の差は見える）。
    始点はペンを下ろして待つ間に墨がにじむので、その分だけ手前（負）に出る。
    """
    N, L = lay.lines, lay.length
    q_s, q_e, width = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    comb = (np.arange(N) - (N - 1) / 2) * lay.pitch
    tk = np.nanmean(tt, axis=0) if len(tt) else comb
    tk = np.where(np.isnan(tk), comb, tk)
    edges = np.arange(-L / 2 - 2.0, L / 2 + 2.0 + bin_w / 2, bin_w)
    cen = edges[:-1] + bin_w / 2
    i0 = int(np.argmin(np.abs(cen)))
    tw = bin_w
    t_edges = np.arange(-lay.pitch / 2, lay.pitch / 2 + tw / 2, tw)
    t_cen = t_edges[:-1] + tw / 2
    j0 = int(np.argmin(np.abs(t_cen)))
    mid_s = np.abs(s) < L / 2 - 2.0
    for k in range(N):
        # 線幅: 中央部の横方向の集計
        tprof = _profile((t - tk[k])[mid_s], w[mid_s], t_edges, tw)
        if tprof.max() <= 0:
            continue
        cross = _crossings(tprof, t_cen, j0, END_LEVEL * float(tprof.max()), tw)
        if cross is None:
            continue
        width[k] = cross[1] - cross[0]
        r_pen = width[k] / 2 if pen_radius is None else pen_radius
        sel = np.abs(t - tk[k]) <= r_pen + 0.1
        prof = _profile(s[sel], w[sel], edges, bin_w)
        mid = prof[np.abs(cen) < L / 2 - 2.0]
        plateau = float(np.median(mid)) if len(mid) else 0.0
        if plateau <= 0:
            continue
        cross = _crossings(prof, cen, i0, END_LEVEL * plateau, bin_w)
        if cross is None:
            continue
        s_min, s_max = cross
        sgn = 1 if k % 2 == 0 else -1
        start_ext, end_ext = (s_min, s_max) if sgn > 0 else (s_max, s_min)
        q_s[k] = sgn * ((start_ext + sgn * r_pen) - (-sgn * L / 2))
        q_e[k] = sgn * ((end_ext - sgn * r_pen) - (sgn * L / 2))
    return q_s, q_e, width


def measure_patch(gray: np.ndarray, px: float, T: Transform, patch: Patch, lay: Layout,
                  column: float = 0.25, band: float = 0.5, margin: float = 1.5,
                  guess: tuple[float, float] | None = None, threshold: int = 128,
                  pen_radius: float | None = None) -> PatchResult:
    """パッチ 1 つの各線の横位置を線に沿って追い、隣接ペアの間隔から D と λ を出す。

    gray は濃淡画像（uint8、白地に黒い線）。横位置は濃さを重みにした重心なので画素より細かく出る。
    guess は中心の見当（紙の座標。省略時は設計位置）。中心と向きはパッチの墨から合わせ直す。
    線の向きは、各線の重心の傾きから出す（t のヒストグラムの鋭さで探すと、線が画素の行と平行なときに
    画素格子に引き込まれる）。
    線ごとにペンを上げたパッチ（lay.separate）では端点の位置（q_start, q_end。line_ends）も出す。
    """
    N, L, p = lay.lines, lay.length, lay.pitch
    W = lay.width
    gx, gy = guess if guess is not None else (patch.x, patch.y)
    cx, cy = T.apply(np.array([[gx, gy]]))[0]
    rc = math.hypot(L / 2, W / 2) + 2.0
    r0, r1 = max(0, int((cy - rc) * px)), min(gray.shape[0], int((cy + rc) * px) + 1)
    c0, c1 = max(0, int((cx - rc) * px)), min(gray.shape[1], int((cx + rc) * px) + 1)
    crop = gray[r0:r1, c0:c1].astype(float)
    bad = PatchResult(patch, (patch.x, patch.y), patch.theta, p, math.nan, math.nan, math.nan, math.nan, ok=False)
    if crop.size == 0 or (crop < threshold).sum() < 100:
        bad.note = "ink not found"
        return bad
    # 濃さの重み。紙の地の明るさ（60 パーセンタイル）を 0 とし、ノイズぶんを引く
    white, black = np.percentile(crop, 60), np.percentile(crop, 5)
    w = np.clip((white - crop) - 0.2 * (white - black), 0, None).ravel()
    rows, cols = np.mgrid[r0:r1, c0:c1]
    XY = np.column_stack([(cols.ravel() + 0.5) / px, (rows.ravel() + 0.5) / px])
    keep = w > 0
    XY, w = XY[keep], w[keep]
    xy = T.invert(XY) - np.array([gx, gy])
    # 設計の範囲（少し余裕を持たせる）の外の墨は隣のパッチなので捨てる
    s0, t0 = xy @ line_frame(patch.theta)[0], xy @ line_frame(patch.theta)[1]
    keep = (np.abs(s0) < L / 2 + 2.0) & (np.abs(t0) < W / 2 + 1.5)
    xy, w = xy[keep], w[keep]
    ink = w > 0.5 * (white - black) - 0.2 * (white - black)  # 2 値（中心の見当と櫛合わせに使う）
    if ink.sum() < 100:
        bad.note = "ink not found in patch"
        return bad

    def frame(theta):
        v, m = line_frame(theta)
        return xy @ v, xy @ m

    comb = (np.arange(N) - (N - 1) / 2) * p

    def locate(theta):
        """向き theta で、中心（s は墨の広がりの中点、t は N 本の櫛と合う位置）を決める。"""
        s, t = frame(theta)
        sel = ink & (np.abs(t) < W / 2 + 1.0)
        s_c = (np.percentile(s[sel], 0.5) + np.percentile(s[sel], 99.5)) / 2
        core = sel & (np.abs(s - s_c) < L / 2 - margin)
        edges = np.arange(-W / 2 - 1.5, W / 2 + 1.5, 0.02)
        h, _ = np.histogram(t[core], bins=edges)
        h = np.convolve(h.astype(float), np.ones(25), mode="same")  # 0.5 mm の箱で平滑化
        centers = edges[:-1] + 0.01
        offs = np.arange(-p / 2, p / 2, 0.02)
        score = [np.interp(comb + o, centers, h).sum() for o in offs]
        t_c = float(offs[int(np.argmax(score))])
        return s - s_c, t - t_c, s_c, t_c

    half = L / 2 - margin
    edges_s = np.arange(-half, half + column / 2, column)
    s_mid = edges_s[:-1] + column / 2
    min_w = 0.3 * column * 0.5 * px * px * (white - black) * 0.5  # 列×線幅の 3 割ぶんの墨

    def track(s, t):
        """列ごとに各線の横位置（帯の中の墨の重心）。2 回目は 1 回目の平均位置を帯の中心にする。"""
        col_idx = np.digitize(s, edges_s) - 1
        in_s = (col_idx >= 0) & (col_idx < len(s_mid))
        tk = comb.copy()
        tt = np.full((len(s_mid), N), np.nan)
        for _ in range(2):
            for k in range(N):
                inb = in_s & (np.abs(t - tk[k]) <= band)
                sums = np.bincount(col_idx[inb], weights=(t * w)[inb], minlength=len(s_mid))
                tot = np.bincount(col_idx[inb], weights=w[inb], minlength=len(s_mid))
                good = tot >= min_w
                tt[:, k] = np.where(good, sums / np.maximum(tot, 1e-9), np.nan)
            tk = np.nanmean(tt, axis=0)
            tk = np.where(np.isnan(tk), comb, tk)
        return tt

    th = patch.theta
    for _ in range(3):
        s, t, s_c, t_c = locate(th)
        tt = track(s, t)
        slopes = []
        for k in range(N):
            ok = ~np.isnan(tt[:, k])
            if ok.sum() >= 10:
                slopes.append(np.polyfit(s_mid[ok], tt[ok, k], 1)[0])
        if not slopes:
            bad.note = "lines not tracked"
            return bad
        dth = math.degrees(math.atan(float(np.median(slopes))))
        th += dth
        if abs(dth) < 0.01:
            break
    s, t, s_c, t_c = locate(th)
    tt = track(s, t)
    v, m = line_frame(th)
    center = np.array([gx, gy]) + s_c * v + t_c * m
    r = PatchResult(patch, (float(center[0]), float(center[1])), th, p, math.nan, math.nan, math.nan, math.nan,
                    s_mid, tt)
    if lay.separate:  # ジグザグでは端が折り返しでつながっていて端点に意味がない
        r.q_start, r.q_end, r.width = line_ends(s, t, w, tt, lay, pen_radius)
    return fit_gaps(r, lay)


def fit_gaps(r: PatchResult, lay: Layout, lam: float | None = None) -> PatchResult:
    """隣接ペアの間隔に gap_model を当てはめて D（と λ、start）を入れる。lam を与えればそれに固定する。

    ペア 0 は線 0 の始点がペン下ろし直後で状態が不定なので除く。
    """
    from scipy.optimize import least_squares

    N, L, p = lay.lines, lay.length, lay.pitch
    half = float(r.s.max()) if len(r.s) else L / 2
    gaps = r.t[:, 1:] - r.t[:, :-1]  # [列, ペア]
    S = np.repeat(r.s[:, None], N - 1, axis=1)
    K = np.repeat(np.arange(N - 1)[None, :], len(r.s), axis=0)
    use = ~np.isnan(gaps) & (K >= 1)
    if use.sum() < 30:
        r.ok, r.note = False, "too few gap samples"
        return r
    mid = use & (np.abs(S) < max(half - 4.0, 1.0))
    even, odd = mid & (K % 2 == 0), mid & (K % 2 == 1)
    r.D_mid = float((np.mean(gaps[odd]) - np.mean(gaps[even])) / 4) if even.any() and odd.any() else math.nan

    def resid(q):
        lam_ = q[2] if lam is None else lam
        return gap_model(S[use], K[use], L, q[0], q[1], lam_, q[3]) - gaps[use]

    q0 = [float(np.nanmedian(gaps[mid])) if mid.any() else p, r.D_mid if not math.isnan(r.D_mid) else 0.0,
          2.0 if lam is None else lam, 0.0]
    # λ の上限は測れる範囲（半分の長さ）より短くしておく。長いと D と start が縮退する
    lo, hi = [p * 0.8, -p / 2, 0.3, -1.0], [p * 1.2, p / 2, min(8.0, half), 0.6]
    if lam is not None:
        lo[2], hi[2] = lam - 1e-6, lam + 1e-6
    res = least_squares(resid, q0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05)
    r.pitch, r.D, r.lam, r.start = float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3])
    r.rms, r.ok, r.note = float(np.sqrt(np.mean(res.fun ** 2))), True, ""
    return r


def start_hooks(r: PatchResult, lay: Layout, lam: float = 1.4, dmax: float = 4.0, dmin: float = 0.3) -> np.ndarray:
    """線ごとの始点の曲がり c_k（mm、進行方向の左が正）。

    線 k の始点側の「横位置 − 線の中央（|s| < 4）の平均」を c·exp(−d/λ)（d は始点からの距離）で当てはめる。
    c は「始点でのずれの状態 − その線の定常値」。ペン下ろし直後や折り返し直後の振りの大きさを表す。
    両端を除く margin を dmin より小さくして測ること（measure_patch(margin=0.3) など）。
    """
    N, L = lay.lines, lay.length
    out = np.full(N, np.nan)
    if len(r.s) == 0:
        return out
    for k in range(N):
        t = r.t[:, k]
        sgn = 1 if k % 2 == 0 else -1
        mid = (np.abs(r.s) < 4) & ~np.isnan(t)
        d = r.s * sgn + L / 2
        sel = (d >= dmin) & (d < dmax) & ~np.isnan(t)
        if mid.sum() < 5 or sel.sum() < 5:
            continue
        y = sgn * (t[sel] - np.mean(t[mid]))
        x = np.exp(-d[sel] / lam)
        out[k] = float((x @ y) / (x @ x))
    return out


def start_excursion(r: PatchResult, lay: Layout, dmin: float = 0.3, dmax: float = 3.5) -> np.ndarray:
    """線ごとの、始点から dmin〜dmax mm の区間での「横位置 − 線の中央の平均」の最大の絶対値（mm）。

    始点の曲がりが目に見える量をそのまま表す（start_hooks の c は指数の当てはめなので、
    線の途中まで続くオフセットにも反応する）。
    """
    N, L = lay.lines, lay.length
    out = np.full(N, np.nan)
    for k in range(N):
        t = r.t[:, k]
        sgn = 1 if k % 2 == 0 else -1
        mid = (np.abs(r.s) < 4) & ~np.isnan(t)
        d = r.s * sgn + L / 2
        sel = (d >= dmin) & (d < dmax) & ~np.isnan(t)
        if mid.sum() >= 5 and sel.any():
            out[k] = float(np.abs(t[sel] - np.mean(t[mid])).max())
    return out


def common_lambda(results: list[PatchResult], min_D: float = 0.05) -> float:
    """ずれが大きく λ が上限に掛かっていないパッチの λ の中央値。"""
    ok = [r.lam for r in results if r.ok and abs(r.D) >= min_D and r.lam < 7.9]
    return float(np.median(ok)) if ok else math.nan


def measure_scan(gray: np.ndarray, px: float, lay: Layout, threshold: int = 128,
                 **kw) -> tuple[Registration, list[PatchResult]]:
    """スキャン全体を測る。1 段目で各パッチの λ を自由に出し、その中央値に固定して 2 段目で D を出し直す
    （λ は機械の性質で共通のはず。ずれが小さいパッチでは λ と D が縮退するので固定が要る）。"""
    gray = np.asarray(gray)
    if gray.dtype == bool:  # 2 値画像でも動く
        gray = np.where(gray, 0, 255).astype(np.uint8)
    blobs, areas = find_blobs(gray < threshold, px, dilate_mm=(lay.pitch * 0.6 if lay.separate else 0.35))
    reg = register(blobs, areas, lay)
    results = []
    for patch, bi in zip(lay.patches(), reg.matched):
        if bi < 0:
            results.append(PatchResult(patch, (patch.x, patch.y), patch.theta, lay.pitch, math.nan, math.nan,
                                       math.nan, math.nan, ok=False, note="no blob"))
            continue
        # blob の重心から入って、中心と向きはパッチの中で合わせ直す
        bx, by = reg.transform.invert(blobs[bi:bi + 1])[0]
        results.append(measure_patch(gray, px, reg.transform, patch, lay, guess=(float(bx), float(by)),
                                     threshold=threshold, **kw))
    lam = common_lambda(results)
    if not math.isnan(lam):
        for r in results:
            if r.ok:
                r.lam_free = r.lam
                fit_gaps(r, lay, lam=lam)
    return reg, results


@dataclass
class StaticCheck:
    """紙 1 枚の静的な健康診断。パッチ中心の設計位置からのずれ（向きに依存しない成分）と間隔の倍率。"""
    radial_mean: float  # 半径方向のずれの平均（mm、外向きが正）
    radial_std: float
    tangential_mean: float  # 接線方向（反時計回りが正）
    delta_equiv: float  # 半径方向のずれを原点 Δ のずれに換算した値（度）
    pitch_scale: float  # 測った間隔 ÷ 設計の間隔（紙の伸縮やスキャナの倍率）
    n: int

    def describe(self) -> str:
        return (f"静的なずれ: 半径方向 {self.radial_mean:+.2f} mm（ばらつき {self.radial_std:.2f}）、"
                f"接線方向 {self.tangential_mean:+.2f} mm、原点 Δ に換算 {self.delta_equiv:+.2f}°。"
                f"間隔の倍率 {self.pitch_scale:.4f}（半径 100 mm で {100 * (self.pitch_scale - 1):+.2f} mm 相当）")


def static_check(kin: Kinematics, lay: Layout, results: list[PatchResult]) -> StaticCheck | None:
    """パッチ中心の実測と設計の差を半径方向と接線方向に分け、半径方向を原点 Δ のずれに換算する。

    位置合わせで平行移動と回転は吸収済みなので、残るのは倍率と半径方向のずれ。
    原点 Δ のずれは絵全体の半径を変えるので、紙ごとの値を追うと原点合わせやペンの滑りの変化が分かる。
    """
    rad, tan, dd, pitches = [], [], [], []
    for r in results:
        if not r.ok:
            continue
        nx, ny = r.patch.x, r.patch.y
        rr = math.hypot(nx, ny)
        if rr < 1e-6:
            continue
        dx, dy = r.center[0] - nx, r.center[1] - ny
        dr = (dx * nx + dy * ny) / rr
        rad.append(dr)
        tan.append((-dx * ny + dy * nx) / rr)
        try:
            al, be = kin.inverse(nx, ny)
            drdD = (math.hypot(*kin.forward(al, be + 0.1)) - rr) / 0.1
            dd.append(dr / drdD)
        except Unreachable:
            pass
        pitches.append(r.pitch)
    if not rad:
        return None
    return StaticCheck(float(np.mean(rad)), float(np.std(rad)), float(np.mean(tan)),
                       float(np.mean(dd)) if dd else math.nan, float(np.mean(pitches) / lay.pitch), len(rad))


CSV_FIELDS = ["site_x", "site_y", "theta_nom", "x", "y", "theta", "pitch", "D", "D_mid", "lam", "lam_free", "start",
              "q_start", "q_end", "width", "rms", "ok", "note"]


def _nanmean(a: np.ndarray) -> float:
    a = a[~np.isnan(a)] if len(a) else a
    return float(np.mean(a)) if len(a) else math.nan


def result_row(r: PatchResult) -> dict:
    return dict(site_x=r.patch.site_x, site_y=r.patch.site_y, theta_nom=r.patch.theta, x=r.center[0], y=r.center[1],
                theta=r.theta, pitch=r.pitch, D=r.D, D_mid=r.D_mid, lam=r.lam, lam_free=r.lam_free, start=r.start,
                q_start=_nanmean(r.q_start), q_end=_nanmean(r.q_end), width=_nanmean(r.width), rms=r.rms,
                ok=int(r.ok), note=r.note)


# --- 力モデル ----------------------------------------------------------------
@dataclass
class RodGeometry:
    u: np.ndarray  # 棒の方向（A→B）
    n: np.ndarray  # u の左
    c: float  # 弦の長さ
    vu: float  # v·u
    vn: float  # v·n
    sign_cdot: float  # 弦の伸縮の符号
    sign_adot: float
    sign_bdot: float
    rA: np.ndarray  # A の半径方向
    um: float  # u·m（m は進行方向の左）。= −vn
    sin2: float  # sin²(Δ/2) = (弦/2R)²。A の半径方向の動きが棒に沿って出る割合
    dal_ds: float = 0.0  # v 方向に 1 mm 進むときの α の変化（度/mm）
    dbe_ds: float = 0.0  # 同じく β


def rod_geometry(kin: Kinematics, x: float, y: float, vx: float, vy: float, eps: float = 0.05) -> RodGeometry:
    """点 (x, y) を v 方向に進むときの棒の向きと各符号。"""
    al, be = kin.inverse(x, y)
    a1, b1 = kin.inverse(x - eps * vx, y - eps * vy)
    a2, b2 = kin.inverse(x + eps * vx, y + eps * vy)
    a1 += 360 * round((al - a1) / 360)
    a2 += 360 * round((al - a2) / 360)
    b1, b2 = a1 + (b1 - a1) % 360, a2 + (b2 - a2) % 360
    dal, dbe = a2 - a1, b2 - b1
    a, b = math.radians(al - 90), math.radians(be - 90)
    PA = np.array([math.cos(a), math.sin(a)]) * kin.R
    PB = np.array([math.cos(b), math.sin(b)]) * kin.R
    c = float(np.linalg.norm(PB - PA))
    u = (PB - PA) / c
    n = np.array([-u[1], u[0]])
    vlen = math.hypot(vx, vy)
    v = np.array([vx, vy]) / vlen
    m = np.array([-v[1], v[0]])
    return RodGeometry(u, n, c, float(v @ u), float(v @ n), float(np.sign(dbe - dal)), float(np.sign(dal)),
                       float(np.sign(dbe)), PA / kin.R, float(u @ m), (c / (2 * kin.R)) ** 2,
                       dal / (2 * eps * vlen), dbe / (2 * eps * vlen))


def cruise_speed(kin: Kinematics, x: float, y: float, vx: float, vy: float, speed: float, fmax: float,
                 max_rate: float = 2000.0) -> float:
    """点 (x, y) を v 方向に走るときのペン先の巡航速度（mm/s）。

    gcode.convert と同じく、F = (α, β) の合成速度で speed に合わせ、fmax と GRBL の軸ごとの上限（max_rate）で
    頭打ちにする。中心に近いほど α の変化が速いので上限に掛かって遅くなる。
    """
    g = rod_geometry(kin, x, y, vx, vy)
    f_need = math.hypot(g.dal_ds, g.dbe_ds) * speed * 60  # 度/分
    f = min(fmax, f_need)
    f = min(f, max_rate * math.hypot(g.dal_ds, g.dbe_ds) / max(abs(g.dal_ds), abs(g.dbe_ds), 1e-12))
    return speed * f / f_need if f_need > 0 else speed


MODELS = ("rod_axial_sin2", "rod_axial_sin2_sqrt", "rod_axial")
VU_EPS = 0.05  # rod_axial_sin2_sqrt で v·u = 0 の傾きを有限にする丸め。|v·u| = 1 で 0.976 倍


def _vu_term(g: RodGeometry, model: str) -> float:
    """a に掛かる進行方向の項（a·_vu_term が棒方向のずれ）。"""
    if model == "rod_axial_sin2":
        return g.sin2 * g.vu
    if model == "rod_axial_sin2_sqrt":
        return g.sin2 * g.vu / math.sqrt(abs(g.vu) + VU_EPS)
    if model == "rod_axial":
        return g.vu
    raise ValueError(f"unknown deflection model: {model}")


@dataclass
class DeflectionModel:
    """補正に使う係数。fit_deflection.py の JSON を読む。"""
    a: float  # 進行方向の棒方向成分で決まる項（mm）。rod_axial_sin2 では sin²(Δ/2)·(v·u)、
    # rod_axial_sin2_sqrt では sin²(Δ/2)·sign(v·u)·√|v·u| に掛かる
    b: float  # 弦の伸縮の符号で決まる項（mm）
    lam: float  # 距離定数（mm）。ずれは走った距離で exp(−ds/λ) で目標値に近づく
    reset_on_down: bool = False  # ペンを下ろすときにずれの状態を 0 に戻す（既定はペン上げ移動中も更新し続ける）
    model: str = "rod_axial_sin2"
    # ペン上げ移動中の目標値に掛ける倍率。10 月中旬に 0.65 を試したが効かず（線の両端の効果の取り違え）、1.0 のまま使う
    gain_up: float = 1.0
    # 速度の係数 f(v) = 1 − c_speed·max(0, 1 − v/v_ref)。加減速区間と F の上限で遅い線でずれが小さい
    c_speed: float = 0.0  # 0 で無効
    v_ref: float = 30.0  # mm/s
    # 端の係数: 巡航速度 v_c に対する落ち込み (1 − c_end·max(0, 1 − v/v_c)) をさらに掛ける。
    # 10 月中旬: c_speed = 0.17（巡航の当てはめ）では端の落ち込みが機械の半分しか出なかったので分けた
    c_end: float = 0.0  # 0 で無効

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown deflection model: {self.model}（{', '.join(MODELS)} のどれか）")

    @classmethod
    def load(cls, path: str, reset_on_down: bool = False, gain_up: float | None = None) -> "DeflectionModel":
        d = json.load(open(path, encoding="utf-8-sig"))  # PowerShell が書いた BOM 付きも読む
        return cls(float(d["a"]), float(d["b"]), float(d["lam"]), reset_on_down, d.get("model", "rod_axial"),
                   float(d.get("gain_up", 1.0)) if gain_up is None else gain_up,
                   float(d.get("c_speed", 0.0)), float(d.get("v_ref", 30.0)), float(d.get("c_end", 0.0)))

    def to_json(self) -> dict:
        return dict(model=self.model, a=self.a, b=self.b, lam=self.lam, c_speed=self.c_speed, v_ref=self.v_ref,
                    c_end=self.c_end, gain_up=self.gain_up)

    def speed_factor(self, speed: float | None, cruise: float | None = None) -> float:
        """速度 speed [mm/s] での目標値の倍率。cruise はその区間の巡航速度（端の係数 c_end に使う）。
        speed が None（不明）なら 1。"""
        if speed is None:
            return 1.0
        f = 1.0 - self.c_speed * max(0.0, 1.0 - speed / self.v_ref)
        if self.c_end and cruise is not None and cruise > 0:
            f *= 1.0 - self.c_end * max(0.0, 1.0 - speed / cruise)
        return f

    def steady(self, kin: Kinematics, x: float, y: float, vx: float, vy: float) -> np.ndarray:
        """点 (x, y) を v 方向に走り続けたときのずれ（速度の係数を掛ける前、紙の座標、mm）。"""
        g = rod_geometry(kin, x, y, vx, vy)
        return (self.a * _vu_term(g, self.model) + self.b * g.sign_cdot) * g.u

    def target(self, kin: Kinematics, x: float, y: float, vx: float, vy: float,
               speed: float | None = None, cruise: float | None = None) -> np.ndarray:
        """点 (x, y) を v 方向に速度 speed [mm/s]（巡航 cruise）で走ったときのずれ（紙の座標、mm）。"""
        return self.steady(kin, x, y, vx, vy) * self.speed_factor(speed, cruise)

    def describe(self, ascii_only: bool = False) -> str:
        lam = "lam" if ascii_only else "λ"
        return (f"deflection {self.model} a={self.a:+.3f} b={self.b:+.3f} {lam}={self.lam:.2f} mm"
                + (f" c-speed={self.c_speed:.2f} v-ref={self.v_ref:g}" if self.c_speed else "")
                + (f" c-end={self.c_end:.2f}" if self.c_end else "")
                + (f" gain-up={self.gain_up:.2f}" if self.gain_up != 1.0 else "")
                + (" reset-on-down" if self.reset_on_down else ""))


class Compensator:
    """経路を歩きながらずれの状態を更新し、指令する点（目標 − ずれ）を返す。

    状態は紙の座標の 2 次元ベクトル。ペン上げ移動（角度空間の直線）も xy に戻して歩く。
    """

    def __init__(self, model: DeflectionModel, kin: Kinematics):
        self.model, self.kin = model, kin
        self.state = np.zeros(2)
        self.max_abs = 0.0  # 統計用: ずれの大きさの最大
        self.fallbacks = 0  # 補正した点が届かず、補正なしにした回数
        self.fallback_r = []  # そのときの中心からの距離（どこで起きたかを報告する）

    def reset(self) -> None:
        self.state = np.zeros(2)

    def advance(self, x: float, y: float, vx: float, vy: float, ds: float, gain: float = 1.0,
                speed: float | None = None, cruise: float | None = None) -> None:
        """(x, y) まで v 方向に ds だけ進んだあとの状態にする。gain は目標値に掛ける倍率（ペン上げ移動で使う）。
        speed はこの区間のペン先の速度、cruise はその移動の巡航速度（mm/s。速度の係数に使う。None なら係数 1）。"""
        if ds <= 0:
            return
        try:
            tgt = self.model.target(self.kin, x, y, vx, vy, speed, cruise) * gain
        except Unreachable:
            return  # 届く範囲の縁。状態は据え置く
        self.state = self.state + (tgt - self.state) * (1 - math.exp(-ds / self.model.lam))
        self.max_abs = max(self.max_abs, float(np.linalg.norm(self.state)))

    def travel(self, ab0: tuple[float, float], ab1: tuple[float, float], substep_deg: float = 0.5,
               profile=None) -> None:
        """角度空間の直線移動（ペン上げ）に沿って状態を更新する。

        profile はこの移動の timing.SegmentProfile（合成方向、度）。与えれば各小区間のペン先の速度を
        出して速度係数に使う。
        """
        da = max(abs(ab1[0] - ab0[0]), abs(ab1[1] - ab0[1]))
        n = max(1, math.ceil(da / substep_deg))
        d_deg = math.hypot(ab1[0] - ab0[0], ab1[1] - ab0[1])
        prev = self.kin.forward(*ab0)
        for i in range(1, n + 1):
            ab = (ab0[0] + (ab1[0] - ab0[0]) * i / n, ab0[1] + (ab1[1] - ab0[1]) * i / n)
            p = self.kin.forward(*ab)
            dx, dy = p[0] - prev[0], p[1] - prev[1]
            ds = math.hypot(dx, dy)
            if ds > 1e-9:
                speed = cruise = None
                if profile is not None and d_deg > 0:
                    # 小区間の中央での角速度（度/秒）を、その小区間の xy の長さ / 角度の長さ で mm/s にする
                    mm_per_deg = ds / (d_deg / n)
                    speed = profile.speed_at(d_deg * (i - 0.5) / n) * mm_per_deg
                    cruise = profile.vn * mm_per_deg
                self.advance(p[0], p[1], dx / ds, dy / ds, ds, gain=self.model.gain_up, speed=speed, cruise=cruise)
            prev = p

    def angles(self, p: tuple[float, float], ab_ref: tuple[float, float]) -> tuple[float, float]:
        """点 p の指令角度。p − 状態 を逆運動学に通し、巻き数は ab_ref（補正なしの角度）に合わせる。
        届かなければ補正なしの ab_ref を返す。"""
        r = math.hypot(p[0], p[1])
        try:
            al, be = self.kin.inverse(p[0] - self.state[0], p[1] - self.state[1], near_alpha=ab_ref[0])
        except Unreachable:
            self.fallbacks += 1
            self.fallback_r.append(r)
            return ab_ref
        k = round((ab_ref[0] - al) / 360)
        al, be = al + 360 * k, be + 360 * k
        # 逆運動学の枝が変わっていないか。中心に近いほど小さな移動で角度が大きく変わるので許容幅を広げる
        tol = 2.0 + math.degrees(1.0 / max(r, 0.5))
        if abs(al - ab_ref[0]) > tol or abs(be - ab_ref[1]) > tol:
            self.fallbacks += 1
            self.fallback_r.append(r)
            return ab_ref
        return al, be


def add_deflection_args(ap) -> None:
    ap.add_argument("--deflection", help="fit_deflection.py の JSON（a, b, lam）。与えると進行方向で決まるずれを補正する")
    ap.add_argument("--deflection-reset", action="store_true",
                    help="ペンを下ろすたびにずれの状態を 0 に戻す（既定はペン上げ移動中も更新し続ける）")
    ap.add_argument("--deflection-gain-up", type=float, default=None, metavar="K",
                    help="ペン上げ移動中の目標値に掛ける倍率（JSON の gain_up を上書き。1.0 のまま使う）")
    ap.add_argument("--deflection-c-end", type=float, default=None, metavar="C",
                    help="端の係数 c_end（JSON の値を上書き。fit_end_effect.py で当てはめる前に試すとき用）")


def load_deflection(args) -> DeflectionModel | None:
    if not args.deflection:
        return None
    m = DeflectionModel.load(args.deflection, args.deflection_reset, getattr(args, "deflection_gain_up", None))
    c_end = getattr(args, "deflection_c_end", None)
    if c_end is not None:
        m.c_end = c_end
    return m


def predict_D(kin: Kinematics, x: float, y: float, theta: float, a: float, b: float,
              model: str = "rod_axial_sin2", factor: float = 1.0) -> float:
    """力モデルによる、向き θ の線の進行方向左へのずれ。factor は速度の係数 f(v)（DeflectionModel.speed_factor）。"""
    g = rod_geometry(kin, x, y, math.cos(math.radians(theta)), math.sin(math.radians(theta)))
    return (a * _vu_term(g, model) + b * g.sign_cdot) * factor * g.um


def predict_along(kin: Kinematics, x: float, y: float, theta: float, a: float, b: float,
                  model: str = "rod_axial_sin2", factor: float = 1.0) -> float:
    """力モデルによる、ずれの線に沿った成分 δ·v（進行方向が正）。向きを反転しても同じ値になる。"""
    g = rod_geometry(kin, x, y, math.cos(math.radians(theta)), math.sin(math.radians(theta)))
    return (a * _vu_term(g, model) + b * g.sign_cdot) * factor * g.vu


LINE_WIDTH = 0.5  # ペン 1 本の線幅（mm）


def saturated(D: float, pitch: float, line_width: float = LINE_WIDTH) -> bool:
    """狭い側の間隔 p − 2|D| が線幅に近づくと隣の線と融合して追えない（測った |D| は信用できない）。"""
    return not math.isnan(D) and abs(D) > (pitch - line_width - 0.1) / 2


def fit_model(kin: Kinematics, rows: list[dict], model: str = "rod_axial_sin2", speed_factor: bool = False,
              v_ref: float = 30.0) -> dict:
    """測定結果（result_row の列）に力モデルを当てはめる。線形最小二乗。

    融合したパッチ（saturated）は除いて当てはめる。戻り値の used にどの行を使ったかを入れる。
    行に applied（補正を掛けて描いたときの DeflectionModel）があれば、その補正量を足し戻して真のずれにする。
    speed_factor なら各行の v（巡航速度、mm/s。cruise_speed で出す）を使って、目標値に掛かる
    f(v) = 1 − c·max(0, 1 − v/v_ref) の c も当てはめる（c を 0〜1 で走査し、a, b は c ごとに線形解）。
    """
    A, y, vs, used = [], [], [], []
    for r in rows:
        if not r.get("ok", 1) or r["D"] is None or math.isnan(r["D"]):
            used.append(False)
            continue
        if saturated(r["D"], r.get("pitch", math.nan) or math.nan):
            used.append(False)
            continue
        v = r.get("v", math.nan)
        if speed_factor and (v is None or math.isnan(v)):
            raise ValueError("速度の係数を当てはめるには各行に巡航速度 v が要ります（fit_deflection.py --speed/--fmax）")
        used.append(True)
        g = rod_geometry(kin, r["x"], r["y"], math.cos(math.radians(r["theta"])), math.sin(math.radians(r["theta"])))
        A.append([_vu_term(g, model) * g.um, g.sign_cdot * g.um])
        D = r["D"]
        if r.get("applied") is not None:
            ap = r["applied"]
            D += predict_D(kin, r["x"], r["y"], r["theta"], ap.a, ap.b, ap.model,
                           ap.speed_factor(None if v is None or math.isnan(v) else v))
        y.append(D)
        vs.append(v if v is not None else math.nan)
    A, y, vs = np.array(A), np.array(y), np.array(vs, dtype=float)
    if len(y) < 3:
        raise ValueError("フィットに使えるパッチが足りません")

    def factors(c):
        return 1.0 - c * np.maximum(0.0, 1.0 - vs / v_ref) if speed_factor else np.ones(len(y))

    def solve(c, keep=None):
        Ac = A * factors(c)[:, None]
        if keep is not None:
            Ac, yy = Ac[keep], y[keep]
        else:
            yy = y
        coef, *_ = np.linalg.lstsq(Ac, yy, rcond=None)
        return coef, Ac

    c = 0.0
    if speed_factor:
        from scipy.optimize import minimize_scalar

        def cost(cc):
            coef, Ac = solve(cc)
            return float(np.mean((y - Ac @ coef) ** 2))

        # 粗く走査してから局所で詰める（c と a が縮退気味なので一山とは限らない）
        grid = np.linspace(0.0, 1.0, 41)
        c0 = float(grid[int(np.argmin([cost(cc) for cc in grid]))])
        res = minimize_scalar(cost, bounds=(max(0.0, c0 - 0.05), min(1.0, c0 + 0.05)), method="bounded")
        c = float(res.x)
    coef, Ac = solve(c)
    pred = Ac @ coef
    res = y - pred
    loo = []  # 1 点抜き（c は固定）
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        c_i, _ = solve(c, keep)
        loo.append(y[i] - Ac[i] @ c_i)
    # λ は補正なしで描いた紙から取る（補正入りの紙では残差の形から出た値で意味がない）
    lams = [r["lam"] for r in rows if r.get("ok", 1) and not math.isnan(r.get("lam", math.nan))
            and r.get("applied") is None]
    return dict(model=model, a=float(coef[0]), b=float(coef[1]), c_speed=c, v_ref=v_ref,
                rms=float(np.sqrt(np.mean(res ** 2))),
                rms_loo=float(np.sqrt(np.mean(np.array(loo) ** 2))), rms_zero=float(np.sqrt(np.mean(y ** 2))),
                n=int(len(y)), n_saturated=int(sum(1 for r, u in zip(rows, used) if not u and r.get("ok", 1)
                                                    and not math.isnan(r["D"]))),
                lam=float(np.median(lams)) if lams else math.nan, pred=pred.tolist(), resid=res.tolist(), used=used,
                D=y.tolist(), v=vs.tolist(), factor=factors(c).tolist())
