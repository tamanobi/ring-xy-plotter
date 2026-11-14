"""折れ線（mm）の列を、リング式プロッタ用の G-code にする。

処理の順序:
  1. 各折れ線を step [mm] 以下に細分化する
  2. 各点を逆運動学で (α, β) にする。届かない点は捨てて、折れ線をそこで分割する（clip）
  3. ペン上げ移動が短くなるよう、折れ線を貪欲法で並べ替える（sort）
  4. 折れ線ごとに、360° の巻き数と描く向きを、ケーブルのねじれ上限（原点位置から ±twist、
     つまり |α| ≤ twist かつ |β − home| ≤ twist）の中で直前の位置から最も近くなるように選ぶ
  5. セグメントごとに送り速度 F を決める。ペン先の速度が speed [mm/s] になるように、
     (α, β) の移動量と xy の移動量の比から計算し、fmax で頭打ちにする
  6. 所要時間を推定する。GRBL の加減速（timing.py）を入れたものと、等速のものの両方

G-code の約束事（gen_shapes.py と同じ）:
  X = α、Y = β（度）。G90 絶対。ペン上げ `M5`、下げ `M3 S110`、そのあと `G4 P<dwell>`
  （上げたあとの待ちは dwell_up、下げたあとは dwell）。
  開始位置はストッパー原点（X0 Y<home>）、終了時にそこへ戻る。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .kinematics import Kinematics, Unreachable
from .timing import Motion, Segment, block_profile, block_time, format_time

if TYPE_CHECKING:
    from .deflection import DeflectionModel

Polyline = list[tuple[float, float]]
Angles = list[tuple[float, float]]


@dataclass
class GcodeOptions:
    speed: float = 15.0  # ペン先の速度（mm/s）
    step: float = 1.0  # 経路の細分化（mm）
    fmax: float = 1500.0  # 描画中の F の上限（度/分）
    fmin: float = 30.0  # F の下限（度/分）
    travel: float = 1000.0  # ペン上げ移動の F（度/分）
    down: str = "M3 S110"
    up: str = "M5"
    dwell: float = 0.3  # ペンを下げたあとの待ち（秒）
    dwell_up: float | None = None  # ペンを上げたあとの待ち（秒）。None なら dwell と同じ
    twist: float = 400.0  # |α| の上限（ケーブルのねじれ、度）
    home: float = 88.0  # 原点（ストッパー位置）の Δ
    unreachable: str = "clip"  # clip: 届かない点を捨てて分割 / error: 止める
    sort: bool = True  # 折れ線を並べ替える
    keep_direction: bool = False  # 折れ線を与えた向きのまま描く（並べ替えや巻き数の選択で逆向きにしない）
    return_home: bool = True
    motion: Motion = field(default_factory=Motion)  # GRBL の最大速度・加速度（所要時間の推定に使う）
    deflection: "DeflectionModel | None" = None  # 進行方向で決まるずれの補正（deflection.py）
    # リードイン: 各折れ線の始点に、最初の向きの後ろ lead_in [mm] からペンを上げたまま直線で入る。
    # ずれの状態（deflection.py）は直前の移動の向きで決まるので、線に沿って入れば始点の振りが消える。
    lead_in: float = 0.0  # 0 で無効
    lead_in_min_travel: float = 5.0  # 直前のペン上げ移動（xy の直線距離）がこれ未満なら入れない
    lead_in_min_swing: float = 0.1  # 補正ありのとき、予測した始点の振りがこれ未満なら入れない（mm）
    # リードアウト: 各折れ線の終点を、最後の向きに lead_out [mm] だけペンを下ろしたまま延ばす。
    # ペン先はホルダーの後ろへ 0.2 mm ほど引きずられ、台車が止まってもそこに留まるので終点が一律に足りない
    lead_out: float = 0.0  # 0 で無効
    # 脱調の試験: 折れ線はペンを上げたまま（M3 の代わりに M5）同じ動きで流し、その前と後に (x, y) に十字の印を
    # ペンを下ろして描く。2 つの十字が重なれば一周で歩数を落としていない（半径 100 mm で 0.05 mm = 0.03°）
    dry_run: tuple[float, float] | None = None
    dry_run_size: float = 4.0  # 十字の線の長さ（mm）

    @property
    def dwell_up_s(self) -> float:
        return self.dwell if self.dwell_up is None else self.dwell_up


@dataclass
class Stats:
    paths_in: int = 0
    paths_out: int = 0
    points_dropped: int = 0
    paths_dropped: int = 0
    segments: int = 0
    draw_mm: float = 0.0
    travel_deg: float = 0.0
    pen_downs: int = 0
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    beta_min: float = 0.0
    beta_max: float = 0.0
    delta_min: float = math.inf
    delta_max: float = -math.inf
    est_time_s: float = 0.0  # GRBL の加減速を入れた推定（秒）
    est_time_const_s: float = 0.0  # 等速とみなした推定（秒）
    lines: int = 0
    dropped_bbox: tuple[float, float, float, float] | None = None
    branch_splits: int = 0  # 逆運動学の枝が変わる点で折れ線を切った回数
    lead_ins: int = 0  # リードインを入れた折れ線の数
    lead_in_skipped: int = 0  # 条件は満たしたが入れられなかった数（届かない、枝が変わる）
    lead_outs: int = 0  # リードアウトを入れた折れ線の数
    lead_out_skipped: int = 0  # 入れられなかった数（届かない、枝が変わる）
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        out = [
            f"折れ線 {self.paths_in} → {self.paths_out}本、セグメント {self.segments}、G-code {self.lines}行",
            f"描画 {self.draw_mm:.0f} mm、ペン上げ移動 {self.travel_deg:.0f}°、ペン下げ {self.pen_downs}回、"
            f"推定 {format_time(self.est_time_s)}（加減速込み。等速なら {format_time(self.est_time_const_s)}）",
            f"α {self.alpha_min:.0f}〜{self.alpha_max:.0f}°、β {self.beta_min:.0f}〜{self.beta_max:.0f}°、Δ {self.delta_min:.1f}〜{self.delta_max:.1f}°",
        ]
        if self.points_dropped:
            x0, y0, x1, y1 = self.dropped_bbox
            out.append(f"届かない点 {self.points_dropped}個を捨てた（折れ線 {self.paths_dropped}本が消えた）。"
                       f"範囲 x {x0:.1f}〜{x1:.1f}, y {y0:.1f}〜{y1:.1f} mm")
        if self.branch_splits:
            out.append(f"逆運動学の枝が変わる点で折れ線を {self.branch_splits} 回切った（ペンを上げて移動する）")
        if self.lead_ins or self.lead_in_skipped:
            out.append(f"リードイン {self.lead_ins} 本"
                       + (f"（届かないか枝が変わるので入れなかったもの {self.lead_in_skipped} 本）" if self.lead_in_skipped else ""))
        if self.lead_outs or self.lead_out_skipped:
            out.append(f"リードアウト {self.lead_outs} 本"
                       + (f"（届かないか枝が変わるので入れなかったもの {self.lead_out_skipped} 本）" if self.lead_out_skipped else ""))
        out += self.notes
        return "\n".join(out)


class TwistError(ValueError):
    """ケーブルのねじれ上限の中に収まらない折れ線。"""


# --- 1. 細分化 -------------------------------------------------------------
def subdivide(path: Polyline, step: float) -> Polyline:
    out = [path[0]]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
    return out


# --- 2. 逆運動学と分割 ------------------------------------------------------
def to_angles(kin: Kinematics, pts: Polyline, policy: str, stats: Stats) -> list[tuple[Polyline, Angles]]:
    """折れ線を (xy点列, αβ点列) の組に。届かない点で分割し、αを連続にする。"""
    runs: list[tuple[Polyline, Angles]] = []
    cur_p: Polyline = []
    cur_a: Angles = []

    def flush():
        if len(cur_p) >= 2:
            runs.append((list(cur_p), list(cur_a)))
        cur_p.clear()
        cur_a.clear()

    dropped = []
    for p in pts:
        try:
            # 直前の α に近い枝を選ぶ（中心の近くでは 2 つの枝が両方有効で、点ごとに選ぶと α が跳ぶ）
            al, be = kin.inverse(*p, near_alpha=cur_a[-1][0] if cur_a else None)
        except Unreachable as e:
            if policy == "error":
                raise
            dropped.append(p)
            flush()
            continue
        if cur_a:
            k = round((cur_a[-1][0] - al) / 360)
            al, be = al + 360 * k, be + 360 * k
            if abs(al - cur_a[-1][0]) > 90:
                # それでも枝が変わる（今の枝が届かなくなった）。輪を描かずに、ここで切ってペンを上げる
                stats.branch_splits += 1
                flush()
        cur_p.append(p)
        cur_a.append((al, be))
    flush()

    if dropped:
        stats.points_dropped += len(dropped)
        xs, ys = [p[0] for p in dropped], [p[1] for p in dropped]
        b = (min(xs), min(ys), max(xs), max(ys))
        if stats.dropped_bbox is None:
            stats.dropped_bbox = b
        else:
            o = stats.dropped_bbox
            stats.dropped_bbox = (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
    if not runs:
        stats.paths_dropped += 1
    return runs


# --- 3. 並べ替え -------------------------------------------------------------
def sort_runs(runs: list[tuple[Polyline, Angles]], start_xy: tuple[float, float],
              allow_reverse: bool = True) -> list[tuple[Polyline, Angles]]:
    """貪欲法。現在位置に近い端点を持つ折れ線を次に選ぶ（allow_reverse なら逆向きも可）。"""
    if len(runs) <= 1:
        return list(runs)
    import numpy as np

    heads = np.array([r[0][0] for r in runs])
    tails = np.array([r[0][-1] for r in runs])
    left = np.ones(len(runs), dtype=bool)
    cur = np.array(start_xy, dtype=float)
    out = []
    for _ in range(len(runs)):
        dh = np.linalg.norm(heads - cur, axis=1)
        dt = np.linalg.norm(tails - cur, axis=1)
        dh[~left] = np.inf
        dt[~left] = np.inf
        ih, it = int(dh.argmin()), int(dt.argmin())
        if not allow_reverse or dh[ih] <= dt[it]:
            pts, ang = runs[ih]
            left[ih] = False
            cur = tails[ih]
        else:
            pts, ang = runs[it]
            pts, ang = pts[::-1], ang[::-1]
            left[it] = False
            cur = heads[it]
        out.append((pts, ang))
    return out


# --- 4. 巻き数と向き -----------------------------------------------------------
def choose_wrap(ang: Angles, cur_alpha: float, twist: float, home: float = 0.0,
                allow_reverse: bool = True) -> tuple[int, bool]:
    """360° の巻き数 k と、逆向きに描くか rev を選ぶ。

    ケーブルのねじれは、A が |α − 360k| ≤ twist、B が |β − 360k − home| ≤ twist
    （それぞれ原点位置からの回転角）。その中で、直前の α からいちばん近い始点になるものを選ぶ。
    allow_reverse=False なら rev は常に False（進行方向で結果が変わる試験パターン用）。
    """
    lo, hi = min(a for a, _ in ang), max(a for a, _ in ang)
    blo, bhi = min(b for _, b in ang) - home, max(b for _, b in ang) - home
    k0 = round((lo + bhi) / 2 / 360)
    best = None
    for k in (k0 - 1, k0, k0 + 1):
        if max(abs(lo - 360 * k), abs(hi - 360 * k), abs(blo - 360 * k), abs(bhi - 360 * k)) > twist:
            continue
        for rev in ((False, True) if allow_reverse else (False,)):
            start = ang[-1][0] if rev else ang[0][0]
            cost = abs(start - 360 * k - cur_alpha)
            if best is None or cost < best[0]:
                best = (cost, k, rev)
    if best is None:
        raise TwistError(
            f"α {lo:.0f}〜{hi:.0f}°、β−{home:g} {blo:.0f}〜{bhi:.0f}°（幅 {bhi - lo:.0f}°）の折れ線は、"
            f"ねじれ上限 ±{twist:g}° に収まりません。折れ線を分割するか、--twist を見直してください")
    return best[1], best[2]


# --- 5. 出力 ---------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{v:.3f}"


def first_direction(pts: Polyline) -> tuple[float, float] | None:
    """折れ線の最初の向き（単位ベクトル）。始点に重なる点は飛ばす。"""
    x0, y0 = pts[0]
    for x1, y1 in pts[1:]:
        d = math.hypot(x1 - x0, y1 - y0)
        if d > 1e-6:
            return (x1 - x0) / d, (y1 - y0) / d
    return None


def lead_in_point(kin: Kinematics, pts: Polyline, ab0: tuple[float, float], opt: GcodeOptions,
                  cur: tuple[float, float], comp=None, stats: Stats | None = None) -> tuple[float, float] | None:
    """折れ線 pts の始点に入るためのリードイン点の角度。条件に合わなければ None。

    始点の手前 lead_in [mm]（最初の向きの後ろ）。cur は今の位置（角度）。
    補正器 comp があれば、直接入ったときの始点の振り |状態 − 目標値| を予測して、小さければ入れない。
    届かない、または逆運動学の枝が変わる点なら入れない（stats.lead_in_skipped に数える）。
    """
    if opt.lead_in <= 0:
        return None
    v = first_direction(pts)
    if v is None:
        return None
    x0, y0 = pts[0]
    px, py = kin.forward(*cur)
    if math.hypot(px - x0, py - y0) < opt.lead_in_min_travel:
        return None
    if comp is not None and opt.lead_in_min_swing > 0:
        saved = comp.state.copy()
        prof = None
        if comp.model.c_speed:
            prof = block_profile([(ab0[0] - cur[0], ab0[1] - cur[1], opt.travel)], opt.motion)[0]
        comp.travel(cur, ab0, profile=prof)
        try:
            tgt = comp.model.target(kin, x0, y0, v[0], v[1])
            swing = float(math.hypot(*(comp.state - tgt)))
        except Unreachable:
            swing = math.inf
        comp.state = saved
        if swing < opt.lead_in_min_swing:
            return None
    try:
        al, be = kin.inverse(x0 - opt.lead_in * v[0], y0 - opt.lead_in * v[1], near_alpha=ab0[0])
    except Unreachable:
        if stats is not None:
            stats.lead_in_skipped += 1
        return None
    k = round((ab0[0] - al) / 360)
    al, be = al + 360 * k, be + 360 * k
    if (abs(al - ab0[0]) > 45 or abs(be - ab0[1]) > 45 or abs(al) > opt.twist
            or abs(be - opt.home) > opt.twist):
        if stats is not None:
            stats.lead_in_skipped += 1
        return None
    return al, be


def last_direction(pts: Polyline) -> tuple[float, float] | None:
    """折れ線の最後の向き（単位ベクトル）。終点に重なる点は飛ばす。"""
    x1, y1 = pts[-1]
    for x0, y0 in pts[-2::-1]:
        d = math.hypot(x1 - x0, y1 - y0)
        if d > 1e-6:
            return (x1 - x0) / d, (y1 - y0) / d
    return None


def lead_out_point(kin: Kinematics, pts: Polyline, ang: Angles, opt: GcodeOptions,
                   stats: Stats | None = None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """折れ線 pts の終点を最後の向きに lead_out [mm] 延ばした点（xy と角度）。届かないか枝が変わるなら None。"""
    if opt.lead_out <= 0:
        return None
    v = last_direction(pts)
    if v is None:
        return None
    x1, y1 = pts[-1]
    q = (x1 + opt.lead_out * v[0], y1 + opt.lead_out * v[1])
    ab1 = ang[-1]
    try:
        al, be = kin.inverse(q[0], q[1], near_alpha=ab1[0])
    except Unreachable:
        if stats is not None:
            stats.lead_out_skipped += 1
        return None
    k = round((ab1[0] - al) / 360)
    al, be = al + 360 * k, be + 360 * k
    if (abs(al - ab1[0]) > 45 or abs(be - ab1[1]) > 45 or abs(al) > opt.twist
            or abs(be - opt.home) > opt.twist):
        if stats is not None:
            stats.lead_out_skipped += 1
        return None
    return q, (al, be)


def add_lead_in_args(ap) -> None:
    ap.add_argument("--lead-in", type=float, default=0.0, metavar="MM",
                    help="各折れ線の始点に、最初の向きの後ろ MM mm からペンを上げたまま直線で入る（始点の曲がり対策）")
    ap.add_argument("--lead-in-min-travel", type=float, default=5.0, metavar="MM",
                    help="直前のペン上げ移動がこれ未満ならリードインを入れない（既定 5。0 で全折れ線）")
    ap.add_argument("--lead-in-min-swing", type=float, default=0.1, metavar="MM",
                    help="--deflection ありのとき、予測した始点の振りがこれ未満なら入れない（既定 0.1）")
    ap.add_argument("--lead-out", type=float, default=0.0, metavar="MM",
                    help="各折れ線の終点を、最後の向きに MM mm ペンを下ろしたまま延ばす（終点の不足 0.2 mm の対策）")


def lead_in_options(args) -> dict:
    return dict(lead_in=args.lead_in, lead_in_min_travel=args.lead_in_min_travel,
                lead_in_min_swing=args.lead_in_min_swing, lead_out=args.lead_out)


def add_dry_run_args(ap) -> None:
    ap.add_argument("--dry-run", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="脱調の試験: 折れ線はペンを上げたまま同じ動きで流し、その前と後に (X, Y) に十字の印をペンを下ろして"
                         "描く（mm、紙の座標）。2 つの十字が重なれば歩数を落としていない")


def dry_run_option(args) -> dict:
    d = getattr(args, "dry_run", None)
    return dict(dry_run=(float(d[0]), float(d[1])) if d else None)


def convert(kin: Kinematics, polylines: list[Polyline], opt: GcodeOptions | None = None,
            header: list[str] | None = None) -> tuple[list[str], Stats]:
    opt = opt or GcodeOptions()
    stats = Stats(paths_in=len(polylines))

    runs: list[tuple[Polyline, Angles]] = []
    for pl in polylines:
        if len(pl) < 2:
            continue
        runs += to_angles(kin, subdivide(pl, opt.step), opt.unreachable, stats)
    stats.paths_out = len(runs)

    cur = (0.0, opt.home)  # ストッパー原点
    if opt.sort:
        runs = sort_runs(runs, kin.forward(*cur), allow_reverse=not opt.keep_direction)
    # (折れ線, ペンを下ろすか)。dry_run なら本体はペン上げで、前後に十字の印
    jobs: list[tuple[Polyline, Angles, bool]] = [(p, a, True) for p, a in runs]
    if opt.dry_run is not None:
        x, y, h = opt.dry_run[0], opt.dry_run[1], opt.dry_run_size / 2
        cross = [[(x - h, y), (x + h, y)], [(x, y - h), (x, y + h)]]
        mark = [(p, a, True) for pl in cross for p, a in to_angles(kin, subdivide(pl, opt.step), "error", stats)]
        jobs = mark + [(p, a, False) for p, a in runs] + mark
        stats.notes.append(f"脱調の試験: 折れ線はペン上げのまま、前後に ({x:g}, {y:g}) へ十字の印")

    out = list(header or [])
    out += ["G90", opt.up, f"G4 P{opt.dwell_up_s:g}"]
    t_const = 0.0  # 等速の推定（移動だけ。待ちは t_dwell）
    t_dwell = opt.dwell_up_s
    blocks: list[list[Segment]] = []  # 静止から静止までの移動の列（加減速込みの推定に使う）
    block: list[Segment] = []
    lo = hi = 0.0
    blo, bhi = opt.home, opt.home

    def move(a: float, b: float, f: float):
        nonlocal cur, t_const, lo, hi, blo, bhi
        f = float(round(f))  # 出力と同じ丸めで推定する
        da = math.hypot(a - cur[0], b - cur[1])
        t_const += da / f * 60 if f > 0 else 0.0
        block.append((a - cur[0], b - cur[1], f))
        out.append(f"G1 X{_fmt(a)} Y{_fmt(b)} F{f:.0f}")
        cur = (a, b)
        lo, hi = min(lo, a), max(hi, a)
        blo, bhi = min(blo, b), max(bhi, b)
        d = b - a
        stats.delta_min, stats.delta_max = min(stats.delta_min, d), max(stats.delta_max, d)
        return da

    def stop():
        """M3/M5 と G4 の前で GRBL は動きを止める。"""
        nonlocal block
        if block:
            blocks.append(block)
            block = []

    comp = None
    if opt.deflection is not None:
        from .deflection import Compensator

        comp = Compensator(opt.deflection, kin)

    # 速度の係数（deflection.py）には各点のペン先の速度が要る。補正前の経路で GRBL の台形を回して出す
    use_speed = comp is not None and opt.deflection.c_speed != 0.0

    def draw_feed(ds: float, da: float) -> float:
        f = min(opt.fmax, da / (ds / opt.speed) * 60) if ds > 0 else opt.fmax
        return max(f, opt.fmin)

    for pts, ang, pen_down in jobs:
        k, rev = choose_wrap(ang, cur[0], opt.twist, opt.home, allow_reverse=not opt.keep_direction)
        ang = [(a - 360 * k, b - 360 * k) for a, b in ang]
        if rev:
            pts, ang = pts[::-1], ang[::-1]
        tail = lead_out_point(kin, pts, ang, opt, stats)
        if tail is not None:
            pts, ang = pts + [tail[0]], ang + [tail[1]]
            stats.lead_outs += 1
        start = ang[0]
        lead = lead_in_point(kin, pts, ang[0], opt, cur, comp, stats)
        # ペン上げ移動の台形（リードインがあれば 2 セグメントで 1 ブロック）
        legs = ([(cur, lead)] if lead is not None else []) + [((cur if lead is None else lead), ang[0])]
        profs = (block_profile([(b[0] - a[0], b[1] - a[1], opt.travel) for a, b in legs], opt.motion)
                 if use_speed else [None] * len(legs))
        if lead is not None:
            # リードイン点までペンを上げたまま移動し、そこから線の向きに沿って始点に入る
            if comp is not None:
                comp.travel(cur, lead, profile=profs[0])
            stats.travel_deg += move(lead[0], lead[1], opt.travel)
            stats.lead_ins += 1
        if comp is not None:
            # ペン上げ移動に沿って状態を進め、着いたところの状態で始点を補正する
            comp.travel(cur, ang[0], profile=profs[-1])
            if opt.deflection.reset_on_down:
                comp.reset()
            start = comp.angles(pts[0], ang[0])
        stats.travel_deg += move(start[0], start[1], opt.travel)
        stop()
        out += [opt.down if pen_down else opt.up, f"G4 P{opt.dwell:g}"]
        t_dwell += opt.dwell
        stats.pen_downs += int(pen_down)
        draw_profs = None
        if use_speed:
            segs, prev = [], ang[0]
            for p0, p1, ab in zip(pts, pts[1:], ang[1:]):
                ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                da = math.hypot(ab[0] - prev[0], ab[1] - prev[1])
                segs.append((ab[0] - prev[0], ab[1] - prev[1], float(round(draw_feed(ds, da)))))
                prev = ab
            draw_profs = block_profile(segs, opt.motion)
        for i, (p0, p1, ab) in enumerate(zip(pts, pts[1:], ang[1:])):
            ds = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if comp is not None and ds > 0:
                speed = cruise = None
                if draw_profs is not None and draw_profs[i] is not None:
                    pr = draw_profs[i]
                    speed = pr.speed_at(pr.d / 2) * ds / pr.d  # 区間の中央の速度を mm/s に
                    cruise = pr.vn * ds / pr.d
                comp.advance(p1[0], p1[1], (p1[0] - p0[0]) / ds, (p1[1] - p0[1]) / ds, ds, speed=speed, cruise=cruise)
                ab = comp.angles(p1, ab)
            da = math.hypot(ab[0] - cur[0], ab[1] - cur[1])
            if ds <= 0 and da <= 0:
                continue
            move(ab[0], ab[1], draw_feed(ds, da))
            stats.segments += 1
            if pen_down:
                stats.draw_mm += ds
        stop()
        out += [opt.up, f"G4 P{opt.dwell_up_s:g}"]
        t_dwell += opt.dwell_up_s

    if opt.return_home:
        stats.travel_deg += move(0.0, opt.home + 2, opt.travel)  # ストッパーの2°手前まで
        move(0.0, opt.home, 200.0)  # ゆっくり当たり位置へ（当たりを確認する）
        stop()

    if comp is not None:
        note = f"ずれの補正: {opt.deflection.describe()}、補正量の最大 {comp.max_abs:.2f} mm"
        if comp.fallbacks:
            note += (f"、補正なし {comp.fallbacks} 点（補正した点が届かないか枝が変わる。"
                     f"中心から {min(comp.fallback_r):.1f}〜{max(comp.fallback_r):.1f} mm）")
        stats.notes.append(note)
    stats.alpha_min, stats.alpha_max = lo, hi
    stats.beta_min, stats.beta_max = blo, bhi
    stats.est_time_const_s = t_const + t_dwell
    stats.est_time_s = sum(block_time(b, opt.motion) for b in blocks) + t_dwell
    stats.lines = len(out)
    if stats.segments == 0:
        warnings.warn("描くものがありません", stacklevel=2)
    return out, stats
