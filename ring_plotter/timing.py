"""GRBL の加減速を入れた所要時間の推定。

GRBL は軸ごとの最大速度（$110/$111）と加速度（$120/$121）で台形の速度プロファイルを作り、
連続する移動の継ぎ目は junction deviation（$11）で減速する。M3/M5（レーザーモード以外）と
G4 の前では動きが止まる。ここでは、静止から静止までの移動の列（ブロック）ごとに
GRBL の planner と同じ考え方（継ぎ目の上限 → 後ろ向き → 前向きの2パス）で時間を出す。

F は (α, β) の合成速度で、$110/$111 は軸ごとの上限なので、GRBL は速いほうの軸が上限に
収まるように合成速度を抑える。加速度も同じく軸ごとに掛かる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Segment = tuple[float, float, float]  # (Δα, Δβ, F) 度、度、度/分


@dataclass(frozen=True)
class Motion:
    max_rate: float = 2000.0  # $110/$111（度/分、軸ごと）
    accel: float = 200.0  # $120/$121（度/秒²、軸ごと）。実機の設定（10 月中旬に 100 → 200、脱調なしを確認）
    junction_deviation: float = 0.01  # $11（度）


@dataclass(frozen=True)
class SegmentProfile:
    """1 セグメントの台形プロファイル（合成方向、度と度/秒）。"""
    d: float  # 距離（度）
    v0: float  # 入口の速度
    v1: float  # 出口の速度
    vn: float  # 巡航の上限
    a: float  # 加速度（合成方向）

    def speed_at(self, s: float) -> float:
        """始点から s 度進んだ地点の速度（度/秒）。台形でも三角形でも、3 つの上限の最小になる。"""
        s = min(max(s, 0.0), self.d)
        return min(self.vn, math.sqrt(self.v0 ** 2 + 2 * self.a * s),
                   math.sqrt(self.v1 ** 2 + 2 * self.a * (self.d - s)))

    def time(self) -> float:
        return _profile_time(self.d, self.v0, self.v1, self.vn, self.a)


def axis_rate(da: float, db: float, f: float) -> float:
    """合成速度 F で (da, db) 動くとき、速いほうの軸の速度（度/分）。"""
    d = math.hypot(da, db)
    return f * max(abs(da), abs(db)) / d if d > 0 else 0.0


def limited_rate(da: float, db: float, f: float, max_rate: float) -> float:
    """GRBL が軸ごとの上限で抑えたあとの合成速度（度/分）。"""
    d = math.hypot(da, db)
    if d <= 0:
        return 0.0
    return min(f, max_rate * d / max(abs(da), abs(db)))


def _profile_time(d: float, v0: float, v1: float, vn: float, a: float) -> float:
    """距離 d を、入口 v0、出口 v1、巡航 vn、加速度 a の台形（届かなければ三角形）で動く時間。"""
    if d <= 0:
        return 0.0
    d_acc = (vn * vn - v0 * v0) / (2 * a)
    d_dec = (vn * vn - v1 * v1) / (2 * a)
    if d_acc + d_dec <= d:
        return (vn - v0) / a + (vn - v1) / a + (d - d_acc - d_dec) / vn
    vp = math.sqrt(max((2 * a * d + v0 * v0 + v1 * v1) / 2, 0.0))
    return (vp - v0) / a + (vp - v1) / a


def block_profile(segs: list[Segment], motion: Motion) -> list[SegmentProfile | None]:
    """静止から静止までの移動の列の、セグメントごとの台形プロファイル。F は度/分。

    戻り値は segs と同じ長さ（長さ 0 のセグメントは None）。GRBL の planner と同じ考え方で、
    継ぎ目の上限（junction deviation）→ 後ろ向き → 前向きの 2 パスで入口・出口の速度を決める。
    """
    d, vn, acc, unit, idx = [], [], [], [], []
    for i, (da, db, f) in enumerate(segs):
        dist = math.hypot(da, db)
        if dist <= 0:
            continue
        u = (da / dist, db / dist)
        m = max(abs(u[0]), abs(u[1]))
        d.append(dist)
        vn.append(max(min(f, motion.max_rate / m) / 60, 1e-9))
        acc.append(motion.accel / m)
        unit.append(u)
        idx.append(i)
    n = len(d)
    out: list[SegmentProfile | None] = [None] * len(segs)
    if n == 0:
        return out

    # 継ぎ目の上限（GRBL planner の junction deviation。両端は静止）
    vj = [0.0] * (n + 1)
    for i in range(1, n):
        cos_t = -(unit[i - 1][0] * unit[i][0] + unit[i - 1][1] * unit[i][1])
        if cos_t > 0.999999:  # 折り返し
            v = 0.0
        elif cos_t < -0.999999:  # 直進
            v = math.inf
        else:
            s = math.sqrt(0.5 * (1.0 - cos_t))
            v = math.sqrt(min(acc[i - 1], acc[i]) * motion.junction_deviation * s / (1.0 - s))
        vj[i] = min(v, vn[i - 1], vn[i])
    for i in range(n - 1, -1, -1):  # 後ろ向き: 出口速度から減速で入れる速度
        vj[i] = min(vj[i], math.sqrt(vj[i + 1] ** 2 + 2 * acc[i] * d[i]))
    for i in range(n):  # 前向き: 入口速度から加速で出せる速度
        vj[i + 1] = min(vj[i + 1], math.sqrt(vj[i] ** 2 + 2 * acc[i] * d[i]))
    for i in range(n):
        out[idx[i]] = SegmentProfile(d[i], vj[i], vj[i + 1], vn[i], acc[i])
    return out


def block_time(segs: list[Segment], motion: Motion) -> float:
    """静止から静止までの移動の列の所要時間（秒）。F は度/分。"""
    return sum(p.time() for p in block_profile(segs, motion) if p is not None)


def format_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}分{s:02d}秒"
