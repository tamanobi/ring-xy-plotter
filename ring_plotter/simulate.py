"""G-code を読んで、ペン先の動きを順運動学で紙面に戻す。

対応する G-code は、このプロジェクトのスクリプトが出すもの:
  G90 / G91、G0 / G1 X.. Y.. F..、G92 X.. Y..、G4 P..、M3 S.. / M5、コメント（; と ( )）
開始位置は、最初に G92 が来なければ (X0, Y<home>) とみなす。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .kinematics import Kinematics
from .timing import Motion, Segment, axis_rate, block_time, format_time, limited_rate

WORD = re.compile(r"([A-Z])\s*(-?\d+\.?\d*|-?\.\d+)")


@dataclass
class Stroke:
    xy: list[tuple[float, float]]
    pen_down: bool
    line_no: int


@dataclass(frozen=True)
class Move:
    """G0/G1 の 1 移動（角度空間の直線）。block は静止から静止までの移動の列の番号、index はその中の位置。"""
    ab0: tuple[float, float]
    ab1: tuple[float, float]
    f: float  # 度/分
    pen_down: bool
    block: int
    index: int
    line_no: int


@dataclass
class Simulation:
    strokes: list[Stroke] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)  # 移動の列（ずれの順方向シミュレーションに使う）
    blocks: list[list[Segment]] = field(default_factory=list)  # 静止から静止までの移動の列（timing.block_profile 用）
    problems: list[str] = field(default_factory=list)
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    beta_min: float = 0.0
    beta_max: float = 0.0
    delta_min: float = math.inf
    delta_max: float = -math.inf
    f_max: float = 0.0  # 指令した F（合成速度）の最大
    axis_rate_max: float = 0.0  # 軸ごとの速度の最大（$110/$111 と比べる値）
    draw_mm: float = 0.0
    travel_deg: float = 0.0
    pen_downs: int = 0
    est_time_s: float = 0.0  # GRBL の加減速を入れた推定（秒）
    est_time_const_s: float = 0.0  # 等速とみなした推定（秒）
    lines: int = 0

    def summary(self) -> str:
        out = [
            f"G-code {self.lines}行、描画 {self.draw_mm:.0f} mm、ペン上げ移動 {self.travel_deg:.0f}°、"
            f"ペン下げ {self.pen_downs}回、推定 {format_time(self.est_time_s)}"
            f"（加減速込み。等速なら {format_time(self.est_time_const_s)}）",
            f"α {self.alpha_min:.0f}〜{self.alpha_max:.0f}°、β {self.beta_min:.0f}〜{self.beta_max:.0f}°、"
            f"Δ {self.delta_min:.1f}〜{self.delta_max:.1f}°、F の最大 {self.f_max:.0f}（軸では {self.axis_rate_max:.0f}）",
        ]
        if self.problems:
            out.append(f"問題 {len(self.problems)}件:")
            out += ["  " + p for p in self.problems[:20]]
            if len(self.problems) > 20:
                out.append(f"  …ほか {len(self.problems) - 20}件")
        else:
            out.append("問題なし")
        return "\n".join(out)


def simulate(lines: list[str], kin: Kinematics, home: float = 88.0, guard: tuple[float, float] = (88.0, 180.0),
             twist: float = 400.0, max_rate: float | None = None, substep_deg: float = 0.5,
             motion: Motion | None = None) -> Simulation:
    """max_rate は motion.max_rate の省略形（両方あれば max_rate を優先）。"""
    if motion is None:
        motion = Motion()
    if max_rate is not None:
        motion = Motion(max_rate, motion.accel, motion.junction_deviation)
    max_rate = motion.max_rate
    sim = Simulation(lines=len(lines))
    pos = (0.0, home)
    absolute = True
    pen = False
    f = 0.0
    strokes: list[Stroke] = []
    cur_stroke: Stroke | None = None
    gmin, gmax = guard
    block: list[Segment] = []  # 静止から静止までの移動の列
    t_move = 0.0

    def stop():
        """M3/M5 と G4 の前で GRBL は動きを止める。"""
        nonlocal block, t_move
        if block:
            t_move += block_time(block, motion)
            sim.blocks.append(block)
            block = []

    def note_pos(a, b, line_no):
        sim.alpha_min, sim.alpha_max = min(sim.alpha_min, a), max(sim.alpha_max, a)
        sim.beta_min, sim.beta_max = min(sim.beta_min, b), max(sim.beta_max, b)
        d = b - a
        sim.delta_min, sim.delta_max = min(sim.delta_min, d), max(sim.delta_max, d)
        if not gmin - 1e-6 <= d <= gmax + 1e-6:
            sim.problems.append(f"{line_no}行目: Δ={d:.3f}° がガード範囲（{gmin:g}〜{gmax:g}°）の外（grbl_send.py が止める）")
        if abs(a) > twist + 1e-6:
            sim.problems.append(f"{line_no}行目: α={a:.1f}° がねじれ上限 ±{twist:g}° を超える")
        if abs(b - home) > twist + 1e-6:
            sim.problems.append(f"{line_no}行目: β={b:.1f}° が B のねじれ上限 {home:g}±{twist:g}° を超える")

    note_pos(*pos, 0)
    for n, raw in enumerate(lines, 1):
        code = re.sub(r"\(.*?\)", "", raw.split(";")[0]).strip().upper()
        if not code or code.startswith("$"):
            continue
        words = WORD.findall(code)
        gs = {float(v) for k, v in words if k == "G"}
        ms = {float(v) for k, v in words if k == "M"}
        x = next((float(v) for k, v in words if k == "X"), None)
        y = next((float(v) for k, v in words if k == "Y"), None)
        fw = next((float(v) for k, v in words if k == "F"), None)
        pw = next((float(v) for k, v in words if k == "P"), None)
        if 90 in gs:
            absolute = True
        if 91 in gs:
            absolute = False
        if 4 in gs and pw is not None:
            stop()
            sim.est_time_s += pw
            sim.est_time_const_s += pw
        if 3 in ms or 4 in ms:
            stop()
            if not pen:
                sim.pen_downs += 1
            pen = True
            cur_stroke = None
        if 5 in ms:
            stop()
            pen = False
            cur_stroke = None
        if 92 in gs:
            stop()
            pos = (pos[0] if x is None else x, pos[1] if y is None else y)
            note_pos(*pos, n)
            cur_stroke = None
            continue
        if fw is not None:
            f = fw
            sim.f_max = max(sim.f_max, f)
        if x is None and y is None:
            continue
        if not (0 in gs or 1 in gs):
            continue
        if absolute:
            tgt = (pos[0] if x is None else x, pos[1] if y is None else y)
        else:
            tgt = (pos[0] + (x or 0.0), pos[1] + (y or 0.0))
        dab = (tgt[0] - pos[0], tgt[1] - pos[1])
        da = math.hypot(*dab)
        rate = max_rate if 0 in gs else f
        if rate > 0:
            if da > 0:
                # F は (α, β) の合成速度。$110/$111 は軸ごとの上限なので、速いほうの軸で比べる
                ar = axis_rate(*dab, rate)
                sim.axis_rate_max = max(sim.axis_rate_max, ar)
                if 1 in gs and ar > max_rate + 1e-6:
                    sim.problems.append(f"{n}行目: F{rate:.0f} で軸の速度が {ar:.0f}°/分になり、"
                                        f"GRBL の最大速度 {max_rate:g} を超える（$110/$111 で頭打ち）")
                sim.est_time_const_s += da / limited_rate(*dab, rate, max_rate) * 60
                sim.moves.append(Move(pos, tgt, rate, pen, len(sim.blocks), len(block), n))
                block.append((dab[0], dab[1], rate))
        elif da > 0:
            sim.problems.append(f"{n}行目: F が未指定のまま移動")
        if pen:
            sim.draw_mm += 0.0  # 下で xy 長を足す
        else:
            sim.travel_deg += da

        # 角度空間で直線に動くので、細かく刻んで xy に戻す
        steps = max(1, math.ceil(da / substep_deg))
        if cur_stroke is None or cur_stroke.pen_down != pen:
            cur_stroke = Stroke([kin.forward(*pos)], pen, n)
            strokes.append(cur_stroke)
        for i in range(1, steps + 1):
            a = pos[0] + (tgt[0] - pos[0]) * i / steps
            b = pos[1] + (tgt[1] - pos[1]) * i / steps
            p = kin.forward(a, b)
            if pen:
                q = cur_stroke.xy[-1]
                sim.draw_mm += math.hypot(p[0] - q[0], p[1] - q[1])
            cur_stroke.xy.append(p)
        pos = tgt
        note_pos(*pos, n)

    stop()
    sim.est_time_s += t_move
    sim.strokes = strokes
    return sim
