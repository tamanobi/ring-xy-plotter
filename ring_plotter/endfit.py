"""G-code の指令経路に沿って機械のずれを順方向に再現し、スキャンの線の形（特に両端）と比べて係数を当てはめる。

補正器（deflection.Compensator）は「目標 − ずれ」を指令にする。ここでは逆に、出来上がった G-code の指令経路を
GRBL の台形（timing.block_profile）の速度つきで歩き、同じ力モデルでずれの状態を更新して「描かれる位置 =
指令 + ずれ」を出す。測った線（measure_scan の r.s, r.t）と、線ごとに中央の平均を引いて比べれば、
始点・終点の 0.3〜3.5 mm の落ち込みがモデルで出るかが分かる。端の係数 c_end、巡航の係数 c_speed、
全体の倍率（a, b に掛かる）を当てはめられる（scripts/fit_end_effect.py）。

線の対応は、パッチの線 k の設計の始点から 1 mm 以内で始まるペン下げストロークを探す（線ごとにペンを上げた
パッチ、Layout.separate 用。ジグザグでも線 0 だけは対応がつく）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .deflection import DeflectionModel, Layout, PatchResult, line_frame, zigzag
from .kinematics import Kinematics, Unreachable
from .simulate import Simulation, simulate
from .timing import Motion, block_profile


@dataclass
class Trace:
    """指令経路を刻んだもの。配列は刻みごと（刻みの終点）。"""
    xy: np.ndarray  # (N, 2) 指令の位置
    steady: np.ndarray  # (N, 2) その点・向きでの定常のずれ（速度の係数を掛ける前）。届かない点は nan
    ds: np.ndarray  # 刻みの長さ（mm）
    speed: np.ndarray  # ペン先の速度（mm/s、刻みの中央）
    cruise: np.ndarray  # その移動の巡航速度（mm/s）
    pen: np.ndarray  # ペン下げなら True
    stroke: np.ndarray  # ペン下げストロークの番号（ペン上げは −1）
    starts: list[tuple[float, float]] = field(default_factory=list)  # ストロークごとの指令の始点（ペンを下ろした位置）


def trace_gcode(lines: list[str], kin: Kinematics, model: DeflectionModel, motion: Motion | None = None,
                substep_deg: float = 0.5, sim: Simulation | None = None) -> Trace:
    """G-code を歩いて Trace を作る。model は定常のずれ（a, b, model）にだけ使う。"""
    motion = motion or Motion()
    if sim is None:
        sim = simulate(lines, kin, motion=motion)
    profiles = [block_profile(b, motion) for b in sim.blocks]
    xy, steady, dss, speeds, cruises, pens, strokes = [], [], [], [], [], [], []
    starts: list[tuple[float, float]] = []
    pen_prev = False
    stroke = -1
    for m in sim.moves:
        prof = profiles[m.block][m.index] if m.block < len(profiles) else None
        if m.pen_down and not pen_prev:
            stroke += 1
            starts.append(kin.forward(*m.ab0))
        pen_prev = m.pen_down
        d_deg = math.hypot(m.ab1[0] - m.ab0[0], m.ab1[1] - m.ab0[1])
        n = max(1, math.ceil(max(abs(m.ab1[0] - m.ab0[0]), abs(m.ab1[1] - m.ab0[1])) / substep_deg))
        prev = kin.forward(*m.ab0)
        for i in range(1, n + 1):
            ab = (m.ab0[0] + (m.ab1[0] - m.ab0[0]) * i / n, m.ab0[1] + (m.ab1[1] - m.ab0[1]) * i / n)
            p = kin.forward(*ab)
            dx, dy = p[0] - prev[0], p[1] - prev[1]
            ds = math.hypot(dx, dy)
            prev = p
            if ds <= 1e-9:
                continue
            try:
                st = model.steady(kin, p[0], p[1], dx / ds, dy / ds)
            except Unreachable:
                st = np.array([math.nan, math.nan])
            if prof is not None and d_deg > 0:
                mm_per_deg = ds / (d_deg / n)
                sp = prof.speed_at(d_deg * (i - 0.5) / n) * mm_per_deg
                cr = prof.vn * mm_per_deg
            else:
                sp = cr = math.nan
            xy.append(p)
            steady.append(st)
            dss.append(ds)
            speeds.append(sp)
            cruises.append(cr)
            pens.append(m.pen_down)
            strokes.append(stroke if m.pen_down else -1)
    return Trace(np.array(xy).reshape(-1, 2), np.array(steady).reshape(-1, 2), np.array(dss), np.array(speeds),
                 np.array(cruises), np.array(pens, dtype=bool), np.array(strokes, dtype=int), starts)


def speed_factors(tr: Trace, c_speed: float, v_ref: float, c_end: float) -> np.ndarray:
    """刻みごとの速度の係数 f（DeflectionModel.speed_factor と同じ式）。速度が不明な刻みは 1。"""
    f = np.ones(len(tr.ds))
    known = ~np.isnan(tr.speed)
    f[known] = 1.0 - c_speed * np.maximum(0.0, 1.0 - tr.speed[known] / v_ref)
    if c_end:
        ok = known & ~np.isnan(tr.cruise) & (tr.cruise > 0)
        f[ok] *= 1.0 - c_end * np.maximum(0.0, 1.0 - tr.speed[ok] / tr.cruise[ok])
    return f


def run_state(tr: Trace, lam: float, factors: np.ndarray, scale: float = 1.0, gain_up: float = 1.0) -> np.ndarray:
    """ずれの状態を刻みごとに更新して返す（(N, 2)、各刻みの終点での値）。届かない刻みは状態を据え置く。"""
    tgt = tr.steady * (scale * factors)[:, None]
    tgt[~tr.pen] *= gain_up
    k = 1.0 - np.exp(-tr.ds / lam)
    out = np.zeros_like(tgt)
    state = np.zeros(2)
    for i in range(len(k)):
        if not math.isnan(tgt[i, 0]):
            state = state + (tgt[i] - state) * k[i]
        out[i] = state
    return out


@dataclass
class LineMatch:
    """測った線 1 本と、その指令ストロークの対応。"""
    patch_index: int
    k: int
    sgn: int
    s: np.ndarray  # 測った列の位置（線に沿って、パッチ中心が 0）
    t: np.ndarray  # 測った横位置（nan は欠測）
    idx: np.ndarray  # Trace の刻みの添字（このストロークのペン下げ刻み）
    s_cmd: np.ndarray  # 刻みの指令位置の、線に沿った位置
    t_cmd: np.ndarray  # 同、横位置
    m0: np.ndarray  # 横方向の単位ベクトル（測った枠）


def match_lines(results: list[PatchResult], lay: Layout, tr: Trace, tol: float = 1.0) -> list[LineMatch]:
    """パッチの各線に、設計の始点から tol 以内で始まるペン下げストロークを対応づける。"""
    out = []
    starts = np.array(tr.starts).reshape(-1, 2)
    for pi, r in enumerate(results):
        if not r.ok or len(r.s) == 0:
            continue
        v0, m0 = line_frame(r.theta)
        c = np.array(r.center)
        zz = zigzag(r.patch, lay)
        for k in range(lay.lines):
            t = r.t[:, k]
            if np.sum(~np.isnan(t)) < 10:
                continue
            start = np.array(zz[2 * k])
            if len(starts) == 0:
                break
            j = int(np.argmin(np.linalg.norm(starts - start, axis=1)))
            if np.linalg.norm(starts[j] - start) > tol:
                continue
            idx = np.nonzero(tr.stroke == j)[0]
            if len(idx) < 3:
                continue
            rel = tr.xy[idx] - c
            out.append(LineMatch(pi, k, 1 if k % 2 == 0 else -1, r.s, t, idx, rel @ v0, rel @ m0, m0))
    return out


def predicted_tracks(matches: list[LineMatch], tr: Trace, state: np.ndarray) -> list[np.ndarray]:
    """各線について、描かれる横位置（指令 + ずれ）を測った列 s に補間したもの。"""
    out = []
    for lm in matches:
        t_pred = lm.t_cmd + state[lm.idx] @ lm.m0
        order = np.argsort(lm.s_cmd)
        out.append(np.interp(lm.s, lm.s_cmd[order], t_pred[order]))
    return out


def end_residuals(matches: list[LineMatch], preds: list[np.ndarray], length: float, dmin: float = 0.3,
                  dmax: float = 3.5, mid: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """線ごとに中央（|s| < mid）の平均を引いたうえで、始点側と終点側の dmin〜dmax の『測った − 予測』を集める。

    戻り値: (全残差, 線ごとの始点側の平均残差, 線ごとの終点側の平均残差)。
    """
    res, rs, re_ = [], [], []
    for lm, pr in zip(matches, preds):
        ok = ~np.isnan(lm.t)
        m = ok & (np.abs(lm.s) < mid)
        if m.sum() < 5:
            rs.append(math.nan)
            re_.append(math.nan)
            continue
        meas = lm.t - np.mean(lm.t[m])
        pred = pr - np.mean(pr[m])
        d = lm.s * lm.sgn + length / 2
        a = ok & (d >= dmin) & (d < dmax)
        b = ok & (d > length - dmax) & (d <= length - dmin)
        diff = meas - pred
        res += list(diff[a]) + list(diff[b])
        rs.append(float(np.mean(diff[a])) if a.any() else math.nan)
        re_.append(float(np.mean(diff[b])) if b.any() else math.nan)
    return np.array(res), np.array(rs), np.array(re_)


@dataclass
class Sheet:
    name: str
    trace: Trace
    matches: list[LineMatch]
    length: float


def fit_coefficients(sheets: list[Sheet], model: DeflectionModel, fit: tuple[str, ...] = ("c_end",),
                     dmin: float = 0.3, dmax: float = 3.5) -> dict:
    """c_end / c_speed / scale のうち fit にあるものを、両端の残差 rms が最小になるように当てはめる。"""
    from scipy.optimize import minimize

    names = tuple(fit)
    x0 = {"c_end": model.c_end, "c_speed": model.c_speed, "scale": 1.0}

    def unpack(x):
        p = dict(x0)
        for n, v in zip(names, x):
            p[n] = float(v)
        return p

    def resid(p):
        allr = []
        for sh in sheets:
            f = speed_factors(sh.trace, p["c_speed"], model.v_ref, p["c_end"])
            st = run_state(sh.trace, model.lam, f, p["scale"], model.gain_up)
            r, _, _ = end_residuals(sh.matches, predicted_tracks(sh.matches, sh.trace, st), sh.length, dmin, dmax)
            allr.append(r)
        return np.concatenate(allr)

    def cost(x):
        p = unpack(x)
        if not (0.0 <= p["c_end"] <= 1.0 and 0.0 <= p["c_speed"] <= 1.0 and 0.3 <= p["scale"] <= 3.0):
            return 1e3
        return float(np.sqrt(np.mean(resid(p) ** 2)))

    before = cost([x0[n] for n in names])
    best = None
    if names == ("c_end",):
        grid = np.linspace(0.0, 1.0, 21)
        c = [cost([g]) for g in grid]
        start = [float(grid[int(np.argmin(c))])]
    else:
        start = [x0[n] for n in names]
        if "c_end" in names and x0["c_end"] == 0.0:
            start[names.index("c_end")] = 0.3
    res = minimize(cost, start, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
    best = unpack(res.x)
    return dict(params=best, rms=float(res.fun), rms_before=before, n=int(len(resid(best))))


def sheet_summary(sh: Sheet, model: DeflectionModel, p: dict, kin: Kinematics, lay: Layout,
                  results: list[PatchResult], dmin: float = 0.3, dmax: float = 3.5) -> dict:
    """1 枚ぶんの、測った端の落ち込みと予測の落ち込み（D予測の向きに合わせた平均、mm）。"""
    from .deflection import predict_D

    f = speed_factors(sh.trace, p["c_speed"], model.v_ref, p["c_end"])
    st = run_state(sh.trace, model.lam, f, p["scale"], model.gain_up)
    preds = predicted_tracks(sh.matches, sh.trace, st)
    meas_s, meas_e, pred_s, pred_e = [], [], [], []
    for lm, pr in zip(sh.matches, preds):
        r = results[lm.patch_index]
        dp = predict_D(kin, r.center[0], r.center[1], r.theta, model.a, model.b, model.model)
        if abs(dp) < 0.1:
            continue
        ok = ~np.isnan(lm.t)
        m = ok & (np.abs(lm.s) < 4.0)
        if m.sum() < 5:
            continue
        sg = lm.sgn * np.sign(dp)
        meas = (lm.t - np.mean(lm.t[m])) * sg
        pred = (pr - np.mean(pr[m])) * sg
        d = lm.s * lm.sgn + sh.length / 2
        a = ok & (d >= dmin) & (d < dmax)
        b = ok & (d > sh.length - dmax) & (d <= sh.length - dmin)
        if a.any():
            meas_s.append(np.mean(meas[a]))
            pred_s.append(np.mean(pred[a]))
        if b.any():
            meas_e.append(np.mean(meas[b]))
            pred_e.append(np.mean(pred[b]))
    return dict(meas_start=float(np.mean(meas_s)) if meas_s else math.nan,
                pred_start=float(np.mean(pred_s)) if pred_s else math.nan,
                meas_end=float(np.mean(meas_e)) if meas_e else math.nan,
                pred_end=float(np.mean(pred_e)) if pred_e else math.nan, n=len(meas_s))
