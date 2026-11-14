"""G-code の指令経路に沿ったずれの順方向の再現（ring_plotter.endfit）と端の係数の当てはめ。"""
import math

import numpy as np
import pytest

from ring_plotter.deflection import DeflectionModel, Layout, PatchResult, line_frame, patch_polylines
from ring_plotter.endfit import (end_residuals, fit_coefficients, match_lines, predicted_tracks, run_state, Sheet,
                                 speed_factors, trace_gcode)
from ring_plotter.gcode import GcodeOptions, convert
from ring_plotter.kinematics import Kinematics
from ring_plotter.timing import Motion

MOTION = Motion(accel=100.0)


@pytest.fixture(scope="module")
def kin():
    return Kinematics.from_params(dmin=89, dmax=179)


@pytest.fixture(scope="module")
def layout():
    return Layout(grid_x=(-40.0,), grid_y=(60.0,), angles=(0.0, 90.0), separate="alt", mark=False)


def gcode_for(kin, layout, model):
    polys = [pl for p in layout.patches() for pl in patch_polylines(p, layout)]
    opt = GcodeOptions(sort=True, keep_direction=True, step=1.0, speed=30.0, fmax=2250.0, return_home=False,
                       deflection=model, lead_in=3.0, lead_in_min_travel=0.0, motion=MOTION)
    lines, _ = convert(kin, polys, opt)
    return lines


def nominal_distance(p, polys):
    best = math.inf
    for poly in polys:
        a, b = np.array(poly[0]), np.array(poly[1])
        ab = b - a
        t = float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def test_trace_reproduces_compensation(kin, layout):
    """補正に使ったのと同じモデルで順方向に再現すると、描かれる位置は元の線に戻る。別の係数ならずれる。"""
    m = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", c_speed=0.17, c_end=0.3)
    lines = gcode_for(kin, layout, m)
    tr = trace_gcode(lines, kin, m, MOTION)
    assert tr.pen.any() and len(tr.starts) == layout.lines * len(layout.angles)
    assert np.nanmax(tr.speed[tr.pen]) == pytest.approx(30.0, abs=1.0)
    st = run_state(tr, m.lam, speed_factors(tr, m.c_speed, m.v_ref, m.c_end))
    drawn = tr.xy[tr.pen] + st[tr.pen]
    polys = [pl for p in layout.patches() for pl in patch_polylines(p, layout)]
    errs = np.array([nominal_distance(q, polys) for q in drawn])
    assert errs.max() < 0.03, errs.max()
    # 端の係数を落とした「機械」では、線の端が 0.02 mm 以上ずれる
    st0 = run_state(tr, m.lam, speed_factors(tr, m.c_speed, m.v_ref, 0.0))
    errs0 = np.array([nominal_distance(q, polys) for q in tr.xy[tr.pen] + st0[tr.pen]])
    assert errs0.max() > 0.02


def synth_results(kin, layout, tr, state, column=0.25, noise=0.0, seed=0):
    """順方向の再現から、measure_scan が返すのと同じ形の PatchResult（s と t の列）を合成する。"""
    rng = np.random.default_rng(seed)
    starts = np.array(tr.starts)
    out = []
    half = layout.length / 2 - 0.3
    s_mid = np.arange(-half, half + column / 2, column)[:-1] + column / 2
    for p in layout.patches():
        v0, m0 = line_frame(p.theta)
        c = np.array([p.x, p.y])
        tt = np.full((len(s_mid), layout.lines), np.nan)
        for k, poly in enumerate(patch_polylines(p, layout)):
            j = int(np.argmin(np.linalg.norm(starts - np.array(poly[0]), axis=1)))
            idx = np.nonzero(tr.stroke == j)[0]
            drawn = tr.xy[idx] + state[idx] - c
            s, t = drawn @ v0, drawn @ m0
            order = np.argsort(s)
            tt[:, k] = np.interp(s_mid, s[order], t[order]) + noise * rng.standard_normal(len(s_mid))
        out.append(PatchResult(p, (p.x, p.y), p.theta, layout.pitch, 0.0, 0.0, 1.4, 0.01, s_mid, tt))
    return out


def test_fit_recovers_end_coefficient(kin, layout):
    """c_end = 0 で補正して描いた G-code を、c_end = 0.3 の「機械」で描いた線から、c_end が戻る。"""
    applied = DeflectionModel(0.5, 0.08, 1.4, model="rod_axial_sin2_sqrt", c_speed=0.17)
    lines = gcode_for(kin, layout, applied)
    tr = trace_gcode(lines, kin, applied, MOTION)
    truth = run_state(tr, applied.lam, speed_factors(tr, applied.c_speed, applied.v_ref, 0.3))
    results = synth_results(kin, layout, tr, truth, noise=0.01)
    matches = match_lines(results, layout, tr)
    assert len(matches) == layout.lines * len(layout.angles)
    sheet = Sheet("synth", tr, matches, layout.length)
    fit = fit_coefficients([sheet], applied, ("c_end",))
    assert abs(fit["params"]["c_end"] - 0.3) < 0.04, fit
    assert fit["rms"] < fit["rms_before"]
    # 当てはめ前は端に系統的な残差があり、当てはめ後は消える
    st_b = run_state(tr, applied.lam, speed_factors(tr, applied.c_speed, applied.v_ref, 0.0))
    _, rs_b, _ = end_residuals(matches, predicted_tracks(matches, tr, st_b), layout.length)
    st_a = run_state(tr, applied.lam, speed_factors(tr, applied.c_speed, applied.v_ref, fit["params"]["c_end"]))
    _, rs_a, _ = end_residuals(matches, predicted_tracks(matches, tr, st_a), layout.length)
    assert np.nanmean(np.abs(rs_b)) > 3 * np.nanmean(np.abs(rs_a))
    # 全体の倍率も同時に当てはめられる（真値 1.0）
    fit2 = fit_coefficients([sheet], applied, ("c_end", "scale"))
    assert abs(fit2["params"]["scale"] - 1.0) < 0.1 and abs(fit2["params"]["c_end"] - 0.3) < 0.08
