import math

import pytest

from ring_plotter.gcode import GcodeOptions, convert
from ring_plotter.kinematics import Kinematics
from ring_plotter.simulate import simulate
from ring_plotter.timing import Motion, axis_rate, block_time, limited_rate

M = Motion(max_rate=2000, accel=25, junction_deviation=0.01)


def test_axis_rate_and_limit():
    # 1軸だけ動くなら軸速度 = F。2軸が同量なら F/√2
    assert axis_rate(10, 0, 1500) == pytest.approx(1500)
    assert axis_rate(10, 10, 2828) == pytest.approx(2000, rel=1e-3)
    assert limited_rate(10, 0, 2500, 2000) == pytest.approx(2000)
    assert limited_rate(10, 10, 2500, 2000) == pytest.approx(2500)  # 軸では 1768 なので抑えない


def test_short_move_is_triangle():
    # 4.4° を F1000（16.7°/s）で。加速度 25 では巡航に届かず、2√(d/a)
    d = 4.4
    assert block_time([(d, 0, 1000)], M) == pytest.approx(2 * math.sqrt(d / 25))


def test_long_move_is_trapezoid():
    d, v = 100.0, 1000 / 60
    assert block_time([(d, 0, 1000)], M) == pytest.approx(d / v + v / 25)


def test_diagonal_uses_axis_limits():
    # 斜め移動は軸ごとの上限と加速度が合成方向で √2 倍になる
    d, v, a = 100.0, 2000 / 60, 25.0
    t_diag = block_time([(d / math.sqrt(2), d / math.sqrt(2), 1e9)], M)
    vn, an = v * math.sqrt(2), a * math.sqrt(2)
    assert t_diag == pytest.approx(d / vn + vn / an)


def test_straight_segments_join_without_slowing():
    one = block_time([(20, 0, 1000)], M)
    many = block_time([(1, 0, 1000)] * 20, M)
    assert many == pytest.approx(one, rel=1e-6)


def test_reversal_stops_at_junction():
    one = block_time([(5, 0, 1000)], M)
    back = block_time([(5, 0, 1000), (-5, 0, 1000)], M)
    assert back == pytest.approx(2 * one, rel=1e-6)


def test_sharp_corner_is_slower_than_gentle_curve():
    curve = block_time([(1, 0, 1000), (math.cos(0.05), math.sin(0.05), 1000)], M)
    corner = block_time([(1, 0, 1000), (0, 1, 1000)], M)
    assert corner > curve


def test_profile_speed_along_segment():
    """台形: 両端で 0、加速区間は √(2as)、中央は巡航。継ぎ目は直進なら巡航のまま、折り返しなら 0。"""
    from ring_plotter.timing import block_profile

    p = block_profile([(100.0, 0, 1000)], M)[0]
    v, a = 1000 / 60, 25.0
    assert p.speed_at(0) == 0 and p.speed_at(p.d) == 0
    assert p.speed_at(p.d / 2) == pytest.approx(v)
    assert p.speed_at(1.0) == pytest.approx(math.sqrt(2 * a * 1.0))
    assert p.speed_at(p.d - 1.0) == pytest.approx(math.sqrt(2 * a * 1.0))
    assert p.time() == pytest.approx(block_time([(100.0, 0, 1000)], M))
    straight = block_profile([(50.0, 0, 1000), (50.0, 0, 1000)], M)
    assert straight[0].speed_at(50.0) == pytest.approx(v) and straight[1].speed_at(0) == pytest.approx(v)
    back = block_profile([(50.0, 0, 1000), (-50.0, 0, 1000)], M)
    assert back[0].speed_at(50.0) == 0 and back[1].speed_at(0) == 0
    # 三角形（巡航に届かない）: 頂点は √(a·d)
    tri = block_profile([(4.0, 0, 1000)], M)[0]
    assert tri.speed_at(2.0) == pytest.approx(math.sqrt(a * 4.0))
    # 長さ 0 のセグメントは None で、他の添字はずれない
    prof = block_profile([(10.0, 0, 1000), (0.0, 0.0, 1000), (10.0, 0, 1000)], M)
    assert prof[1] is None and prof[0] is not None and prof[2] is not None


@pytest.fixture
def kin():
    return Kinematics.from_params(dmin=89, dmax=179)


def test_dwell_up_and_down(kin):
    lines, stats = convert(kin, [[(100, 0), (100, 10)]], GcodeOptions(dwell=0.3, dwell_up=0.1))
    i_down = lines.index("M3 S110")
    assert lines[i_down + 1] == "G4 P0.3"
    i_up = max(i for i, l in enumerate(lines) if l == "M5")
    assert lines[i_up + 1] == "G4 P0.1"
    assert lines[1:3] == ["M5", "G4 P0.1"]
    # 待ちだけで 0.1 + 0.3 + 0.1 秒
    assert stats.est_time_s > 0.5
    # 既定では上げも下げも同じ
    lines2, _ = convert(kin, [[(100, 0), (100, 10)]], GcodeOptions(dwell=0.3))
    assert all(l == "G4 P0.3" for l in lines2 if l.startswith("G4"))


def test_estimates_agree_between_convert_and_simulate(kin):
    pls = [[(100, 0), (100, 10)], [(50, 50), (60, 60), (70, 50)]]
    lines, stats = convert(kin, pls, GcodeOptions(motion=M))
    sim = simulate(lines, kin, motion=M)
    # G-code の座標は 3 桁に丸めてあるので、わずかに違う
    assert sim.est_time_s == pytest.approx(stats.est_time_s, rel=1e-3)
    assert sim.est_time_const_s == pytest.approx(stats.est_time_const_s, rel=1e-3)
    assert stats.est_time_s > stats.est_time_const_s


def test_axis_rate_check(kin):
    head = ["G90", "M5", "G4 P0.1"]  # 原点は (0, 88)
    sim = simulate(head + ["G1 X-10 Y88 F2500"], kin)  # α だけ動く: 軸速度 2500
    assert any("軸の速度" in p for p in sim.problems)
    sim = simulate(head + ["G1 X10 Y98 F2500"], kin)  # 2 軸が同量: 軸では 1768
    assert not any("軸の速度" in p for p in sim.problems)
    assert sim.axis_rate_max == pytest.approx(2500 / math.sqrt(2), rel=1e-3)
    sim = simulate(head + ["G1 X-10 Y88 F2500"], kin, max_rate=3000)
    assert not sim.problems
