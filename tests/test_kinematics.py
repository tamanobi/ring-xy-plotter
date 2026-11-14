import math

import pytest

from ring_plotter.kinematics import Kinematics, Unreachable


@pytest.fixture
def kin():
    return Kinematics.from_params(dmin=88, dmax=180)


def test_roundtrip_grid(kin):
    """届く範囲の格子で inverse → forward が元に戻る。"""
    n = 0
    for xi in range(-170, 171, 10):
        for yi in range(-170, 171, 10):
            r = math.hypot(xi, yi)
            if r < 3 or r > 170:
                continue
            al, be = kin.inverse(xi, yi)
            assert kin.dmin - 1e-9 <= be - al <= kin.dmax + 1e-9
            assert 0 <= al < 360
            x, y = kin.forward(al, be)
            assert math.hypot(x - xi, y - yi) < 1e-6
            n += 1
    assert n > 500


def test_alpha_zero_is_bottom(kin):
    """α=0 のとき A は下（−y）。Δ=180 近くでペン先はほぼ中心。"""
    x, y = kin.forward(0, 176.79)
    assert math.hypot(x, y) < 1.6


def test_reach(kin):
    rc = kin.reach()
    assert abs(rc.r_min - abs(kin.R - kin.L)) < 0.01
    assert 170 < rc.r_max < 180


def test_unreachable(kin):
    with pytest.raises(Unreachable):
        kin.inverse(0.5, 0.5)  # 穴の中
    with pytest.raises(Unreachable):
        kin.inverse(200, 0)  # 届かない
    k2 = Kinematics.from_params(dmin=130, dmax=180)
    with pytest.raises(Unreachable):
        k2.inverse(150, 0)  # Δの範囲外


def test_inverse_keeps_branch_near_center():
    """中心の近くでは 2 つの枝が両方有効。near_alpha を渡せば近い枝、渡さなければ最初の枝。"""
    kin = Kinematics.from_params({"R": 230.705, "d": 230.340, "e": 10.852}, dmin=89, dmax=179)
    cands = kin.inverse_all(3.0, 4.0)
    assert len(cands) == 2
    assert abs((cands[0][0] - cands[1][0] + 180) % 360 - 180) > 150  # 枝の差は α で 180° 近い
    for ab in cands:
        x, y = kin.forward(*ab)
        assert math.hypot(x - 3.0, y - 4.0) < 1e-6
    # 既定は Δ が範囲の端から遠い枝（ここでは Δ ≈ 172 の枝。もう一方は 177 で上限 179 に近い）
    margin = [min(b - a - kin.dmin, kin.dmax - (b - a)) for a, b in cands]
    assert kin.inverse(3.0, 4.0) == cands[margin.index(max(margin))]
    for ab in cands:
        assert kin.inverse(3.0, 4.0, near_alpha=ab[0] + 360 * 2 + 5) == ab
    # 外側（半径 9 mm より外）では片方だけが有効
    assert len(kin.inverse_all(20.0, 0.0)) == 1


def test_matches_fit_dots_model(kin):
    """fit_dots.py の forward と同じ式になっていること（θ=−90°、s=+1、平行移動なし）。"""
    import numpy as np

    from scripts_fit_forward import fit_forward

    al = np.radians([0, 30, 100, 250])
    be = np.radians([150, 170, 190, 400])
    q = fit_forward(np.array([kin.R, kin.d, kin.e, -math.pi / 2, 0, 0]), 1, al, be)
    for (x, y), a, b in zip(q, np.degrees(al), np.degrees(be)):
        fx, fy = kin.forward(a, b)
        assert math.hypot(fx - x, fy - y) < 1e-9
