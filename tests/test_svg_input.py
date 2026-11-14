import math
import os

import pytest

from ring_plotter.svg_input import load_svg, polyline_length, transform

HERE = os.path.dirname(os.path.abspath(__file__))


def write(tmp_path, body, size='width="100mm" height="80mm" viewBox="0 0 100 80"'):
    p = tmp_path / "t.svg"
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" {size}>{body}</svg>', encoding="utf-8")
    return str(p)


def test_center_and_orientation(tmp_path):
    # SVG 左上の点 (0,0) → リング座標では (-50, +40)。下端中央 (50,80) → (0, -40)
    p = write(tmp_path, '<line x1="0" y1="0" x2="50" y2="80"/>')
    d = load_svg(p)
    assert d.width_mm == pytest.approx(100, abs=1e-3) and d.height_mm == pytest.approx(80, abs=1e-3)
    (a, b), = d.polylines
    assert a == pytest.approx((-50, 40), abs=1e-3) and b == pytest.approx((0, -40), abs=1e-3)


def test_rect_closed_and_transform(tmp_path):
    p = write(tmp_path, '<g transform="translate(10,10)"><rect x="0" y="0" width="20" height="10"/></g>')
    (pl,) = load_svg(p).polylines
    assert pl[0] == pytest.approx(pl[-1])
    assert polyline_length(pl) == pytest.approx(60)
    assert min(x for x, _ in pl) == pytest.approx(-40) and max(y for _, y in pl) == pytest.approx(30)


def test_circle_flattened(tmp_path):
    p = write(tmp_path, '<circle cx="50" cy="40" r="10"/>')
    (pl,) = load_svg(p, step=1.0).polylines
    assert all(abs(math.hypot(x, y) - 10) < 0.02 for x, y in pl)
    assert len(pl) >= 60
    assert polyline_length(pl) == pytest.approx(2 * math.pi * 10, rel=1e-3)


def test_text_skipped_and_display_none(tmp_path):
    p = write(tmp_path, '<text x="1" y="1">hi</text><path style="display:none" d="M0 0 L5 5"/><path d="M0 0 L5 5"/>')
    with pytest.warns(UserWarning):
        d = load_svg(p)
    assert len(d.polylines) == 1 and len(d.skipped) == 1


def test_viewbox_scaling(tmp_path):
    # viewBox の単位が mm と一致しない場合も mm に直る
    p = write(tmp_path, '<line x1="0" y1="0" x2="1000" y2="0"/>', size='width="100mm" height="80mm" viewBox="0 0 1000 800"')
    (pl,) = load_svg(p).polylines
    assert polyline_length(pl) == pytest.approx(100)


def test_transform():
    pl = transform([[(10, 0)]], scale=2, rotate=90, dx=1, dy=2)
    assert pl[0][0] == pytest.approx((1, 22))
    pl = transform([[(10, 0)]], mirror=True)
    assert pl[0][0] == pytest.approx((-10, 0))


def test_example_pattern():
    d = load_svg(os.path.join(HERE, "..", "examples", "test_pattern.svg"))
    assert d.width_mm == pytest.approx(210) and d.height_mm == pytest.approx(297)
    assert len(d.polylines) == 11
