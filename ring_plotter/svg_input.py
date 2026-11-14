"""SVG を読み、mm 単位の折れ線（ポリライン）の列にする。

- path、rect、circle、ellipse、line、polyline、polygon を扱う。transform は適用済み
- 塗りは無視し、輪郭（線）だけを取り出す
- text は扱わない（Inkscape でパスに変換しておく）。見つけたら警告する
- display:none の要素は飛ばす
- 曲線は最大弦長 `step` で折れ線にする

座標系の変換:
  SVG は左上原点・y 下向き。描画領域（width/height、なければ viewBox）の中心を
  リングの中心（原点）に合わせ、y を上向きに反転する。
  SVG の下が α=0° の方向（−y）になる。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

from svgelements import SVG, Close, Line, Move, Path, Shape, Text

PX_TO_MM = 25.4 / 96.0

Polyline = list[tuple[float, float]]


@dataclass
class SvgDrawing:
    polylines: list[Polyline]
    width_mm: float
    height_mm: float
    skipped: list[str] = field(default_factory=list)  # 扱えなかった要素の説明

    @property
    def total_length(self) -> float:
        return sum(polyline_length(p) for p in self.polylines)

    def bbox(self) -> tuple[float, float, float, float] | None:
        pts = [p for pl in self.polylines for p in pl]
        if not pts:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


def polyline_length(pl: Polyline) -> float:
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pl, pl[1:]))


def _flatten(path: Path, step_px: float) -> list[Polyline]:
    """Path を折れ線の列に。Move で分割、曲線は最大弦長 step_px で分ける。"""
    out: list[Polyline] = []
    cur: Polyline = []
    for seg in path.segments():
        if isinstance(seg, Move):
            if len(cur) >= 2:
                out.append(cur)
            cur = [(seg.end.x, seg.end.y)]
            continue
        if seg.start is None or seg.end is None:
            continue
        if not cur:
            cur = [(seg.start.x, seg.start.y)]
        if isinstance(seg, (Line, Close)):
            cur.append((seg.end.x, seg.end.y))
            continue
        # Arc, QuadraticBezier, CubicBezier
        length = seg.length(error=1e-4)
        n = max(2, math.ceil(length / step_px))
        for i in range(1, n + 1):
            p = seg.point(i / n)
            cur.append((p.x, p.y))
    if len(cur) >= 2:
        out.append(cur)
    return out


def _dedupe(pl: Polyline, eps: float = 1e-6) -> Polyline:
    out = [pl[0]]
    for p in pl[1:]:
        if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    return out


def load_svg(path: str, step: float = 1.0) -> SvgDrawing:
    """SVG ファイルを読み、リング座標系（中心原点、y上向き、mm）の折れ線にする。"""
    svg = SVG.parse(path, reify=True)
    width_px, height_px = float(svg.width), float(svg.height)
    if not (width_px > 0 and height_px > 0):
        raise ValueError("SVG の大きさ（width/height または viewBox）が取れません")
    cx, cy = width_px / 2, height_px / 2
    step_px = step / PX_TO_MM

    polylines: list[Polyline] = []
    skipped: list[str] = []
    for e in svg.elements():
        if isinstance(e, Text):
            skipped.append(f"text: {e.text!r}（Inkscape でパスに変換してください）")
            continue
        if not isinstance(e, Shape):
            continue
        p = e if isinstance(e, Path) else Path(e)
        for pl in _flatten(p, step_px):
            pl = _dedupe([((x - cx) * PX_TO_MM, -(y - cy) * PX_TO_MM) for x, y in pl])
            if len(pl) >= 2:
                polylines.append(pl)
    for s in skipped:
        warnings.warn("扱えない要素を飛ばしました " + s, stacklevel=2)
    return SvgDrawing(polylines, width_px * PX_TO_MM, height_px * PX_TO_MM, skipped)


def transform(polylines: list[Polyline], scale: float = 1.0, rotate: float = 0.0,
              dx: float = 0.0, dy: float = 0.0, mirror: bool = False) -> list[Polyline]:
    """原点を中心に拡大・回転（度、反時計回り）し、平行移動する。mirror は x を反転。"""
    c, s = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
    out = []
    for pl in polylines:
        q = []
        for x, y in pl:
            if mirror:
                x = -x
            x, y = x * scale, y * scale
            q.append((c * x - s * y + dx, s * x + c * y + dy))
        out.append(q)
    return out
