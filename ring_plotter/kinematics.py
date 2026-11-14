"""リング式プロッタの運動学。

座標系（gen_shapes.py と同じ）:
  - リングの中心が原点、単位は mm
  - α=0° のときのユニットAの方向が下（−y）
  - X = ユニットAの角度 α、Y = ユニットBの角度 β（どちらも度）、Δ = β − α

モデル（fit_dots.py と同じ）:
  Aのピン = R·(cos α', sin α')、Bのピン = R·(cos β', sin β')   （α' = α − 90°）
  ペン先 = Aのピン + d·u + e·n   （u: A→Bの単位ベクトル、n: uに垂直）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 10 月上旬のフィッティング結果（ミリペン、ストッパー原点 Y88）
DEFAULT_PARAMS = {"R": 230.778, "d": 229.190, "e": 6.428}


class Unreachable(ValueError):
    """ペン先が届かない点。"""


@dataclass(frozen=True)
class Reach:
    r_min: float  # 中心の描けない穴の半径（Δの範囲内で届く最小半径）
    r_max: float  # 届く最大半径（Δ=dmin のとき）


class Kinematics:
    def __init__(self, R: float, d: float, e: float, dmin: float = 88.0, dmax: float = 180.0):
        if not 0 < dmin < dmax <= 180.0:
            raise ValueError("0 < dmin < dmax <= 180 である必要があります")
        self.R, self.d, self.e = R, d, e
        self.L = math.hypot(d, e)  # Aのピンからペン先までの距離
        self.gamma = math.atan2(e, d)  # A→ペンの方向と棒の方向の差
        self.dmin, self.dmax = dmin, dmax

    @classmethod
    def from_params(cls, params: dict | None = None, **kw) -> "Kinematics":
        prm = dict(DEFAULT_PARAMS)
        if params:
            prm.update({k: float(v) for k, v in params.items() if k in prm})
        return cls(prm["R"], prm["d"], prm["e"], **kw)

    # --- 順運動学 ----------------------------------------------------------
    def forward(self, alpha: float, beta: float) -> tuple[float, float]:
        """(α, β)[度] → ペン先 (x, y)[mm]"""
        a, b = math.radians(alpha - 90), math.radians(beta - 90)
        ax, ay = self.R * math.cos(a), self.R * math.sin(a)
        ux, uy = self.R * math.cos(b) - ax, self.R * math.sin(b) - ay
        c = math.hypot(ux, uy)
        if c < 1e-12:
            raise ValueError("α=β ではペン先の位置が定まりません")
        ux, uy = ux / c, uy / c
        return ax + self.d * ux - self.e * uy, ay + self.d * uy + self.e * ux

    # --- 逆運動学 ----------------------------------------------------------
    def inverse(self, x: float, y: float, near_alpha: float | None = None) -> tuple[float, float]:
        """ペン先 (x, y)[mm] → (α, β)[度]。αは 0〜360、β = α + Δ。

        A の候補は 2 つ（鏡像）あり、Δ が範囲に入るものを返す。両方入るときは、near_alpha を
        与えればそれに（360° を法として）近い α のほうを、与えなければ Δ が範囲の端から遠いほう
        （そこから動いても範囲に留まりやすい枝）を返す。
        中心の近く（この機械では半径 9 mm 前後より内側）では両方が有効なので、経路をたどるときは
        直前の α を渡して枝を保つ。渡さないと点ごとに枝が飛び、α が 180° 近く跳ぶ（紙の上では輪になる）。
        届かない点は Unreachable を投げる。
        """
        cands = self.inverse_all(x, y)
        if not cands:
            raise Unreachable(
                f"({x:.1f}, {y:.1f}) はΔの範囲（{self.dmin}〜{self.dmax}°）では描けません"
            )
        if len(cands) == 1:
            return cands[0]
        if near_alpha is None:
            return max(cands, key=lambda ab: min(ab[1] - ab[0] - self.dmin, self.dmax - (ab[1] - ab[0])))
        return min(cands, key=lambda ab: abs((ab[0] - near_alpha + 180) % 360 - 180))

    def inverse_all(self, x: float, y: float) -> list[tuple[float, float]]:
        """ペン先 (x, y) に対する (α, β) の候補すべて（Δ が範囲に入るもの。0〜2 個）。
        穴の中や届かない点は Unreachable を投げる。"""
        R, L = self.R, self.L
        r = math.hypot(x, y)
        if r < abs(R - L) + 1e-6:
            raise Unreachable(f"({x:.1f}, {y:.1f}) は中心の描けない穴の中です")
        cospsi = (R * R + r * r - L * L) / (2 * R * r)
        if abs(cospsi) > 1:
            raise Unreachable(f"({x:.1f}, {y:.1f}) には届きません")
        phi, psi = math.atan2(y, x), math.acos(cospsi)
        out = []
        for sign in (1, -1):
            a = phi + sign * psi
            ax, ay = R * math.cos(a), R * math.sin(a)
            w = math.atan2(y - ay, x - ax) - self.gamma  # 棒の方向
            ux, uy = math.cos(w), math.sin(w)
            t = -2 * (ax * ux + ay * uy)  # Aから棒の方向に進んでリングと交わる距離
            if t <= 0:
                continue
            b = math.atan2(ay + t * uy, ax + t * ux)
            delta = math.degrees(b - a) % 360
            if self.dmin - 1e-9 <= delta <= self.dmax + 1e-9:
                alpha = (math.degrees(a) + 90) % 360
                out.append((alpha, alpha + delta))
        return out

    def reachable(self, x: float, y: float) -> bool:
        try:
            self.inverse(x, y)
            return True
        except Unreachable:
            return False

    # --- 到達範囲 ----------------------------------------------------------
    def radius_at(self, delta: float) -> float:
        """Δ[度] のときのペン先の中心からの距離。"""
        return math.hypot(*self.forward(0.0, delta))

    def reach(self, n: int = 2000) -> Reach:
        """Δの範囲で届く半径の最小・最大（数値で求める）。"""
        rs = [self.radius_at(self.dmin + (self.dmax - self.dmin) * i / n) for i in range(n + 1)]
        return Reach(r_min=min(rs), r_max=max(rs))

    def describe(self, ascii_only: bool = False) -> str:
        """パラメータの要約。ascii_only は G-code のコメント用（GRBL は 0x80 以上のバイトを
        リアルタイムコマンドとして解釈するので、日本語や記号を G-code に入れてはいけない）。"""
        rc = self.reach()
        if ascii_only:
            return (f"R={self.R:.3f} d={self.d:.3f} e={self.e:.3f} mm, gamma={math.degrees(self.gamma):.3f} deg, "
                    f"delta={self.dmin:g}-{self.dmax:g} deg, reach {rc.r_min:.2f}-{rc.r_max:.1f} mm")
        return (f"R={self.R:.3f} d={self.d:.3f} e={self.e:.3f} mm, γ={math.degrees(self.gamma):.3f}°, "
                f"Δ={self.dmin:g}〜{self.dmax:g}°, 届く半径 {rc.r_min:.2f}〜{rc.r_max:.1f} mm")
