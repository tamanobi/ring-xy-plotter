"""fit_dots.py の forward を、scipy/PIL を読み込まずに取り出すための写し。"""
import numpy as np


def fit_forward(p, s, al, be, ellipse=False):
    R, d, e, th, tx, ty = p[:6]
    ra = rb = R
    if ellipse:
        c2, s2 = p[6:8]
        ra = R + c2 * np.cos(2 * al) + s2 * np.sin(2 * al)
        rb = R + c2 * np.cos(2 * be) + s2 * np.sin(2 * be)
    a, b = s * al + th, s * be + th
    ax, ay = ra * np.cos(a), ra * np.sin(a)
    ux, uy = rb * np.cos(b) - ax, rb * np.sin(b) - ay
    c = np.hypot(ux, uy)
    ux, uy = ux / c, uy / c
    return np.c_[ax + d * ux - e * s * uy + tx, ay + d * uy + e * s * ux + ty]
