#!/usr/bin/env python3
"""measure_hatch_patches.py の CSV に力モデルを当てはめ、係数を JSON に出す。

モデル（--model）:
  rod_axial_sin2（既定）: δ = (a·sin²(Δ/2)·(v·u) + b·sign ċ)·u
  rod_axial_sin2_sqrt:    δ = (a·sin²(Δ/2)·sign(v·u)·√|v·u| + b·sign ċ)·u（10 月中旬。小さい v·u で早く立ち上がる実測に合う。
                          v·u = 0 の傾きは v·u/√(|v·u| + 0.05) で丸める）
  rod_axial:              δ = (a·(v·u) + b·sign ċ)·u（10 月中旬の最初の形）
a: A 本体の半径方向の動きが棒に沿って出る項。Δ は 2 台のなす角で、向かい合うほど大きく効く
b: 弦の伸縮の符号で決まる項（スリーブの摩擦）。この機械では sign ċ = sign(v·n) なので、
   棒と直交する力が棒方向に漏れる項とは区別できない（補正には区別は要らない）
λ: 折り返し後にずれが育つ距離定数（パッチごとの値の中央値）

速度の係数（--speed-factor）: 目標値に f(v) = 1 − c·max(0, 1 − v/v_ref) を掛け、c も当てはめる。
  v は各パッチの巡航速度（mm/s）で、描いたときの --speed と --fmax（と GRBL の軸ごとの上限 --max-rate）から
  出す。F の上限で遅くなる中心付近のパッチの残差がこれで消える。線の両端（加減速区間）の補正にも同じ c を使う。
  紙ごとに speed や fmax が違うなら、別々に当てはめること（指定は全 CSV に共通）。

CSV は複数まとめて当てはめられる。補正を掛けて描いた紙の CSV は `CSV=掛けた係数.json` と書くと、
その補正量を足し戻して真のずれとして使う。

  python scripts/fit_deflection.py scan_patches.csv --params scripts/fit_params.json -o deflect_params.json
  python scripts/fit_deflection.py a_patches.csv b_patches.csv comp_patches.csv=deflect_params.json -o all.json
  python scripts/fit_deflection.py a.csv b.csv --model rod_axial_sin2_sqrt --speed-factor --speed 30 --fmax 2000 -o v.json
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ring_plotter.deflection import MODELS, DeflectionModel, cruise_speed, fit_model, predict_D, rod_geometry  # noqa: E402
from ring_plotter.kinematics import Kinematics  # noqa: E402


def load_rows(spec: str) -> list[dict]:
    path, _, applied = spec.partition("=")
    model = DeflectionModel.load(applied) if applied else None
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        row = {k: (float(v) if k != "note" and v != "" else (math.nan if k != "note" else v)) for k, v in r.items()}
        row["ok"] = int(row["ok"]) if not math.isnan(row["ok"]) else 0
        row["applied"] = model
        row["sheet"] = os.path.basename(path)
        rows.append(row)
    return rows


def add_speed(kin: Kinematics, rows: list[dict], speed: float, fmax: float, max_rate: float) -> None:
    """各行にパッチの巡航速度 v（mm/s）を入れる。"""
    for r in rows:
        if r["ok"] and not math.isnan(r.get("x", math.nan)):
            th = math.radians(r["theta"])
            try:
                r["v"] = cruise_speed(kin, r["x"], r["y"], math.cos(th), math.sin(th), speed, fmax, max_rate)
            except Exception:
                r["v"] = math.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("csv", nargs="+", help="measure_hatch_patches.py の出力。補正入りなら CSV=係数.json")
    ap.add_argument("--params", help="fit_dots.py の fit_params.json（R, d, e）")
    ap.add_argument("--model", choices=MODELS, default="rod_axial_sin2")
    ap.add_argument("--lam", type=float, default=None,
                    help="距離定数（mm）を指定する。省略時は補正なしで描いた紙のパッチごとの値の中央値")
    ap.add_argument("--speed-factor", action="store_true",
                    help="速度の係数 c（f(v) = 1 − c·max(0, 1 − v/v_ref)）も当てはめる。--speed と --fmax は描いたときの値")
    ap.add_argument("--speed", type=float, default=30.0, help="描いたときのペン先の速度（mm/s）")
    ap.add_argument("--fmax", type=float, default=2250.0, help="描いたときの F の上限（度/分）")
    ap.add_argument("--max-rate", type=float, default=2000.0, help="GRBL の $110/$111（度/分、軸ごと）")
    ap.add_argument("--v-ref", type=float, default=30.0, help="速度の係数の基準速度 v_ref（mm/s）")
    ap.add_argument("-o", "--output", default="deflect_params.json")
    args = ap.parse_args(argv)

    prm = json.load(open(args.params, encoding="utf-8")) if args.params else None
    kin = Kinematics.from_params(prm, dmin=89, dmax=179)
    rows = [r for spec in args.csv for r in load_rows(spec)]
    add_speed(kin, rows, args.speed, args.fmax, args.max_rate)
    fit = fit_model(kin, rows, args.model, speed_factor=args.speed_factor, v_ref=args.v_ref)
    if args.lam is not None:
        fit["lam"] = args.lam
    if math.isnan(fit["lam"]):
        # 補正入りの紙だけなら、掛けた係数の λ を引き継ぐ
        applied = [r["applied"] for r in rows if r.get("applied") is not None]
        if not applied:
            print("λ が決まりません（補正なしで描いた紙の CSV を含めるか、--lam を指定してください）")
            return 1
        fit["lam"] = applied[0].lam
    used = [r for r, u in zip(rows, fit["used"]) if u]

    print(f"{args.model}: パッチ {fit['n']} 個でフィット（融合して除外 {fit['n_saturated']} 個）。"
          f"a = {fit['a']:+.3f} mm、b = {fit['b']:+.3f} mm、λ = {fit['lam']:.1f} mm"
          + (f"、速度の係数 c = {fit['c_speed']:.3f}（v_ref {args.v_ref:g} mm/s、"
             f"speed {args.speed:g} fmax {args.fmax:g}）" if args.speed_factor else ""))
    print(f"残差 rms {fit['rms']:.4f} mm（1 点抜き {fit['rms_loo']:.4f}、ずれ 0 とした場合 {fit['rms_zero']:.3f}）")
    if args.speed_factor:
        fit0 = fit_model(kin, rows, args.model)
        print(f"速度の係数なしなら a = {fit0['a']:+.3f}、b = {fit0['b']:+.3f}、残差 rms {fit0['rms']:.4f}")
    multi = len(args.csv) > 1
    print(f"{'sheet ' if multi else ''}{'site':>12s} {'theta':>6s} {'v.u':>6s} {'v.n':>6s} {'cdot':>4s} {'v':>5s} "
          f"{'f(v)':>5s} {'D':>7s} {'model':>7s} {'resid':>7s} {'lam':>5s}")
    for r, D, p, e, v, fv in zip(used, fit["D"], fit["pred"], fit["resid"], fit["v"], fit["factor"]):
        g = rod_geometry(kin, r["x"], r["y"], math.cos(math.radians(r["theta"])), math.sin(math.radians(r["theta"])))
        tag = f"{r['sheet'][:5]:5s} " if multi else ""
        print(f"{tag}({r['site_x']:+4.0f},{r['site_y']:+4.0f}) {r['theta_nom']:6.0f} {g.vu:+6.2f} {g.vn:+6.2f} "
              f"{'+' if g.sign_cdot > 0 else '-':>4s} {v:5.1f} {fv:5.2f} {D:+7.3f} {p:+7.3f} {e:+7.3f} {r['lam']:5.1f}")
    for r, u in zip(rows, fit["used"]):
        if not u and r["ok"] and not math.isnan(r["D"]):
            p = predict_D(kin, r["x"], r["y"], r["theta"], fit["a"], fit["b"], args.model)
            print(f"({r['site_x']:+4.0f},{r['site_y']:+4.0f}) {r['theta_nom']:6.0f} {'':30s} {r['D']:+7.3f} {p:+7.3f}"
                  f"   融合（線が重なって測れない。--pitch を大きくして測り直す）")

    out = dict(model=args.model, a=fit["a"], b=fit["b"], lam=fit["lam"], c_speed=fit["c_speed"], v_ref=args.v_ref,
               rms=fit["rms"], rms_loo=fit["rms_loo"], n=fit["n"],
               source=" ".join(os.path.basename(s) for s in args.csv),
               fit_speed=args.speed, fit_fmax=args.fmax, fit_max_rate=args.max_rate,
               note="delta = f(v)*(a*f*g(v.u) + b*sign(cdot))*u [mm], f = sin^2(delta/2) (1 for rod_axial), "
                    "g(x) = x, or x/sqrt(|x|+0.05) for rod_axial_sin2_sqrt; f(v) = 1 - c_speed*max(0, 1 - v/v_ref), "
                    "v = pen speed [mm/s]; lag exp(-s/lam); D = delta.m (left of travel)")
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=1)
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
