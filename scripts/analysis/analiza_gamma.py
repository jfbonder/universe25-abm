"""Análisis de gamma_v3.py: Gamma con y sin dilución, O10/O10b sobre adultos >= a_w y fundadores, barrido
gamma x s_base, estructura generacional de los brazos de robustez, y la serie para la Fig. 2c.
Lee los Parquet de gamma_v3.py (escalares y, al lado, <nombre>_series.parquet).
Uso: python3 scripts/analysis/analiza_gamma.py \
         results/fisico_v3/gamma_v3.parquet [results/fisico_v3/fig2c_series.csv]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from gamma_v3 import read_results

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
D, SER = read_results(sys.argv[1])
for k, nm in (("ad", "t_ad"), ("f0", "t_f0"), ("aw", "t_aw")):
    D[f"tdet_{k}"] = D[nm] - D.t_on
    D[f"G_{k}"] = D[f"tdet_{k}"] / D.T_gen
rng = np.random.default_rng(0)


def boot_ratio(g, num, den, B=2000):
    """Gamma como cociente de medianas (t_det mediano / T_gen mediano) con IC bootstrap sobre corridas."""
    a, b = g[num].to_numpy(float), g[den].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 5:
        return np.nan, np.nan, np.nan
    est = np.median(a) / np.median(b)
    bs = []
    for _ in range(B):
        i = rng.integers(0, a.size, a.size)
        bs.append(np.median(a[i]) / np.median(b[i]))
    return est, *np.percentile(bs, [2.5, 97.5])


def gamma_table(sub):
    rows = []
    for lab, g in sub.groupby("label", sort=False):
        row = dict(label=lab, n=len(g))
        for c in ("t_on", "t_ad", "t_f0", "t_aw", "T_gen", "T_gen_mat", "s_f0_at_tad", "frac_f0_ad_at_tad",
                  "gB", "F1", "Rnet0", "Rnet1", "rho_pk"):
            row[c] = g[c].median()
        for k in ("ad", "f0", "aw"):
            e, lo, hi = boot_ratio(g, f"tdet_{k}", "T_gen")
            row[f"G_{k}"] = e
            row[f"G_{k}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        row["G_f0_runmed"] = g.G_f0.median()
        row["O17p"] = (g.O17p == "PASS").mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


def frac(g, c):
    return (g[c] == "PASS").mean()


print("=== Gamma: ensamble de referencia (réplicas exactas de f2b_base) ===")
print(gamma_table(D[D.block == "base"]).round(3).T.to_string())
print("\n=== Gamma: barrido gamma x s_base (C1, 20 corridas por punto) ===")
print(gamma_table(D[D.block == "barrido"]).round(3).T.to_string())
print("\n=== Gamma: brazos de robustez (réplicas exactas, 20 por brazo) ===")
print(gamma_table(D[D.block == "rob"]).round(3).T.to_string())
print("\n=== Gamma: nulos ===")
print(gamma_table(D[D.block == "nulos"]).round(3).T.to_string())

print("\n=== O10 / O10b: adultos >= a_min (v2), >= a_w = 12 y fundadores (fracción PASS) ===")
rows = []
for lab, g in D.groupby("label", sort=False):
    rows.append(dict(label=lab, block=g.block.iloc[0], n=len(g),
                     O10=frac(g, "O10"), O10_aw=frac(g, "O10_aw"), O10_f0=frac(g, "O10_f0"),
                     O10b=frac(g, "O10b"), O10b_aw=frac(g, "O10b_aw"), O10b_f0=frac(g, "O10b_f0"),
                     O10b_aw_marg=(g.O10b_aw == "MARG").mean(),
                     r10b=(g.t_ad / g.t_pk).median(), r10b_aw=(g.t_aw / g.t_pk).median(),
                     r10b_f0=(g.t_f0 / g.t_pk).median(), t_pk=g.t_pk.median()))
print(pd.DataFrame(rows).set_index("label").round(3).to_string())

print("\n=== Estructura generacional (medianas) y O5 (nu5, Delta5) ===")
rows = []
for lab, g in D.groupby("label", sort=False):
    rows.append(dict(label=lab, n=len(g), F1=100 * g.F1.median(), F2=100 * g.F2.median(), F3p=100 * g.F3p.median(),
                     gB=g.gB.median(), Rnet0=g.Rnet0.median(), Rnet1=g.Rnet1.median(), t_ad=g.t_ad.median(),
                     t_f0=g.t_f0.median(), plateau=g.plateau.median(), O17p=frac(g, "O17p"),
                     O5P=(g.O5 == "PASS").sum(), O5M=(g.O5 == "MARG").sum(), O5F=(g.O5 == "FAIL").sum(),
                     nu5=g.nu5.median(), nu5_min=g.nu5.min(), d5=g.d5.median(),
                     d5_marg_min=g.d5[g.O5 == "MARG"].min(), d5_marg_max=g.d5[g.O5 == "MARG"].max(),
                     tpk1=g.t_pk.median(), tB99=g.t_B99.median()))
print(pd.DataFrame(rows).set_index("label").round(3).to_string())
print("\nmáx |Sad registro - S_adult serie|:", D.sad_align_maxdiff.max())

# serie para la Fig. 2c (C1, réplicas exactas de f2b_base punto 0)
ser = [v for (b, pt, _), v in SER.items() if b == "base" and pt == 0]
if ser and len(sys.argv) > 2:
    tmax = int(min(s["t"].max() for s in ser))
    out = {"t": np.arange(0, tmax + 1)}
    for c in ("s_f0", "s_ad", "s_aw", "frac_f0_ad"):
        M = np.full((len(ser), tmax + 1), np.nan)
        for i, s in enumerate(ser):
            tt = s["t"].astype(int)
            k = tt <= tmax
            M[i, tt[k]] = s[c][k]
        for q, nm in ((50, "med"), (10, "q10"), (90, "q90")):
            out[f"{c}_{nm}"] = np.nanpercentile(M, q, axis=0)
    F = pd.DataFrame(out).round(4)
    F.to_csv(sys.argv[2], index=False)
    print("serie Fig. 2c:", sys.argv[2], len(F), "semanas,", len(ser), "corridas")
