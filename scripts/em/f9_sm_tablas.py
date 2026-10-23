"""Fase 9a (A9): tablas del SM de EM con P_ext(eps_ad), umbral eps^c(L), apareamiento y cohortes, a partir de
datos ya archivados. No corre simulaciones. Escribe results_em/sm_tables/tab_sm_*.tex (cohorts, mate y pext son idénticas a manuscript/em/tables/; tab_sm_epsc.tex del manuscrito es posterior: lleva los intervalos de verosimilitud perfilada de results_em/robustness_checks/epsc_profile.csv y se compuso a mano) e imprime los controles.

Fuentes:
  results_em/adult_recovery/summary.csv     (barrido de recuperación adulta; P_ext a 1000 semanas, Wilson)
  results_em/adult_recovery/transicion.csv  (punto medio logístico eps^c y ancho)
  results_em/adult_recovery/escaleo.csv     (pendiente dT_ext/d ln|G| con eps_ad = 0)
  results_em/mate_search/tabla_mate.csv (C1 y SL bien mezclados, tres reglas de apareamiento)
  results_em/kinetic/cohortes_C1.csv, dicot_orbit_C1.csv (órbita cinética de C1)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = lambda *p: os.path.join(ROOT, "results_em", *p)
OUT = os.path.join(ROOT, "results_em", "sm_tables")
os.makedirs(OUT, exist_ok=True)


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(p):
    if p == 1:
        return "1"
    if p == 0:
        return "0"
    return f"{p:.2f}".lstrip("0")


# ---------------------------------------------------------------- 1. P_ext(eps_ad), lambda in {0, 1}
S = pd.read_csv(R("adult_recovery", "summary.csv"))
S = S[(S.model == "C1") & (S.variant == "cell")]
EPS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
# control: los IC archivados son de Wilson
for _, row in S.sample(20, random_state=1).iterrows():
    k = round(row.P_ext_1000 * row.n)
    lo, hi = wilson(k, row.n)
    assert abs(lo - row.P_ext_1000_lo) < 1e-4 and abs(hi - row.P_ext_1000_hi) < 1e-4, row.cfg
lines = []
for space, nm in (("sp", "spatial"), ("wm", "well mixed")):
    for lam in (1.0, 0.0):
        for L in (15, 30, 60):
            g = S[(S.space == space) & (S.lam == lam) & (S.L == L)].set_index("out")
            if g.empty:
                continue
            n = int(g.n.iloc[0])
            cells = []
            for e in EPS:
                if e in g.index:
                    p = g.loc[e, "P_ext_1000"]
                    cells.append(fmt(p))
                else:
                    cells.append("--")
            lines.append(f"{nm} & {lam:g} & {L} & {n} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
lines = lines[:-1]
open(os.path.join(OUT, "tab_sm_pext.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")

# conteos agregados que cita el texto
low = S[(S.out <= 0.02) & (S.L <= 30)]
print("eps<=0.02, L=15,30, celda (sp y wm):", int(low.n_ext.sum()), "/", int(low.n.sum()))
S_all = pd.read_csv(R("adult_recovery", "summary.csv"))
low = S_all[(S_all.out <= 0.02) & (S_all.L <= 30)]
print("eps<=0.02, L=15,30, todas las variantes:", int(low.n_ext.sum()), "/", int(low.n.sum()))
z = S_all[(S_all.out == 0)]
print("eps=0, todos los L y variantes:", int(z.n_ext.sum()), "/", int(z.n.sum()))
z = S_all[(S_all.out == 0) & (S_all.space == "sp")]
print("eps=0, espacial, L=15-60:", int(z.n_ext.sum()), "/", int(z.n.sum()))

# ---------------------------------------------------------------- 2. eps^c(L)
T = pd.read_csv(R("adult_recovery", "transicion.csv"))
rows = [("sp", "cell", "spatial", (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)),
        ("wm", "cell", "well mixed, cell", (0.0, 0.5, 1.0)),
        ("wm", "moore", r"well mixed, $3\times3$ block", (0.0, 0.5, 1.0)),
        ("wm", "global", "well mixed, global", (0.0, 0.5, 1.0))]
lines = []
for sp, var, nm, lams in rows:
    for lam in lams:
        g = T[(T.space == sp) & (T.variant == var) & (T.lam == lam)].set_index("L")
        cells = []
        for L in (15, 30, 60):
            if L not in g.index:
                cells.append("n.s.")
                continue
            c, w = g.loc[L, "out_c"], g.loc[L, "width"]
            if not np.isfinite(c) or c > 1:
                cells.append(r"$>1$")
            else:
                cells.append(f"{c:.2f} ({w:.3f})")
        lines.append(f"{nm} & {lam:g} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
lines = lines[:-1]
open(os.path.join(OUT, "tab_sm_epsc.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- 3. apareamiento
M = pd.read_csv(R("mate_search", "tabla_mate.csv"))
lines = []
MN = {"cell": "cell", "moore": r"$3\times3$ block", "global": "global"}
for model in ("C1", "SL"):
    for mate in ("cell", "moore", "global"):
        g = M[(M.model == model) & (M.mate == mate)].sort_values("L")
        Ls = g.L.tolist()
        k, n = int(g.n_ext.sum()), int(g.n.sum())
        lo, hi = wilson(k, n)
        per = ", ".join(f"{int(a)}/{int(b)}" for a, b in zip(g.n_ext, g.n))
        tm = g.Text_med.dropna()
        trange = f"{tm.min():g}--{tm.max():g}" if len(tm) else "--"
        p300 = ", ".join(fmt(x) for x in g.P_ext_300)
        mdl = r"\textsf{C1}" if model == "C1" else r"\textsf{SL}"
        agg = f"{k}/{n} [{lo:.2f}, {hi:.2f}]" if k == n else "--"
        lines.append(f"{mdl} & {MN[mate]} & {Ls[0]}--{Ls[-1]} & {per} & {agg} & {p300} & {trange}" + r" \\")
        if model == "C1":
            print("mate", model, mate, k, n, round(lo, 3))
    lines.append(r"\addlinespace")
lines = lines[:-1]
open(os.path.join(OUT, "tab_sm_mate.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- 4. cohortes en la órbita cinética
C = pd.read_csv(R("kinetic", "cohortes_C1.csv")).set_index("b")
D = pd.read_csv(R("kinetic", "dicot_orbit_C1.csv"))
nustar = 6.9210
above = D[D.nu_pre > nustar].t
print("nu_pre > nu*: semanas", int(above.min()), "-", int(above.max()))
lines = []
for b in (3, 12, 16, 19, 22):
    r = C.loc[b]
    lines.append(f"{b} & {r.ratio:.3g} & {r.theta_bar:.2f} & {r.kappa_mf:.2f} & "
                 + ("yes" if r.theta_bar >= 2 / 3 else "no") + r" \\")
    print("cohorte", b, round(r.ratio, 4), round(r.theta_bar, 3), round(r.kappa_mf, 3))
open(os.path.join(OUT, "tab_sm_cohorts.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- 5. pendientes de T_ext con eps_ad = 0
E = pd.read_csv(R("adult_recovery", "escaleo.csv"))
for sp, var in (("sp", "cell"), ("wm", "cell"), ("wm", "moore"), ("wm", "global")):
    g = E[(E.space == sp) & (E.variant == var) & (E.out == 0)]
    print("pendiente b, eps=0", sp, var, dict(zip(g.lam, g.b_lnG.round(2))))
print("escrito en", OUT)
