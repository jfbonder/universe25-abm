"""(c) de dicotomia.md §7: competencia de cohortes en la órbita de C1-trunc. Para la cohorte nacida en b:
m_b = s media de las madres (= S_a1(b+1)/w), c_b = S de la cohorte al salir de la ventana (S_a13(b+13)),
cociente c_b/m_b frente a κ_eff = w + r a_w θ̄ (θ̄ = media de θ en b+1..b+12) y frente a
w + r Σ_u S_u (1 - e^{-9ν_u}) / m_b (ganancia de campo medio, sin daño). Guarda results_em/kinetic/cohortes_C1.csv"""
import os
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K = os.path.join(ROOT, "results_em", "kinetic")
d = pd.read_csv(os.path.join(K, "dicot_orbit_C1.csv")).set_index("t")
w, r, aw = 0.2, 0.1, 12
rows = []
for b in range(3, 60):
    if b + 13 not in d.index or not np.isfinite(d.loc[b + 1, "S_a1"]) or d.loc[b, "births"] < 1e-14:
        continue
    m = d.loc[b + 1, "S_a1"] / w
    c = d.loc[b + 13, "S_a13"]
    win = d.loc[b + 1: b + 12]
    th = win.theta.mean()
    gain = r * np.sum(win.S * (1 - np.exp(-9 * win.nu_pre)))
    rows.append(dict(b=b, births=d.loc[b, "births"], m_madres=m, c_cohorte=c, ratio=c / m if m > 0 else np.nan,
                     theta_bar=th, kappa_eff=w + r * aw * th, kappa_mf=w + gain / m if m > 0 else np.nan))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(K, "cohortes_C1.csv"), index=False)
pd.set_option("display.width", 200)
print(t.round(4).to_string(index=False))
