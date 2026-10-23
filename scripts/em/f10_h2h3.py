"""Fase 10a, punto 1: descomposición de la caída de la natalidad (crecimiento -> meseta) en concepción y supervivencia
de las crías (H2), y fracción de hembras que nunca conciben (H3), con IC95 bootstrap sobre corridas.
    python3 scripts/em/f10_h2h3.py
Entradas: results_em/robustness_checks/{rev1000,c1x,revages}_litters.csv, *_cohorts.csv (f10_reversible.py).
Salida: results_em/robustness_checks/h2h3.json
births/(hembra repr. y semana) = conc_rate * Lbar * phi  =>  ln(B_g/B_p) = ln(C_g/C_p) + ln(phi_g/phi_p) (exacto).
"""
import json, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results_em", "robustness_checks")
rng = np.random.default_rng(11)

def ci(x, B=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    m = rng.choice(x, (B, x.size)).mean(1)
    return [round(float(x.mean()), 4), round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4), int(x.size)]

res = {}
for nm in ("rev1000", "c1x", "revages"):
    L = pd.read_csv(os.path.join(OUT, f"{nm}_litters.csv")); C = pd.read_csv(os.path.join(OUT, f"{nm}_cohorts.csv"))
    lc = np.log(L.conc_rate_grow / L.conc_rate_plat); lp = np.log(L.phi_exact_grow / L.phi_exact_plat)
    d = dict(conc_drop_factor=ci(L.conc_rate_grow / L.conc_rate_plat), phi_drop_factor=ci(L.phi_exact_grow / L.phi_exact_plat),
             share_conception=ci(lc / (lc + lp)), share_survival=ci(lp / (lc + lp)))
    for w in ("grow", "plat", "decl", "late"):
        if f"fail_exact_{w}" in L:
            d[f"fail_{w}"] = ci(L[f"fail_exact_{w}"]); d[f"phi_{w}"] = ci(L[f"phi_exact_{w}"]); d[f"conc_{w}"] = ci(L[f"conc_rate_{w}"])
    for w in ("boom", "o11b", "plat", "decl", "late"):
        d[f"never_{w}"] = ci(C[f"never_{w}"])
        d[f"nborn_{w}_total"] = int(C[f"nborn_{w}"].sum())
    res[nm] = d
    print(nm, json.dumps(d))
json.dump(res, open(os.path.join(OUT, "h2h3.json"), "w"), indent=1)
