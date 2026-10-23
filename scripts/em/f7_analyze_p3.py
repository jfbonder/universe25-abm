"""P3 del ecólogo (E4): separar theta* de nu*. Tabla de supervivencia y T_ext por manipulación.
Entrada: results_em/juvenile_protection/*/seed*.npz (f7_sweep.py p3). Salida: results_em/juvenile_protection/tabla_p3.csv
Uso: python3 scripts/em/f7_analyze_p3.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT
from f7_aggregate import wilson

BASE = os.path.join(ROOT, "results_em", "juvenile_protection")


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(BASE, "C1_*"))):
        cfg = os.path.basename(d)
        _, sp, kind, t1, w, L = cfg.split("_")
        fs = [f for f in glob.glob(os.path.join(d, "seed*.npz")) if not f.endswith(".tmp.npz")]
        if not fs:
            continue
        ext = np.array([bool(np.load(f)["extinct"]) for f in fs], bool)
        te = np.array([float(np.load(f)["t_ext"]) for f in fs])
        n = len(fs); k = int((~ext).sum())
        lo, hi = wilson(k, n)
        rows.append(dict(space=sp, kind=kind, t1=int(t1[1:]), w=int(w[1:]), n=n, P_surv=k / n, lo=lo, hi=hi,
                         Text_med=float(np.median(te[ext])) if ext.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "tabla_p3.csv"), index=False)
    for (sp, kind), g in df.groupby(["space", "kind"]):
        print(sp, kind)
        print(g.pivot_table(index="t1", columns="w", values="P_surv").round(2).to_string())
        print(g.pivot_table(index="t1", columns="w", values="Text_med").round(0).to_string())


if __name__ == "__main__":
    main()
