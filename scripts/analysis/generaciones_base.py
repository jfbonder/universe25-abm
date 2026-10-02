"""Tabla por generaciones sobre el ensamble de referencia de la V2: re-simula las réplicas de results_v2/f2b_base (mismos Params y
semillas que u25.sweep) con el registro generacional de generaciones.GenU, y verifica que la trayectoria coincide
con el summary.csv (N_max, t_ext). n_jobs = 2.
Uso: nice -n 10 python3 scripts/analysis/generaciones_base.py OUT.pkl PUNTOS NREPS
     ej.: ... /tmp/gen_base.pkl 0,1,2 200"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from multiprocessing import Pool
from u25.params import Params
from u25.sweep import rep_seed
from generaciones import GenU

BASE = "results_v2/f2b_base"


def one(a):
    point, rep, pdict, base_seed = a
    p = Params.from_dict(pdict)
    U = GenU(p, rng=np.random.default_rng(rep_seed(base_seed, point, rep, False)))
    S, summ = U.run(600)
    n = U.next_id
    reg = {k: getattr(U, "r_" + k)[:n].copy() for k in ("gen", "male", "birth", "sbirth", "s6", "s12", "s26", "pups", "daught")}
    return dict(tag=f"base p{point}", point=point, seed=rep, log=U.log, reg=reg,
                N_max=int(S.N.max()), t_end=int(S.t.iloc[-1]))


if __name__ == "__main__":
    out, pts, nrep = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
    meta = json.load(open(f"{BASE}/meta.json"))
    plist = json.load(open(f"{BASE}/params.json"))
    jobs = [(pt, r, plist[pt], meta["base_seed"]) for pt in pts for r in range(nrep)]
    with Pool(2) as P:
        res = P.map(one, jobs, chunksize=2)
    summ = pd.read_csv(f"{BASE}/summary.csv", keep_default_na=False)
    chk = summ.set_index(["point", "rep"])
    bad = [(r["point"], r["seed"]) for r in res
           if int(chk.loc[(r["point"], r["seed"]), "N_max"]) != r["N_max"]]
    print("réplicas:", len(res), " con N_max distinto del summary:", len(bad), bad[:5])
    with open(out, "wb") as f:
        pickle.dump(res, f)
