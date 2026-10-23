"""Fase 10a (revisión EM, ronda 2): re-corridas con semillas declaradas y reproducción de los chequeos de la revisión.

    python3 scripts/em/f10_runs.py GRUPO

Grupos:
  rev1000   nulo reversible (punto 0 de results_em/spatial_reversible_null/params.json, eps_ad = 0.8, lambda = 0.9)
            a T = 1000, n = 200, semillas SeedSequence([20261018, 0, rep]) = las de f9_sp / f9_sp1000. Es una
            RE-GENERACIÓN bit a bit (se verifica contra f9_sp1000/summary.csv), no un ensamble nuevo: f9_sp1000
            no guardó series. Además registra, sin consumir números aleatorios, cada camada (crías nacidas y
            supervivientes) -> results_em/reversible_null_T1000/
  c1base    lo mismo para C1 espacial de producción (results_v2/f2b_base punto 0, semillas [25301, 0, rep]),
            T = 600, n = 200: comparación de las camadas fallidas -> results_em/c1_litters/
  refr      C1 de producción con refractory_weeks = 3, n = 60, T = 600, semillas de f2b_base ([25301, 0, rep];
            pareado con las primeras 60 corridas de C1) -> results_em/refractory_check/
  refr777   el mismo con la semilla 777 (base_seed) -> results_em/refractory_check_seed777/
  revages   nulo reversible con edades iniciales U{4..80}, T = 1000, n = 60, semilla 777
            -> results_em/reversible_ages_seed777/ ; revages_prod: semilla [20261018, 0, rep]
            -> results_em/reversible_ages/
Todos con el evaluador primario (results_v2/f2b_base/eval_kw.json) y el runner U2v2 (f2b_model_v2).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "scripts" / "production"), str(ROOT / "scripts" / "analysis"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from u25.params import Params, preset    # noqa: E402
from u25.ensemble import rep_seed        # noqa: E402
from f2b_model_v2 import U2v2            # noqa: E402

RES = ROOT / "results_em"
EVAL_KW = json.load(open(ROOT / "results_v2" / "f2b_base" / "eval_kw.json"))
KEEP = ["t", "N", "N_f", "births", "deaths", "conceptions", "pups", "S_adult", "N_repro_f", "N_adult", "mean_age",
        "occ_max", "occ_var", "mstar", "f_over", "phi_w", "coh_f_born", "coh_f_ever_conceived"]


class _RecRNG:
    """Proxy del Generator: delega todo y registra las llamadas a binomial (única llamada: supervivencia de las
    crías de cada camada, model.py l. 857). No altera la secuencia aleatoria."""

    def __init__(self, rng, log):
        self._rng, self._log = rng, log

    def binomial(self, n, p, size=None):
        out = self._rng.binomial(n, p, size)
        self._log.append((np.asarray(n).copy(), np.asarray(out).copy()))
        return out

    def __getattr__(self, k):
        return getattr(self._rng, k)


class U10(U2v2):
    def __init__(self, params=None, rng=None):
        self._lit_log = []
        super().__init__(params, rng=_RecRNG(rng, self._lit_log))
        self.lit_rows = []

    def step(self):
        self._lit_log.clear()
        super().step()
        nl = nf = nl_all_born = 0
        hist = np.zeros(11, np.int64)
        for L, S in self._lit_log:
            nl += L.size
            nf += int((S == 0).sum())
            hist += np.bincount(np.minimum(S, 10), minlength=11)
        self.lit_rows.append((self.t, nl, nf, *hist.tolist()))


def _one(task):
    pdict, point, rep, T, base_seed = task
    from u25.objectives import evaluate_run
    p = Params.from_dict(pdict)
    U = U10(p, rng=np.random.default_rng(rep_seed(base_seed, point, rep)))
    series, summ = U.run(T)
    res = evaluate_run(series, p, summ, **EVAL_KW)
    row = {"point": point, "rep": rep, "base_seed": base_seed,
           **{k: v for k, v in summ.items() if not isinstance(v, (list, dict))}}
    for k, v in res.items():
        if k != "_times":
            row["obj_O17_run" if k == "calhoun_like" else f"obj_{k}"] = getattr(v, "status", v)
    lit = pd.DataFrame(U.lit_rows, columns=["t", "litters", "litters_failed"] + [f"surv{k}" for k in range(11)])
    S = series[[c for c in KEEP if c in series.columns]].merge(lit, on="t", how="left").assign(point=point, rep=rep)
    return row, S


def run(name, pdict, T, base_seed, reps, point=0, n_jobs=4):
    o = RES / name
    (o / "series").mkdir(parents=True, exist_ok=True)
    tasks = [(pdict, point, r, T, base_seed) for r in range(reps)]
    rows = []
    with Pool(n_jobs) as pool:
        for row, S in pool.imap_unordered(_one, tasks):
            S.to_parquet(o / "series" / f"p{point}_r{row['rep']}.parquet", index=False)
            rows.append(row)
            print(name, row["rep"], row.get("N_max"), row.get("t_end"), flush=True)
    pd.DataFrame(rows).sort_values("rep").to_csv(o / "summary.csv", index=False)
    json.dump(dict(T=T, base_seed=base_seed, reps=reps, point=point, params=pdict, eval_kw=EVAL_KW,
                   runner="scripts/em/f10_runs.py (U2v2 + registro de camadas)"),
              open(o / "meta.json", "w"), indent=1)


def main(g):
    c1 = preset("calhoun_C1")
    base = json.load(open(ROOT / "results_v2" / "f2b_base" / "params.json"))[0]
    rev = json.load(open(RES / "spatial_reversible_null" / "params.json"))[0]
    if g == "rev1000":
        run("reversible_null_T1000", dict(rev, T=1000), 1000, 20261018, 200)
    elif g == "c1base":
        run("c1_litters", dict(base, T=600), 600, 25301, 200)
    elif g == "refr":
        run("refractory_check", dict(base, T=600, refractory_weeks=3), 600, 25301, 60)
    elif g == "refr777":
        run("refractory_check_seed777", c1.replace(refractory_weeks=3, T=600).to_dict(), 600, 777, 60)
    elif g == "revages":
        pr = c1.replace(plast_outside_rec=0.8, social_rec_lambda=0.9, initial_age_dist="uniform",
                        initial_age_hi=80, T=1000).to_dict()
        run("reversible_ages_seed777", pr, 1000, 777, 60)
    elif g == "revages_prod":
        pr = c1.replace(plast_outside_rec=0.8, social_rec_lambda=0.9, initial_age_dist="uniform",
                        initial_age_hi=80, T=1000).to_dict()
        run("reversible_ages", pr, 1000, 20261018, 60)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1])
