"""Fase 9a: análisis de los ensambles para la revisión de EM (A1, A2, A5, C17). No corre simulaciones nuevas salvo
re-evaluaciones desde las series guardadas.

    python3 scripts/em/em_f9_analysis.py [pasos]

Pasos (por defecto todos):
  scores     O11 a la edad a_w y duración de la fase C, puntuados desde las series de cada corrida
             (f2b_base, em_wm canónico, f9_sp, f9_nulos) -> results_em/pattern_tables/series_scores.csv
             (verifica que la re-evaluación reproduce obj_O11 primario corrida por corrida)
  matrix     matriz patrón x estructura, UNA fuente por columna, PASS y MARG -> .../matrix.csv
  ablations  qué patrón falla en cada ablación (f2b_abl1, f2b_abl2; evaluador primario) -> .../ablations.csv
  survival   P_ext y T_ext (KM) de C1, núcleo, SL y nulo reversible: espacio, b.m. celda, b.m. global (L = 30, 45,
             60) -> .../survival.csv
  lhs        fracción de los hipercubos (f2b_lhs, f2b_lhs2; 400 puntos x 10) que reproduce la envolvente del
             colapso frente a O17p -> .../lhs.json
  deposit    series reducidas de las Figs. 2, 3(a-d) y 5(a-d) -> results_em/series_deposit/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "scripts" / "production"), str(ROOT / "scripts" / "analysis"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from u25.params import Params             # noqa: E402

RES, RES2 = ROOT / "results_em", ROOT / "results_v2"
OUT = RES / "pattern_tables"
OUT.mkdir(parents=True, exist_ok=True)
EVAL_KW = json.load(open(RES2 / "f2b_base" / "eval_kw.json"))


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------------------- 1. puntajes desde las series
SERIES_SRC = {  # nombre: (directorio, puntos)
    "C1_sp": (RES / "spatial_c1_rerun", [0]),   # = f2b_base p0 bit a bit, con s a la edad a_w
    "C1_wm_cell": (RES / "wellmixed_c1", [0, 1]),
    "f9_sp": (RES / "spatial_reversible_null", [0, 1]),
    "f9_nulos": (RES / "spatial_nulls_rerun", [0, 3]),
}


def _score_dir(name, d, points):
    from u25.objectives import evaluate_run
    from em_f9_runs import phase_c
    if not (d / "series.parquet").exists():
        print("sin", d / "series.parquet", "(las series crudas no se versionan: correr antes em_f9_runs.py / em_wm_run.py)")
        return []
    P = json.load(open(d / "params.json"))
    summ = pd.read_csv(d / "summary.csv").set_index(["point", "rep"])
    S = pd.read_parquet(d / "series.parquet")
    rows = []
    for (pt, rep), g in S[S.point.isin(points)].groupby(["point", "rep"]):
        p = Params.from_dict(P[pt])
        g = g.sort_values("t").reset_index(drop=True)
        sm = summ.loc[(pt, rep)].to_dict()
        r6 = evaluate_run(g, p, sm, **EVAL_KW)["O11"]
        ra = evaluate_run(g, p, sm, **EVAL_KW, o11_age="aw")["O11"]
        rows.append(dict(src=name, point=pt, rep=rep, O11_re=r6.status, O11aw=ra.status, O11aw_val=ra.value,
                         O11_obj=str(sm.get("obj_O11")), **phase_c(g)))
    return rows


def step_scores():
    rows = []
    for name, (d, pts) in SERIES_SRC.items():
        rows += _score_dir(name, d, pts)
        print("scores", name, len(rows))
    if not rows:
        print("scores: sin series crudas; se conserva series_scores.csv depositado")
        return
    df = pd.DataFrame(rows)
    df["O11_same"] = df.O11_re.fillna("NA").astype(str) == df.O11_obj.replace("nan", "NA").fillna("NA").astype(str)
    df.to_csv(OUT / "series_scores.csv", index=False)
    print("re-evaluación de O11 primario idéntica en", int(df.O11_same.sum()), "de", len(df))


# ----------------------------------------------------------------------------- 2. matriz
ROWS = [  # (código, fila, fase, nota)
    ("O10b", "Socially functional growth", "B", ""),
    ("ceiling", "Peak below $m_c|G|$ animals", "C", ""),
    ("O4", "Peak below capacity, empty nests", "C", ""),
    ("O10", "Adults deteriorate before peak", "C", ""),
    ("O14", "Aggregation with free space", "C", ""),
    ("O13", "Crowded and withdrawn coexist", "C", ""),
    ("O13p", "Withdrawn class persists", "C", "constr"),
    ("O13b", "Withdrawn class born late", "C", ""),
    ("phC", "Gradual stagnation phase", "C", "post"),
    ("O11", "Late cohorts fail (age 6 wk)", "C–D", ""),
    ("O11aw", "Late cohorts fail (age $a_w$)", "C–D", "post"),
    ("O5", "Natality ceases near peak", "D", ""),
    ("O6", "No recovery of natality", "D", ""),
    ("O7", "Extinction", "D", ""),
    ("O8", "Long senescent decline", "D", ""),
    ("O9", "Old female-biased remnant", "D", ""),
    ("O15", "Total infant mortality", "D", ""),
    ("O17p", "All essential patterns", "all", ""),
]
TRUNC_OK = {"O10b", "ceiling", "O4", "O10", "O14"}     # filas válidas en las corridas cortadas por max_pop (gamma=0)
COLS = [  # (clave, grupo, encabezado, fuente)
    ("g0_sp", "space", "$\\gamma=0$†", "f9_nulos p0 (= f2b_nulos p0, bit a bit)"),
    ("core_sp", "space", "core", "f9_nulos p3 (= f2b_nulos p3)"),
    ("SL_sp", "space", "short-\nlived", "f9_sp p1 (parámetros de f2b_ref p0)"),
    ("rev_sp", "space", "C1, adults\nrecover", "f9_sp p0 (eps_ad = 0.8, lambda = 0.9)"),
    ("C1_sp", "space", "C1", "f2b_base p0"),
    ("core_wmc", "well mixed, local mating", "core", "f9_wmcell WM_none (= wellmixed_cell_reference, bit a bit)"),
    ("SL_wmc", "well mixed, local mating", "short-\nlived", "f9_wmcell WM_SL (= wellmixed_cell_reference)"),
    ("C1_wmc", "well mixed, local mating", "C1", "wellmixed_c1 p0 (canónico)"),
    ("core_wmg", "well mixed, global mating", "core", "f9_wmglobal core L30"),
    ("SL_wmg", "well mixed, global mating", "short-\nlived", "f9_wmglobal SL L30"),
    ("C1_wmg", "well mixed, global mating", "C1", "f9_wmglobal C1_ref L30"),
]


def _from_sweep(d, point, scores_src=None, scores_point=None):
    s = pd.read_csv(d / "summary.csv")
    s = s[s.point == point].sort_values("rep").reset_index(drop=True)
    out = pd.DataFrame({"rep": s.rep})
    for code, *_ in ROWS:
        c = f"obj_{code}"
        if c in s:
            out[code] = s[c].fillna("NA").astype(str).to_numpy()
    p = Params.from_dict(json.load(open(d / "params.json"))[point])
    out["ceiling"] = np.where(s.N_max.to_numpy() < p.m_c * p.grid_size ** 2, "PASS", "FAIL")
    if scores_src is not None:
        sc = pd.read_csv(OUT / "series_scores.csv")
        sc = sc[(sc.src == scores_src) & (sc.point == scores_point)].set_index("rep").reindex(out.rep)
        out["O11aw"] = sc.O11aw.fillna("NA").astype(str).to_numpy()
        out["phC"] = np.where(sc.ph_Cfrac.to_numpy() >= 0.10, "PASS", "FAIL")
        out.loc[sc.ph_Cfrac.isna().to_numpy(), "phC"] = "NA"
    out["exploded"] = s.exploded.to_numpy() if "exploded" in s else False
    return out


def _from_perrun(df, L=30):
    out = pd.DataFrame({"rep": df.rep.to_numpy()})
    for code, *_ in ROWS:
        if code in df:
            out[code] = df[code].fillna("NA").astype(str).to_numpy()
    out["ceiling"] = np.where(df.N_max.to_numpy() < 8 * L * L, "PASS", "FAIL") if "N_max" in df else "NA"
    out["phC"] = np.where(df.ph_Cfrac.to_numpy() >= 0.10, "PASS", "FAIL")
    out.loc[df.ph_Cfrac.isna().to_numpy(), "phC"] = "NA"
    return out


def column_runs():
    C = {}
    C["g0_sp"] = _from_sweep(RES / "spatial_nulls_rerun", 0, "f9_nulos", 0)
    C["core_sp"] = _from_sweep(RES / "spatial_nulls_rerun", 3, "f9_nulos", 3)
    C["SL_sp"] = _from_sweep(RES / "spatial_reversible_null", 1, "f9_sp", 1)
    C["rev_sp"] = _from_sweep(RES / "spatial_reversible_null", 0, "f9_sp", 0)
    C["C1_sp"] = _from_sweep(RES2 / "f2b_base", 0, "C1_sp", 0)
    C["C1_wmc"] = _from_sweep(RES / "wellmixed_c1", 0, "C1_wm_cell", 0)
    wc = pd.read_csv(RES / "wellmixed_cell_rerun" / "per_run.csv")
    eco = pd.read_csv(RES / "wellmixed_cell_reference" / "per_run.csv")
    for key, nm in (("core_wmc", "WM_none"), ("SL_wmc", "WM_SL")):
        g = wc[wc.cfg == nm].sort_values("rep").reset_index(drop=True)
        e = eco[eco.cfg == nm].set_index("rep").reindex(g.rep)
        g["N_max"] = (e.rho_pk.to_numpy() * 8 * 900).round()
        C[key] = _from_perrun(g)
    wg = pd.read_csv(RES / "wellmixed_global_mating" / "per_run.csv")
    wg = wg[wg.L == 30]
    for key, nm in (("core_wmg", "core"), ("SL_wmg", "SL"), ("C1_wmg", "C1_ref")):
        C[key] = _from_perrun(wg[wg.cfg == nm].sort_values("rep").reset_index(drop=True))
    return C


def step_matrix():
    C = column_runs()
    rows = []
    for key, grp, head, src in COLS:
        R = C[key]
        n = len(R)
        for code, lab, ph, note in ROWS:
            if code not in R:
                rows.append(dict(col=key, row=code, n=n, status="ne"))
                continue
            v = R[code].astype(str)
            if key == "g0_sp" and code not in TRUNC_OK:
                rows.append(dict(col=key, row=code, n=n, status="trunc", k_pass=int((v == "PASS").sum()),
                                 k_marg=int((v == "MARG").sum())))
                continue
            kp, km, kna = int((v == "PASS").sum()), int((v == "MARG").sum()), int(v.isin(["NA", "nan"]).sum())
            lo, hi = wilson(kp, n)
            rows.append(dict(col=key, row=code, n=n, status="ok", k_pass=kp, k_marg=km, k_na=kna, f_pass=kp / n,
                             f_pm=(kp + km) / n, lo=lo, hi=hi))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "matrix.csv", index=False)
    json.dump(dict(rows=ROWS, cols=COLS, trunc_ok=sorted(TRUNC_OK)), open(OUT / "matrix_layout.json", "w"),
              indent=1)
    piv = M.assign(txt=lambda d: d.apply(lambda r: r.status if r.status != "ok" else
                                          f"{r.f_pass:.2f}" + (f"+{r.k_marg / r.n:.2f}m" if r.k_marg else ""),
                                          axis=1)).pivot(index="row", columns="col", values="txt")
    print(piv.reindex([r[0] for r in ROWS])[[c[0] for c in COLS]].to_string())


# ----------------------------------------------------------------------------- 3. ablaciones
ABL = [("abl1", 0, "C1"), ("abl1", 1, "−R1"), ("abl1", 2, "−R2"), ("abl1", 3, "−AV"), ("abl1", 4, "−R7"),
       ("abl1", 6, "$w=0$"), ("abl1", 7, "R7, no\npreference"), ("abl1", 8, "R7 soft\ncapacity"),
       ("abl2", 0, "−R1\n−R2"), ("abl2", 1, "−R1\n−AV"), ("abl2", 2, "−R1\n−R7"), ("abl2", 3, "−R2\n−AV"),
       ("abl2", 4, "−R2\n−R7"), ("abl2", 5, "−AV\n−R7"), ("abl2", 10, "no\nrules")]
ABL_ROWS = ["O10b", "O4", "O10", "O14", "O13", "O13p", "O11", "O5", "O6", "O7", "O8", "O9", "O17p"]


def step_ablations():
    rows = []
    for st, pt, lab in ABL:
        s = pd.read_csv(RES2 / f"f2b_{st}" / "summary.csv")
        s = s[s.point == pt]
        n = len(s)
        for code in ABL_ROWS:
            v = s[f"obj_{code}"].fillna("NA").astype(str)
            kp, km = int((v == "PASS").sum()), int((v == "MARG").sum())
            rows.append(dict(src=st, point=pt, label=lab, row=code, n=n, k_pass=kp, k_marg=km, f_pass=kp / n,
                             f_pm=(kp + km) / n))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "ablations.csv", index=False)
    print(A.pivot_table(index="row", columns="label", values="f_pass", sort=False).round(2).reindex(ABL_ROWS)
          .to_string())


# ----------------------------------------------------------------------------- 4. supervivencia
def _surv_row(name, variant, model, L, T, t_ext):
    from pipeline import survival
    t_ext = np.asarray(t_ext, float)
    n = t_ext.size
    ev = np.isfinite(t_ext)
    tt = np.where(ev, t_ext, T)
    km = survival.kaplan_meier(tt, ev)
    m = survival.km_median(km)
    k = int(ev.sum())
    lo, hi = wilson(k, n)
    q = np.quantile(t_ext[ev], [0.1, 0.5, 0.9]) if k else [np.nan] * 3
    return dict(name=name, variant=variant, model=model, L=L, T=T, n=n, n_ext=k, P_ext=k / n, P_lo=lo, P_hi=hi,
                km_med=float(m["median"]), km_lo=float(m["lo"]), km_hi=float(m["hi"]), p10=q[0], p50=q[1],
                p90=q[2])


def step_survival():
    rows = []
    s = pd.read_csv(RES2 / "f2b_base" / "summary.csv")
    rows.append(_surv_row("C1", "space", "C1", 30, 600, s[s.point == 0].t_ext))
    s = pd.read_csv(RES2 / "f2b_nulos" / "summary.csv")
    rows.append(_surv_row("core", "space", "core", 30, 600, s[s.point == 3].t_ext))
    s = pd.read_csv(RES / "spatial_reversible_null_T1000" / "summary.csv")
    rows.append(_surv_row("SL", "space", "SL", 30, 1000, s[s.point == 1].t_ext))
    rows.append(_surv_row("C1 adults recover", "space", "rev", 30, 1000, s[s.point == 0].t_ext))
    s = pd.read_csv(RES2 / "f2b_ref" / "summary.csv")
    rows.append(_surv_row("SL (f2b_ref, PRE)", "space", "SL_pre", 30, 1000, s[s.point == 0].t_ext))
    s = pd.read_csv(RES / "wellmixed_c1" / "summary.csv")
    rows.append(_surv_row("C1", "wm_cell", "C1", 30, 600, s[s.point == 0].t_ext))
    rows.append(_surv_row("C1 no retreat sites", "wm_cell", "C1_noref", 30, 600, s[s.point == 1].t_ext))
    e = pd.read_csv(RES / "wellmixed_cell_reference" / "per_run.csv")
    rows.append(_surv_row("core", "wm_cell", "core", 30, 600, e[e.cfg == "WM_none"].t_ext))
    rows.append(_surv_row("SL", "wm_cell", "SL", 30, 600, e[e.cfg == "WM_SL"].t_ext))
    g = pd.read_csv(RES / "wellmixed_global_mating" / "per_run.csv")
    for (cfg, L), d in g.groupby(["cfg", "L"]):
        rows.append(_surv_row({"C1_ref": "C1", "C1_noref": "C1 no retreat sites", "SL": "SL", "core": "core"}[cfg],
                              "wm_global", cfg if cfg != "C1_ref" else "C1", int(L), 1000, d.t_ext))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "survival.csv", index=False)
    print(D.round(3).to_string())


# ----------------------------------------------------------------------------- 5. hipercubos
def step_lhs():
    s = pd.concat([pd.read_csv(RES2 / d / "summary.csv").assign(src=d) for d in ("f2b_lhs", "f2b_lhs2")],
                  ignore_index=True).copy()

    def P(c):
        return (s[f"obj_{c}"].fillna("NA").astype(str) == "PASS").to_numpy()
    anti = np.all([(s[f"obj_A{i}"].astype(str) == "CLEAR").to_numpy() for i in range(1, 5)], axis=0)
    env = P("O5") & P("O6") & P("O7") & P("O10") & anti
    defs = {"O7 (extinction by T = 400)": P("O7"), "collapse envelope (O5, O6, O7, O10; A1-A4 clear)": env,
            "envelope + O11": env & P("O11"), "envelope + O11 + O4 + O14": env & P("O11") & P("O4") & P("O14"),
            "envelope + O11 + O4 + O14 + O13": env & P("O11") & P("O4") & P("O14") & P("O13"),
            "O17p (all essential patterns)": P("O17p")}
    key = s.src + ":" + s.point.astype(str)
    out = {"n_points": int(key.nunique()), "runs_per_point": int(s.groupby(key).size().median()), "T": 400,
           "rule": "point reproduces X if k/n >= 0.8 of its runs", "rows": {}}
    rng = np.random.default_rng(1)
    for nm, v in defs.items():
        fr = pd.Series(v.astype(float)).groupby(key.to_numpy()).mean()
        ok = (fr >= 0.8).to_numpy()
        boots = [ok[rng.integers(0, ok.size, ok.size)].mean() for _ in range(2000)]
        out["rows"][nm] = dict(k=int(ok.sum()), n=int(ok.size), f=float(ok.mean()),
                               lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                               run_frac=float(v.mean()))
        print(f"{nm:55s} {ok.sum():4d}/{ok.size}  {ok.mean():.3f}  runs {v.mean():.3f}")
    json.dump(out, open(OUT / "lhs.json", "w"), indent=1)


# ----------------------------------------------------------------------------- 6. depósito de series (C17)
DEP = RES / "series_deposit"
DEP_COLS = ["point", "rep", "t", "N", "births", "pups", "conceptions", "S_adult", "mean_age", "N_adult",
            "frac_withdrawn", "frac_crowded_low", "phi_BO", "low_s_frac", "phi_w", "coh_f_born",
            "coh_f_ever_conceived"]


def _reduce(path, points):
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    s = pd.read_parquet(path, columns=[c for c in DEP_COLS if c in have])
    s = s[s.point.isin(points)].sort_values(["point", "rep", "t"])
    # serie completa (T = 600) para que las bandas de em_figs.py salgan idénticas a las de las series originales
    for c in s.columns:
        if s[c].dtype.kind == "f":
            s[c] = s[c].astype(np.float32)
        elif s[c].dtype.kind in "iu":
            s[c] = s[c].astype(np.int32)
    return s.reset_index(drop=True)


def step_deposit():
    DEP.mkdir(parents=True, exist_ok=True)
    meta = {}
    SRC = (("C1_spatial", RES2 / "f2b_base" / "series.parquet", [0]),
           ("C1_wellmixed", RES / "wellmixed_c1" / "series.parquet", [0, 1]))
    import pyarrow.parquet as pq
    if not all(p.exists() for _, p, _ in SRC) or "frac_withdrawn" not in pq.read_schema(SRC[0][1]).names:
        print("deposit: faltan las series completas de f2b_base (make fig-inputs sólo restituye 8 columnas; ver "
              "scripts/figures/fig_inputs.py para regenerarlas) o de wellmixed_c1 (em_wm_run.py); no se escribe nada")
        return
    for name, path, pts in SRC:
        if not path.exists():
            print("deposit: sin", path, "(correr antes em_wm_run.py; las series crudas no se versionan)")
            return
        s = _reduce(path, pts)
        fn = DEP / f"em_series_{name}.parquet"
        s.to_parquet(fn, compression="zstd", index=False)
        meta[fn.name] = dict(source=str(path.relative_to(ROOT)), points=pts, rows=int(len(s)),
                             runs=int(s.groupby(["point", "rep"]).ngroups), columns=list(s.columns),
                             bytes=fn.stat().st_size)
        print(fn, len(s), f"{fn.stat().st_size / 1e6:.2f} MB")
    json.dump(meta, open(DEP / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["scores", "matrix", "ablations", "survival", "lhs", "deposit"]
    for st in steps:
        globals()[f"step_{st}"]()
