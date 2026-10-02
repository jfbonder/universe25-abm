"""Generador de los diseños de la Fase 2b (v1), parametrizado por el preset base.

    python3 scripts/production/gen_designs.py all   --preset calhoun_C1 [--out DIR]
    python3 scripts/production/gen_designs.py check --preset calhoun_C1 [--des DIR]      # verificación previa
    python3 scripts/production/gen_designs.py stagevars --des DIR --stage base         # variables para el orquestador
    python3 scripts/production/gen_designs.py sobol-from-morris --preset P --des DIR --morris RUN_DIR
    python3 scripts/production/gen_designs.py refine --preset P --des DIR --runs DIR1 DIR2 ...
    python3 scripts/production/gen_designs.py measure --out RUN_DIR --c-est 1.8           # costo medido / estimado
    python3 scripts/production/gen_designs.py smoke --des DIR --stage p1 --points 0,7 --out DIR

Cada etapa queda en <out>/<etapa>/ con
  design.csv   puntos x columnas de Params (escala física; NaN = inf sólo en plast_age_max);
  labels.json  etiqueta legible por punto;
  stage.json   réplicas, T, CRN, series, semilla, preset de la etapa, prioridad, costo estimado, Q que alimenta;
  problem.json, X.npy  (sólo morris / lhs / sobol: formato de scripts/analysis/pipeline).
Los diseños están *resueltos* contra el preset: toda columna que alguna fila cambia lleva, en las demás filas, el
valor del preset. Por eso un cambio de preset requiere regenerar (run_2b_v2.sh lo hace si falta el directorio).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "scripts" / "production"
for p in (str(ROOT), str(ROOT / "scripts" / "analysis"), str(MAT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from u25.params import Params, PRESETS  # noqa: E402

FIELDS = set(Params.__dataclass_fields__)
C_RUN = float(os.environ.get("C_RUN", "1.5"))      # s de CPU por corrida del candidato (C1 medido 1.4–1.8; C0 ~ 0.95)

# --------------------------------------------------------------------------- factores del screening
# (nombre, lo, hi, escala, tipo, grupo). Rangos de la calibración + m_c, eta, refugios.
FACTORS = [
    ("alpha", 2.0, 8.0, "lin", "float", "base"),
    ("crowd_aversion", 0.05, 0.30, "lin", "float", "base"),
    ("crowd_power", 1.0, 8.0, "log", "float", "base"),
    ("rec", 0.03, 0.30, "log", "float", "base"),
    ("inherit_weight", 0.0, 0.6, "lin", "float", "base"),
    ("plast_age_max", 4.0, 24.0, "log", "float", "base"),
    ("gamma", 0.03, 0.30, "log", "float", "base"),
    ("a_max", 90.0, 150.0, "lin", "float", "base"),      # + repro_age_max = round(2/3 a_max)
    ("p0", 0.15, 0.45, "lin", "float", "base"),
    ("delta", 0.003, 0.05, "log", "float", "base"),
    ("m_c", 4, 16, "lin", "int", "base"),
    ("eta", 0.05, 0.50, "lin", "float", "base"),
    ("refuge_frac", 0.05, 0.40, "lin", "float", "refugios"),
    ("refuge_pref", 3.0, 1000.0, "log", "float", "refugios"),
    ("refuge_capacity", 1, 3, "lin", "int", "refugios"),
]
# predicción para Sobol (si no hay Morris): lo más sensible en la OAT de la calibración
SOBOL_PRED = ["rec", "crowd_power", "crowd_aversion", "alpha", "gamma", "inherit_weight", "plast_age_max"]
SOBOL_K = 8


# --------------------------------------------------------------------------- preset y reglas
def preset_dict(name: str) -> dict:
    if name not in PRESETS:
        raise SystemExit(f"preset desconocido: {name}; opciones: {sorted(PRESETS)}")
    return dict(PRESETS[name])


def base_params(name: str) -> Params:
    return Params(**preset_dict(name))


def active_rules(p: Params) -> list[str]:
    r = []
    if math.isfinite(p.plast_age_max) and p.plast_outside_rec == 0.0:
        r.append("R1")
    if p.social_rec_lambda > 0:
        r.append("R2")
    if p.crowd_aversion > 0:
        r.append("AV")
    if getattr(p, "refuge_frac", 0) > 0 and getattr(p, "refuge_pref", 0) > 0:
        r.append("R7")
    return r


# ablación de cada regla ("sin R2" = herencia completa con rec = 0.01, como en s3)
ABLATE = {
    "R1": dict(plast_age_max=math.inf, plast_outside_rec=1.0),
    "R2": dict(social_rec_lambda=0.0, inherit_weight=1.0, s_newborn_base=1.0, rec=0.01),
    "AV": dict(crowd_aversion=0.0),
    "R7": dict(refuge_frac=0.0, refuge_pref=0.0, refuge_hard=False),
}


def has_refuges(p: Params) -> bool:
    return "refuge_pref" in FIELDS and "refuge_hard" in FIELDS


# --------------------------------------------------------------------------- escritura
def _clean(v):
    if isinstance(v, float) and math.isinf(v):
        return np.nan
    return v


def write_rows(out: Path, rows: list[tuple[str, dict]], base: Params, meta: dict):
    """rows: [(label, overrides)] -> design.csv completo (columnas = unión de claves; el resto del preset)."""
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for _, kw in rows:
        for k in kw:
            if k not in keys:
                keys.append(k)
    bad = [k for k in keys if k not in FIELDS]
    if bad:
        raise SystemExit(f"[{meta['stage']}] columnas que no son campos de Params: {bad}")
    data = []
    for i, (_, kw) in enumerate(rows):
        r = {"point": i}
        for k in keys:
            r[k] = _clean(kw.get(k, getattr(base, k)))
        data.append(r)
    df = pd.DataFrame(data, columns=["point", *keys])
    df.to_csv(out / "design.csv", index=False)
    json.dump([lab for lab, _ in rows], open(out / "labels.json", "w"), indent=0, ensure_ascii=False)
    meta = {**meta, "n_points": len(df), "runs": len(df) * meta.get("reps", 0), "columns": keys}
    json.dump(meta, open(out / "stage.json", "w"), indent=1, ensure_ascii=False)
    return df


def problem_for(names: list[str]) -> dict:
    F = {f[0]: f for f in FACTORS}
    return {"names": names, "bounds": [[F[n][1], F[n][2]] for n in names],
            "scale": [F[n][3] for n in names], "type": [F[n][4] for n in names], "fixed": {}}


def factor_names(p: Params) -> list[str]:
    names = [f[0] for f in FACTORS if f[5] == "base"]
    if has_refuges(p) and p.refuge_frac > 0:
        names += [f[0] for f in FACTORS if f[5] == "refugios"]
    return names


def write_sa(out: Path, kind: str, names: list[str], meta: dict, **kw):
    from pipeline import design as pdesign
    problem = problem_for(names)
    if kind == "morris":
        X, df = pdesign.morris_design(problem, r=kw["r"], num_levels=4, seed=kw.get("seed", 0))
        pdesign.save_design(out, "morris", problem, X, df, r=kw["r"], num_levels=4, seed=kw.get("seed", 0))
    elif kind == "lhs":
        X, df = pdesign.lhs_design(problem, n=kw["n"], seed=kw.get("seed", 0))
        pdesign.save_design(out, "lhs", problem, X, df, n=kw["n"], seed=kw.get("seed", 0))
    else:
        X, df = pdesign.saltelli_design(problem, N=kw["N"], calc_second_order=False, seed=kw.get("seed", 0))
        pdesign.save_design(out, "sobol", problem, X, df, N=kw["N"], calc_second_order=False, seed=kw.get("seed", 0))
    d = pd.read_csv(out / "design.csv")
    if "a_max" in names:                         # longevidad: ventana fértil proporcional
        d["repro_age_max"] = np.round(2.0 * d["a_max"] / 3.0).astype(int)
    d.to_csv(out / "design.csv", index=False)
    json.dump([f"{kind}{i}" for i in range(len(d))], open(out / "labels.json", "w"))
    meta = {**meta, "n_points": len(d), "runs": len(d) * meta["reps"], "factors": names,
            "columns": [c for c in d.columns if c != "point"]}
    json.dump(meta, open(out / "stage.json", "w"), indent=1, ensure_ascii=False)
    return d


def M(stage, reps, T, c, prio, q="", kind="sweep", crn=False, series_every=0, full_series=False,
      preset=None, analysis=None, seed=None, **extra):
    """Metadatos de una etapa. c = costo estimado en s de CPU por corrida *relativo a C_RUN = 1.2*."""
    return dict(stage=stage, kind=kind, reps=reps, T=T, crn=crn, series_every=series_every,
                full_series=full_series, preset=preset, priority=prio, c_cpu=round(c * C_RUN / 1.2, 3),
                analysis=analysis or stage, q=q, base_seed=seed, **extra)


# --------------------------------------------------------------------------- etapas
ALT = {"calhoun_C1": "calhoun_C2", "calhoun_C2": "calhoun_C1"}     # contraste confirmatorio


def preset_diff(name: str, P: Params) -> dict:
    """Campos en que el preset `name` difiere de P (para escribirlo como fila de un diseño resuelto contra P)."""
    Q = Params(**preset_dict(name))
    return {k: getattr(Q, k) for k in sorted(FIELDS) if getattr(Q, k) != getattr(P, k)}


SEEDS = dict(c1c2=25118, p6=25119, base=25101, ref=25102, abl1=25103, abl2=25104, amax=25105, trans=25106, morris=25107, p2=25108,
             p1=25109, p1aw=25110, p5=25111, lhs=25112, lhs2=25113, sobol=25114, p4=25115, trgrid=25116,
             refine=25117)
ORDER = ["base", "c1c2", "ref", "abl1", "trans", "amax", "abl2", "morris", "p2", "p1", "p1aw", "p5", "lhs",
         "sobol", "lhs2", "p4", "trgrid", "refine", "p6", "sobolx"]


def gen_all(preset: str, out: Path):
    P = base_params(preset)
    rules = active_rules(P)
    tag = preset.replace("calhoun_", "")
    out.mkdir(parents=True, exist_ok=True)
    made = {}

    # E0 base: candidato, C0 de control y control alpha = 0 (O14, G_B), serie completa (evaluador oficial)
    rows = [(tag, {})]
    if preset != "calhoun_C0" and "calhoun_C0" in PRESETS:
        rows.append(("C0 (control)", preset_diff("calhoun_C0", P)))
    rows.append((f"{tag} alpha=0 (control O14)", {"alpha": 0.0}))
    made["base"] = write_rows(out / "base", rows, P, M("base", 200, 600, 1.8, 0, "Q3", series_every=1,
                                                       full_series=True, seed=SEEDS["base"]))

    # E0' línea base: notebook (alpha = 1.4) y longevo puro, preset notebook
    NB = Params()
    made["ref"] = write_rows(out / "ref", [("notebook alpha=1.4", {}), ("longevo", dict(a_max=120.0, repro_age_max=80))],
                             NB, M("ref", 100, 1000, 4.0, 0, "Q3", series_every=1, preset="notebook", seed=SEEDS["ref"]))

    # E1 ablaciones: factorial 2^m de las reglas activas; distancia <= 1 a n = 120 (abl1), resto a n = 40 (abl2)
    cells1, cells2 = [], []
    for k in range(len(rules) + 1):
        for drop in itertools.combinations(rules, k):
            kw = {}
            for r in drop:
                kw.update(ABLATE[r])
            lab = tag if not drop else tag + " -" + " -".join(drop)
            (cells1 if k <= 1 else cells2).append((lab, kw))
    if "R1" in rules:
        cells1.append((f"{tag} -R1 alpha=6", {**ABLATE["R1"], "alpha": 6.0}))
    if "R2" in rules:
        cells1.append((f"{tag} w=0 rec=0.15", {"inherit_weight": 0.0, "rec": 0.15}))
    if "R7" in rules:
        cells1.append((f"{tag} -pref (refugios sin preferencia)", {"refuge_pref": 0.0}))
        cells1.append((f"{tag} -hard (capacidad blanda)", {"refuge_hard": False}))
    cells2.append((f"{tag} gamma=0 (sin hacinamiento)", {"gamma": 0.0}))
    meta_f = dict(rules=rules)
    made["abl1"] = write_rows(out / "abl1", cells1, P, M("abl1", 120, 600, 1.6, 0, "Q2", seed=SEEDS["abl1"], **meta_f))
    made["abl2"] = write_rows(out / "abl2", cells2, P, M("abl2", 40, 600, 4.0, 0, "Q2", seed=SEEDS["abl2"], **meta_f))

    # E2 transplantes (O12, O12b, Q1): candidato y ablaciones de la irreversibilidad
    arms = [(tag, {})]
    for drop in (["R1"], ["R2"], ["R1", "R2"]):
        if all(r in rules for r in drop):
            kw = {}
            for r in drop:
                kw.update(ABLATE[r])
            arms.append((tag + " -" + " -".join(drop), kw))
    # Q1: transplante cruzado (madre = candidato, punto 0; colonia sin R1 ni R2 y con recuperación individual
    # rápida rec del candidato): mismos animales D (pareado), sólo cambian las reglas de la colonia nueva
    cross = {}
    if "R1" in rules and "R2" in rules:
        cross[0] = 0                      # el brazo del candidato corre por el mismo camino (cruce identidad)
        cross[len(arms)] = 0
        arms.append((f"X: madre {tag}, colonia -R1 -R2 rec={P.rec:g}", {**ABLATE["R1"], **ABLATE["R2"], "rec": P.rec}))
    if preset != "calhoun_C0" and "calhoun_C0" in PRESETS:
        arms.append(("C0 (control)", preset_diff("calhoun_C0", P)))
    alt = ALT.get(preset)
    if alt and alt in PRESETS:
        arms.append((alt.replace("calhoun_", "") + " (contraste)", preset_diff(alt, P)))
    made["trans"] = write_rows(out / "trans", arms, P, M("trans", 60, 300, 5.0, 0, "Q1", kind="transplant",
                                                        T_main=300, T_after=150, seed=SEEDS["trans"], cross=cross))
    json.dump({str(k): v for k, v in cross.items()}, open(out / "trans" / "cross.json", "w"))

    # E2' contraste confirmatorio C1 vs C2: todos los objetivos, Holm; n = 200 cada uno
    if alt and alt in PRESETS:
        made["c1c2"] = write_rows(out / "c1c2", [(tag, {}), (alt.replace("calhoun_", ""), preset_diff(alt, P))], P,
                                  M("c1c2", 200, 700, 1.8, 0, "QC", seed=SEEDS["c1c2"], alt=alt))

    # E3 ley de la fase D (Q4): a_max con ventana fértil 2/3
    am = [(f"a_max={a:g}", {"a_max": float(a), "repro_age_max": int(round(2 * a / 3))}) for a in (90, 105, 120, 135, 150)]
    made["amax"] = write_rows(out / "amax", am, P, M("amax", 60, 600, 1.3, 0, "Q4", seed=SEEDS["amax"]))

    # E4 Morris
    names = factor_names(P)
    made["morris"] = write_sa(out / "morris", "morris", names,
                              M("morris", 10, 400, 2.0, 0, "", crn=True, seed=SEEDS["morris"]), r=20, seed=1)

    # E5 planos de la calibración
    al = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    ka = [0.0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.30]
    rows = [(f"a={a:g} k={k:g}", {"alpha": a, "crowd_aversion": k, "crowd_power": P.crowd_power}) for a in al for k in ka]
    rows += [(f"a=4 k={k:g} p={pw:g}", {"alpha": 4.0, "crowd_aversion": k, "crowd_power": pw}) for pw in (2.0, 8.0) for k in ka]
    made["p2"] = write_rows(out / "p2", rows, P, M("p2", 30, 600, 1.5, 1, "Q5", seed=SEEDS["p2"],
                                                  axes=["alpha", "crowd_aversion"], layer="crowd_power"))
    recs = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
    ws = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows = [(f"rec={r:g} w={w:g} g={g:g}", {"rec": r, "inherit_weight": w, "gamma": g})
            for g in (P.gamma, 0.0) for r in recs for w in ws]
    made["p1"] = write_rows(out / "p1", rows, P, M("p1", 30, 600, 1.8, 1, "Q7", seed=SEEDS["p1"],
                                                  axes=["rec", "inherit_weight"], layer="gamma"))
    aws = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, math.inf]
    rows = [(f"a_w={a:g} rec={r:g}", {"plast_age_max": a, "rec": r}) for a in aws for r in recs]
    made["p1aw"] = write_rows(out / "p1aw", rows, P, M("p1aw", 30, 600, 2.0, 1, "Q6", seed=SEEDS["p1aw"],
                                                      axes=["rec", "plast_age_max"]))
    if has_refuges(P):
        rows = [(f"fr={fr:g} cap={c} pref={pf:g}", {"refuge_frac": fr, "refuge_capacity": c, "refuge_pref": pf,
                                                    "refuge_hard": True})
                for fr in (0.05, 0.10, 0.20, 0.30, 0.40) for c in (1, 2) for pf in (0.0, 10.0, 30.0, 100.0, 1000.0)]
        made["p5"] = write_rows(out / "p5", rows, P, M("p5", 30, 600, 1.3, 1, "", seed=SEEDS["p5"],
                                                      axes=["refuge_frac", "refuge_pref"], layer="refuge_capacity"))

    # E6 LHS (dos mitades independientes de 200: la segunda es recortable) y Sobol
    made["lhs"] = write_sa(out / "lhs", "lhs", names, M("lhs", 10, 400, 2.0, 1, "", seed=SEEDS["lhs"]), n=200, seed=2)
    made["lhs2"] = write_sa(out / "lhs2", "lhs", names, M("lhs2", 10, 400, 2.0, 2, "", seed=SEEDS["lhs2"]), n=200, seed=3)
    sob = SOBOL_PRED + (["refuge_frac"] if "refuge_frac" in names else ["a_max"])
    made["sobol"] = gen_sobol(out / "sobol", sob[:SOBOL_K])

    # E7 siembra y escala + capa de apareamiento en Moore (mate_radius = 1) si existe
    rows = []
    for L in (15, 20, 30):
        for N0 in ("F", 20, 50, 100, 200, 400, 800):
            if N0 == "F":
                kw = dict(n_males=4, n_females=4, initial_age=7, founders_same_cell=True)
            else:
                kw = dict(n_males=N0 // 2, n_females=N0 // 2, initial_age=P.initial_age, founders_same_cell=False)
            rows.append((f"L={L} N0={N0}", {"grid_size": L, **kw}))
    if "mate_radius" in FIELDS:
        for N0 in ("F", 20, 50):
            kw = (dict(n_males=4, n_females=4, initial_age=7, founders_same_cell=True) if N0 == "F" else
                  dict(n_males=N0 // 2, n_females=N0 // 2, initial_age=P.initial_age, founders_same_cell=False))
            rows.append((f"L=30 N0={N0} mate_radius=1", {"grid_size": 30, **kw, "mate_radius": 1}))
    made["p4"] = write_rows(out / "p4", rows, P, M("p4", 30, 600, 1.5, 2, "", seed=SEEDS["p4"],
                                                  axes=["grid_size", "n_males"]))
    # E8 sub-grilla de O12 (transplantes) en alpha x kappa
    rows = [(f"a={a:g} k={k:g}", {"alpha": a, "crowd_aversion": k}) for a in (2.0, 4.0, 6.0, 8.0) for k in (0.10, 0.15, 0.20)]
    made["trgrid"] = write_rows(out / "trgrid", rows, P, M("trgrid", 40, 300, 4.0, 2, "", kind="transplant",
                                                          T_main=300, T_after=150, seed=SEEDS["trgrid"]))
    # E9 fundadores (exploratorio, prioridad 3): founders_longevo + mate_radius = 1
    if "mate_radius" in FIELDS:
        rows = []
        for s0 in (0.2, 0.3, 0.5):
            for k in (1, 2):
                for am in (120.0, 200.0):
                    rows.append((f"F s0={s0:g} k={k} a_max={am:g}",
                                 dict(n_males=4, n_females=4, initial_age=7, founders_same_cell=True, mate_radius=1,
                                      initial_s=s0, move_substeps=k, a_max=am, repro_age_max=int(round(2 * am / 3)))))
        made["p6"] = write_rows(out / "p6", rows, P, M("p6", 30, 900, 2.5, 3, "", seed=SEEDS["p6"],
                                                      axes=["initial_s", "a_max"], layer="move_substeps"))
    pending = [] if (alt is None or alt in PRESETS) else ["c1c2"]
    manifest = dict(preset=preset, tag=tag, pending=pending, primary=("O17p" if "R7" in rules else "O17_run"), rules=rules, order=[s for s in ORDER if (out / s).exists() or s in ("refine", "sobolx")],
                    c_run=C_RUN, factors=names, generated_by="scripts/production/gen_designs.py")
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    stamp(out)
    return made


def stamp(des: Path):
    """Copia preset, etiqueta y criterio primario del manifiesto a cada stage.json del directorio."""
    m = json.load(open(des / "manifest.json"))
    for f in des.glob("*/stage.json"):
        st = json.load(open(f))
        st.update(preset_main=m["preset"], tag=m["tag"], primary=m["primary"])
        json.dump(st, open(f, "w"), indent=1, ensure_ascii=False)


def gen_sobol(out: Path, names: list[str], reps=4, src="prediccion"):
    meta = M("sobol", reps, 400, 2.0, 1, "", crn=True, seed=SEEDS["sobol"], reps_min=3, source=src, ext_reps=8)
    return write_sa(out, "sobol", names, meta, N=256, seed=4)


# --------------------------------------------------------------------------- utilidades para el orquestador (run_2b_v2.sh)
def stagevars(des: Path, stage: str):
    st = json.load(open(des / stage / "stage.json"))
    sh = dict(KIND=st["kind"], REPS=st["reps"], T=st["T"], CRN=int(bool(st["crn"])), SERIES_EVERY=st["series_every"],
              FULL=int(bool(st["full_series"])), BASE_SEED=st["base_seed"], SPRESET=st["preset"] or "",
              PRIORITY=st["priority"], RUNS=st["runs"], NPTS=st["n_points"], CCPU=st["c_cpu"],
              TMAIN=st.get("T_main", 300), TAFTER=st.get("T_after", 150), REPS_MIN=st.get("reps_min", st["reps"]),
              ANALYSIS=st["analysis"], TFIX=st.get("T_fix", 95), BAND=",".join(str(x) for x in st.get("band", [0.2, 0.5])),
              CONT=st.get("cont_after", 40))
    for k, v in sh.items():
        print(f"{k}={v!s}")


def check(preset: str, des: Path):
    P = base_params(preset)
    P.validate()
    probs = []
    for sd in sorted(des.iterdir()):
        if not (sd / "stage.json").exists():
            continue
        d = pd.read_csv(sd / "design.csv")
        bad = [c for c in d.columns if c != "point" and c not in FIELDS]
        if bad:
            probs.append(f"{sd.name}: columnas desconocidas {bad}")
            continue
        from u25.sweep import design_to_params
        st = json.load(open(sd / "stage.json"))
        try:
            base = base_params(st["preset"]) if st.get("preset") else P
            design_to_params(d, base)
        except Exception as e:  # noqa: BLE001
            probs.append(f"{sd.name}: {type(e).__name__}: {e}")
    m = json.load(open(des / "manifest.json"))
    if m["preset"] != preset:
        probs.append(f"manifest.json es de {m['preset']}, no de {preset}")
    if probs:
        print("PROBLEMAS:\n  " + "\n  ".join(probs))
        sys.exit(2)
    print(f"ok: {preset}, reglas {active_rules(P)}, {len(list(des.glob('*/stage.json')))} etapas")


def sobol_from_morris(preset: str, des: Path, morris_run: Path, k=SOBOL_K):
    """Elige los k factores con mayor mu* relativo medio (sobre las métricas primarias) y regenera Sobol."""
    sel = morris_run / "analisis" / "seleccion_sobol.json"
    if not sel.exists():
        print(f"no hay {sel}; se usa el Sobol predicho")
        return None
    names = json.load(open(sel))["factores"][:k]
    out = des / "sobol_auto"
    if out.exists():
        old = json.load(open(out / "stage.json"))["factors"]
        if old == names:
            print(f"sobol_auto ya existe con {names}")
            return out
    gen_sobol(out, names, src=str(sel))
    stamp(des)
    print(f"sobol_auto: {names}")
    return out


def refine(preset: str, des: Path, runs: list[Path], lo=0.1, hi=0.9, max_points=40, reps=100):
    """Puntos de frontera de los planos (0.1 < f < 0.9 en el criterio primario) -> diseño 'refine'."""
    P = base_params(preset)
    rows = []
    for rd in runs:
        tp = rd / "analisis" / "tabla_puntos.csv"
        if not tp.exists():
            continue
        T = pd.read_csv(tp)
        col = "f_PRIMARIO" if "f_PRIMARIO" in T else "f_O17_run"
        B = T[(T[col] > lo) & (T[col] < hi)].copy()
        B["dist"] = (B[col] - 0.5).abs()
        dcols = json.load(open(rd / "meta.json"))["design_columns"]
        for _, r in B.sort_values("dist").iterrows():
            kw = {c: (math.inf if (c == "plast_age_max" and not np.isfinite(r[c])) else r[c]) for c in dcols}
            for c, v in list(kw.items()):
                cur = getattr(P, c)
                if isinstance(cur, bool):
                    kw[c] = str(v).lower() in ("true", "1")
                elif isinstance(cur, int):
                    kw[c] = int(round(float(v)))
            rows.append((f"{rd.name}:{r['label']}", kw))
    rows = rows[:max_points]
    if not rows:
        print("refine: no hay puntos de frontera (o faltan análisis)")
        return None
    out = des / "refine"
    write_rows(out, rows, P, M("refine", reps, 600, 1.8, 2, "", seed=SEEDS["refine"]))
    stamp(des)
    print(f"refine: {len(rows)} puntos -> {out}")
    return out


def budget(sd: Path, out: Path, n_jobs: int, remaining: float, scale: float, margin: float = 300.0,
           reps_override: int | None = None):
    """Decide si una etapa corre y con cuántas réplicas (orden de recorte del plan, §6).
    P0 y P1 nunca se saltean (el vigía las corta en el plazo y quedan reanudables); Sobol baja réplicas hasta
    reps_min; P2/P3 se saltean si no entran en el tiempo restante."""
    st = json.load(open(sd / "stage.json"))
    reps = reps_override or st["reps"]
    prio = 3 if reps_override else st["priority"]          # extensión (sobolx): P3
    npts, c = st["n_points"], st["c_cpu"] * (scale if np.isfinite(scale) and scale > 0 else 1.0)
    done = 0
    if str(st["kind"]).startswith("transplant"):
        done = sum(1 for d in out.glob("arm_*") if (d / "O12.txt").exists()) * st["reps"]
    elif (out / "summary.csv").exists():
        try:
            s = pd.read_csv(out / "summary.csv", usecols=["point", "rep"])
            done = int(((s.rep < reps)).sum())
        except Exception:  # noqa: BLE001
            done = 0
    todo = max(0, npts * reps - done)
    est = todo * c / n_jobs
    avail = remaining - margin
    dec, r = "RUN", reps
    if todo == 0:
        dec = "DONE"
    elif est > avail:
        if "reps_min" in st and not reps_override:
            fit = int(avail * n_jobs / (npts * c)) if avail > 0 else 0
            if fit >= st["reps_min"]:
                r = min(reps, fit)
                est = max(0, npts * r - done) * c / n_jobs
            elif prio >= 2:
                dec = "SKIP"
            else:
                r = st["reps_min"]
                est = max(0, npts * r - done) * c / n_jobs
        elif prio >= 2:
            dec = "SKIP"
    print(f"DECISION={dec}\nREPS={r}\nEST={int(est)}\nTODO={todo}\nSCALE={c / st['c_cpu']:.3f}")


def measure(out: Path, c_est: float) -> float:
    """Cociente costo medido / estimado (s de CPU por corrida, columna wall_time_s del summary)."""
    f = out / "summary.csv"
    if not f.exists():
        return float("nan")
    s = pd.read_csv(f, usecols=lambda c: c in ("wall_time_s",))
    if not len(s):
        return float("nan")
    return float(s["wall_time_s"].mean() / c_est)


def smoke(des: Path, stage: str, points: list[int], out: Path, reps: int, T: int):
    sd = des / stage
    d = pd.read_csv(sd / "design.csv")
    lab = json.load(open(sd / "labels.json"))
    st = json.load(open(sd / "stage.json"))
    keep = [p for p in points if p < len(d)] or list(range(min(4, len(d))))
    d2 = d[d.point.isin(keep)].copy()
    d2_old = [int(x) for x in d2.point]
    lab2 = [lab[i] for i in d2.point]
    d2["point"] = np.arange(len(d2))
    out.mkdir(parents=True, exist_ok=True)
    d2.to_csv(out / "design.csv", index=False)
    json.dump(lab2, open(out / "labels.json", "w"))
    st.update(reps=min(reps, st["reps"]), T=min(T, st["T"]), n_points=len(d2), runs=len(d2) * min(reps, st["reps"]),
              smoke=True, smoke_from=str(sd), smoke_points=keep)
    json.dump(st, open(out / "stage.json", "w"), indent=1)
    if (sd / "cross.json").exists():          # transplante cruzado: reindexar los brazos que quedan
        new = {old: i for i, old in enumerate(d2_old)}
        cr = {int(k): int(v) for k, v in json.load(open(sd / "cross.json")).items()}
        json.dump({str(new[k]): new[v] for k, v in cr.items() if k in new and v in new}, open(out / "cross.json", "w"))
    print(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "check", "stagevars", "sobol-from-morris", "refine", "measure", "smoke",
                                    "budget"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--remaining", type=float, default=6 * 3600)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--reps-override", type=int, default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--preset", default=os.environ.get("PRESET", "calhoun_C1"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--des", default=None)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--morris", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--c-est", type=float, default=1.0)
    ap.add_argument("--points", default="")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--T", type=int, default=250)
    a = ap.parse_args(argv)
    des = Path(a.des or a.out or (MAT / "designs" / a.preset))   # v1; los diseños vigentes están en designs_v2/
    if a.cmd == "all":
        made = gen_all(a.preset, des)
        tot = 0.0
        for s, df in made.items():
            st = json.load(open(des / s / "stage.json"))
            h = st["runs"] * st["c_cpu"] / 4 / 3600
            tot += h
            print(f"{s:8s} P{st['priority']} {st['n_points']:5d} pts x {st['reps']:3d} rép = {st['runs']:6d} corridas,"
                  f" ~{h:5.2f} h de pared (4 núcleos)")
        print(f"total estimado (sin sobolx/refine): {tot:.2f} h -> {des}")
    elif a.cmd == "check":
        check(a.preset, des)
    elif a.cmd == "stagevars":
        stagevars(des, a.stage)
    elif a.cmd == "sobol-from-morris":
        r = sobol_from_morris(a.preset, des, Path(a.morris))
        print(r or "")
    elif a.cmd == "refine":
        refine(a.preset, des, [Path(x) for x in a.runs])
    elif a.cmd == "measure":
        r = measure(Path(a.out), a.c_est)
        if a.state:                       # mediana móvil de los cocientes medido/estimado
            st = Path(a.state)
            vals = [float(x) for x in st.read_text().split()] if st.exists() else []
            if np.isfinite(r):
                vals.append(r)
                st.write_text(" ".join(f"{v:.4f}" for v in vals))
            r = float(np.median(vals)) if vals else 1.0
        print(f"{r:.3f}")
    elif a.cmd == "budget":
        budget(des / a.stage if a.stage else des, Path(a.out), a.n_jobs, a.remaining, a.scale,
               reps_override=a.reps_override)
    elif a.cmd == "smoke":
        smoke(des, a.stage, [int(x) for x in a.points.split(",") if x], Path(a.out), a.reps, a.T)


if __name__ == "__main__":
    main()
