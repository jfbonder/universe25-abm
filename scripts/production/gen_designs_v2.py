"""Diseños de la V2: re-corrida de la Fase 2b con R7 corregido + etapas nuevas.

    python3 scripts/production/gen_designs_v2.py all   --preset calhoun_C1 [--out scripts/production/designs_v2/calhoun_C1]
    python3 scripts/production/gen_designs_v2.py check --preset calhoun_C1 [--des DIR]     # incluye el viaje CSV -> Params
    python3 scripts/production/gen_designs_v2.py sobol-from-morris --preset P --des DIR --morris RUN_DIR
    python3 scripts/production/gen_designs_v2.py lhsx --preset P --des DIR --runs LHS_DIR LHS2_DIR
    (stagevars, refine, measure, budget, smoke: los de gen_designs.py, que este módulo reexporta)

Reutiliza gen_designs.py (v1) con semillas nuevas (25301–25349: la V2 es una réplica independiente) y modifica:
  base   + C1′ (calhoun_C1p): un único ensamble de referencia para C1, C1′, C0 y α = 0 (cifras únicas);
  trans  protocolo trans_v2 (kind transplant2): tiempos de muestreo igualados, banda de s, + brazo C1′;
  sobol  k = 11: los 8 de Morris ∪ {w, r, a_w}, N = 256, 3 réplicas (extensión a 6 en sobolx);
  lhs2   prioridad 1 (el estimador de la región factible usa los dos hipercubos);
y agrega nulos, rob, robmu, ci, size, q7aw y lhsx (ésta se genera después de lhs y lhs2).
Los campos nuevos de Params se leen de v2_api.json; un brazo cuyo campo no existe queda en
manifest.json['pendiente_api'] y no entra al diseño.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "scripts" / "production"
for p_ in (str(ROOT), str(ROOT / "scripts" / "analysis"), str(MAT)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import gen_designs as G1  # noqa: E402
from gen_designs import (M, write_rows, write_sa, base_params, preset_diff, active_rules, ABLATE, FIELDS,  # noqa: E402,F401
                         stagevars, refine, budget, measure, smoke, stamp)
from u25.params import Params, PRESETS  # noqa: E402

API = json.load(open(MAT / "v2_api.json"))
G1.C_RUN = 0.70           # s de CPU por corrida de C1 medidos en la Fase 2b (escala 0,40–0,48 sobre 1,5)
SEEDS_V2 = {k: v + 200 for k, v in G1.SEEDS.items()}
SEEDS_V2.update(nulos=25330, rob=25331, robmu=25332, ci=25333, size=25334, q7aw=25335, lhsx=25336)
ORDER_V2 = ["base", "c1c2", "ref", "nulos", "abl1", "trans", "amax", "rob", "ci", "abl2",
            "size", "q7aw", "p1", "p1aw", "p2", "p5", "lhs", "lhs2", "lhsx", "morris", "sobol",
            "robmu", "p4", "trgrid", "refine", "p6", "sobolx"]
SOBOL_FORCE = ["inherit_weight", "rec", "plast_age_max"]       # w, r y a_w entran siempre
SOBOL_V1_SEL = ["m_c", "crowd_aversion", "alpha", "refuge_frac", "refuge_capacity", "eta", "gamma", "crowd_power"]
SOBOL_K2 = 11
T_FIX = 95                # tiempo absoluto común de muestreo de los transplantes (semanas; ≈ 1,75 t_pk de C1 v1)
BAND = (0.2, 0.5)


def api_ok(kw: dict) -> list[str]:
    """Campos de kw que no son campos de Params (vacío = disponible)."""
    return [k for k in kw if k not in FIELDS]


def add_rows(rows, pend, label, kw):
    miss = api_ok(kw)
    if miss:
        pend.append(dict(brazo=label, faltan=miss))
    else:
        rows.append((label, kw))


def gen_all_v2(preset: str, out: Path):
    G1.SEEDS.clear(); G1.SEEDS.update(SEEDS_V2)
    made = G1.gen_all(preset, out)
    P = base_params(preset)
    tag = preset.replace("calhoun_", "")
    rules = active_rules(P)
    pend = []
    alt_p = "calhoun_C1p" if preset == "calhoun_C1" else None       # variante mínima (sin AV)
    bases = [(tag, {})] + ([("C1p", preset_diff(alt_p, P))] if alt_p in PRESETS else [])

    # base: C1, C1′, C0 y alpha = 0 (serie completa; ensamble de referencia de las cifras del texto)
    rows = [(tag, {})]
    if alt_p in PRESETS:
        rows.append(("C1p", preset_diff(alt_p, P)))
    if preset != "calhoun_C0" and "calhoun_C0" in PRESETS:
        rows.append(("C0 (control)", preset_diff("calhoun_C0", P)))
    rows.append((f"{tag} alpha=0 (control O14)", {"alpha": 0.0}))
    made["base"] = write_rows(out / "base", rows, P, M("base", 200, 600, 1.8, 0, "Q3", series_every=1,
                                                      full_series=True, seed=SEEDS_V2["base"]))

    # trans (V2): mismos brazos que la v1 + C1′; protocolo trans_v2 (Dfix, Dtp, BAND)
    arms = [(tag, {})]
    for drop in (["R1"], ["R2"], ["R1", "R2"]):
        if all(r in rules for r in drop):
            kw = {}
            for r in drop:
                kw.update(ABLATE[r])
            arms.append((tag + " -" + " -".join(drop), kw))
    cross = {}
    if "R1" in rules and "R2" in rules:         # Q1: transplante cruzado pareado (madre = candidato)
        cross[len(arms)] = 0
        arms.append((f"X: madre {tag}, colonia -R1 -R2 rec={P.rec:g}", {**ABLATE["R1"], **ABLATE["R2"], "rec": P.rec}))
    if preset != "calhoun_C0" and "calhoun_C0" in PRESETS:
        arms.append(("C0 (control)", preset_diff("calhoun_C0", P)))
    alt = G1.ALT.get(preset)
    if alt and alt in PRESETS:
        arms.append((alt.replace("calhoun_", "") + " (contraste)", preset_diff(alt, P)))
    if alt_p in PRESETS:
        arms.append(("C1p", preset_diff(alt_p, P)))
    made["trans"] = write_rows(out / "trans", arms, P, M("trans", 60, 300, 6.0, 0, "Q1", kind="transplant2",
                                                        T_main=300, T_after=150, T_fix=T_FIX, band=list(BAND),
                                                        cont_after=40, seed=SEEDS_V2["trans"]))
    json.dump({str(k): v for k, v in cross.items()}, open(out / "trans" / "cross.json", "w"))

    # nulos (i): poder discriminante de cada criterio (C0 y alpha = 0 vienen de base)
    nb = Params(**PRESETS["notebook"]) if "notebook" in PRESETS else Params()
    longevo = {k: getattr(nb.replace(a_max=120.0, repro_age_max=80), k) for k in sorted(FIELDS)
               if getattr(nb.replace(a_max=120.0, repro_age_max=80), k) != getattr(P, k)
               and k not in ("n_males", "n_females", "initial_age", "max_pop", "T", "seed")}
    rows = [(f"{tag} gamma=0 (sin hacinamiento)", {"gamma": 0.0}),
            (f"{tag} -R2 (w=1, sin aprendizaje social)", dict(ABLATE["R2"])),
            ("longevo (notebook a_max=120, CI de C1)", longevo)]
    allk = {}
    for r in rules:
        allk.update(ABLATE[r])
    rows.append((f"{tag} sin reglas (" + " ".join("-" + r for r in rules) + ")", allk))
    made["nulos"] = write_rows(out / "nulos", rows, P, M("nulos", 200, 600, 2.0, 0, "V2-nulos", seed=SEEDS_V2["nulos"]))

    # rob (iii): robustez demográfica, de actualización y de implementación de R7, en C1 y C1′
    rows = []
    for b, bkw in bases:
        add_rows(rows, pend, f"{b} R7 legacy (v1)", {**bkw, **API["R7_legacy"]})
        for mu in API["mu_bg"]["rob"]:
            add_rows(rows, pend, f"{b} mu_bg={mu:g}", {**bkw, API["mu_bg"]["field"]: mu})
        add_rows(rows, pend, f"{b} nido={API['nido'].get('nest_period', 3)}", {**bkw, **API["nido"]})
        add_rows(rows, pend, f"{b} async", {**bkw, **API["async"]})
        add_rows(rows, pend, f"{b} async blanda", {**bkw, **API["async"], **API["R7_soft"]})
        for s0 in API.get("s0_valores", [0.3]):
            add_rows(rows, pend, f"{b} s0={s0:g}", {**bkw, "s_newborn_base": float(s0)})
    if rows:
        made["rob"] = write_rows(out / "rob", rows, P, M("rob", 200, 700, 2.4, 0, "V2-rob", seed=SEEDS_V2["rob"]))
    rows = []
    for b, bkw in bases:
        for mu in API["mu_bg"]["barrido"]:
            add_rows(rows, pend, f"{b} mu_bg={mu:g}", {**bkw, API["mu_bg"]["field"]: mu})
    if rows:
        made["robmu"] = write_rows(out / "robmu", rows, P, M("robmu", 60, 700, 1.6, 2, "V2-rob", seed=SEEDS_V2["robmu"],
                                                            axes=[API["mu_bg"]["field"], "crowd_aversion"]))

    # ci (iv): condición inicial no sincrónica
    rows = []
    for b, bkw in bases:
        for nm, kw in API["ci_variantes"].items():
            add_rows(rows, pend, f"{b} CI {nm}", {**bkw, **kw})
    if rows:
        made["ci"] = write_rows(out / "ci", rows, P, M("ci", 200, 700, 1.8, 0, "V2-ci", series_every=1,
                                                      seed=SEEDS_V2["ci"]))

    # size: densidad fija N0/L^2 = 400/900
    rows = []
    for L in (15, 20, 30, 45, 60):
        n0 = int(round(400 * (L / 30) ** 2 / 2))
        rows.append((f"L={L} N0={2 * n0}", dict(grid_size=L, n_males=n0, n_females=n0,
                                                max_pop=int(P.max_pop * (L / 30) ** 2))))
    made["size"] = write_rows(out / "size", rows, P, M("size", 60, 600, 5.0, 1, "V2-size", seed=SEEDS_V2["size"],
                                                      axes=["grid_size", "n_males"]))

    # q7aw (vii): Λ = r·a_w con a_w variable, sin hacinamiento (γ = 0)
    rows = []
    for aw in (6.0, 12.0, 24.0):           # grilla sobre la frontera de la v1 (w <= 0,1, Λ ∈ [0,24; 1,2])
        for lam in (0.24, 0.36, 0.48, 0.6, 0.84, 1.2):
            for w in (0.0, 0.05, 0.1):
                rows.append((f"a_w={aw:g} Lam={lam:g} w={w:g}", dict(gamma=0.0, plast_age_max=aw, rec=lam / aw,
                                                                     inherit_weight=w)))
    made["q7aw"] = write_rows(out / "q7aw", rows, P, M("q7aw", 20, 600, 6.0, 1, "V2-Q7aw", seed=SEEDS_V2["q7aw"],
                                                      axes=["rec", "inherit_weight"], layer="plast_age_max"))

    # sobol: k = 11 con la selección de la v1 (si falta Morris) ∪ {w, r, a_w}
    made["sobol"] = gen_sobol_v2(out / "sobol", union_names(SOBOL_V1_SEL, G1.factor_names(P)), src="v1+forzados")

    # prioridades de la V2
    for st, pr in dict(lhs2=1, size=1, q7aw=1, morris=1).items():
        f = out / st / "stage.json"
        if f.exists():
            d = json.load(open(f)); d["priority"] = pr; json.dump(d, open(f, "w"), indent=1, ensure_ascii=False)

    man = json.load(open(out / "manifest.json"))
    man.update(order=[s for s in ORDER_V2 if (out / s).exists() or s in ("lhsx", "refine", "sobolx")],
               version="v2", pendiente_api=pend, api=API.get("estado", ""), t_fix=T_FIX, band=list(BAND),
               generated_by="scripts/production/gen_designs_v2.py")
    json.dump(man, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)
    stamp(out)
    return made, pend


def _restore(kw, P):
    """Valores de design.csv (strings, NaN = inf) a los tipos de Params."""
    out = {}
    for k, v in kw.items():
        cur = getattr(P, k)
        if isinstance(v, float) and np.isnan(v):
            out[k] = math.inf
        elif isinstance(cur, bool):
            out[k] = str(v).lower() in ("true", "1")
        elif isinstance(cur, int):
            out[k] = int(round(float(v)))
        elif isinstance(cur, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def union_names(sel, allowed, k=SOBOL_K2, ranking=None):
    """sel (ordenado) ∪ SOBOL_FORCE, completado con `ranking` hasta k factores."""
    names = [x for x in SOBOL_FORCE if x in allowed]
    for x in list(sel) + list(ranking or []):
        if len(names) >= k:
            break
        if x in allowed and x not in names:
            names.append(x)
    return names


def gen_sobol_v2(out: Path, names, src):
    meta = M("sobol", 3, 400, 2.0, 1, "", crn=True, seed=SEEDS_V2["sobol"], reps_min=2, source=src, ext_reps=6)
    return write_sa(out, "sobol", names, meta, N=256, seed=4)


def sobol_from_morris_v2(preset, des: Path, morris_run: Path):
    sel = morris_run / "analisis" / "seleccion_sobol.json"
    if not sel.exists():
        print(f"no hay {sel}; se usa el Sobol por defecto (selección v1 ∪ forzados)")
        return None
    rank = json.load(open(sel))["factores"]
    names = union_names(rank[:G1.SOBOL_K], G1.factor_names(base_params(preset)), ranking=rank)
    out = des / "sobol_auto"
    if out.exists() and json.load(open(out / "stage.json"))["factors"] == names:
        print(f"sobol_auto ya existe con {names}")
        return out
    gen_sobol_v2(out, names, src=str(sel))
    stamp(des)
    print(f"sobol_auto: {names}")
    return out


def lhsx(preset, des: Path, runs, lo=0.4, hi=0.95, max_points=60, reps=40):
    """(vi) Réplicas extra en los puntos frontera de los LHS: 0,4 <= k/n <= 0,95 en el criterio primario
    (regla fija antes de ver los datos de la V2). Las réplicas nuevas usan semillas propias."""
    P = base_params(preset)
    rows = []
    for rd in runs:
        tp = Path(rd) / "analisis" / "tabla_puntos.csv"
        if not tp.exists():
            continue
        T = pd.read_csv(tp)
        col = "f_PRIMARIO" if "f_PRIMARIO" in T else "f_O17_run"
        B = T[(T[col] >= lo) & (T[col] <= hi)].copy()
        B["dist"] = (B[col] - 0.8).abs()
        dcols = json.load(open(Path(rd) / "meta.json"))["design_columns"]
        for _, r in B.sort_values("dist").iterrows():
            kw = _restore({c: r[c] for c in dcols}, P)
            rows.append((f"{Path(rd).name}:{int(r['point'])}", kw))
    rows = rows[:max_points]
    if not rows:
        print("lhsx: no hay puntos frontera (o faltan análisis)")
        return None
    out = des / "lhsx"
    write_rows(out, rows, P, M("lhsx", reps, 400, 2.0, 1, "V2-region", seed=SEEDS_V2["lhsx"]))
    stamp(des)
    print(f"lhsx: {len(rows)} puntos -> {out}")
    return out


def roundtrip(des: Path, preset: str) -> list[str]:
    """Verifica que cada fila de cada design.csv produzca, vía u25.sweep.design_to_params, los valores
    escritos (p. ej. refuge_hard='legacy' como string; NaN = inf sólo en plast_age_max)."""
    from u25.sweep import design_to_params
    P = base_params(preset)
    probs = []
    for sd in sorted(des.iterdir()):
        if not (sd / "stage.json").exists():
            continue
        st = json.load(open(sd / "stage.json"))
        base = base_params(st["preset"]) if st.get("preset") else P
        d = pd.read_csv(sd / "design.csv")
        try:
            pm = design_to_params(d, base)
        except Exception as e:  # noqa: BLE001
            probs.append(f"{sd.name}: {type(e).__name__}: {e}")
            continue
        for _, r in d.iterrows():
            q = pm[int(r["point"])]
            for c in d.columns:
                if c == "point":
                    continue
                v, got = r[c], getattr(q, c)
                if isinstance(v, float) and np.isnan(v):
                    ok = (isinstance(got, float) and math.isinf(got)) or got is None
                elif isinstance(got, bool):
                    ok = (str(v).strip().lower() in ("true", "1", "1.0")) == got
                elif str(v).lower() in ("true", "false"):
                    ok = str(got).lower() == str(v).lower() or (str(v).lower() == "false" and str(got).lower() in ("off", "false"))
                elif isinstance(got, (int, float)) and not isinstance(got, bool):
                    try:
                        ok = math.isclose(float(v), float(got), rel_tol=1e-9, abs_tol=1e-12)
                    except ValueError:
                        ok = False
                else:
                    ok = str(v) == str(got)
                if not ok:
                    probs.append(f"{sd.name} punto {int(r['point'])}: {c} escrito {v!r} -> Params {got!r}")
    return probs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("stagevars", "refine", "measure", "smoke", "budget"):
        G1.SEEDS.clear(); G1.SEEDS.update(SEEDS_V2)       # refine con la semilla de la V2
        return G1.main(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "check", "sobol-from-morris", "lhsx"])
    ap.add_argument("--preset", default="calhoun_C1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--des", default=None)
    ap.add_argument("--morris", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    des = Path(a.des or a.out or (MAT / "designs_v2" / a.preset))
    if a.cmd == "all":
        made, pend = gen_all_v2(a.preset, des)
        tot = 0.0
        man = json.load(open(des / "manifest.json"))
        for s in man["order"]:
            f = des / s / "stage.json"
            if not f.exists():
                continue
            st = json.load(open(f))
            h = st["runs"] * st["c_cpu"] / 4 / 3600
            tot += h
            print(f"{s:7s} P{st['priority']} {st['kind']:11s} {st['n_points']:5d} pts x {st['reps']:3d} rép = "
                  f"{st['runs']:6d}, ~{h * 60:5.1f} min de pared (4 núcleos)")
        print(f"total estimado (sin lhsx/refine/sobolx): {tot:.2f} h -> {des}")
        if pend:
            print("PENDIENTE_API:", json.dumps(pend, ensure_ascii=False))
    elif a.cmd == "check":
        G1.check(a.preset, des)
        probs = roundtrip(des, a.preset)
        man = json.load(open(des / "manifest.json"))
        if man.get("pendiente_api"):
            print("aviso: brazos pendientes de API:", json.dumps(man["pendiente_api"], ensure_ascii=False))
        if probs:
            print("PROBLEMAS (viaje CSV -> Params):\n  " + "\n  ".join(probs[:40]))
            sys.exit(2)
        print("ok: viaje CSV -> Params")
    elif a.cmd == "sobol-from-morris":
        r = sobol_from_morris_v2(a.preset, des, Path(a.morris))
        print(r or "")
    elif a.cmd == "lhsx":
        lhsx(a.preset, des, a.runs)


if __name__ == "__main__":
    main()
