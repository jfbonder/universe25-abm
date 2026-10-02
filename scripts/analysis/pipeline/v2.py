"""Análisis de la V2: etapas nuevas, re-puntuaciones e informe consolidado.

    python -m pipeline v2 STAGE RUN_DIR            # nulos, rob, robmu, ci, size, q7aw, lhsx, trans (transplant2)
    python -m pipeline v2 final RESULTS_DIR         # informe_v2.md + tablas en RESULTS_DIR/f2b_final/v2/

Reglas de decisión fijadas antes de las corridas de producción:
  * criterio "genérico": pasa (f >= 0,8) en al menos dos de los nulos verdaderos (γ = 0, −R2, longevo, sin reglas);
    "no discrimina frente al nulo M": f_E(M) >= 0,8 (el nulo cumple el umbral esencial); "discrimina":
    cota superior de Wilson de f_E(M) < 0,8 y Δ = f_E(C1) − f_E(M) con IC de Newcombe que excluye 0;
  * variante de robustez "preserva el resultado": O17p >= 0,8 (estimación puntual), informando el IC;
  * región factible: P(p_i >= 0,8 | datos) con prior beta centrado en el emulador logístico (empírico-Bayes).
Nunca falla por una etapa o columna faltante: lo que no se puede calcular queda como NaN y se anota.
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import multiple, survival
from .f2b import (ESS, OBJ, _fisher, firth_logit, dispersion, frac, label_of, load, point_table)
from .report import _md

V2_CODES = ["O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p", "O17_run", "O17p", "O17ps", "O10b", "O8", "O8b",
            "O9", "O11b", "O14"]
THETAS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]


# =========================================================================== utilidades
def newcombe(k1, n1, k2, n2):
    """IC 95 % de Newcombe (híbrido de Wilson) para p1 − p2."""
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = survival.wilson(k1, n1)
    l2, u2 = survival.wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def add_strict(s):
    """O17ps: O17p y además todos los esenciales aplicables con N0 = 400 en PASS (sin NA: O11 o O10 vacíos no
    cuentan como cumplidos). Se agrega como columna obj_O17ps."""
    if "obj_O17p" not in s:
        return s
    ok = s["obj_O17p"].astype(str) == "PASS"
    for c in ("O4", "O5", "O6", "O7", "O10", "O11", "O13"):
        if f"obj_{c}" in s:
            ok &= s[f"obj_{c}"].astype(str) == "PASS"
    s["obj_O17ps"] = np.where(ok, "PASS", "FAIL")     # en la v2 coincide con O17p (estricto); control
    return s


def _run(root, stage, sfx=""):
    d = Path(root) / f"f2b_{stage}{sfx}"
    if not (d / "summary.csv").exists():
        return None
    r = load(d)
    add_strict(r["summary"])
    return r


def _rows_of(run, labels=None):
    """{etiqueta: DataFrame de corridas} de una etapa."""
    s = run["summary"]
    out = {}
    for pt in sorted(s.point.unique()):
        lab = label_of(run, pt)
        if labels is None or lab in labels:
            out[lab] = s[s.point == pt]
    return out


def fr(g, code, na_fail=False):
    """Fracción PASS con Wilson. Por defecto sobre las corridas evaluables (PASS/MARG/FAIL); con na_fail=True sobre
    todas (NA y ERR cuentan como 'no pasa'; así se puntúa la tabla de poder discriminante)."""
    col = f"obj_{code}"
    if col not in g:
        return dict(k=0, n=0, f=np.nan, lo=np.nan, hi=np.nan, n_na=np.nan)
    v = g[col].astype(str)
    if na_fail or code in ("O17_run", "O17p", "O17ps"):
        k, n = int((v == "PASS").sum()), int(len(v))
        lo, hi = survival.wilson(k, n) if n else (np.nan, np.nan)
        d = dict(k=k, n=n, f=k / n if n else np.nan, lo=lo, hi=hi)
    else:
        d = frac(g[col], code)
    d["n_na"] = int((~v.isin(["PASS", "MARG", "FAIL"])).sum())
    return d


def _fmt(d):
    if not d or not np.isfinite(d.get("f", np.nan)):
        return "NA"
    return f"{d['f']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]" + (f" ({d['n_na']} NA)" if d.get("n_na") else "")


# =========================================================================== (i) poder discriminante
NULL_SPEC = [  # (columna, etapa, etiqueta en la etapa)
    ("C1", "base", "C1"), ("C1′", "base", "C1p"),
    ("γ=0", "nulos", "C1 gamma=0 (sin hacinamiento)"), ("α=0", "base", "C1 alpha=0 (control O14)"),
    ("C0 (−R7)", "base", "C0 (control)"), ("−R2", "nulos", "C1 -R2 (w=1, sin aprendizaje social)"),
    ("longevo", "nulos", "longevo (notebook a_max=120, CI de C1)"),
    ("sin reglas", "nulos", None), ("notebook", "ref", "notebook alpha=1.4"),
]
PRIMARY_NULLS = ["γ=0", "−R2", "longevo", "sin reglas"]      # nulos verdaderos; α=0 y C0 son ablaciones parciales
# respaldo (vista previa con los datos de la v1, sin la etapa nulos): abl2, abl1 y ref, con n menor y, en ref,
# otra condición inicial (N0 del notebook) y T = 1000
NULL_FALLBACK = {"γ=0": ("abl2", "C1 gamma=0 (sin hacinamiento)"), "−R2": ("abl1", "C1 -R2"),
                 "longevo": ("ref", "longevo"), "sin reglas": ("abl2", "C1 -R1 -R2 -AV -R7")}


def discriminant_table(root, sfx=""):
    groups = {}
    for col, st, lab in NULL_SPEC:
        r = _run(root, st, sfx)
        if r is None:
            continue
        rows = _rows_of(r)
        if lab is None:
            lab = next((k for k in rows if "sin reglas" in k), None)
        if lab in rows:
            groups[col] = rows[lab]
    used_fb = []
    for col, (st, lab) in NULL_FALLBACK.items():
        if col not in groups:
            r = _run(root, st, sfx)
            if r is not None and lab in _rows_of(r):
                groups[col] = _rows_of(r)[lab]
                used_fb.append(f"{col} ← {st}:{lab}")
    if "C1" not in groups:
        return None, ["(sin el ensamble de referencia de C1)"]
    ref = groups["C1"]
    recs = []
    for code in V2_CODES:
        dc = fr(ref, code, na_fail=True)
        for col, g in groups.items():
            d = fr(g, code, na_fail=True)
            r = dict(criterio=code, modelo=col, k=d["k"], n=d["n"], f=d["f"], lo=d["lo"], hi=d["hi"], n_na=d["n_na"])
            if col != "C1" and d["n"] and dc["n"]:
                lo, hi = newcombe(dc["k"], dc["n"], d["k"], d["n"])
                r.update(delta=dc["f"] - d["f"], delta_lo=lo, delta_hi=hi,
                         p=_fisher(dc["k"], dc["n"], d["k"], d["n"], "greater"),
                         pasa_nulo=bool(d["f"] >= 0.8), discrimina=bool(d["hi"] < 0.8 and lo > 0))
            recs.append(r)
    D = pd.DataFrame(recs)
    if "p" in D:
        m = D["p"].notna()
        D.loc[m, "p_bh"] = multiple.bh(D.loc[m, "p"].to_numpy())
    # resumen por criterio
    summ = []
    for c_ in ("pasa_nulo", "discrimina"):
        D[c_] = D[c_].astype("boolean").fillna(False).astype(bool) if c_ in D else False
    for code, g in D[D.modelo != "C1"].groupby("criterio", sort=False):
        gp = g[g.modelo.isin(PRIMARY_NULLS)]
        summ.append(dict(criterio=code, f_C1=float(D[(D.criterio == code) & (D.modelo == "C1")]["f"].iloc[0]),
                         pasa_en_nulos=", ".join(gp.loc[gp.pasa_nulo.astype(bool), "modelo"]) or "—",
                         n_nulos_que_pasa=int(gp.pasa_nulo.astype(bool).sum()),
                         generico=bool(gp.pasa_nulo.astype(bool).sum() >= 2),
                         discrimina_frente_a=", ".join(g.loc[g.discrimina.astype(bool), "modelo"]) or "—"))
    S = pd.DataFrame(summ)
    W = D.pivot_table(index="criterio", columns="modelo", values="f", aggfunc="first", sort=False)
    W = W[[c for c, _, _ in NULL_SPEC if c in W.columns]]
    lines = ["## (i) Poder discriminante de cada criterio frente a modelos nulos", "",
             "f = fracción PASS sobre TODAS las corridas (NA y ERR cuentan como 'no pasa'; el número de NA va en la"
             " tabla larga). Genérico: pasa"
             " (f ≥ 0,8) en al menos dos de los nulos verdaderos (γ = 0, −R2, longevo, sin reglas). Discrimina frente a M:"
             " cota superior de Wilson de f(M) < 0,8 y IC de Newcombe de f(C1) − f(M) que excluye 0.", "",
             _md(W.reset_index(), index=False, floatfmt=".2f"), "", _md(S, index=False), ""]
    if used_fb:
        lines += ["Respaldo (sin la etapa nulos): " + "; ".join(used_fb), ""]
    return dict(tabla=D, resumen=S, ancho=W), lines


# =========================================================================== (ii) re-puntuaciones
VAR_PREF = [("primario v2", ""), ("evaluador v1", "v1_"), ("argmax", "ta_"), ("fin meseta", "te_"),
            ("m > m_c", "mc_"), ("O5 con t_last", "l5_"), ("O4 v1", "o4a_")]


def rescoring_table(root, sfx=""):
    """O5, O8, O11, O13p, O17_run y O17p con t_pk, t'_pk, centro de la meseta y umbral fijo."""
    recs = []
    for st in ("base", "c1c2", "nulos", "abl1", "rob", "ci"):
        r = _run(root, st, sfx)
        if r is None:
            continue
        for lab, g in _rows_of(r).items():
            for vname, pref in VAR_PREF:
                for code in ("O4", "O5", "O8", "O10", "O11", "O13", "O13p", "O17_run", "O17p"):
                    col = f"obj_{pref}{code}"
                    if col in g:
                        d = frac(g[col], "O17p" if code.startswith("O17") else code)
                        recs.append(dict(etapa=st, brazo=lab, variante=vname, criterio=code, k=d["k"], n=d["n"],
                                         f=d["f"], lo=d["lo"], hi=d["hi"]))
    R = pd.DataFrame(recs)
    if not len(R):
        return None, ["## (ii) Re-puntuaciones: sin datos", ""]
    P = R[R.criterio == "O17p"].pivot_table(index=["etapa", "brazo"], columns="variante", values="f", aggfunc="first")
    P = P[[v for v, _ in VAR_PREF if v in P.columns]]
    lines = ["## (ii) Re-puntuaciones: evaluador primario v2 (t'_pk = plateau_start, O5 con t_B99,"
             " O4 v2, m > 8 fijo, O10/O11 v2) frente al evaluador v1 completo y a cada opción por separado", "",
             "O17p por brazo (fracción de corridas):", "", _md(P.reset_index(), index=False, floatfmt=".3f"), ""]
    for code in ("O4", "O5", "O8", "O10", "O11", "O13p"):
        Q = R[R.criterio == code].pivot_table(index=["etapa", "brazo"], columns="variante", values="f", aggfunc="first")
        Q = Q[[v for v, _ in VAR_PREF if v in Q.columns]]
        if len(Q):
            lines += [f"{code}:", "", _md(Q.reset_index(), index=False, floatfmt=".3f"), ""]
    return R, lines


def o17p_theta(g, theta, tpk_col_suffix=""):
    """Re-puntúa O13p y O17p con el umbral de clase θ (en lugar de 0,10) desde las fracciones por instante.
    O17_run se mantiene; O13 instantáneo también se re-puntúa (BOi/CR) porque es esencial."""
    need = [f"x_BO4_{k}" for k in (0, 10, 20)]
    if any(c not in g for c in need) or "obj_O17_run" not in g:
        return None
    ok13 = np.zeros(len(g), bool); ok13p = np.zeros(len(g), bool); seen = np.zeros(len(g), bool)
    crn = "CRf" if _primary_fixed_m() else "CR"          # clase hacinada del evaluador primario
    for k in (0, 10, 20):
        bo4, boi, cr = (g[f"x_{n}_{k}"].to_numpy(float) for n in ("BO4", "BOi", crn))
        nad = g[f"x_nad_{k}"].to_numpy(float)
        v = np.isfinite(bo4) & np.isfinite(cr) & (nad >= 20)
        seen |= v
        ok13p |= v & (bo4 >= theta) & (cr >= theta)
        ok13 |= v & (boi >= theta) & (cr >= theta)
    # O17_run con O13 re-puntuado: los demás esenciales PASS o NA y ningún anti-patrón disparado
    others = _only_o13_fails(g)
    o17 = others & (ok13 | ~seen)
    return float((o17 & ok13p).mean()), float(ok13p[seen].mean()) if seen.any() else np.nan


def _primary_fixed_m():
    try:
        api = json.load(open(Path(__file__).resolve().parents[2] / "production" / "v2_api.json"))
        return api["evaluador"].get("primario", {}).get("crowd_class_threshold") is not None
    except Exception:  # noqa: BLE001
        return False


def _only_o13_fails(g):
    ess = [c for c in ("O4", "O5", "O6", "O7", "O10", "O11") if f"obj_{c}" in g]
    ok = np.ones(len(g), bool)
    for c in ess:
        ok &= g[f"obj_{c}"].astype(str).isin(["PASS", "NA"]).to_numpy()
    for a in ("A1", "A2", "A3", "A4"):
        if f"obj_{a}" in g:
            ok &= (g[f"obj_{a}"].astype(str) != "TRIGGERED").to_numpy()
    return ok


def theta_curves(root, sfx=""):
    """O17p en función del umbral de clase θ, y la necesidad de cada regla en función de (θ, τ)."""
    recs = []
    for st in ("base", "c1c2", "abl1", "rob"):
        r = _run(root, st, sfx)
        if r is None:
            continue
        for lab, g in _rows_of(r).items():
            for th in THETAS:
                v = o17p_theta(g, th)
                if v is not None:
                    recs.append(dict(etapa=st, brazo=lab, theta=th, O17p=v[0], O13p=v[1], n=len(g)))
    C = pd.DataFrame(recs)
    if not len(C):
        return None, ["## Curvas de umbral θ: sin datos", ""]
    lines = ["## O17p en función del umbral de clase θ (re-puntuado sin simular)", "",
             _md(C.pivot_table(index=["etapa", "brazo"], columns="theta", values="O17p").reset_index(), index=False,
                 floatfmt=".2f"), ""]
    # necesidad de cada regla (abl1): f(C1 − r) < τ para cada (θ, τ); y C1 por encima de τ
    A = C[C.etapa == "abl1"]
    if len(A):
        rows = []
        for (lab, th), g in A.groupby(["brazo", "theta"]):
            for tau in TAUS:
                rows.append(dict(brazo=lab, theta=th, tau=tau, O17p=float(g.O17p.iloc[0]), bajo_tau=bool(g.O17p.iloc[0] < tau)))
        N = pd.DataFrame(rows)
        piv = N.pivot_table(index=["brazo", "theta"], columns="tau", values="bajo_tau", aggfunc="first")
        lines += ["## Necesidad de cada regla en función de θ (filas) y del umbral de ensamble τ (columnas):"
                  " True = la ablación queda por debajo de τ", "", _md(piv.reset_index(), index=False), ""]
    return C, lines


# =========================================================================== (iii)/(iv) robustez y CI
ROB_CODES = ["O17p", "O17ps", "O17_run", "O13p", "O11", "O10", "O5", "O8", "O4"]
ROB_X = ["x_rho_pk", "x_f0_pk", "x_BO4", "x_CR", "x_plateau", "x_tpk2", "t_peak", "x_D_last", "x_tB90", "x_t_s05ad"]


def compare_arms(groups: dict, refs: dict):
    """groups: {etiqueta: corridas}; refs: {prefijo de etiqueta: (etiqueta_ref, corridas_ref)}."""
    recs = []
    for lab, g in groups.items():
        key = next((k for k in sorted(refs, key=len, reverse=True) if lab.startswith(k)), None)
        if key is None:
            continue
        rlab, rg = refs[key]
        r = dict(brazo=lab, referencia=rlab, n=len(g))
        for code in ROB_CODES:
            d, dr = fr(g, code), fr(rg, code)
            r[f"{code}"] = _fmt(d)
            if d["n"] and dr["n"]:
                lo, hi = newcombe(d["k"], d["n"], dr["k"], dr["n"])
                r[f"d_{code}"] = d["f"] - dr["f"]
                r[f"d_{code}_ic"] = f"[{lo:+.2f}, {hi:+.2f}]"
                r[f"p_{code}"] = _fisher(d["k"], d["n"], dr["k"], dr["n"], "two-sided")
        for c in ROB_X:
            if c in g and c in rg:
                x, y = pd.to_numeric(g[c], errors="coerce").dropna(), pd.to_numeric(rg[c], errors="coerce").dropna()
                if len(x) > 3 and len(y) > 3:
                    r[f"{c}_med"] = float(x.median())
                    r[f"{c}_med_ref"] = float(y.median())
                    r[f"p_{c}"] = float(stats.mannwhitneyu(x, y).pvalue)
        ev = g["extinct"].to_numpy(bool)
        km = survival.kaplan_meier(np.where(ev, g.t_ext, g.t_end).astype(float), ev)
        r["t_ext_med"] = survival.km_median(km)["median"]
        r["P_ext"] = float(ev.mean())
        r["preserva_O17p_0.8"] = bool(fr(g, "O17p")["f"] >= 0.8)
        recs.append(r)
    return pd.DataFrame(recs)


def _refs(root, sfx=""):
    b = _run(root, "base", sfx)
    if b is None:
        return {}
    rows = _rows_of(b)
    out = {}
    if "C1" in rows:
        out["C1 "] = ("C1 (base)", rows["C1"])
    if "C1p" in rows:
        out["C1p "] = ("C1p (base)", rows["C1p"])
    return out


def analyze_rob(run, out, lines, root=None):
    root = root or run["dir"].parent
    sfx = run["dir"].name.split(run["stage"].get("stage", "rob"))[-1] if run.get("stage") else ""
    add_strict(run["summary"])
    C = compare_arms(_rows_of(run), _refs(root, sfx))
    if len(C):
        pc = [c for c in C.columns if c.startswith("p_")]
        allp = C[pc].to_numpy(float).ravel()
        m = np.isfinite(allp)
        bh = np.full(allp.shape, np.nan)
        bh[m] = multiple.bh(allp[m])
        C[[c.replace("p_", "pbh_") for c in pc]] = bh.reshape(len(C), len(pc))
        C.to_csv(out / "contraste_variantes.csv", index=False)
        show = ["brazo", "referencia", "n", "O17p", "d_O17p", "d_O17p_ic", "O17ps", "O13p", "O11", "O5", "O8",
                "x_rho_pk_med", "x_rho_pk_med_ref", "x_plateau_med", "t_ext_med", "preserva_O17p_0.8"]
        lines += ["## Variantes frente a la referencia (base, misma semilla de etapa distinta; Fisher bilateral,"
                  " Newcombe; BH sobre la tabla en contraste_variantes.csv)", "",
                  _md(C[[c for c in show if c in C]], index=False, floatfmt=".3g"), ""]
    return C


# =========================================================================== (v) transplantes V2
def analyze_trans2(run_dir, out, lines):
    d = Path(run_dir)
    tr = pd.read_csv(d / "transplant.csv")
    labels = json.load(open(d / "labels.json")) if (d / "labels.json").exists() else None
    rows = []
    for (pt, ph), g in tr.groupby(["point", "phase"]):
        x = g["success"].dropna().astype(float)
        r = dict(point=int(pt), brazo=labels[int(pt)] if labels else f"p{pt}", fase=ph, n=len(x),
                 k=int(x.sum()), P_T=x.mean() if len(x) else np.nan)
        if len(x):
            r["lo"], r["hi"] = survival.wilson(int(x.sum()), len(x))
        for c in ("t_snap", "s_med", "s_max", "frac_s0", "age_med", "age_min", "age_max", "s_desc_ad", "s_desc_mat",
                  "n_desc_ad"):
            if c in g:
                v = pd.to_numeric(g[c], errors="coerce")
                r[f"{c}_mediana"] = float(v.median()) if v.notna().any() else np.nan
        if "s_max" in g:
            r["s_max_max"] = float(pd.to_numeric(g["s_max"], errors="coerce").max())
            r["frac_ensayos_s0_todos"] = float((pd.to_numeric(g["frac_s0"], errors="coerce") == 1).mean())
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(out / "tabla_brazos_fases.csv", index=False)
    lines += ["## Transplantes V2 por brazo y fase (P_T con Wilson; distribución de s y edad de los trasladados)", "",
              _md(A, index=False, floatfmt=".3g"), ""]
    # comparaciones a tiempo igual (Dfix) contra el candidato: Fisher bilateral, BH
    if labels:
        a = 0
        ga = tr[(tr.point == a) & (tr.phase == "Dfix")]["success"].dropna().astype(int)
        recs = []
        for pt in sorted(tr.point.unique()):
            if pt == a:
                continue
            for ph in ("Dfix", "MIXfix_M", "MIXfix_F", "BAND", "BAND_MIX_M", "BAND_MIX_F"):
                ref = tr[(tr.point == a) & (tr.phase == ph)]["success"].dropna().astype(int)
                gb = tr[(tr.point == pt) & (tr.phase == ph)]["success"].dropna().astype(int)
                if len(ref) and len(gb):
                    recs.append(dict(contraste=f"{labels[pt]} vs {labels[a]}", fase=ph, P_T=gb.mean(), P_T_ref=ref.mean(),
                                     p=_fisher(int(gb.sum()), len(gb), int(ref.sum()), len(ref), "two-sided")))
        E = pd.DataFrame(recs)
        if len(E):
            E["p_bh"] = multiple.bh(E["p"].fillna(1).to_numpy())
            E.to_csv(out / "contraste_tiempo_igual.csv", index=False)
            lines += ["## Brazos frente al candidato a tiempo igual (t_fix) y en la banda de s (Fisher bilateral, BH)", "",
                      _md(E, index=False, floatfmt=".3g"), ""]
        # McNemar del cruzado X a tiempo igual (además del preregistrado en t_D)
        b = next((i for i, l in enumerate(labels) if l.startswith("X: madre")), None)
        if b is not None:
            for ph in ("D", "Darg", "Dfix", "BAND"):
                pa = tr[(tr.point == a) & (tr.phase == ph)].set_index("rep")["success"].dropna().astype(int)
                pb = tr[(tr.point == b) & (tr.phase == ph)].set_index("rep")["success"].dropna().astype(int)
                j = pa.index.intersection(pb.index)
                n01 = int(((pa[j] == 0) & (pb[j] == 1)).sum()); n10 = int(((pa[j] == 1) & (pb[j] == 0)).sum())
                p = float(stats.binomtest(n01, n01 + n10, 0.5, alternative="greater").pvalue) if n01 + n10 else 1.0
                lines += [f"- McNemar candidato vs X en {ph}: {int(pa.sum())}/{len(pa)} vs {int(pb.sum())}/{len(pb)},"
                          f" discordantes 0→1 = {n01}, 1→0 = {n10}, p = {p:.3g}"]
            lines += [""]
    return A


# =========================================================================== tamaño
def km_spread(time, ev, B=1000, seed=0):
    """(Q75 − Q25)/Q50 de T_ext con Kaplan–Meier (censura en T) y IC bootstrap percentil."""
    def one(t, e):
        km = survival.kaplan_meier(t, e)
        q = [survival.km_median(km, q_)["median"] for q_ in (0.75, 0.5, 0.25)]   # S <= 0.75 -> Q25 de T
        return (q[2] - q[0]) / q[1] if all(np.isfinite(q)) and q[1] > 0 else np.nan
    t = np.asarray(time, float); e = np.asarray(ev, bool)
    est = one(t, e)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(t), len(t))
        bs.append(one(t[i], e[i]))
    bs = np.asarray(bs, float)
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), bs


def analyze_size(run, out, lines):
    s = run["summary"]
    rows = []
    for pt in sorted(s.point.unique()):
        g = s[s.point == pt]
        ev = g["extinct"].to_numpy(bool)
        t = np.where(ev, g.t_ext, g.t_end).astype(float)
        sp, lo, hi, _ = km_spread(t, ev, B=500)
        tl = pd.to_numeric(g.get("x_t_lastbirth"), errors="coerce")
        r = dict(point=int(pt), label=label_of(run, pt), L=int(g.grid_size.iloc[0]) if "grid_size" in g else np.nan,
                 n=len(g), O17p=_fmt(fr(g, "O17p")), O17ps=_fmt(fr(g, "O17ps")),
                 t_ext_med=survival.km_median(survival.kaplan_meier(t, ev))["median"],
                 iqr_med_KM=sp, iqr_lo=lo, iqr_hi=hi,
                 t_last_med=float(tl.median()), t_last_iqr=float(tl.quantile(0.75) - tl.quantile(0.25)),
                 D_last_med=float(pd.to_numeric(g.get("x_D_last"), errors="coerce").median()),
                 rho_pk_med=float(pd.to_numeric(g.get("x_rho_pk"), errors="coerce").median()),
                 f0_med=float(pd.to_numeric(g.get("x_f0_pk"), errors="coerce").median()),
                 BO4_med=float(pd.to_numeric(g.get("x_BO4"), errors="coerce").median()))
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "tabla_tamano.csv", index=False)
    lines += ["## Tamaño del sistema a densidad fija", "", _md(T, index=False, floatfmt=".3g"), ""]
    for c in ("x_rho_pk", "x_f0_pk", "x_BO4"):
        if c in s:
            grp = [pd.to_numeric(s[s.point == pt][c], errors="coerce").dropna() for pt in sorted(s.point.unique())]
            if all(len(x) > 2 for x in grp):
                kw = stats.kruskal(*grp)
                lines += [f"- Kruskal–Wallis de {c} entre tamaños (densidad fija): H = {kw.statistic:.2f}, p = {kw.pvalue:.3g}"]
    lines += [""]
    return T


# =========================================================================== (vii) Q7 con a_w variable
def analyze_q7aw(run, out, lines, B=300, seed=0):
    """logit P_ext(γ = 0) = b0 + b_w w + b_r log r + b_a log a_w. El mapa generacional (Λ = r a_w) predice
    θ = b_a / b_r = 1. Perfil de θ en logit = b0 + b_w w + b_1 (log r + θ log a_w) (+ cuadrático), LR de θ = 1,
    TOST con márgenes [2/3, 3/2]; frontera en (w, Λ) con IC bootstrap paramétrico de σ̂ y φ̂."""
    T = point_table(run)
    T.to_csv(out / "tabla_puntos.csv", index=False)
    w = T.inherit_weight.astype(float).to_numpy()
    lr_ = np.log(T.rec.astype(float).to_numpy()); la = np.log(T.plast_age_max.astype(float).to_numpy())
    k = T.k_ext.to_numpy(float); n = T.n.to_numpy(float)

    def X_of(th, quad=False):
        z = lr_ + th * la
        cols = [np.ones_like(w), w, z] + ([z ** 2] if quad else [])
        return np.column_stack(cols)
    res = {}
    for quad in (False, True):
        grid = np.linspace(-0.5, 3.0, 141)
        ll = np.array([firth_logit(X_of(t, quad), k, n)[2] for t in grid])
        th = float(grid[np.argmax(ll)])
        Xh = X_of(th, quad)
        beta = firth_logit(Xh, k, n)[0]
        phi, _, _ = dispersion(Xh, k, n, beta)
        stat1 = max(0.0, 2 * (ll.max() - firth_logit(X_of(1.0, quad), k, n)[2]))
        inside = grid[2 * (ll.max() - ll) / phi <= stats.chi2.ppf(0.95, 1)]
        in90 = grid[2 * (ll.max() - ll) / phi <= stats.chi2.ppf(0.90, 1)]
        tost = bool(in90.size and in90.min() >= 2 / 3 and in90.max() <= 1.5)
        res["cuadratico" if quad else "lineal"] = dict(theta=th, ic95_lo=float(inside.min()), ic95_hi=float(inside.max()),
                                                       ic90_lo=float(in90.min()), ic90_hi=float(in90.max()), phi=phi,
                                                       LR_theta1=stat1 / phi, p_theta1=float(stats.chi2.sf(stat1 / phi, 1)),
                                                       tost_equiv_2_3=tost)
    R = pd.DataFrame(res).T
    R.to_csv(out / "q7aw_theta.csv")
    # frontera en (w, Λ): logit = c0 + c_w w + c_Λ Λ -> w + (c_Λ/c_w) Λ = −c0/c_w; forma σ w + φ Λ = 1
    Lam = np.exp(lr_ + la)
    Xf = np.column_stack([np.ones_like(w), w, Lam])
    bf = firth_logit(Xf, k, n)[0]

    def sp(b):
        c = -b[0] / b[1] if b[1] != 0 else np.nan
        return 1 / c, (b[2] / b[1]) / c
    sig, ph = sp(bf)
    rng = np.random.default_rng(seed)
    p_hat = np.clip(k / np.maximum(n, 1), 1e-3, 1 - 1e-3)
    bs = []
    for _ in range(B):
        kb = rng.binomial(n.astype(int), p_hat)
        bs.append(sp(firth_logit(Xf, kb, n)[0]))
    bs = np.array(bs, float)
    lines += ["## (vii) Q7 con a_w variable: ¿r y a_w entran sólo como Λ = r·a_w? (γ = 0)", "",
              "θ = peso de log a_w relativo a log r en el predictor de P_ext; el mapa generacional con φ constante"
              " predice θ = 1. IC de perfil cuasi-binomiales (LR/φ); TOST de equivalencia con márgenes [2/3, 3/2]"
              " (IC 90 % dentro de los márgenes).", "", _md(R.reset_index(), index=False, floatfmt=".3g"), "",
              f"Frontera P_ext = 1/2 en la forma σw + φΛ = 1: σ̂ = {sig:.2f} [{np.nanpercentile(bs[:, 0], 2.5):.2f},"
              f" {np.nanpercentile(bs[:, 0], 97.5):.2f}], φ̂ = {ph:.2f} [{np.nanpercentile(bs[:, 1], 2.5):.2f},"
              f" {np.nanpercentile(bs[:, 1], 97.5):.2f}] (bootstrap paramétrico, {B} réplicas; exploratorio).", ""]
    try:
        W = q7aw_widths(T)
        W.to_csv(out / "q7aw_ancho.csv", index=False)
        lines += ["Ancho de la transición de P_ext en log Λ con w = 0 (f₋ + (f₊ − f₋)/(1 + e^{−(x − x_c)/Δ}),"
                  " máxima verosimilitud binomial, IC bootstrap paramétrico):", "",
                  _md(W, index=False, floatfmt=".3g"), ""]
    except Exception as e:  # noqa: BLE001
        lines += [f"(ancho de transición no disponible: {type(e).__name__}: {e})", ""]
    json.dump(dict(theta=res, sigma=sig, phi=ph, sigma_ic=[float(np.nanpercentile(bs[:, 0], q)) for q in (2.5, 97.5)],
                   phi_ic=[float(np.nanpercentile(bs[:, 1], q)) for q in (2.5, 97.5)]),
              open(out / "q7aw.json", "w"), indent=1, default=float)
    return T


# =========================================================================== (vi) región factible
def _betabin_ll(logphi, k, n, mu):
    phi = math.exp(logphi)
    a, b = mu * phi, (1 - mu) * phi
    return float(np.sum(special.betaln(k + a, n - k + b) - special.betaln(a, b)))


def feasible_region(T, names, scales, thr=0.8, B=300, seed=0, extra=None):
    """Estimadores de la fracción tipo Calhoun del hipercubo.
    T: tabla por punto (k_PRIMARIO, n_PRIMARIO y factores); extra: {punto: (k, n)} réplicas nuevas (lhsx).
    Devuelve: ingenuo (k/n >= thr), volumen ponderado (media de k/n: insesgado para E_x[p(x)]) y
    empírico-Bayes: p_i ~ Beta(μ_i φ, (1 − μ_i) φ) con μ_i del emulador logístico (efectos principales) y φ por
    máxima verosimilitud beta-binomial; F̂ = media de P(p_i >= thr | datos). IC: bootstrap sobre puntos (refit)."""
    k = T["k_PRIMARIO"].to_numpy(float).copy(); n = T["n_PRIMARIO"].to_numpy(float).copy()
    if extra:
        for i, (kk, nn) in extra.items():
            k[i] += kk; n[i] += nn
    Z = []
    for nm, sc in zip(names, scales):
        x = pd.to_numeric(T[nm], errors="coerce").to_numpy(float)
        z = np.log10(x) if sc == "log" else x
        Z.append((z - np.nanmean(z)) / (np.nanstd(z) or 1))
    X = np.column_stack([np.ones(len(T))] + Z)

    def fit(idx):
        beta = firth_logit(X[idx], k[idx], n[idx])[0]
        mu = np.clip(1 / (1 + np.exp(-(X[idx] @ beta))), 1e-4, 1 - 1e-4)
        r = optimize.minimize_scalar(lambda lp: -_betabin_ll(lp, k[idx], n[idx], mu), bounds=(-3, 8), method="bounded")
        phi = math.exp(r.x)
        post = stats.beta.sf(thr, k[idx] + mu * phi, n[idx] - k[idx] + (1 - mu) * phi)
        return post, phi
    idx = np.arange(len(T))
    post, phi = fit(idx)
    rng = np.random.default_rng(seed)
    f = k / np.maximum(n, 1)
    bs_eb, bs_naive, bs_vol = [], [], []
    for _ in range(B):
        i = rng.integers(0, len(T), len(T))
        pb, _ = fit(i)
        bs_eb.append(pb.mean()); bs_naive.append((f[i] >= thr).mean()); bs_vol.append(f[i].mean())
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return dict(n_puntos=len(T), ingenuo=float((f >= thr).mean()), ingenuo_ic=q(bs_naive),
                volumen=float(f.mean()), volumen_ic=q(bs_vol), eb=float(post.mean()), eb_ic=q(bs_eb), phi=phi,
                n_inciertos=int(((post > 0.05) & (post < 0.95)).sum()), post=post)


def region_report(root, sfx=""):
    lines = ["## (vi) Región factible del hipercubo (relativa a la caja y a las escalas de muestreo)", ""]
    tabs = []
    for st in ("lhs", "lhs2"):
        d = Path(root) / f"f2b_{st}{sfx}"
        if not (d / "analisis" / "tabla_puntos.csv").exists():
            continue
        T = pd.read_csv(d / "analisis" / "tabla_puntos.csv")
        info = json.load(open(d / "problem.json"))["problem"]
        T["_src"] = st
        tabs.append((T, info))
    if not tabs:
        return None, lines + ["(faltan los LHS)", ""]
    names, scales = tabs[0][1]["names"], tabs[0][1]["scale"]
    T = pd.concat([t for t, _ in tabs], ignore_index=True)
    extra = {}
    dx = Path(root) / f"f2b_lhsx{sfx}"
    if (dx / "summary.csv").exists():
        rx = load(dx)
        Tx = point_table(rx)
        for _, r in Tx.iterrows():
            src, pt = str(r["label"]).split(":")
            src = src.replace("f2b_", "").replace(sfx, "") if sfx else src.replace("f2b_", "")
            j = T.index[(T._src == src) & (T.point == int(pt))]
            if len(j):
                extra[int(j[0])] = (float(r["k_PRIMARIO"]), float(r["n_PRIMARIO"]))
    res = {}
    for lab, ex in (("sin lhsx", None), ("con lhsx", extra or None)):
        if lab == "con lhsx" and not extra:
            continue
        rr = feasible_region(T, names, scales, extra=ex)
        res[lab] = {k_: v for k_, v in rr.items() if k_ != "post"}
    R = pd.DataFrame(res).T
    lines += ["Ingenuo: fracción de puntos con k/n ≥ 0,8 (sesgado por el error de clasificación con 10 corridas);"
              " volumen: media de O17p sobre la caja (insesgado para E_x[p(x)]); EB: media de P(p ≥ 0,8 | datos) con prior"
              " beta centrado en el emulador logístico (empírico-Bayes). IC bootstrap sobre puntos con re-ajuste.", "",
              _md(R.reset_index(), index=False, floatfmt=".3f"), ""]
    return R, lines


# =========================================================================== Q6: índice único frente a aditivo
def q6_models(root, sfx=""):
    d = Path(root) / f"f2b_p1aw{sfx}"
    if not (d / "summary.csv").exists():
        return None, ["## Q6: falta p1aw", ""]
    T = point_table(load(d))
    T = T[np.isfinite(pd.to_numeric(T.plast_age_max, errors="coerce"))]
    lr_, la = np.log(T.rec.astype(float).to_numpy()), np.log(T.plast_age_max.astype(float).to_numpy())
    k, n = T.k_PRIMARIO.to_numpy(float), T.n_PRIMARIO.to_numpy(float)
    std = lambda z: (z - z.mean()) / (z.std() or 1)  # noqa: E731
    out = {}

    def single(mask):
        best = None
        for th in np.linspace(-1, 3, 81):
            z = std(lr_[mask] + th * la[mask])
            X = np.column_stack([np.ones(mask.sum()), z, z ** 2])
            ll = firth_logit(X, k[mask], n[mask])[2]
            if best is None or ll > best[1]:
                best = (th, ll, X)
        return best
    for name, mask in (("todos", np.ones(len(T), bool)),
                       ("sin fallas de persistencia (f_O13p >= 0,8)",
                        (T["f_O13p"].to_numpy(float) >= 0.8) if "f_O13p" in T else np.ones(len(T), bool))):
        if mask.sum() < 8:
            continue
        th, ll1, X1 = single(mask)
        a, b = std(lr_[mask]), std(la[mask])
        X2 = np.column_stack([np.ones(mask.sum()), a, a ** 2, b, b ** 2])
        b2 = firth_logit(X2, k[mask], n[mask])
        phi = dispersion(X2, k[mask], n[mask], b2[0])[0]
        p1, p2 = 4, 5          # parámetros: (b0, b1, b2, θ) y (b0, b1, b2, b3, b4)
        qaic1, qaic2 = -2 * ll1 / phi + 2 * p1, -2 * b2[2] / phi + 2 * p2
        out[name] = dict(n_puntos=int(mask.sum()), theta=th, QAIC_indice=qaic1, QAIC_aditivo=qaic2,
                         dQAIC_aditivo_menos_indice=qaic2 - qaic1, phi_aditivo=phi)
    R = pd.DataFrame(out).T
    return R, ["## Q6: modelo de índice único frente al aditivo en log r y log a_w (QAIC con φ del aditivo)", "",
               _md(R.reset_index(), index=False, floatfmt=".3g"), ""]


# =========================================================================== dispersión, mezcla y tabla de ensambles
def spread_table(root, sfx=""):
    """(Q75 − Q25)/Q50 de T_ext con KM y la diferencia C1′ − longevo con IC bootstrap (margen ±0,05)."""
    groups = {}
    for st, labs in (("base", ("C1", "C1p", "C0 (control)", "C1 alpha=0 (control O14)")),
                     ("c1c2", ("C1", "C2")), ("nulos", None), ("ref", None)):
        r = _run(root, st, sfx)
        if r is None:
            continue
        for lab, g in _rows_of(r).items():
            if labs is None or lab in labs:
                groups[f"{st}:{lab}"] = g
    rows, boots = [], {}
    for lab, g in groups.items():
        ev = g["extinct"].to_numpy(bool)
        t = np.where(ev, g.t_ext, g.t_end).astype(float)
        est, lo, hi, bs = km_spread(t, ev, B=1000)
        boots[lab] = bs
        tl = pd.to_numeric(g.get("x_t_lastbirth"), errors="coerce")
        rows.append(dict(ensamble=lab, n=len(g), iqr_med_KM=est, lo=lo, hi=hi, censuradas=int((~ev).sum()),
                         t_last_iqr=float(tl.quantile(0.75) - tl.quantile(0.25)) if tl.notna().any() else np.nan))
    R = pd.DataFrame(rows)
    lines = ["## Dispersión del tiempo de extinción con cuantiles de Kaplan–Meier", "",
             _md(R, index=False, floatfmt=".3f"), ""]
    a, b = "base:C1p", "nulos:longevo (notebook a_max=120, CI de C1)"
    if a in boots and b in boots:
        d = boots[a] - boots[b]
        lines += [f"- C1′ − longevo: {np.nanmedian(d):+.3f} [{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]"
                  " (bootstrap; equivalencia si el IC está dentro de ±0,05)", ""]
    return R, lines


def tsplit_sensitivity(root, sfx="", factors=(1.25, 1.5, 2.0, 2.5)):
    """Mezcla temprana/tardía del notebook con distintos cortes t_split = f × mediana de los eventos."""
    r = _run(root, "ref", sfx)
    if r is None:
        return None, []
    s = r["summary"]
    g = s[s.point == 0]
    ev = g.extinct.to_numpy(bool); t = np.where(ev, g.t_ext, g.t_end).astype(float)
    med = float(np.median(t[ev]))
    rows = []
    for f_ in factors:
        m = survival.early_late_mixture(t, ev, f_ * med)
        rows.append(dict(factor=f_, t_split=f_ * med, p_temprana=m["p_early"], lo=m["lo"], hi=m["hi"],
                         hazard_cola=m["tail_rate"], eventos_cola=m["tail_events"]))
    R = pd.DataFrame(rows)
    return R, ["## Sensibilidad de la mezcla temprana/tardía del notebook al corte t_split", "",
               _md(R, index=False, floatfmt=".3g"), ""]


def ensembles_table(root, sfx=""):
    """Etapa, brazo, n, T y semilla base de cada ensamble (para citar cada cifra con su fuente)."""
    rows = []
    for d in sorted(Path(root).glob(f"f2b_*{sfx}")):
        if not (d / "summary.csv").exists():
            continue
        try:
            r = load(d)
        except Exception:  # noqa: BLE001
            continue
        s = r["summary"]
        seed = int(s["base_seed"].iloc[0]) if "base_seed" in s else np.nan
        npts = s.point.nunique()
        if npts <= 16:
            for lab, g in _rows_of(r).items():
                rows.append(dict(etapa=d.name, brazo=lab, n=len(g), T=int(g["T"].max()), semilla_base=seed))
        else:
            rows.append(dict(etapa=d.name, brazo=f"{npts} puntos", n=len(s), T=int(s["T"].max()), semilla_base=seed))
    return pd.DataFrame(rows)


# =========================================================================== orquestación
STAGE_FN_V2 = dict(rob=analyze_rob, ci=analyze_rob, size=analyze_size, q7aw=analyze_q7aw)


def analyze_stage_v2(stage, run_dir, out=None):
    from .f2b import analyze_generic, analyze_plane
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "analisis"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# V2, etapa `{stage}`: `{run_dir}`", ""]
    try:
        if stage == "trans" or (run_dir / "transplant.csv").exists() and not (run_dir / "summary.csv").exists():
            analyze_trans2(run_dir, out, lines)
        else:
            run = load(run_dir)
            add_strict(run["summary"])
            lines += [f"{run['summary']['point'].nunique()} puntos, {len(run['summary'])} corridas", ""]
            if stage in STAGE_FN_V2:
                analyze_generic(run, out, lines)
                STAGE_FN_V2[stage](run, out, lines)
            elif stage == "robmu":
                analyze_plane(run, out, lines)
                analyze_rob(run, out, lines)
            else:
                analyze_generic(run, out, lines)
    except Exception:  # noqa: BLE001
        lines += ["**ERROR en el análisis**", "", "```", traceback.format_exc(), "```"]
    (out / "informe.md").write_text("\n".join(str(x) for x in lines))
    return out


def final_v2(root, sfx=""):
    root = Path(root)
    out = root / f"f2b_final{sfx}" / "v2"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# V2: informe de las etapas nuevas y re-puntuaciones", ""]
    for name, fn in (("discriminante", discriminant_table), ("repuntuacion", rescoring_table),
                     ("theta", theta_curves), ("region", region_report), ("q6_modelos", q6_models),
                     ("dispersion", spread_table), ("tsplit", tsplit_sensitivity)):
        try:
            res, ls = fn(root, sfx)
            lines += ls
            if isinstance(res, dict):
                for k_, v in res.items():
                    if isinstance(v, pd.DataFrame):
                        v.to_csv(out / f"{name}_{k_}.csv", index=k_ == "ancho")
            elif isinstance(res, pd.DataFrame):
                res.to_csv(out / f"{name}.csv")
        except Exception:  # noqa: BLE001
            lines += [f"## {name}: ERROR", "", "```", traceback.format_exc(), "```", ""]
    try:
        E = ensembles_table(root, sfx)
        E.to_csv(out / "ensambles.csv", index=False)
        lines += ["## Ensambles", "", _md(E, index=False), ""]
    except Exception:  # noqa: BLE001
        lines += ["## ensambles: ERROR", "", "```", traceback.format_exc(), "```", ""]
    (out / "informe_v2.md").write_text("\n".join(str(x) for x in lines))
    return out


# =========================================================================== ancho de transición
def transition_width(x, k, n, B=300, seed=0):
    """f(x) = f₋ + (f₊ − f₋)/(1 + exp(−(x − x_c)/Δ)) por máxima verosimilitud binomial; IC bootstrap paramétrico
    de x_c y Δ. x en la escala en que se quiere el ancho (p. ej. log Λ)."""
    x = np.asarray(x, float); k = np.asarray(k, float); n = np.asarray(n, float)

    def f(th, xx):
        lo, hi = special.expit(th[0]), special.expit(th[1])
        return lo + (hi - lo) * special.expit((xx - th[2]) / math.exp(th[3]))

    def nll(th, kk):
        p = np.clip(f(th, x), 1e-9, 1 - 1e-9)
        return -float(np.sum(kk * np.log(p) + (n - kk) * np.log(1 - p)))

    def fit(kk):
        fr_ = kk / np.maximum(n, 1)
        x0 = float(x[np.argmin(np.abs(fr_ - 0.5))])
        sg = 1.0 if fr_[np.argmax(x)] >= fr_[np.argmin(x)] else -1.0
        best = None
        for d0 in (0.1, 0.3, 1.0):
            th0 = [-4.0, 4.0, x0, math.log(d0)] if sg > 0 else [4.0, -4.0, x0, math.log(d0)]
            r = optimize.minimize(nll, th0, args=(kk,), method="Nelder-Mead", options=dict(maxiter=4000))
            if best is None or r.fun < best.fun:
                best = r
        return best.x
    th = fit(k)
    rng = np.random.default_rng(seed)
    p = np.clip(f(th, x), 1e-6, 1 - 1e-6)
    bs = np.array([fit(rng.binomial(n.astype(int), p).astype(float)) for _ in range(B)])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return dict(x_c=float(th[2]), x_c_ic=q(bs[:, 2]), ancho=float(math.exp(th[3])), ancho_ic=q(np.exp(bs[:, 3])),
                f_menos=float(special.expit(th[0])), f_mas=float(special.expit(th[1])))


def q7aw_widths(T, B=200):
    """Ancho de la transición de P_ext en log Λ a w = 0, por a_w (etapa q7aw)."""
    rows = []
    for aw, g in T[np.isclose(T.inherit_weight.astype(float), 0.0)].groupby("plast_age_max"):
        lam = np.log(g.rec.astype(float) * float(aw))
        if len(g) >= 4:
            r = transition_width(lam, g.k_ext.to_numpy(float), g.n.to_numpy(float), B=B)
            rows.append(dict(a_w=float(aw), **r))
    return pd.DataFrame(rows)
