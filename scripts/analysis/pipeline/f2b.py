"""Análisis de la Fase 2b (plan prerregistrado: docs/preregistered_plan.md): una función por tipo de etapa y el informe final Q1–Q7.

    python -m pipeline f2b STAGE RUN_DIR [--out RUN_DIR/analisis]
    python -m pipeline f2b final RESULTS_DIR [--suffix ""] [--out RESULTS_DIR/f2b_final]

Lee la salida de `scripts/production/u25x.py` (= `python -m u25 sweep` + columnas obj_O13p y x_* de
f2b_model.py) y, si está, `stage.json` (copiado por run_2b_v2.sh). Nunca falla por una columna o etapa faltante:
lo que no se puede calcular queda como NaN y se anota en el informe.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import survival, multiple
from .report import _md

OBJ = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O8b", "O9", "O10", "O11", "O11b", "O13", "O13b", "O13p",
       "O14", "O15", "O15b"]
ESS = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O10", "O11", "O13"]
ANTI = ["A1", "A2", "A3", "A4"]
XMEAN = ["N_max", "t_peak", "x_rho_pk", "x_f0_pk", "x_BO", "x_BO4", "x_CR", "x_in_ref_pk", "x_S_adult_pk", "x_rho8",
         "x_D_last", "x_smat_B", "x_smat_sen", "S_at_peak", "mstar_at_peak", "wall_time_s"]


# =========================================================================== carga
def load(run_dir):
    d = Path(run_dir)
    s = pd.read_csv(d / "summary.csv", low_memory=False)
    s = s.drop_duplicates(["point", "rep"], keep="last").sort_values(["point", "rep"]).reset_index(drop=True)
    for c in ("extinct", "censored", "exploded"):
        if c in s and s[c].dtype != bool:
            s[c] = s[c].astype(str).str.lower().isin(["true", "1"])
    params = json.load(open(d / "params.json")) if (d / "params.json").exists() else None
    labels = json.load(open(d / "labels.json")) if (d / "labels.json").exists() else None
    meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
    stage = json.load(open(d / "stage.json")) if (d / "stage.json").exists() else {}
    if "T" not in s:
        s["T"] = meta.get("T", np.nan)
    apply_variant(s)
    add_O17p(s)
    return dict(dir=d, summary=s, params=params, labels=labels, meta=meta, stage=stage)


def add_O17p(s):
    """O17p = corrida tipo Calhoun y además O13 persistente, ESTRICTO (u25.objectives.o17p_from_summary): un NA en
    un esencial (salvo O1–O3 con N0 > 20), en O13p o en un anti-patrón cuenta como FAIL y una excepción da ERR.
    Si la columna obj_O17p ya viene del barrido (u25 v2), se respeta: fuente única. En los barridos
    de la v1 no hay ningún NA de O13p ni de los esenciales fuera de O1–O3, así que coincide con la regla vieja."""
    if "obj_O17p" in s and s["obj_O17p"].astype(str).isin(["PASS", "FAIL", "ERR"]).any():
        return
    if "obj_O17_run" in s:
        from u25.objectives import o17p_from_summary
        s["obj_O17p"] = o17p_from_summary(s).to_numpy()


VARIANT = os.environ.get("F2B_VARIANT", "")       # p. ej. "ta": usar obj_ta_* (t_pk = argmax) como obj_*


def apply_variant(s, pref=None):
    """Re-puntuación alternativa de la V2 (f2b_model_v2): copia obj_<pref>_<código> sobre obj_<código>."""
    pref = VARIANT if pref is None else pref
    if not pref:
        return s
    p = f"obj_{pref}_"
    for c in [c for c in s.columns if c.startswith(p)]:
        s[f"obj_{c[len(p):]}"] = s[c]
    return s


def primary_code(run):
    st = run.get("stage") or {}
    if st.get("primary"):
        return st["primary"]
    return "O17_run"


def label_of(run, pt):
    lab = run.get("labels")
    return lab[int(pt)] if lab is not None and int(pt) < len(lab) else f"p{int(pt)}"


# =========================================================================== fracciones
def frac(series, code):
    v = series.astype(str)
    if code in ("O17_run", "O17p"):
        k, n = int((v == "PASS").sum()), int(len(v))
    else:
        n = int(v.isin(["PASS", "MARG", "FAIL"]).sum())
        k = int((v == "PASS").sum())
    if n == 0:
        return dict(k=0, n=0, f=np.nan, lo=np.nan, hi=np.nan)
    lo, hi = survival.wilson(k, n)
    return dict(k=k, n=n, f=k / n, lo=lo, hi=hi)


def point_table(run, primary=None):
    s = run["summary"]
    primary = primary or primary_code(run)
    dcols = run["meta"].get("design_columns") or []
    rows = []
    for pt, g in s.groupby("point", sort=True):
        r = dict(point=int(pt), label=label_of(run, pt), n=len(g))
        for c in dcols:
            if c in g:
                r[c] = g[c].iloc[0]
        for code in OBJ + ["O17_run", "O17p"]:
            col = f"obj_{code}"
            if col in g:
                fr = frac(g[col], code)
                r[f"f_{code}"], r[f"lo_{code}"], r[f"hi_{code}"], r[f"n_{code}"], r[f"k_{code}"] = \
                    fr["f"], fr["lo"], fr["hi"], fr["n"], fr["k"]
        for a in ANTI:
            if f"obj_{a}" in g:
                r[f"trig_{a}"] = float((g[f"obj_{a}"].astype(str) == "TRIGGERED").mean())
        if f"f_{primary}" in r:
            r["f_PRIMARIO"], r["lo_PRIMARIO"], r["hi_PRIMARIO"] = r[f"f_{primary}"], r[f"lo_{primary}"], r[f"hi_{primary}"]
            r["k_PRIMARIO"], r["n_PRIMARIO"] = r[f"k_{primary}"], r[f"n_{primary}"]
        ev = g["extinct"].to_numpy(bool)
        k = int(ev.sum())
        lo, hi = survival.wilson(k, len(g))
        r.update(P_ext=k / len(g), P_ext_lo=lo, P_ext_hi=hi, k_ext=k,
                 f_exploded=float(g["exploded"].mean()) if "exploded" in g else np.nan)
        time = np.where(ev, g["t_ext"], g["t_end"]).astype(float)
        km = survival.kaplan_meier(time, ev)
        med = survival.km_median(km)
        r.update(t_ext_med=med["median"], t_ext_med_lo=med["lo"], t_ext_med_hi=med["hi"])
        te = g.loc[ev, "t_ext"].to_numpy(float)
        r.update(t_ext_mean=np.mean(te) if te.size else np.nan, t_ext_sd=np.std(te, ddof=1) if te.size > 1 else np.nan)
        r["t_ext_cv"] = r["t_ext_sd"] / r["t_ext_mean"] if te.size > 1 else np.nan
        for c in XMEAN:
            if c in g:
                v = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
                r[f"{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        if "x_rho8" in g:
            v = pd.to_numeric(g["x_rho8"], errors="coerce")
            r["f_rho8_ge_1.84"] = float((v >= 1.84).sum() / max(1, v.notna().sum()))
        if "x_D_last" in g:
            v = pd.to_numeric(g["x_D_last"], errors="coerce")
            r["x_D_last_sd"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


SHORT = ["point", "label", "n", "f_PRIMARIO", "f_O17_run", "f_O13", "f_O13p", "f_O4", "f_O7", "f_O11", "f_O8",
         "P_ext", "t_ext_med", "t_ext_cv", "x_rho_pk_mean", "x_f0_pk_mean", "x_BO4_mean", "x_CR_mean", "wall_time_s_mean"]


def _fmt_ci(T, code):
    if f"f_{code}" not in T:
        return None
    return [("NA" if not np.isfinite(f) else f"{f:.2f} [{lo:.2f}, {hi:.2f}]")
            for f, lo, hi in zip(T[f"f_{code}"], T[f"lo_{code}"], T[f"hi_{code}"])]


def objectives_md(T, codes):
    D = T[["point", "label", "n"]].copy()
    for c in codes:
        v = _fmt_ci(T, c)
        if v is not None:
            D[c] = v
    return _md(D, index=False)


# =========================================================================== regresión logística (Firth)
def firth_logit(X, k, n, maxit=100, tol=1e-8):
    """Logística binomial agrupada con penalización de Firth (estable con separación).
    X: (m, p) con columna de unos; k éxitos; n ensayos. Devuelve beta, se, loglik penalizada, cov."""
    X = np.asarray(X, float); k = np.asarray(k, float); n = np.asarray(n, float)
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = n * p * (1 - p) + 1e-12
        XtWX = X.T @ (X * W[:, None])
        try:
            inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtWX)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], inv, X * np.sqrt(W)[:, None])
        U = X.T @ (k - n * p + h * (0.5 - p))
        step = inv @ U
        if np.max(np.abs(step)) > 5:
            step = step * 5 / np.max(np.abs(step))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    W = n * p * (1 - p) + 1e-12
    XtWX = X.T @ (X * W[:, None])
    sign, logdet = np.linalg.slogdet(XtWX)
    ll = float(np.sum(k * np.log(np.clip(p, 1e-300, 1)) + (n - k) * np.log(np.clip(1 - p, 1e-300, 1))) + 0.5 * logdet)
    cov = np.linalg.pinv(XtWX)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), ll, cov


def dispersion(X, k, n, beta):
    """Sobredispersión de Pearson de una logística agrupada: phi = X²/(m − p), acotada abajo por 1.
    Con puntos de diseño distintos (LHS, planos), la falta de ajuste del predictor lineal infla la varianza
    binomial; los EE cuasi-binomiales son EE·sqrt(phi) (y las LR se dividen por phi)."""
    X = np.asarray(X, float); k = np.asarray(k, float); n = np.asarray(n, float)
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-6, 1 - 1e-6)
    chi2 = float(np.sum((k - n * p) ** 2 / (n * p * (1 - p))))
    dof = max(1, X.shape[0] - X.shape[1])
    return max(1.0, chi2 / dof), chi2, dof


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = X.T @ (X * (res / np.clip(1 - h, 1e-6, None))[:, None] ** 2)   # HC3
    cov = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(cov)), res


# =========================================================================== figuras
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def km_figure(groups, path, title="Kaplan–Meier del tiempo de extinción", tmax=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab, (time, ev) in groups.items():
        km = survival.kaplan_meier(time, ev)
        ax.step(km["t"], km["S"], where="post", lw=1.6, label=f"{lab} (n={len(time)})")
    ax.set_xlabel("semana"); ax.set_ylabel("P(colonia viva)"); ax.set_title(title)
    if tmax:
        ax.set_xlim(0, tmax)
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def heatmaps(T, axes, layer, metrics, out, stem):
    """Mapas (ejes del plano) de cada métrica por capa."""
    plt = _plt()
    xa, ya = axes
    if xa not in T or ya not in T:
        return []
    T = T.copy()
    for c in (xa, ya):
        T[c] = pd.to_numeric(T[c], errors="coerce").fillna(np.inf)
    layers = sorted(T[layer].dropna().unique()) if layer and layer in T else [None]
    files = []
    for m in metrics:
        if m not in T:
            continue
        fig, axs = plt.subplots(1, len(layers), figsize=(4.6 * len(layers), 3.8), squeeze=False)
        for ax, L in zip(axs[0], layers):
            d = T if L is None else T[T[layer] == L]
            if not len(d):
                continue
            P = d.pivot_table(index=ya, columns=xa, values=m, aggfunc="mean")
            im = ax.imshow(P.to_numpy(float), origin="lower", aspect="auto", cmap="viridis",
                           vmin=0 if m.startswith(("f_", "P_ext")) else None, vmax=1 if m.startswith(("f_", "P_ext")) else None)
            ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([f"{v:g}" for v in P.columns], rotation=45, fontsize=7)
            ax.set_yticks(range(P.shape[0])); ax.set_yticklabels([f"{v:g}" for v in P.index], fontsize=7)
            ax.set_xlabel(xa); ax.set_ylabel(ya)
            ax.set_title(f"{m}" + ("" if L is None else f" | {layer}={L:g}"), fontsize=9)
            for (i, j), v in np.ndenumerate(P.to_numpy(float)):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v < 0.6 else "k")
            fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fn = out / f"fig_{stem}_{m}.png"
        plt.savefig(fn, dpi=120); plt.close(fig)
        files.append(fn.name)
    return files


# =========================================================================== análisis por etapa
def analyze_generic(run, out, lines):
    T = point_table(run)
    T.to_csv(out / "tabla_puntos.csv", index=False)
    cols = [c for c in SHORT if c in T]
    lines += ["## Tabla por punto (f = fracción PASS de las corridas evaluables; P_ext = extinción antes de T)", "",
              _md(T[cols], index=False, floatfmt=".3g"), ""]
    codes = [c for c in ESS + ["O13p", "O8", "O8b", "O9", "O11b", "O13b", "O14", "O15b", "O17_run", "O17p"] if f"f_{c}" in T]
    lines += ["## Objetivos con IC de Wilson al 95 %", "", objectives_md(T, codes), ""]
    s = run["summary"]
    pts = sorted(s.point.unique())
    if len(pts) <= 16:
        groups = {}
        for pt in pts:
            g = s[s.point == pt]
            ev = g["extinct"].to_numpy(bool)
            groups[label_of(run, pt)] = (np.where(ev, g["t_ext"], g["t_end"]).astype(float), ev)
        km_figure(groups, out / "fig_km.png", tmax=float(s["T"].max()))
        lines += ["![KM](fig_km.png)", ""]
    if 2 <= len(pts) <= 40 and s["extinct"].any():
        ev = s["extinct"].to_numpy(bool)
        lr = survival.logrank(np.where(ev, s["t_ext"], s["t_end"]).astype(float), ev, s["point"].to_numpy())
        lines += [f"Log-rank entre los {len(pts)} puntos: χ² = {lr['chi2']:.1f}, gl = {lr['df']}, p = {lr['p']:.3g}", ""]
    return T


def analyze_plane(run, out, lines):
    T = analyze_generic(run, out, lines)
    st = run["stage"]
    axes = st.get("axes")
    if axes:
        figs = heatmaps(T, axes, st.get("layer"), ["f_PRIMARIO", "f_O13", "f_O13p", "f_O4", "f_O11", "P_ext",
                                                   "x_rho_pk_mean", "x_f0_pk_mean", "x_BO4_mean", "f_exploded"],
                        out, run["dir"].name)
        lines += ["## Mapas", ""] + [f"![{f}]({f})" for f in figs] + [""]
    return T


def main_mode_weibull(time, ev):
    """Modo principal de extinción como modelo de curación (mezcla): S(t) = π + (1 − π)·S_W(t; k, λ).
    π es la fracción de colonias que escapan (persistentes) y (k, λ) la Weibull del colapso casi determinista.
    Un ajuste Weibull sobre todo t sin π deja que unas pocas colonias persistentes (censuradas muy tarde) aplasten k.
    Devuelve k, λ, π, CV de la Weibull y el LR de π = 0 (frontera: p = ½·P(χ²₁ > LR))."""
    from scipy import optimize
    time = np.asarray(time, float); ev = np.asarray(ev, bool)
    if ev.sum() < 3:
        return {}
    x = np.maximum(time, 1e-9)

    def nll(th, cure=True):
        k, lam = np.exp(th[0]), np.exp(th[1])
        pi = 1 / (1 + np.exp(-th[2])) if cure else 0.0
        z = (x / lam) ** k
        logf = np.log(k / lam) + (k - 1) * np.log(x / lam) - z
        ll = np.sum(np.log1p(-pi) + logf[ev]) if pi < 1 else -np.inf
        ll += np.sum(np.log(pi + (1 - pi) * np.exp(-z[~ev]) + 1e-300))
        return -ll
    m = float(np.mean(x[ev])); cv0 = float(np.std(x[ev]) / m) if ev.sum() > 1 else 0.3
    k0 = np.log(max(1.0, 1.2 / max(cv0, 1e-3)))
    th0 = [k0, np.log(m * 1.05), np.log((1 - ev.mean() + 1e-3) / (ev.mean() + 1e-3))]
    f0 = survival.fit_parametric(x, ev, "weibull", t0=0.0)          # sin π (π = 0): MLE de referencia
    starts = [th0, [np.log(f0["params"]["shape"]), np.log(f0["params"]["scale"]), th0[2]],
              [k0 + 0.5, np.log(m), -3.0], [k0 - 0.5, np.log(m), -1.0]]
    r1 = min((optimize.minimize(nll, st, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-9, fatol=1e-10))
              for st in starts), key=lambda r: r.fun)
    k, lam = float(np.exp(r1.x[0])), float(np.exp(r1.x[1]))
    pi = float(1 / (1 + np.exp(-r1.x[2])))
    from math import gamma as G
    cvw = float(np.sqrt(G(1 + 2 / k) / G(1 + 1 / k) ** 2 - 1))
    lr = max(0.0, 2 * (-r1.fun - f0["loglik"]))
    return dict(weibull_k_modo=k, weibull_escala_modo=lam, pi_escape=pi, cv_weibull_modo=cvw,
                LR_pi0=lr, p_pi0=float(0.5 * stats.chi2.sf(lr, 1)), weibull_k_sin_pi=float(f0["params"]["shape"]),
                n_censuradas=int((~ev).sum()))


def analyze_base(run, out, lines):
    T = analyze_generic(run, out, lines)
    s = run["summary"]
    lines += ["## Supervivencia por punto (KM + ajustes Weibull/exponencial con censura)", "",
              "`weibull_k_todo`: Weibull sobre todo t (una sola colonia persistente, censurada en T, basta para bajar k"
              " artificialmente). `weibull_k_modo`: modelo de curación S = π + (1 − π)·S_W: π = fracción que escapa"
              " (persistente), k y λ del colapso; `p_pi0`: LR de π = 0 (½χ²₁).", ""]
    rows = []
    for pt in sorted(s.point.unique()):
        g = s[s.point == pt]
        ev = g["extinct"].to_numpy(bool)
        time = np.where(ev, g["t_ext"], g["t_end"]).astype(float)
        r = dict(point=pt, label=label_of(run, pt), n=len(g), eventos=int(ev.sum()))
        if ev.sum() >= 3:
            fw = survival.fit_parametric(time, ev, "weibull", t0=0.0)
            fe = survival.fit_parametric(time, ev, "exponential", t0=0.0)
            r.update(weibull_k_todo=fw["params"]["shape"], LR_vs_exp_p_todo=survival.lr_test(fe, fw)["p"])
            r.update(main_mode_weibull(time, ev))
        rows.append(r)
    sv = pd.DataFrame(rows)
    sv.to_csv(out / "supervivencia.csv", index=False)
    lines += [_md(sv, index=False, floatfmt=".3g"), ""]
    # evaluador oficial (serie completa) con el control alpha = 0 para O14 (G_B)
    try:
        ser_p = run["dir"] / "series.parquet"
        if ser_p.exists() and run["params"] is not None:
            from u25 import Params
            from u25.objectives import evaluate_ensemble
            ser = pd.read_parquet(ser_p)
            plist = [Params.from_dict(x) for x in run["params"]]
            ctrl = [pt for pt in s.point.unique() if "alpha=0" in label_of(run, pt)]
            cg = s.loc[s.point.isin(ctrl), "obj_G_B"].astype(float).to_numpy() if ctrl and "obj_G_B" in s else None
            tr = _transplants_for_base(run)
            tab, per = evaluate_ensemble(ser, s, plist, tr, cg)
            tab.to_csv(out / "objetivos_oficial.csv", index=False)
            piv = tab.pivot(index="code", columns="point", values="verdict")
            lines += ["## Evaluador oficial (`u25.objectives.evaluate_ensemble`, con O14_GB contra alpha = 0"
                      + (" y O12 de la etapa trans" if tr is not None else "") + ")", "", _md(piv), ""]
    except Exception as e:  # noqa: BLE001
        lines += [f"(evaluador oficial no disponible: {type(e).__name__}: {e})", ""]
    return T


def _transplants_for_base(run):
    """Transplantes de la etapa trans (mismo sufijo) asignados por etiqueta a los puntos de base."""
    d = run["dir"]
    suffix = d.name[len("f2b_base"):]
    tdir = d.parent / f"f2b_trans{suffix}"
    if not (tdir / "transplant.csv").exists() or run["labels"] is None:
        return None
    tr = pd.read_csv(tdir / "transplant.csv")
    tl = json.load(open(tdir / "labels.json")) if (tdir / "labels.json").exists() else []
    rows = []
    for pt, lab in enumerate(run["labels"]):
        if lab in tl:
            rows.append(tr[tr.point == tl.index(lab)].assign(point=pt))
    return pd.concat(rows) if rows else None


def analyze_ref(run, out, lines):
    T = analyze_generic(run, out, lines)
    s = run["summary"]
    from . import peaks
    ser = None
    if (run["dir"] / "series.parquet").exists():
        ser = pd.read_parquet(run["dir"] / "series.parquet", columns=["point", "rep", "t", "N"])
    lines += ["## Línea base: supervivencia (mezcla temprana/tardía) y drift de picos", ""]
    for pt in sorted(s.point.unique()):
        g = s[s.point == pt]
        ev = g["extinct"].to_numpy(bool)
        time = np.where(ev, g["t_ext"], g["t_end"]).astype(float)
        sv = survival.survival_summary(time, ev, B_gof=0)
        lines += [f"### {label_of(run, pt)}", "",
                  f"- eventos {sv['events']}/{sv['n']}; mediana KM {sv['median']['median']}; P_ext(T): {sv['P_ext']}"]
        if "mixture" in sv:
            m = sv["mixture"]
            lines += [f"- mezcla (corte {sv['t_split']:.0f}): p_temprana = {m['p_early']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}],"
                      f" hazard de la cola {m['tail_rate']:.2e}/sem ({m['tail_events']} eventos)"]
        if sv["fits"]:
            lines += [f"- Weibull de la cola: k = {sv['fits']['weibull']['params']['shape']:.2f};"
                      f" LR vs exponencial p = {sv['lr_weibull_vs_exp']['p']:.3g}"]
        if ser is not None:
            cyc = peaks.cycles_per_run(ser[ser.point == pt])
            dr = peaks.ensemble_drift_test(cyc)
            if dr.get("n_runs", 0) >= 3:
                lines += [f"- drift de log(pico) por ciclo: {dr['mean_slope']:+.4f} [{dr['ci_lo']:+.4f}, {dr['ci_hi']:+.4f}],"
                          f" Wilcoxon p = {dr['wilcoxon_p']:.3g} ({dr['n_runs']} corridas con ≥ 4 picos)"]
        lines += [""]
    return T


def analyze_transplant(run_dir, out, lines):
    d = Path(run_dir)
    tr = pd.read_csv(d / "transplant.csv")
    labels = json.load(open(d / "labels.json")) if (d / "labels.json").exists() else None
    rows = []
    for pt, g in tr.groupby("point"):
        r = dict(point=pt, label=labels[pt] if labels else f"p{pt}")
        for ph in ("B", "D", "MIX_M", "MIX_F"):
            x = g.loc[g.phase == ph, "success"].dropna().astype(float)
            if len(x):
                lo, hi = survival.wilson(int(x.sum()), len(x))
                r[f"P_T_{ph}"], r[f"lo_{ph}"], r[f"hi_{ph}"], r[f"n_{ph}"], r[f"k_{ph}"] = x.mean(), lo, hi, len(x), int(x.sum())
        o12 = (d / f"arm_{pt}" / "O12.txt")
        if o12.exists():
            txt = o12.read_text().splitlines()
            r["O12"] = txt[0].split("\t")[0] if txt else ""
            r["O12b"] = txt[1].split("\t")[1] if len(txt) > 1 else ""
        if "early_D" in g:
            r["early_D"] = float(g.loc[g.phase == "D", "early_D"].astype(str).str.lower().isin(["true", "1"]).mean())
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(out / "tabla_brazos.csv", index=False)
    lines += ["## Transplantes por brazo (P_T con IC de Wilson; O12 y O12b del evaluador)", "",
              _md(A, index=False, floatfmt=".3g"), ""]
    return A


def analyze_amax(run, out, lines):
    T = analyze_generic(run, out, lines)
    lines += q4(run)["lines"]
    return T


def split_half(s, code=None, col=None):
    """Por punto: e = (media de réplicas pares − media de réplicas impares)/2. Tiene la misma varianza de ruido
    (y la misma estructura de CRN) que la media por punto, con esperanza 0: sirve de 'piso de ruido'."""
    if code is not None:
        v = (s[f"obj_{code}"].astype(str) == "PASS").astype(float)
    else:
        v = pd.to_numeric(s[col], errors="coerce")
    d = pd.DataFrame(dict(point=s.point, par=(s.rep % 2 == 0), v=v))
    m = d.groupby(["point", "par"])["v"].mean().unstack()
    if True not in m or False not in m:
        return None
    return ((m[True] - m[False]) / 2).rename("e").reset_index()


def split_half_median(s):
    """Piso aproximado para la mediana del tiempo de extinción: e = (mediana de pares − mediana de impares)/2 sobre
    los tiempos con censura en T (t_ext si extinta, t_end si no). Para R = 4 réplicas cada mitad tiene 2 y su
    mediana es la media; la varianza de e aproxima la de la mediana por punto (exacta para la media)."""
    t = np.where(s["extinct"].to_numpy(bool), s["t_ext"], s["t_end"]).astype(float)
    d = pd.DataFrame(dict(point=s.point, par=(s.rep % 2 == 0), v=t))
    m = d.groupby(["point", "par"])["v"].median().unstack()
    if True not in m or False not in m:
        return None
    return ((m[True] - m[False]) / 2).rename("e").reset_index()


def sobol_noise_floor(Y, e, k):
    """Piso de ruido del índice total de Jansen: ST_i^ruido = E[(e_A − e_ABi)^2] / (2 Var(Y)), con el mismo
    ordenamiento de Saltelli que SALib (bloques A, AB_1..AB_k, B; sin segundo orden)."""
    Y = np.asarray(Y, float).reshape(-1, k + 2); e = np.asarray(e, float).reshape(-1, k + 2)
    var = np.var(np.r_[Y[:, 0], Y[:, -1]], ddof=1)
    return np.array([np.nanmean((e[:, 0] - e[:, 1 + i]) ** 2) / (2 * var) for i in range(k)])


def _metrics_for_sa(T, primary):
    m = ["f_PRIMARIO", "f_O13", "f_O13p", "f_O4", "f_O11", "f_O7", "P_ext", "x_rho_pk_mean", "x_f0_pk_mean",
         "t_ext_med", "x_BO4_mean"]
    return [c for c in m if c in T and T[c].notna().any() and np.nanstd(T[c].to_numpy(float)) > 0]


PRIMARY_SA = ["f_PRIMARIO", "f_O13", "f_O13p", "x_rho_pk_mean", "x_f0_pk_mean", "P_ext", "f_O11"]


def analyze_morris(run, out, lines):
    from .sensitivity import morris_analyze
    from . import plots
    T = analyze_generic(run, out, lines)
    info = json.load(open(run["dir"] / "problem.json"))
    X = np.load(run["dir"] / "X.npy")
    mets = _metrics_for_sa(T, primary_code(run))
    e = split_half(run["summary"], code=primary_code(run))
    if e is not None:                       # μ* del ruido puro (piso): mismo diseño, métrica de esperanza 0
        T = T.merge(e.rename(columns={"e": "ruido_PRIMARIO"}), on="point", how="left")
        mets = mets + ["ruido_PRIMARIO"]
    mo = morris_analyze(info["problem"], X, T, mets, num_levels=info.get("num_levels", 4))
    mo.to_csv(out / "morris.csv", index=False)
    for m in mo["metric"].unique():
        d = mo[mo.metric == m].sort_values("mu_star", ascending=False)
        lines += [f"## Morris: {m}", "", _md(d[["name", "mu_star", "mu_star_conf", "sigma", "mu", "rank"]], index=False), ""]
        plots.plot_morris(mo, out / f"fig_morris_{m}.png", m)
    if "ruido_PRIMARIO" in set(mo.metric):
        fl = mo[mo.metric == "ruido_PRIMARIO"].set_index("name")["mu_star"]
        pr = mo[mo.metric == "f_PRIMARIO"].set_index("name")["mu_star"]
        if len(pr):
            D = pd.DataFrame({"mu_star": pr, "mu_star_ruido": fl, "cociente": pr / fl.replace(0, np.nan)})
            lines += ["## Piso de ruido de Morris (f_PRIMARIO): μ* del contraste de mitades de réplicas", "",
                      "Un factor es distinguible del azar si μ* ≫ μ*_ruido (cociente > 2).", "",
                      _md(D.sort_values("mu_star", ascending=False)), ""]
    prim = [m for m in PRIMARY_SA if m in mo.metric.unique()]
    agg = mo[mo.metric.isin(prim)].groupby("name")["mu_star_rel"].mean().sort_values(ascending=False)
    json.dump({"factores": list(agg.index), "mu_star_rel_medio": agg.round(4).to_dict(), "metricas": prim},
              open(out / "seleccion_sobol.json", "w"), indent=1)
    lines += ["## Ranking agregado (μ* relativo medio sobre las métricas primarias) → selección para Sobol", "",
              _md(agg.to_frame("mu_star_rel_medio")), ""]
    return T


def analyze_lhs(run, out, lines):
    T = analyze_generic(run, out, lines)
    info = json.load(open(run["dir"] / "problem.json"))
    names = info["problem"]["names"]
    lines += lhs_block(T, names, info["problem"], out=out)
    return T


def lhs_block(T, names, problem, B=2000, seed=0, out=None):
    rng = np.random.default_rng(seed)
    lines = []
    f = T["f_PRIMARIO"].to_numpy(float)
    feas = f >= 0.8
    bs = np.array([feas[rng.integers(0, len(f), len(f))].mean() for _ in range(B)])
    lines += [f"## Región factible (f_PRIMARIO ≥ 0,8): {feas.mean():.3f} de los {len(f)} puntos "
              f"[IC bootstrap {np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", ""]
    rows = []
    Z = []
    for nm, sc in zip(names, problem["scale"]):
        x = pd.to_numeric(T[nm], errors="coerce").to_numpy(float)
        z = np.log10(x) if sc == "log" else x
        z = (z - np.nanmean(z)) / (np.nanstd(z) or 1)
        Z.append(z)
        for m in ("f_PRIMARIO", "x_rho_pk_mean", "x_f0_pk_mean", "P_ext"):
            if m in T:
                y = T[m].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(z)
                if ok.sum() > 5:
                    rho, p = stats.spearmanr(z[ok], y[ok])
                    rows.append(dict(familia="LHS_spearman", factor=nm, metrica=m, efecto=rho, p=p))
    Z = np.array(Z).T
    phi_txt = ""
    if "k_PRIMARIO" in T:
        X = np.column_stack([np.ones(len(Z)), Z])
        ok = np.isfinite(X).all(1)
        kk, nn = T["k_PRIMARIO"].to_numpy(float)[ok], T["n_PRIMARIO"].to_numpy(float)[ok]
        beta, se, ll, _ = firth_logit(X[ok], kk, nn)
        phi, chi2, dof = dispersion(X[ok], kk, nn, beta)
        phi_txt = (f"Logística de efectos principales: sobredispersión de Pearson φ = {phi:.2f} (X² = {chi2:.0f}, gl = {dof});"
                   " `se` y `p` son cuasi-binomiales (EE × √φ); `p_binom` es el p binomial sin corregir.")
        for nm, b, e in zip(["(intercepto)"] + names, beta, se):
            if nm != "(intercepto)":
                eq = e * np.sqrt(phi)
                rows.append(dict(familia="LHS_logit", factor=nm, metrica="f_PRIMARIO", efecto=b, se=eq,
                                 p=2 * stats.norm.sf(abs(b / eq)) if eq > 0 else np.nan,
                                 p_binom=2 * stats.norm.sf(abs(b / e)) if e > 0 else np.nan))
    E = pd.DataFrame(rows)
    if len(E):
        E["p_bh_familia"] = np.nan
        for fam, g in E.groupby(["familia", "metrica"]):
            E.loc[g.index, "p_bh_familia"] = multiple.bh(g["p"].to_numpy())
        if out is not None:
            E.to_csv(Path(out) / "lhs_efectos.csv", index=False)
    lines += ["## Efectos por factor (Spearman; logística de Firth sobre factores estandarizados); BH por familia", "",
              phi_txt, "", _md(E, index=False, floatfmt=".3g") if len(E) else "(sin datos)", ""]
    return lines


def analyze_sobol(run, out, lines):
    from .sensitivity import sobol_analyze
    from . import plots
    T = analyze_generic(run, out, lines)
    info = json.load(open(run["dir"] / "problem.json"))
    mets = _metrics_for_sa(T, primary_code(run))
    so, _ = sobol_analyze(info["problem"], T, mets)
    so.to_csv(out / "sobol.csv", index=False)
    s = run["summary"]
    k = len(info["problem"]["names"])
    npts = int(T["point"].max()) + 1
    so["ST_ruido"] = np.nan
    so["piso"] = ""
    diag = {}
    for m in so["metric"].unique():
        code = {"f_PRIMARIO": primary_code(run)}.get(m, m[2:] if m.startswith("f_") else None)
        try:
            kind = "exacto"
            if code and f"obj_{code}" in s:
                e = split_half(s, code=code)
            elif m == "P_ext" and "extinct" in s:          # P_ext = media por punto de la indicadora de extinción
                e = split_half(s.assign(_ext=s["extinct"].astype(float)), col="_ext")
            elif m == "t_ext_med":                         # mediana de tiempos censurados: piso aproximado
                e = split_half_median(s)
                kind = "aprox. (medianas)"
            elif m.endswith("_mean") and m[:-5] in s:
                e = split_half(s, col=m[:-5])
            else:
                e = None
            if e is not None:
                Y = T.set_index("point")[m].reindex(range(npts))
                ee = e.set_index("point")["e"].reindex(range(npts))
                fl = sobol_noise_floor(Y.fillna(Y.mean()), ee.fillna(0.0), k)
                so.loc[so.metric == m, "ST_ruido"] = fl
                so.loc[so.metric == m, "piso"] = kind
        except Exception as ex:  # noqa: BLE001
            lines += [f"(piso de ruido de {m} no disponible: {type(ex).__name__}: {ex})"]
        # diagnóstico de degeneración: métrica casi constante en el hipercubo (los estimadores de Saltelli/Jansen
        # se vuelven inestables: ST > 1, IC enormes) -> no interpretable
        Yv = T[m].to_numpy(float)
        Yv = Yv[np.isfinite(Yv)]
        vals, cnt = np.unique(np.round(Yv, 6), return_counts=True)
        modal = float(cnt.max() / cnt.sum()) if cnt.size else np.nan
        dm = so[so.metric == m]
        degen = bool((modal > 0.9) or (dm["ST_conf"].median() > 0.5))      # Σ ST > 1 es legítimo (interacciones)
        diag[m] = dict(frac_modal=modal, suma_ST=float(dm["ST"].sum()), suma_S1=float(dm["S1"].sum()),
                       mediana_ST_conf=float(dm["ST_conf"].median()), degenerada=degen)
        so.loc[so.metric == m, "degenerada"] = degen
    so["ST_corregido"] = so["ST"] - so["ST_ruido"]
    so["relevante"] = ((so["ST_corregido"] > 0.05) & (so["ST"] - so["ST_conf"] > so["ST_ruido"])
                       & so["ST_ruido"].notna() & ~so["degenerada"].astype(bool))
    so.to_csv(out / "sobol.csv", index=False)
    D = pd.DataFrame(diag).T
    D.index.name = "metrica"
    D.to_csv(out / "sobol_diagnostico.csv")
    lines += ["Piso de ruido: ST_ruido = índice de Jansen del contraste de mitades de réplicas (misma varianza de azar y misma"
              " CRN que la media por punto). Relevante: ST − ST_ruido > 0,05, ST − IC > ST_ruido, piso calculable y métrica"
              " no degenerada.", "",
              "## Diagnóstico por métrica (degenerada: > 90 % de los puntos con el mismo valor o mediana del semiancho del IC"
              " de ST > 0,5; sus índices no se interpretan. 1 − Σ S1 es la fracción de varianza en interacciones)", "", _md(D.reset_index(), index=False, floatfmt=".3f"), ""]
    for m in so["metric"].unique():
        d = so[so.metric == m].sort_values("ST", ascending=False)
        lines += [f"## Sobol: {m}" + (" — DEGENERADA, no interpretable" if diag[m]["degenerada"] else ""), "",
                  _md(d[["name", "S1", "S1_conf", "ST", "ST_conf", "ST_ruido", "ST_corregido",
                         "relevante", "interaction"]], index=False, floatfmt=".3f")]
        # fracción de varianza estocástica (intra-punto) de la métrica por corrida
        code = {"f_PRIMARIO": primary_code(run)}.get(m, m[2:] if m.startswith("f_") else None)
        if code and f"obj_{code}" in s:
            y = (s[f"obj_{code}"].astype(str) == "PASS").astype(float)
            w = y.groupby(s.point).var(ddof=1).mean()
            lines += ["", f"fracción de varianza intra-punto (azar) de la corrida: {w / y.var(ddof=1):.3f}"
                      if y.var(ddof=1) > 0 else ""]
        lines += [""]
        plots.plot_sobol(so, out / f"fig_sobol_{m}.png", m)
    return T


def contrast_table(run, a=0, b=1, primary=None):
    """Contraste de dos puntos (C1 vs C2) en todos los objetivos: Fisher bilateral por objetivo y Holm."""
    T = point_table(run, primary)
    A, B = T[T.point == a].iloc[0], T[T.point == b].iloc[0]
    rows = []
    for code in OBJ + ["O17_run", "O17p"]:
        if f"k_{code}" in T and A[f"n_{code}"] > 0 and B[f"n_{code}"] > 0:
            rows.append(dict(objetivo=code, f_A=A[f"f_{code}"], n_A=int(A[f"n_{code}"]), f_B=B[f"f_{code}"],
                             n_B=int(B[f"n_{code}"]), dif=B[f"f_{code}"] - A[f"f_{code}"],
                             p=_fisher(int(B[f"k_{code}"]), int(B[f"n_{code}"]), int(A[f"k_{code}"]), int(A[f"n_{code}"]),
                                       "two-sided")))
    s = run["summary"]
    if "obj_O13b" in s:          # con N0 = 400 O13b sólo puede llegar a MARG (ver la calibración)
        def pm(g):
            v = g["obj_O13b"].astype(str)
            return int(v.isin(["PASS", "MARG"]).sum()), int(v.isin(["PASS", "MARG", "FAIL"]).sum())
        (ka, na), (kb, nb) = pm(s[s.point == a]), pm(s[s.point == b])
        if na and nb:
            rows.append(dict(objetivo="O13b (PASS o MARG)", f_A=ka / na, n_A=na, f_B=kb / nb, n_B=nb,
                             dif=kb / nb - ka / na, p=_fisher(kb, nb, ka, na, "two-sided")))
    for a_ in ANTI:
        if f"obj_{a_}" in s:
            ka = int((s[s.point == a][f"obj_{a_}"].astype(str) == "TRIGGERED").sum()); na = int((s.point == a).sum())
            kb = int((s[s.point == b][f"obj_{a_}"].astype(str) == "TRIGGERED").sum()); nb = int((s.point == b).sum())
            rows.append(dict(objetivo=a_, f_A=ka / na, n_A=na, f_B=kb / nb, n_B=nb, dif=kb / nb - ka / na,
                             p=_fisher(kb, nb, ka, na, "two-sided")))
    # continuas (medianas; Mann–Whitney). x_rho8 y x_D_last sólo existen en colonias extintas: se rotulan
    # "(extintas)" porque condicionan en el evento. t_ext NO se compara entre extintas: log-rank con censura en T
    # y medianas KM (plan §9).
    for c in ("x_rho_pk", "x_f0_pk", "x_BO4", "x_rho8", "x_D_last"):
        if c in s:
            xa = pd.to_numeric(s[s.point == a][c], errors="coerce").dropna()
            xb = pd.to_numeric(s[s.point == b][c], errors="coerce").dropna()
            if len(xa) > 3 and len(xb) > 3:
                lab = c + (" (extintas)" if c in ("x_rho8", "x_D_last") else "")
                rows.append(dict(objetivo=lab, f_A=xa.median(), n_A=len(xa), f_B=xb.median(), n_B=len(xb),
                                 dif=xb.median() - xa.median(), p=float(stats.mannwhitneyu(xb, xa).pvalue)))
    if "t_ext" in s:
        g2 = s[s.point.isin([a, b])]
        ev = g2["extinct"].to_numpy(bool)
        tt = np.where(ev, g2["t_ext"], g2["t_end"]).astype(float)
        lr = survival.logrank(tt, ev, g2["point"].to_numpy())
        med = {}
        for pt in (a, b):
            m_ = g2["point"].to_numpy() == pt
            med[pt] = survival.km_median(survival.kaplan_meier(tt[m_], ev[m_]))["median"]
        rows.append(dict(objetivo="t_ext (mediana KM; log-rank con censura)", f_A=med[a], n_A=int((g2.point == a).sum()),
                         f_B=med[b], n_B=int((g2.point == b).sum()), dif=med[b] - med[a], p=lr["p"]))
    C = pd.DataFrame(rows)
    if len(C):
        C["p_holm"] = multiple.holm(C["p"].fillna(1.0).to_numpy())
        C["difiere_holm_0.05"] = C["p_holm"] < 0.05
    return C, label_of(run, a), label_of(run, b)


def analyze_contrast(run, out, lines):
    T = analyze_generic(run, out, lines)
    C, la, lb = contrast_table(run)
    C.to_csv(out / "contraste.csv", index=False)
    lines += [f"## Contraste {la} (A) frente a {lb} (B): Fisher bilateral por objetivo (Mann–Whitney en continuas), Holm sobre la familia",
              "", _md(C, index=False, floatfmt=".3g"), ""]
    return T


STAGE_FN = dict(c1c2=analyze_contrast, p6=analyze_plane, base=analyze_base, ref=analyze_ref, amax=analyze_amax, morris=analyze_morris, lhs=analyze_lhs,
                lhs2=analyze_lhs, sobol=analyze_sobol, sobolx=analyze_sobol, p1=analyze_plane, p2=analyze_plane,
                p1aw=analyze_plane, p4=analyze_plane, p5=analyze_plane, refine=analyze_generic,
                abl1=analyze_generic, abl2=analyze_generic)


def analyze_stage(stage, run_dir, out=None):
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "analisis"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Fase 2b, etapa `{stage}`: `{run_dir}`", ""]
    try:
        if stage in ("trans", "trgrid"):
            analyze_transplant(run_dir, out, lines)
        else:
            run = load(run_dir)
            st = run["stage"]
            lines += [f"preset {st.get('preset') or st.get('preset_main', '?')}, criterio primario `{primary_code(run)}`, "
                      f"{run['summary']['point'].nunique()} puntos, {len(run['summary'])} corridas, T = {int(run['summary']['T'].max())}", ""]
            STAGE_FN.get(stage, analyze_generic)(run, out, lines)
    except Exception:  # noqa: BLE001
        lines += ["**ERROR en el análisis**", "", "```", traceback.format_exc(), "```"]
    (out / "informe.md").write_text("\n".join(str(x) for x in lines))
    return out


# =========================================================================== Q1–Q7 (confirmatorios)
def _fisher(k1, n1, k2, n2, alt):
    if min(n1, n2) == 0:
        return np.nan
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=alt).pvalue)


def _find(labels, target):
    for i, l in enumerate(labels or []):
        if l == target:
            return i
    return None


def _preset_vals(run):
    """Valores del preset principal de la etapa (stage.json: preset_main), como dict de Params."""
    try:
        from u25.params import Params, PRESETS
        name = (run.get("stage") or {}).get("preset_main")
        return Params(**PRESETS[name]).to_dict() if name in PRESETS else Params().to_dict()
    except Exception:  # noqa: BLE001
        return {}


def q1(root, sfx, tag):
    d = root / f"f2b_trans{sfx}"
    out = dict(Q="Q1", hipotesis="irreversibilidad estructural: los mismos animales D fundan si la colonia nueva no tiene"
               " R1 ni R2 (transplante cruzado, pareado; McNemar exacto)", p=np.nan, lines=[])
    if not (d / "transplant.csv").exists():
        out["nota"] = "falta f2b_trans"
        return out
    tr = pd.read_csv(d / "transplant.csv")
    labels = json.load(open(d / "labels.json"))
    a = _find(labels, tag)
    b = next((i for i, l in enumerate(labels) if l.startswith("X: madre")), None)
    if a is None or b is None:
        out["nota"] = "faltan brazos"
        return out
    ph = "Darg" if (VARIANT in ("ta", "v1") and (tr.phase == "Darg").any()) else "D"   # V2: D = 1,75 t'_pk
    D = tr[tr.phase == ph]
    pa = D[D.point == a][["rep", "success"]].dropna().set_index("rep")["success"].astype(int)
    pb = D[D.point == b][["rep", "success"]].dropna().set_index("rep")["success"].astype(int)
    j = pa.index.intersection(pb.index)
    n01 = int(((pa[j] == 0) & (pb[j] == 1)).sum()); n10 = int(((pa[j] == 1) & (pb[j] == 0)).sum())
    out["p"] = float(stats.binomtest(n01, n01 + n10, 0.5, alternative="greater").pvalue) if n01 + n10 else 1.0
    wa, wb = survival.wilson(int(pa.sum()), len(pa)), survival.wilson(int(pb.sum()), len(pb))
    out["efecto"] = (f"P_T(D) candidato {int(pa.sum())}/{len(pa)} [{wa[0]:.2f}, {wa[1]:.2f}] vs cruzado {int(pb.sum())}/{len(pb)}"
                     f" = {pb.mean():.2f} [{wb[0]:.2f}, {wb[1]:.2f}]; pares discordantes 0→1 = {n01}, 1→0 = {n10}")
    out["prediccion"] = "P_T(D | candidato) ≤ 0,05 y P_T(D | cruzado) ≥ 0,25 (techo por edad: ventana fértil restante 0–33 sem)"
    out["cumple_prediccion"] = bool(pa.mean() <= 0.05 and pb.mean() >= 0.25)
    xa = pa
    # secundarios (exploratorios)
    sec = []
    for i, lab in enumerate(labels):
        if i in (a,):
            continue
        xi = D[D.point == i]["success"].dropna().astype(int)
        sec.append(dict(familia="trans_D_vs_candidato", contraste=f"{lab} vs {tag}", efecto=xi.mean() - xa.mean(),
                        p=_fisher(int(xi.sum()), len(xi), int(xa.sum()), len(xa), "two-sided")))
    out["exploratorio"] = sec
    return out


def q2(root, sfx, tag, primary):
    d = root / f"f2b_abl1{sfx}"
    out = dict(Q="Q2", hipotesis="cada regla es necesaria: quitar cualquiera baja el criterio primario (IUT)", p=np.nan)
    if not (d / "summary.csv").exists():
        out["nota"] = "falta f2b_abl1"
        return out
    run = load(d)
    T = point_table(run, primary)
    c = T[T.label == tag]
    if not len(c):
        out["nota"] = "falta el punto del candidato"
        return out
    c = c.iloc[0]
    ps, eff, sec = [], [], []
    for _, r in T.iterrows():
        lab = r["label"]
        if not lab.startswith(tag + " -") or lab.count(" -") != 1 or " " in lab[len(tag) + 2:]:
            continue
        p = _fisher(int(c["k_PRIMARIO"]), int(c["n_PRIMARIO"]), int(r["k_PRIMARIO"]), int(r["n_PRIMARIO"]), "greater")
        ps.append(p)
        eff.append(f"{lab}: {r['f_PRIMARIO']:.2f} (p={p:.2g})")
        for code in [x for x in ESS + ["O13p", "O8", "O9", "O14", "O15b"] if f"k_{x}" in T]:
            if np.isfinite(c.get(f"f_{code}", np.nan)) and np.isfinite(r.get(f"f_{code}", np.nan)):
                sec.append(dict(familia="ablacion_x_objetivo", contraste=f"{lab} vs {tag}: {code}",
                                efecto=r[f"f_{code}"] - c[f"f_{code}"],
                                p=_fisher(int(r[f"k_{code}"]), int(r[f"n_{code}"]), int(c[f"k_{code}"]), int(c[f"n_{code}"]),
                                          "two-sided")))
    if ps:
        out["p"] = float(np.nanmax(ps))
    out["efecto"] = f"{tag}: {c['f_PRIMARIO']:.2f}; " + "; ".join(eff)
    out["prediccion"] = "f(candidato) ≥ 0,9 y f(sin una regla) ≤ 0,5 para cada regla"
    fs = [float(r["f_PRIMARIO"]) for _, r in T.iterrows() if r["label"].startswith(tag + " -")
          and r["label"].count(" -") == 1 and " " not in r["label"][len(tag) + 2:]]
    out["cumple_prediccion"] = bool(c["f_PRIMARIO"] >= 0.9 and fs and max(fs) <= 0.5)
    out["exploratorio"] = sec + factorial(root, sfx, tag, primary)
    return out


def factorial(root, sfx, tag, primary):
    """Logística de Firth sobre el factorial completo (abl1 + abl2): efectos principales e interacciones dobles."""
    T = []
    for st in ("abl1", "abl2"):
        d = root / f"f2b_{st}{sfx}"
        if (d / "summary.csv").exists():
            T.append(point_table(load(d), primary))
    if len(T) < 2:
        return []
    T = pd.concat(T, ignore_index=True)
    stj = json.load(open(root / f"f2b_abl1{sfx}" / "stage.json")) if (root / f"f2b_abl1{sfx}" / "stage.json").exists() else {}
    rules = stj.get("rules") or ["R1", "R2", "AV", "R7"]
    rows = []
    for _, r in T.iterrows():
        lab = r["label"]
        if lab == tag:
            drop = []
        elif lab.startswith(tag + " -") and all(x in rules for x in lab[len(tag) + 2:].split(" -")):
            drop = lab[len(tag) + 2:].split(" -")
        else:
            continue
        rows.append(dict(k=r["k_PRIMARIO"], n=r["n_PRIMARIO"], **{x: float(x not in drop) for x in rules}))
    F = pd.DataFrame(rows)
    if len(F) < 2 ** len(rules):
        return []
    cols, X = ["(int)"], [np.ones(len(F))]
    for x in rules:
        cols.append(x); X.append(F[x].to_numpy() - 0.5)
    for a, b in [(a, b) for i, a in enumerate(rules) for b in rules[i + 1:]]:
        cols.append(f"{a}:{b}"); X.append((F[a].to_numpy() - 0.5) * (F[b].to_numpy() - 0.5))
    X = np.column_stack(X)
    beta, se, ll, _ = firth_logit(X, F["k"], F["n"])
    return [dict(familia="factorial_logit", contraste=c, efecto=b, se=e, p=2 * stats.norm.sf(abs(b / e)) if e > 0 else np.nan)
            for c, b, e in zip(cols, beta, se) if c != "(int)"]


def q3(root, sfx, tag):
    out = dict(Q="Q3", hipotesis="colapso casi determinista del candidato frente a la línea base del notebook", p=np.nan)
    db, dr = root / f"f2b_base{sfx}", root / f"f2b_ref{sfx}"
    if not ((db / "summary.csv").exists() and (dr / "summary.csv").exists()):
        out["nota"] = "faltan f2b_base o f2b_ref"
        return out
    rb, rr = load(db), load(dr)
    sb, sr = rb["summary"], rr["summary"]
    gb = sb[sb.point == (_find(rb["labels"], tag) or 0)]
    gr = sr[sr.point == (_find(rr["labels"], "notebook alpha=1.4") or 0)]
    kb, kr = int(gb.extinct.sum()), int(gr.extinct.sum())
    out["p"] = _fisher(kb, len(gb), kr, len(gr), "greater")
    te = gb.loc[gb.extinct, "t_ext"].to_numpy(float)
    rng = np.random.default_rng(0)
    cv = np.std(te, ddof=1) / np.mean(te) if te.size > 2 else np.nan
    bs = [np.std(x, ddof=1) / np.mean(x) for x in (rng.choice(te, te.size) for _ in range(2000))] if te.size > 2 else [np.nan]
    out["efecto"] = (f"P_ext {tag} (T={int(gb['T'].max())}) = {kb}/{len(gb)}; notebook (T={int(gr['T'].max())}) = {kr}/{len(gr)};"
                     f" CV(T_ext) {tag} = {cv:.3f} [{np.nanpercentile(bs, 2.5):.3f}, {np.nanpercentile(bs, 97.5):.3f}]")
    out["prediccion"] = "P_ext(candidato) ≥ 0,99, CV(T_ext) ≤ 0,10; notebook P_ext(1000) ≤ 0,7"
    out["cumple_prediccion"] = bool(kb / len(gb) >= 0.99 and cv <= 0.10 and kr / len(gr) <= 0.7)
    sec = []
    for name, g in (("candidato", gb), ("notebook", gr)):
        ev = g.extinct.to_numpy(bool)
        tt = np.where(ev, g.t_ext, g.t_end).astype(float)
        if ev.sum() >= 3:
            fw = survival.fit_parametric(tt, ev, "weibull"); fe = survival.fit_parametric(tt, ev, "exponential")
            sec.append(dict(familia="supervivencia", contraste=f"Weibull k ({name}, todo t) = {fw['params']['shape']:.2f} vs k = 1",
                            efecto=fw["params"]["shape"], p=survival.lr_test(fe, fw)["p"]))
            mm = main_mode_weibull(tt, ev)
            if "weibull_k_modo" in mm:
                sec.append(dict(familia="supervivencia",
                                contraste=(f"modelo de curación ({name}): Weibull k = {mm['weibull_k_modo']:.2f}"
                                           f" (CV {mm['cv_weibull_modo']:.3f}), π_escape = {mm['pi_escape']:.3f}; LR de π = 0"),
                                efecto=mm["pi_escape"], p=mm["p_pi0"]))
    both = pd.concat([gb.assign(grp=0), gr.assign(grp=1)])
    ev = both.extinct.to_numpy(bool)
    lr = survival.logrank(np.where(ev, both.t_ext, both.t_end).astype(float), ev, both.grp.to_numpy())
    sec.append(dict(familia="supervivencia", contraste=f"log-rank {tag} vs notebook", efecto=lr["chi2"], p=lr["p"]))
    gl = sr[sr.point == (_find(rr["labels"], "longevo") or 1)]
    sec.append(dict(familia="supervivencia", contraste=f"P_ext {tag} vs longevo puro",
                    efecto=gb.extinct.mean() - gl.extinct.mean(),
                    p=_fisher(kb, len(gb), int(gl.extinct.sum()), len(gl), "greater")))
    out["exploratorio"] = sec
    return out


def _amax_expected(a_max, k_mort, n):
    """E[máx de n vidas] con el riesgo logístico discreto del modelo (muerte tras envejecer)."""
    a = np.arange(1, int(a_max + 20 / k_mort) + 1, dtype=float)
    h = 1 / (1 + np.exp(-k_mort * (a - a_max)))
    S = np.cumprod(1 - h)
    n = max(1.0, float(n))
    return float(np.sum(1 - (1 - S) ** n))


def q4(run, lo=0.8, hi=1.25):
    out = dict(Q="Q4", hipotesis="ley de la fase D: T_ext − t_última_cría crece 1:1 con a_max (TOST [0,8; 1,25])",
               p=np.nan, lines=[])
    s = run["summary"].copy()
    s = s[s.extinct & s["x_D_last"].notna()] if "x_D_last" in s else s.iloc[0:0]
    if len(s) < 10 or "a_max" not in s:
        out["nota"] = "datos insuficientes"
        return out
    x = s["a_max"].to_numpy(float); y = s["x_D_last"].to_numpy(float)
    beta, se, res = ols(np.column_stack([np.ones_like(x), x]), y)
    b, e = beta[1], se[1]
    p_lo = stats.norm.sf((b - lo) / e); p_hi = stats.norm.sf((hi - b) / e)
    out["p"] = float(max(p_lo, p_hi))
    km = run["params"][0].get("k_mort", 0.15) if run.get("params") else 0.15
    pred = np.array([_amax_expected(a, km, n) for a, n in zip(x, s["x_n_last"].fillna(1).to_numpy(float))])
    r = y - pred
    out["efecto"] = (f"pendiente {b:.3f} ± {e:.3f} (HC3), intercepto {beta[0]:.1f}; residuo frente a E[máx de n_last vidas]"
                     f" = {r.mean():+.1f} ± {r.std(ddof=1) / np.sqrt(len(r)):.1f} sem (sd {r.std(ddof=1):.1f})")
    out["prediccion"] = "pendiente ∈ [0,8; 1,25] (teoría: 1); residuo medio |r| ≤ 10 semanas"
    out["cumple_prediccion"] = bool(lo <= b <= hi and abs(r.mean()) <= 10)
    out["lines"] = [f"## Q4 (ley de la fase D): {out['efecto']}; p_TOST = {out['p']:.3g}", ""]
    out["exploratorio"] = [dict(familia="fase_D", contraste="residuo medio vs 0 (E[máx de n vidas])", efecto=r.mean(),
                                p=float(stats.ttest_1samp(r, 0).pvalue))]
    return out


def q5(root, sfx, primary, code="O13"):
    out = dict(Q="Q5", hipotesis="la banda de O13 está centrada en n* = 1/κ − 1/α ≈ m_c (máximo interior en log n*)",
               p=np.nan)
    d = root / f"f2b_p2{sfx}"
    if not (d / "summary.csv").exists():
        out["nota"] = "falta f2b_p2"
        return out
    run = load(d)
    T = point_table(run, primary)
    stp = _preset_vals(run)
    T = T[(T.crowd_aversion > 0) & (T.alpha > 0)]
    if "crowd_power" in T and stp:
        T = T[np.isclose(T.crowd_power.astype(float), float(stp.get("crowd_power", 4.0)))]
    nstar = 1 / T.crowd_aversion.astype(float) - 1 / T.alpha.astype(float)
    T = T[nstar > 0.5].assign(u=np.log(nstar[nstar > 0.5]))
    if len(T) < 6 or f"k_{code}" not in T:
        out["nota"] = "pocos puntos"
        return out
    X = np.column_stack([np.ones(len(T)), T.u, T.u ** 2])
    k, n = T[f"k_{code}"].to_numpy(float), T[f"n_{code}"].to_numpy(float)
    beta, se, ll, _ = firth_logit(X, k, n)
    out["p"] = float(stats.norm.cdf(beta[2] / se[2]))     # H1: b2 < 0
    rng = np.random.default_rng(0)
    p_hat = np.clip(k / np.maximum(n, 1), 1e-3, 1 - 1e-3)
    opt = []
    for _ in range(500):
        kb = rng.binomial(n.astype(int), p_hat)
        bb, *_ = firth_logit(X, kb, n)
        if bb[2] < 0:
            opt.append(np.exp(-bb[1] / (2 * bb[2])))
    vtx = np.exp(-beta[1] / (2 * beta[2])) if beta[2] < 0 else np.nan
    ci = (np.percentile(opt, 2.5), np.percentile(opt, 97.5)) if len(opt) > 50 else (np.nan, np.nan)
    out["efecto"] = f"b2 = {beta[2]:.2f} ± {se[2]:.2f}; n*_opt = {vtx:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] (bootstrap paramétrico)"
    out["prediccion"] = "b2 < 0 y IC de n*_opt ⊂ [4, 12] (m_c = 8)"
    out["cumple_prediccion"] = bool(beta[2] < 0 and ci[0] >= 4 and ci[1] <= 12)
    fr = k / np.maximum(n, 1)
    phi, _, _ = dispersion(X, k, n, beta)
    low = T.loc[fr < 0.8, ["alpha", "crowd_aversion"]].astype(float)
    out["nota"] = (f"{code} saturado: {int((fr >= 0.95).sum())}/{len(fr)} puntos con f ≥ 0,95; f < 0,8 sólo en "
                   + (", ".join(f"(α={a:g}, κ={c:g})" for a, c in low.to_numpy()) or "ninguno")
                   + f"; sobredispersión φ = {phi:.2f}")
    # exploratorio: f0 crece con alpha y decrece con kappa (Spearman sobre el plano completo)
    T2 = point_table(run, primary)
    sec = []
    for ax, sign in (("alpha", +1), ("crowd_aversion", -1)):
        if "x_f0_pk_mean" in T2:
            r, p = stats.spearmanr(T2[ax].astype(float), T2["x_f0_pk_mean"], nan_policy="omit")
            sec.append(dict(familia="planos_tendencias", contraste=f"f0(t_pk) vs {ax} (signo esperado {sign:+d})", efecto=r, p=p))
    out["exploratorio"] = sec
    return out


def q6(root, sfx, primary):
    out = dict(Q="Q6", hipotesis="la ventana a_w pesa menos que la tasa rec (θ < 1 en z = log rec + θ log a_w)", p=np.nan)
    d = root / f"f2b_p1aw{sfx}"
    if not (d / "summary.csv").exists():
        out["nota"] = "falta f2b_p1aw"
        return out
    T = point_table(load(d), primary)
    T = T[np.isfinite(pd.to_numeric(T.plast_age_max, errors="coerce"))]
    if len(T) < 8:
        out["nota"] = "pocos puntos"
        return out
    lr_, la = np.log(T.rec.astype(float)), np.log(T.plast_age_max.astype(float))
    k, n = T.k_PRIMARIO.to_numpy(float), T.n_PRIMARIO.to_numpy(float)

    def prof(th):
        z = lr_ + th * la
        z = (z - z.mean()) / (z.std() or 1)
        X = np.column_stack([np.ones(len(z)), z, z ** 2])
        return firth_logit(X, k, n)[2]
    grid = np.linspace(-1, 3, 81)
    ll = np.array([prof(t) for t in grid])
    th = grid[np.argmax(ll)]
    ll1 = prof(1.0)
    stat = max(0.0, 2 * (ll.max() - ll1))
    p2 = stats.chi2.sf(stat, 1)
    out["p"] = float(p2 / 2 if th < 1 else 1 - p2 / 2)
    inside = grid[2 * (ll.max() - ll) <= stats.chi2.ppf(0.95, 1)]
    # robustez: sobredispersión de Pearson en θ̂ (los puntos del plano no son réplicas) -> LR/φ y IC con LR/φ
    z = lr_ + th * la
    z = (z - z.mean()) / (z.std() or 1)
    Xh = np.column_stack([np.ones(len(z)), z, z ** 2])
    phi, _, _ = dispersion(Xh, k, n, firth_logit(Xh, k, n)[0])
    p2q = stats.chi2.sf(stat / phi, 1)
    p_q = float(p2q / 2 if th < 1 else 1 - p2q / 2)
    inside_q = grid[2 * (ll.max() - ll) / phi <= stats.chi2.ppf(0.95, 1)]
    out["efecto"] = (f"θ̂ = {th:.2f} [IC perfil 95 % {inside.min():.2f}, {inside.max():.2f}]; LR(θ=1) = {stat:.1f};"
                     f" cuasi-binomial φ = {phi:.2f}: IC [{inside_q.min():.2f}, {inside_q.max():.2f}], p = {p_q:.2g}")
    out["prediccion"] = "θ < 1 (aprendizaje concentrado antes del destete; teoría de campo medio: θ = s̄_V(a_w)/⟨s̄_V⟩)"
    out["cumple_prediccion"] = bool(0 < th < 1 and inside.max() < 1)
    return out


def q7(root, sfx, primary, w0=None, rec0=None):
    out = dict(Q="Q7", hipotesis="colapso endógeno sin hacinamiento (γ = 0) con socialización subcrítica", p=np.nan)
    d = root / f"f2b_p1{sfx}"
    if not (d / "summary.csv").exists():
        out["nota"] = "falta f2b_p1"
        return out
    run = load(d)
    T = point_table(run, primary)
    T = T[T.gamma.astype(float) == 0.0]
    if not len(T):
        out["nota"] = "sin capa γ = 0"
        return out
    pr = _preset_vals(run)
    w0 = w0 if w0 is not None else pr.get("inherit_weight", 0.2)
    rec0 = rec0 if rec0 is not None else pr.get("rec", 0.1)
    corner = T[(T.inherit_weight.astype(float) <= 0.1) & (T.rec.astype(float) <= 0.03)]
    cand = T[np.isclose(T.inherit_weight.astype(float), w0) & np.isclose(T.rec.astype(float), rec0)]
    if not len(corner) or not len(cand):
        out["nota"] = "faltan puntos"
        return out
    k1, n1 = int(corner.k_ext.sum()), int(corner.n.sum())
    k2, n2 = int(cand.k_ext.sum()), int(cand.n.sum())
    out["p"] = _fisher(k1, n1, k2, n2, "greater")
    out["efecto"] = f"P_ext(γ=0) esquina subcrítica {k1}/{n1} vs candidato {k2}/{n2} (explota: {cand.f_exploded.mean():.2f})"
    out["prediccion"] = "esquina ≥ 0,8; candidato ≤ 0,2"
    out["cumple_prediccion"] = bool(k1 / n1 >= 0.8 and k2 / n2 <= 0.2)
    # exploratorio: frontera logística en (w, Λ = rec·a_w)
    a_w = float(pr.get("plast_age_max") or 12.0)
    X = np.column_stack([np.ones(len(T)), T.inherit_weight.astype(float), T.rec.astype(float) * a_w])
    beta, se, ll, _ = firth_logit(X, T.k_ext.to_numpy(float), T.n.to_numpy(float))
    out["exploratorio"] = [dict(familia="frontera_gamma0", contraste=f"logit P_ext ~ w + Λ: coef {c}", efecto=b, se=e,
                                p=2 * stats.norm.sf(abs(b / e)) if e > 0 else np.nan)
                           for c, b, e in zip(["w", "Λ"], beta[1:], se[1:])]
    if beta[2] != 0 and beta[1] != 0:
        c0 = -beta[0] / beta[1]
        out["exploratorio"].append(dict(familia="frontera_gamma0", contraste="φ̂ = b_Λ/b_w (frontera w + φΛ = cte)",
                                        efecto=beta[2] / beta[1], p=np.nan))
        out["exploratorio"].append(dict(familia="frontera_gamma0",
                                        contraste=("cte de la frontera P_ext = 1/2: w + φ̂Λ = c; normalizada a la forma de campo"
                                                   f" medio σw + φΛ = 1: σ̂ = 1/c = {1 / c0:.2f}, φ̂/c = {beta[2] / beta[1] / c0:.2f}"),
                                        efecto=c0, p=np.nan))
    return out


def final(root, sfx="", out=None, tag=None, primary=None):
    root = Path(root)
    out = Path(out) if out else root / f"f2b_final{sfx}"
    out.mkdir(parents=True, exist_ok=True)
    stj = {}
    for st in ("base", "c1c2", "abl1", "abl2", "trans", "amax", "morris", "p2", "p1", "p1aw", "p5", "lhs", "sobol", "p4"):
        f = root / f"f2b_{st}{sfx}" / "stage.json"
        if f.exists():
            stj = json.load(open(f))
            if stj.get("primary"):
                break
    tag = tag or stj.get("tag") or "C1"
    primary = primary or stj.get("primary") or "O17_run"
    lines = [f"# Fase 2b: confirmatorios Q1–Q7 y exploratorios (candidato {tag}, criterio primario {primary})", ""]
    Qs = []
    for fn, args in ((q1, (root, sfx, tag)), (q2, (root, sfx, tag, primary)), (q3, (root, sfx, tag)),
                     (None, None), (q5, (root, sfx, primary)), (q6, (root, sfx, primary)), (q7, (root, sfx, primary))):
        try:
            if fn is None:
                d = root / f"f2b_amax{sfx}"
                Qs.append(q4(load(d)) if (d / "summary.csv").exists() else dict(Q="Q4", p=np.nan, nota="falta f2b_amax"))
            else:
                Qs.append(fn(*args))
        except Exception as e:  # noqa: BLE001
            Qs.append(dict(Q=f"Q{len(Qs) + 1}", p=np.nan, nota=f"error {type(e).__name__}: {e}"))
    Q = pd.DataFrame([{k: v for k, v in q.items() if k not in ("exploratorio", "lines")} for q in Qs])
    ok = Q["p"].notna().to_numpy()
    Q["p_holm"] = np.nan
    if ok.any():
        Q.loc[ok, "p_holm"] = multiple.holm(Q.loc[ok, "p"].to_numpy())
    Q["rechaza_H0_holm_0.05"] = Q["p_holm"] < 0.05
    Q.to_csv(out / "Q.csv", index=False)
    cols = [c for c in ["Q", "hipotesis", "efecto", "prediccion", "p", "p_holm", "rechaza_H0_holm_0.05", "cumple_prediccion", "nota"] if c in Q]
    lines += ["## Familia confirmatoria (Holm, α = 0,05; p faltantes excluidos de la familia)", "", _md(Q[cols], index=False, floatfmt=".3g"), ""]
    dc = root / f"f2b_c1c2{sfx}"
    if (dc / "summary.csv").exists():
        try:
            C, la, lb = contrast_table(load(dc), primary=primary)
            C.to_csv(out / "contraste_c1c2.csv", index=False)
            lines += [f"## Contraste confirmatorio {la} frente a {lb} (todos los objetivos; Holm sobre esta familia)", "",
                      _md(C, index=False, floatfmt=".3g"), ""]
        except Exception as e:  # noqa: BLE001
            lines += [f"(contraste C1–C2: {type(e).__name__}: {e})", ""]
    E = pd.DataFrame([e for q in Qs for e in q.get("exploratorio", [])])
    if len(E):
        E["p_bh_familia"] = np.nan
        for fam, g in E.groupby("familia"):
            m = g["p"].notna()
            if m.any():
                E.loc[g.index[m], "p_bh_familia"] = multiple.bh(g.loc[m, "p"].to_numpy())
        m = E["p"].notna()
        E["p_bh_global"] = np.nan
        E.loc[m, "p_bh_global"] = multiple.bh(E.loc[m, "p"].to_numpy())
        E["descubrimiento_FDR_0.10"] = E["p_bh_familia"] < 0.10
        E.to_csv(out / "exploratorio.csv", index=False)
        lines += ["## Exploratorios (BH al 10 % por familia y global)", "", _md(E, index=False, floatfmt=".3g"), ""]
    # KM consolidado
    try:
        groups = {}
        for st, labs in (("base", None), ("ref", None), ("abl1", None)):
            d = root / f"f2b_{st}{sfx}"
            if not (d / "summary.csv").exists():
                continue
            r = load(d)
            s = r["summary"]
            for pt in sorted(s.point.unique()):
                lab = label_of(r, pt)
                if st == "abl1" and (lab.count(" -") != 1):
                    continue
                if st == "base" and "alpha=0" in lab:
                    continue
                g = s[s.point == pt]
                ev = g.extinct.to_numpy(bool)
                groups[lab] = (np.where(ev, g.t_ext, g.t_end).astype(float), ev)
        if groups:
            km_figure(groups, out / "fig_km_consolidado.png", tmax=1000)
            lines += ["![KM consolidado](fig_km_consolidado.png)", ""]
    except Exception as e:  # noqa: BLE001
        lines += [f"(KM consolidado: {e})"]
    (out / "informe_Q.md").write_text("\n".join(str(x) for x in lines))
    return out
