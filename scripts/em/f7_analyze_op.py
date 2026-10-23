"""Tarea 4 (E4): definiciones operativas de theta* y nu* y su valor predictivo.

Indicadores semanales (f7_common.op_observables), medidos sobre los juveniles en la ventana (edad 1..12):
  theta_eps  fracción de los otros del bloque 3x3 con s > eps (eps = 0, 0.05, s_v)
  theta_R    fracción de los vecinos en edad reproductiva que criaron en las últimas 4 semanas (medible: hembras
             preñadas/lactantes, machos que engendraron)
  D_J        exceso medio de ocupación (m - m_c)_+ de los juveniles  -> criterio de densidad gamma D_J > r
  H_J        fracción de juveniles en celdas con m <= m_c (bolsones sanos)
  mstar_J    hacinamiento medio experimentado (Lloyd) por los juveniles
y, a nivel poblacional, frac_rep_bred (fracción de los adultos en edad reproductiva que criaron en 4 semanas).

Umbrales:
  theta*_eps = (1 - w)/(r a_w) - eps/s_v   (Teorema 4.5 con theta_eps; ver informe_f7.md §4), = 2/3, 0.461, -0.33
  D_J > r/gamma                            (equivale a nu > nu* = 6.92 con ocupación Poisson)
Alarmas por corrida (t >= 15): primer t con theta_eps < theta*_eps; primer t con D_J > r/gamma sostenido 4 semanas;
"no retorno": frac_rep_bred < 0.02 durante 20 semanas seguidas con D_J < r/gamma (adultos que no vuelven a criar
cuando el hacinamiento ya cedió).
Salidas: results_em/operational_indicators/indicadores.csv (una fila por corrida), tabla_alarmas.csv y figura
results_em/extra_figs/fig_operational.pdf.
Uso: python3 scripts/em/f7_analyze_op.py
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT, S_V, group_dir
from f7_aggregate import parse_cfg, wilson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "results_em", "operational_indicators")
FIG = os.path.join(ROOT, "results_em", "extra_figs")   # figuras de otro artículo (no usadas en el de EM)
W, R, AW, GAM = 0.2, 0.1, 12, 0.1
TH = {0.0: (1 - W) / (R * AW), 0.05: (1 - W) / (R * AW) - 0.05 / S_V}
H = 2000


def first_true(t, cond, tmin=15, sustain=1):
    ok = cond & (t >= tmin)
    if sustain > 1:
        run = np.convolve(ok.astype(int), np.ones(sustain, int), "valid") == sustain
        idx = np.flatnonzero(run)
        return int(t[idx[0]]) if idx.size else -1
    idx = np.flatnonzero(ok)
    return int(t[idx[0]]) if idx.size else -1


def features(z):
    t = z["o_t"].astype(int)
    d = dict(extinct=bool(z["extinct"]), t_ext=float(z["t_ext"]), t_last=int(z["t_last"]))
    T_end = int(z["t_end"])
    d["ext_H"] = bool(z["extinct"]) and float(z["t_ext"]) <= H
    B = z["s_B"].astype(float); tb = z["s_t"]
    ib = int(np.argmax(B))
    off = np.flatnonzero((np.arange(B.size) > ib) & (B < 0.05 * B.max()))
    d["t_off"] = int(tb[off[0]]) if off.size else -1
    dense = t <= 400
    t = t[dense]
    g = {k: z[f"o_{k}"][dense].astype(float) for k in ("theta_0", "theta_0.05", "theta_0.243", "theta_R", "D_J", "H_J",
                                                     "mstar_J", "frac_rep_bred", "N") if f"o_{k}" in z.files}
    def mavg(x, w=AW):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        c = np.convolve(np.where(ok, x, 0), np.ones(w), "same"); n = np.convolve(ok.astype(float), np.ones(w), "same")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= w / 2, c / n, np.nan)
    for e, thr in TH.items():
        key = f"theta_{e:g}"
        if key in g:
            x = np.nan_to_num(g[key], nan=1.0)
            d[f"t_{key}"] = first_true(t, x < thr, sustain=4)
            mv = mavg(g[key])
            d[f"t_{key}_w12"] = first_true(t, np.nan_to_num(mv, nan=1.0) < thr)
            for thr2 in (0.25, 0.1):
                d[f"t_{key}_w12_{thr2:g}"] = first_true(t, np.nan_to_num(mv, nan=1.0) < thr2)
            d[f"min_{key}"] = float(np.nanmin(mv[t >= 10])) if np.isfinite(mv[t >= 10]).any() else np.nan
    for key in ("theta_0.243", "theta_R"):
        if key in g:
            d[f"min_{key}"] = float(np.nanmin(g[key][t >= 10])) if np.isfinite(g[key][t >= 10]).any() else np.nan
            for thr in (0.1, 0.2, 1 / 3):
                d[f"t_{key}_{thr:.2f}"] = first_true(t, np.nan_to_num(g[key], nan=1.0) < thr, sustain=4)
    if "D_J" in g:
        DJ = np.nan_to_num(g["D_J"], nan=0.0)
        d["t_D"] = first_true(t, GAM * DJ > R, sustain=4)
        over = GAM * DJ > R
        # meseta más larga con D_J > r/gamma
        best = cur = 0
        for v in over:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        d["plateau_D"] = best
        d["weeks_D"] = int(over.sum())
        d["min_H_J"] = float(np.nanmin(g["H_J"][t >= 10])) if "H_J" in g else np.nan
        d["max_mstar_J"] = float(np.nanmax(g["mstar_J"])) if "mstar_J" in g else np.nan
        # no retorno: tras el pico, 20 semanas seguidas con frac_rep_bred < 0.02 y D_J < r/gamma
        if "frac_rep_bred" in g:
            fb = np.nan_to_num(g["frac_rep_bred"], nan=0.0)
            cond = (fb < 0.02) & (~over) & (g["N"] > 0)
            d["t_noreturn"] = first_true(t, cond, tmin=20, sustain=20)
            # máxima fracción de cría tardía (t > t_off + 30)
            if d["t_off"] > 0:
                late = t > d["t_off"] + 30
                d["late_bred_max"] = float(np.nanmax(fb[late])) if late.any() else np.nan
    return d


def collect(groups=("f7_e1",), Ls=(30,)):
    rows = []
    for grp in groups:
        for dd in sorted(glob.glob(os.path.join(group_dir(grp), "*"))):
            if not os.path.isdir(dd):
                continue
            cfg = os.path.basename(dd)
            info = parse_cfg(cfg)
            if info.get("L") not in Ls:
                continue
            for f in sorted(glob.glob(os.path.join(dd, "seed*.npz"))):
                if f.endswith(".tmp.npz"):
                    continue
                z = np.load(f)
                if "o_t" not in z.files:
                    continue
                r = features(z)
                r.update(info); r["group"] = grp; r["seed"] = int(re.findall(r"seed(\d+)", f)[0])
                rows.append(r)
    return pd.DataFrame(rows)


def _subsets(df):
    """Todas las corridas, y sólo lambda = 1 (donde theta* está definido: aprendizaje puramente social)."""
    for (sp, var), g in df.groupby(["space", "variant"]):
        yield (sp, var, "todas"), g
        yield (sp, var, "lambda=1"), g[g.lam == 1.0]


def auc(score_ext, score_pers):
    """P(score de una extinta < score de una persistente) + 1/2 empates (Mann-Whitney)."""
    a = np.asarray(score_ext, float); b = np.asarray(score_pers, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    lt = (a[:, None] < b[None, :]).mean(); eq = (a[:, None] == b[None, :]).mean()
    return float(lt + 0.5 * eq)


def auc_table(df):
    rows = []
    scores = [("min_theta_0", "min theta_0 (t>=10)"), ("min_theta_0.05", "min theta_0.05"),
              ("min_theta_0.243", "min theta_sv"), ("min_theta_R", "min theta_R"), ("min_H_J", "min H_J"),
              ("late_bred_max", "late breeding fraction (t > t_off+30)")]
    for (sp, var, sub), g in _subsets(df):
        e = g[g.ext_H]; p = g[~g.ext_H]
        for c, name in scores:
            if c not in g:
                continue
            rows.append(dict(space=sp, variant=var, sub=sub, score=name, n_ext=len(e), n_pers=len(p), AUC=auc(e[c], p[c]),
                             ext_med=float(e[c].median()), ext_p90=float(e[c].quantile(0.9)),
                             pers_med=float(p[c].median()) if len(p) else np.nan,
                             pers_p10=float(p[c].quantile(0.1)) if len(p) else np.nan))
    return pd.DataFrame(rows)


def alarm_table(df):
    """Para cada indicador y variante: tasa de acierto (alarma antes del fin de la natalidad en las que se extinguen),
    tasa de falsas alarmas (alarma en las que persisten) y adelanto mediano."""
    out = []
    inds = [("t_theta_0", "theta_0 < 2/3 (4 sem.)"), ("t_theta_0_w12", "media 12 sem. de theta_0 < 2/3"),
            ("t_theta_0.05", "theta_0.05 < 0.461 (4 sem.)"), ("t_theta_0.05_w12", "media 12 sem. de theta_0.05 < 0.461"),
            ("t_theta_0.05_w12_0.25", "media 12 sem. de theta_0.05 < 0.25"),
            ("t_theta_0.05_w12_0.1", "media 12 sem. de theta_0.05 < 0.1"),
            ("t_theta_R_0.33", "theta_R < 1/3"),
            ("t_theta_R_0.20", "theta_R < 0.2"), ("t_D", "gamma D_J > r (4 sem.)"), ("t_noreturn", "no retorno (20 sem.)")]
    for (sp, var, sub), g in _subsets(df):
        ext = g[g.ext_H]; per = g[~g.ext_H]
        for col, name in inds:
            if col not in g:
                continue
            fired_e = (ext[col] > 0)
            before = fired_e & (ext[col] <= ext.t_off.where(ext.t_off > 0, 10 ** 6))
            lead = (ext.t_off - ext[col])[fired_e & (ext.t_off > 0)]
            lead_ext = (ext.t_ext - ext[col])[fired_e]
            fa = (per[col] > 0)
            out.append(dict(space=sp, variant=var, sub=sub, indicador=name, n_ext=len(ext), n_pers=len(per),
                            hit=float(fired_e.mean()) if len(ext) else np.nan,
                            hit_before_off=float(before.mean()) if len(ext) else np.nan,
                            lead_off_med=float(lead.median()) if len(lead) else np.nan,
                            lead_ext_med=float(lead_ext.median()) if len(lead_ext) else np.nan,
                            false_alarm=float(fa.mean()) if len(per) else np.nan))
    return pd.DataFrame(out)


def fig_operational(df):
    """Trayectorias medias de los indicadores en C1 (espacial y b.m.) y en puntos que persisten."""
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, gridspec_kw=dict(hspace=0.25, wspace=0.42))
    cases = [("sp", "cell", 1.0, 0.0, "#b2182b", "spatial C1 (collapses)"),
             ("sp", "cell", 0.9, 0.6, "#2166ac", r"spatial, $\lambda$=0.9, $\epsilon_{\rm ad}$=0.6 (persists)"),
             ("wm", "cell", 1.0, 0.0, "#ef8a62", "well-mixed C1 (collapses)")]
    keys = [("theta_0", r"$\theta_0$ (neighbours with $s>0$)", TH[0.0]), ("theta_0.05", r"$\theta_{0.05}$", TH[0.05]),
            ("theta_R", r"$\theta_R$ (bred in last 4 wk)", None), ("D_J", r"$D_J$ = mean $(m-m_c)_+$, juveniles", R / GAM),
            ("H_J", r"$H_J$ = juveniles with $m\leq m_c$", None), ("frac_rep_bred", "adults that bred in last 4 wk", None)]
    base = os.path.join(ROOT, "results_em", "adult_recovery")
    for (sp, var, lam, out, col, lab) in cases:
        d = os.path.join(base, f"C1_{sp}_{var}_l{lam:g}_o{out:g}_L30")
        fs = [f for f in glob.glob(os.path.join(d, "seed*.npz")) if not f.endswith(".tmp.npz")]
        if not fs:
            continue
        for ax, (k, name, thr) in zip(axs.ravel(), keys):
            M = np.full((len(fs), 201), np.nan)
            for i, f in enumerate(fs):
                z = np.load(f)
                t = z["o_t"].astype(int); v = z[f"o_{k}"]
                m = t <= 200
                M[i, t[m]] = v[m]
            med = np.nanmedian(M, 0); q1, q9 = np.nanpercentile(M, [10, 90], 0)
            tt = np.arange(201)
            ax.plot(tt, med, color=col, lw=1.2, label=lab)
            ax.fill_between(tt, q1, q9, color=col, alpha=0.18, lw=0)
            if thr is not None:
                ax.axhline(thr, color="#555", lw=0.7, ls="--")
            ax.set_title(name, fontsize=7.5)
    for ax in axs[1]:
        ax.set_xlabel("week")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_operational.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = collect()
    df.to_csv(os.path.join(OUT, "indicadores.csv"), index=False)
    at = alarm_table(df)
    at.to_csv(os.path.join(OUT, "tabla_alarmas.csv"), index=False)
    print(at.round(2).to_string())
    au = auc_table(df)
    au.to_csv(os.path.join(OUT, "tabla_auc.csv"), index=False)
    print(au.round(3).to_string())
    # adelanto del indicador de reversibilidad: medible en t_off + 30, frente a T_ext
    e = df[df.ext_H & (df.t_off > 0)]
    lead = (e.t_ext - (e.t_off + 30))
    print("adelanto (T_ext - (t_off+30)) por variante:")
    print(e.assign(lead=lead).groupby(["space", "variant"]).lead.describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string())
    fig_operational(df)


if __name__ == "__main__":
    main()
