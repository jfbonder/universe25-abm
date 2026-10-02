"""Figuras del pipeline (matplotlib, backend Agg)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_km(surv, path, title="Kaplan-Meier del tiempo de extinción", log=False):
    km = surv["km"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].step(km["t"], km["S"], where="post", lw=2, label="KM")
    ax[0].fill_between(km["t"], km["lo"], km["hi"], step="post", alpha=0.25, label="IC 95% (Greenwood)")
    tmax = km["t"].max() * 1.05
    for dist, f in surv.get("fits", {}).items():
        if f.get("sf") is None:
            continue
        tt = np.linspace(f["t0"], tmax, 200)
        S0 = km["S"].to_numpy()[np.searchsorted(km["t"].to_numpy(), f["t0"], "right") - 1]
        ax[0].plot(tt, S0 * f["sf"](tt), "--", lw=1.2, label=f"{dist} (t>{f['t0']:.0f})")
    ax[0].set_xlabel("semana"); ax[0].set_ylabel("P(colonia viva)"); ax[0].set_title(title); ax[0].legend(fontsize=8)
    if log:
        ax[0].set_yscale("log")
    # Nelson-Aalen y hazard a trozos
    ax[1].step(km["t"], km["H"], where="post", lw=2)
    ax[1].set_xlabel("semana"); ax[1].set_ylabel("H(t) (Nelson-Aalen)"); ax[1].set_title("hazard acumulado")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def plot_morris(df, path, metric):
    d = df[df["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(d["mu_star"], d["sigma"], xerr=d["mu_star_conf"], fmt="o", capsize=3)
    for _, r in d.iterrows():
        ax.annotate(r["name"], (r["mu_star"], r["sigma"]), textcoords="offset points", xytext=(4, 4), fontsize=9)
    m = max(d["mu_star"].max(), d["sigma"].max()) * 1.05
    ax.plot([0, m], [0, m], "k:", lw=0.8, label="σ = μ*")
    ax.set_xlabel("μ* (importancia)"); ax.set_ylabel("σ (interacciones / no linealidad)"); ax.set_title(f"Morris: {metric}")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def plot_sobol(df, path, metric):
    d = df[df["metric"] == metric].sort_values("ST", ascending=False)
    x = np.arange(len(d)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, d["S1"], w, yerr=d["S1_conf"], capsize=3, label="S1")
    ax.bar(x + w / 2, d["ST"], w, yerr=d["ST_conf"], capsize=3, label="ST")
    ax.set_xticks(x); ax.set_xticklabels(d["name"], rotation=30); ax.set_ylabel("índice de Sobol"); ax.set_title(f"Sobol: {metric}")
    ax.axhline(0, color="k", lw=0.5); ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def plot_peaks(series, cyc, path, max_runs=12, **kw):
    from .peaks import find_cycles
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    n = 0
    for (pt, rep), g in series.groupby(["point", "rep"]):
        N = g["N"].to_numpy(float); t = g["t"].to_numpy()
        pk, _ = find_cycles(t, N, **kw)
        if len(pk) >= 3:
            ax[0].plot(np.arange(1, len(pk) + 1), N[pk], "o-", alpha=0.6, lw=1)
            n += 1
        if n >= max_runs:
            break
    ax[0].set_xlabel("número de pico"); ax[0].set_ylabel("N en el pico"); ax[0].set_title(f"picos sucesivos ({n} corridas)")
    d = cyc["slope_log"].dropna()
    if len(d):
        ax[1].hist(d, bins=20, color="C0", alpha=0.8); ax[1].axvline(0, color="k")
        ax[1].set_xlabel("pendiente de log(pico) por ciclo (k≥2)"); ax[1].set_title("drift entre picos por corrida")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def plot_series_sample(series, path, n=6, cols=("N", "S_mean")):
    fig, ax = plt.subplots(len(cols), 1, figsize=(11, 3 * len(cols)), sharex=True)
    ax = np.atleast_1d(ax)
    for i, ((pt, rep), g) in enumerate(series.groupby(["point", "rep"])):
        if i >= n:
            break
        for a, c in zip(ax, cols):
            a.plot(g["t"], g[c], lw=0.9, alpha=0.8, label=f"p{pt} r{rep}")
    for a, c in zip(ax, cols):
        a.set_ylabel(c)
    ax[-1].set_xlabel("semana"); ax[0].legend(fontsize=7, ncol=3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
