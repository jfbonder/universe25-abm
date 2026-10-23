"""Tarea 3 (E4/P2): rescate por retiro de adultos no reproductores (efecto hidra). Análisis y figura.

Entradas: results_em/hydra_removal/summary.csv (f7_aggregate.py hyd), results_em/kinetic_grid/hyd*.jsonl
Salidas:  results_em/hydra_removal/mapa.csv; results_em/extra_figs/fig_hydra.pdf (+ png)
Uso: python3 scripts/em/f7_analyze_hyd.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT
from f7_aggregate import wilson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYD = os.path.join(ROOT, "results_em", "hydra_removal")
KIN = os.path.join(ROOT, "results_em", "kinetic_grid")
FIG = os.path.join(ROOT, "results_em", "extra_figs")   # figura de otro artículo
H = 1500
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def removed_stats(cfg):
    """Fracción de la población retirada y nº de candidatos (media sobre semillas)."""
    fr, nc = [], []
    for f in glob.glob(os.path.join(HYD, cfg, "seed*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        if "removed" in z.files and len(z["removed"]):
            t, nd, ncand, Nafter = z["removed"][0]
            fr.append(nd / max(nd + Nafter, 1)); nc.append(ncand)
    return (float(np.mean(fr)) if fr else np.nan, float(np.mean(nc)) if nc else np.nan)


def kin_map(plane="hyd"):
    fn = os.path.join(KIN, f"{plane}.jsonl")
    if not os.path.exists(fn):
        return None
    rows = []
    for line in open(fn):
        r = json.loads(line)
        rem = r["point"]["removal"]
        rows.append(dict(t_r=rem["t_r"], f=rem["f"], period=rem.get("period", 0), cls=r["cls"],
                         Tfl30=r.get("Tfl_L30", -1), valley=r.get("nu_min_late", np.nan)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(HYD, "summary.csv"))
    df["P_surv"] = 1 - df[f"P_ext_{H}"]
    lo, hi = zip(*[wilson(int(round(p * n)), int(n)) for p, n in zip(df.P_surv, df.n)])
    df["P_surv_lo"], df["P_surv_hi"] = lo, hi
    df["rule"] = df.cfg.str.split("_").str[2]
    rs = [removed_stats(c) for c in df.cfg]
    df["frac_removed"] = [r[0] for r in rs]; df["n_cand"] = [r[1] for r in rs]
    df.to_csv(os.path.join(HYD, "mapa.csv"), index=False)
    km = kin_map("hyd")
    fig = plt.figure(figsize=(7.4, 5.4))
    gsf = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.3, height_ratios=[1, 0.9])
    axs = [fig.add_subplot(gsf[0, 0]), fig.add_subplot(gsf[0, 1]), fig.add_subplot(gsf[1, :])]
    for ax, sp, title in ((axs[0], "sp", "(a) spatial, one removal"), (axs[1], "wm", "(b) well mixed, one removal")):
        g = df[(df.space == sp) & (df.rule == "s") & df.period.isna() & ~df.cfg.str.contains("stop")]
        if g.empty:
            continue
        piv = g.pivot_table(index="f", columns="t_r", values="Text_med")
        sv = g.pivot_table(index="f", columns="t_r", values="P_surv")
        ctrl = df[(df.space == sp) & (df.rule == "none")]
        c0 = float(ctrl.Text_med.iloc[0]) if len(ctrl) else np.nan
        im = ax.imshow(piv.values / c0, origin="lower", aspect="auto", cmap="Blues", vmin=1, vmax=3)
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns.astype(int), rotation=90)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{v:g}" for v in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                lab = f"{v:.0f}" if np.isfinite(v) else "–"
                if sv.values[i, j] > 0:
                    lab += "*"
                ax.text(j, i, lab, ha="center", va="center", fontsize=4.6,
                        color="w" if np.isfinite(v) and v / c0 > 2.2 else "#222")
        ax.set_xlabel(r"removal week $t_r$"); ax.set_ylabel("fraction removed $f$")
        ax.set_title(f"{title}\nmedian $T_{{\\rm ext}}$ (control {c0:.0f})", fontsize=7.2)
    cb = fig.colorbar(im, ax=axs[:2], shrink=0.9, pad=0.02)
    cb.set_label(r"$T_{\rm ext}$ / control", fontsize=7)
    ax = axs[2]
    items = []
    for sp, mk, col in (("sp", "o", "#2a78d6"), ("wm", "s", "#eb6834")):
        ctrl = df[(df.space == sp) & (df.rule == "none")]
        one = df[(df.space == sp) & (df.rule == "s") & df.period.isna() & ~df.cfg.str.contains("stop")]
        best = one.sort_values("P_surv").iloc[-1] if len(one) else None
        per = df[(df.space == sp) & (df.period == 4) & ~df.cfg.str.contains("stop")].sort_values("P_surv")
        stop = df[(df.space == sp) & df.cfg.str.contains("stop")]
        rows = [("no removal", ctrl.iloc[0])] if len(ctrl) else []
        if best is not None:
            rows.append(("best single pulse", best))
        for _, r in per.sort_values(["t_r", "f"]).iterrows():
            rows.append((f"periodic {int(r.t_r)}, {r.f:g}", r))
        stop = stop.assign(dur=[int(c.split("stop")[1].split("_")[0]) for c in stop.cfg]).sort_values(["f", "dur"])
        for _, r in stop.iterrows():
            rows.append((f"periodic 30, {r.f:g}, stop +{int(r.dur)}", r))
        items.append((sp, mk, col, rows))
    labels = []
    for sp, mk, col, rows in items:
        for lab, r in rows:
            if lab not in labels:
                labels.append(lab)
    xpos = {lab: i for i, lab in enumerate(labels)}
    for k, (sp, mk, col, rows) in enumerate(items):
        for lab, r in rows:
            x = xpos[lab] + (k - 0.5) * 0.25
            ax.errorbar(x, r.P_surv, yerr=[[r.P_surv - r.P_surv_lo], [r.P_surv_hi - r.P_surv]], fmt=mk, ms=3.5,
                        color=col, lw=0.9, capsize=1.5,
                        label=("spatial" if sp == "sp" else "well mixed") if lab == rows[0][0] else None)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=5.6, rotation=40, ha="right")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("P(alive at week 1 500)")
    ax.set_title("(c) removal of non-breeders ($s<s_v$): periodic = every 4 weeks from week $t_r$, fraction $f$; stop + weeks", fontsize=7.2)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_hydra.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    cols = ["cfg", "n", "P_surv", "P_surv_lo", "P_surv_hi", "Text_med", "frac_removed", "n_cand"]
    print(df[cols].sort_values("cfg").to_string())


if __name__ == "__main__":
    main()
