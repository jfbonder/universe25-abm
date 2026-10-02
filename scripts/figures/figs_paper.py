"""Figuras vectoriales (PDF) para la sección 'Mean-field analysis' de la versión extendida.
Salida: manuscript/extended/figs/ (fig_mf_ode, fig_mf_vs_abm, fig_km_drift, fig_thresholds).

    python3 scripts/figures/figs_paper.py            # las cuatro figuras
    python3 scripts/figures/figs_paper.py 2 4        # sólo las figuras 2 y 4
Insumos: results/baseline_alpha14/analisis/{km,ciclos}.csv (python -m pipeline baseline) y results/meanfield/
(ens_rw_40x2000.pkl de scripts/analysis/ensamble_rapido.py; kin_sweep_*.log de scripts/analysis/mf_kin.py).
"""
import numpy as np, pandas as pd, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
DATA = ROOT / "results" / "meanfield"
OUT = ROOT / "manuscript" / "extended" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
WANT = set(sys.argv[1:]) or {"1", "2", "3", "4"}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "pdf.fonttype": 42})

# ---------- Fig 1: ODE 2D, plano de fases y N(t) ----------
if "1" in WANT:
    from mf_ode import make_rhs, fixed_point, G2
    rhs, D, b = make_rhs(2.0)
    Ns, Ss = fixed_point(D, b)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    sol = solve_ivp(lambda t, z: rhs(t, z, True), (0, 900), [80, 1.0], max_step=0.25, rtol=1e-8, atol=1e-10)
    ax[0].plot(sol.t, sol.y[0], color="C0"); ax[0].set_xlabel("week"); ax[0].set_ylabel("$N$", color="C0")
    a2 = ax[0].twinx(); a2.plot(sol.t, sol.y[1], color="C3", lw=1); a2.set_ylabel("$S$", color="C3"); a2.set_ylim(0, 1.05)
    ax[0].set_title("(a) clamped mean field, $N_0=80$, $S_0=1$")
    ax[1].plot(sol.y[0], sol.y[1], lw=0.9, color="C0", label="clamped: limit cycle")
    sol2 = solve_ivp(lambda t, z: rhs(t, z, False), (0, 1500), [Ns * 1.1, Ss], max_step=0.25, rtol=1e-9, atol=1e-11)
    ax[1].plot(sol2.y[0], sol2.y[1], lw=0.6, color="C2", alpha=0.8, label="unclamped: unstable focus")
    ax[1].plot([Ns], [Ss], "ko", ms=4, label="equilibrium $(N^*,S^*)$")
    ax[1].axhline(1, color="k", lw=0.5, ls=":"); ax[1].axhline(0, color="k", lw=0.5, ls=":")
    ax[1].set_xscale("log"); ax[1].set_xlim(30, 2e4); ax[1].set_xlabel("$N$ (log)"); ax[1].set_ylabel("$S$"); ax[1].set_ylim(-0.05, 1.3)
    ax[1].legend(loc="upper left"); ax[1].set_title("(b) phase plane (neg. binomial, $\\xi=2$)")
    plt.tight_layout(); plt.savefig(OUT / "fig_mf_ode.pdf"); plt.close(fig)

# ---------- Fig 2: ABM (rw) vs bien mezclado vs campo medio por edades ----------
if "2" in WANT:
    from abm_rapido import run as run_abm
    from mf_edad import run as run_mf
    h = pd.DataFrame(run_abm(T=600, seed=3, mode="rw")); hm = pd.DataFrame(run_abm(T=600, seed=3, mode="mix"))
    mf0 = run_mf(T=600, kappa=np.inf, rho=0.0); mf5 = run_mf(T=600, kappa=2.0, rho=0.05)
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
    ax[0].plot(h.time, h.population, color="C0", label="ABM, random walk ($\\alpha=0$)")
    ax[0].plot(hm.time, hm.population, color="C1", label="ABM, well mixed (Poisson occupancy)")
    ax[0].plot(mf0[:, 0], mf0[:, 1], "--", color="C2", label="age-structured MF, homogeneous")
    ax[0].plot(mf5[:, 0], mf5[:, 1], ":", color="C3", label="age-structured MF, 5% healthy pockets")
    ax[0].set_ylabel("$N$"); ax[0].legend(ncol=2)
    ax[1].plot(h.time, h.mean_social, color="C0"); ax[1].plot(hm.time, hm.mean_social, color="C1")
    ax[1].plot(mf0[:, 0], mf0[:, 2], "--", color="C2"); ax[1].plot(mf5[:, 0], mf5[:, 2], ":", color="C3")
    ax[1].set_ylabel("$S$"); ax[1].set_xlabel("week"); ax[1].set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(OUT / "fig_mf_vs_abm.pdf"); plt.close(fig)

# ---------- Fig 3: KM de la línea base (u25, alpha=1.4) y picos sucesivos (alpha=0 y alpha=1.4) ----------
if "3" in WANT:
    km = pd.read_csv(ROOT / "results/baseline_alpha14/analisis/km.csv")
    cyc14 = pd.read_csv(ROOT / "results/baseline_alpha14/analisis/ciclos.csv")
    ens0 = pd.read_pickle(DATA / "ens_rw_40x2000.pkl")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax[0].step(km.t, km.S, where="post", color="C0"); ax[0].fill_between(km.t, km.lo, km.hi, step="post", alpha=0.25, color="C0")
    ax[0].set_xlabel("week"); ax[0].set_ylabel("$P(\\mathrm{colony\\ alive})$"); ax[0].set_ylim(0, 1.02); ax[0].set_title("(a) KM, short-lived control, $\\alpha=1.4$, $n=100$")
    for p in ens0.picos[:10]:
        ax[1].plot(range(1, len(p) + 1), p, "o-", ms=2.5, lw=0.8, alpha=0.7)
    ax[1].set_xlabel("peak index $k$"); ax[1].set_ylabel("$N$ at peak"); ax[1].set_title("(b) successive peaks, $\\alpha=0$")
    d0 = [np.polyfit(np.arange(1, len(p)), np.log(np.array(p[1:], float)), 1)[0] for p in ens0.picos if len(p) >= 4]
    d14 = cyc14.slope_log.dropna()
    ax[2].hist(d0, bins=12, alpha=0.6, label="$\\alpha=0$ (ABM-fast)", color="C0")
    ax[2].hist(d14, bins=12, alpha=0.6, label="$\\alpha=1.4$ (u25)", color="C1")
    ax[2].axvline(0, color="k", lw=0.8); ax[2].set_xlabel("slope of $\\log N_{\\rm peak}$ per cycle"); ax[2].set_title("(c) no secular drift")
    ax[2].legend()
    plt.tight_layout(); plt.savefig(OUT / "fig_km_drift.pdf"); plt.close(fig)

# ---------- Fig 4: umbrales de R1 y R2 (campo medio cinético) ----------
if "4" in WANT:
    def parse(log):
        txt = open(DATA / log).read()
        blocks = re.split(r"== ", txt)[1:]
        out = {}
        for bl in blocks:
            head, *lines = bl.strip().split("\n")
            rule = "R1" if head.startswith("R1") else "R2"; lon = "True" in head
            for ln in lines:
                m = re.match(r"\s*rec=([\d.]+): (.*)", ln)
                if not m: continue
                rec = float(m.group(1)); xs, ps = [], []
                for tok in re.findall(r"(?:a_w|λ)=([\w.]+): ([\d.]+)\|", m.group(2)):
                    xs.append(np.inf if tok[0] == "inf" else float(tok[0])); ps.append(float(tok[1]))
                out[(rule, lon, rec)] = (np.array(xs), np.array(ps))
        return out
    A = parse("kin_sweep_05_8.log"); B = parse("kin_sweep_1_5.log")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    sty = {(False, 0.01): ("C0", "-"), (False, 0.03): ("C0", "--"), (True, 0.01): ("C3", "-"), (True, 0.03): ("C3", "--")}
    for (lon, rec), (c, ls) in sty.items():
        x, p = A[("R1", lon, rec)]; xx = np.where(np.isinf(x), 96, x)
        ax[0].plot(xx, p, ls, color=c, marker="o", ms=3, label=f"{'long-lived' if lon else 'short life'}, $r={rec}$")
        x, p = B[("R1", lon, rec)]; xx = np.where(np.isinf(x), 96, x)
        ax[0].plot(xx, p, ls, color=c, marker="s", ms=2.5, alpha=0.35)
        x, p = A[("R2", lon, rec)]; ax[1].plot(x, p, ls, color=c, marker="o", ms=3, label=f"{'long-lived' if lon else 'short life'}, $r={rec}$")
        x, p = B[("R2", lon, rec)]; ax[1].plot(x, p, ls, color=c, marker="s", ms=2.5, alpha=0.35)
    ax[0].set_xscale("log"); ax[0].set_xticks([6, 12, 24, 48, 96]); ax[0].set_xticklabels(["6", "12", "24", "48", "$\\infty$"])
    ax[0].set_xlabel("critical window $a_w$ (weeks)"); ax[0].set_ylabel("$P_{\\rm ext}$"); ax[0].set_title("(a) R1"); ax[0].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("social-learning weight $\\lambda$"); ax[1].set_title("(b) R2"); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(loc="center left", framealpha=0.9)
    plt.tight_layout(); plt.savefig(OUT / "fig_thresholds.pdf"); plt.close(fig)
print("ok", sorted(p.name for p in OUT.glob("*.pdf")))
