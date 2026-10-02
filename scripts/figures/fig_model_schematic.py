"""Figura esquemática del modelo (manuscript/extended/figs/model_schematic.pdf).
(a) grilla: movimiento con afinidad, aversión, pool hacinado y refugios de capacidad 2;
(b) trayectorias ilustrativas de s(a) bajo R1+R2 con herencia parcial (iteración determinista de las
    ecuaciones del modelo, no una simulación);
(c) diagrama de fase esperado en el plano (alpha, kappa) con los ensambles de calibración (O17).
Paleta: 3 primeros slots categóricos de la paleta de referencia (validados all-pairs); texto en tinta neutra.
Uso: python3 scripts/figures/fig_model_schematic.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "pdf.fonttype": 42,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans"})
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # raíz del repo
OUT = str(ROOT / "manuscript/extended/figs")
os.makedirs(OUT, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.1]))

# ------------------------------------------------------------------ (a) grilla
ax = axs[0]
G = 7
ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_aspect("equal"); ax.axis("off")
refuges = {(0, 6), (1, 2), (5, 5), (6, 1), (2, 5), (6, 4), (0, 0)}
for i in range(G):
    for j in range(G):
        if (i, j) in refuges:
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor="#eef7f3", edgecolor=AQUA, hatch="////",
                                   linewidth=0.0, alpha=0.9))
        ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor=GRID, linewidth=0.6))
rng = np.random.default_rng(3)


def dots(cell, n, color, r=0.13, ring=False):
    i, j = cell
    k = max(2, int(np.ceil(np.sqrt(n))))
    rr = min(r, 0.42 / k)
    pts = [(i + (a + 0.5) / k, j + (b + 0.5) / k) for a in range(k) for b in range(k)][:n]
    r = rr
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))


dots((4, 2), 11, ORANGE)            # pool hacinado (m > m_c)
dots((5, 2), 3, ORANGE); dots((4, 1), 2, ORANGE)
dots((1, 2), 1, AQUA); dots((5, 5), 2, AQUA); dots((6, 1), 1, AQUA); dots((2, 5), 1, AQUA)
dots((1, 4), 3, BLUE); dots((2, 3), 2, BLUE); dots((3, 4), 1, BLUE)
# focal y su vecindario de Moore
fx, fy = 2.5, 3.5
ax.add_patch(Rectangle((1, 2), 3, 3, fill=False, edgecolor=INK, linewidth=1.1, linestyle=(0, (3, 2)), zorder=4))
ax.add_patch(Circle((fx, fy), 0.11, facecolor=BLUE, edgecolor=INK, linewidth=1.0, zorder=5))
for (tx, ty, lw) in ((1.5, 4.5, 2.2), (3.5, 4.5, 1.0), (2.5, 2.5, 0.6), (1.5, 2.5, 0.6), (3.5, 3.5, 0.8)):
    ax.add_patch(FancyArrowPatch((fx, fy), (tx, ty), arrowstyle="-|>", mutation_scale=7, linewidth=lw,
                                 color=INK2, shrinkA=6, shrinkB=6, zorder=4))
ax.annotate("pool, $m>m_c$", xy=(4.5, 2.5), xytext=(4.6, 0.35), ha="center", fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("refuge, $c_R=2$", xy=(5.5, 5.5), xytext=(5.0, 6.45), ha="center", fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.text(0.0, -0.2, r"$W_i(y)\propto(1+\alpha A_i(y))\,e^{-\kappa(1-s_i)^{p}n_y}\,[1+\pi(1-b_i)^{q}]$",
        ha="left", va="top", fontsize=6.6, color=INK)
# leyenda directa
for k, (c, lab) in enumerate(((BLUE, "socialized, $s\\geq0.2$"), (ORANGE, "deteriorated, crowded"),
                              (AQUA, "deteriorated, isolated"))):
    ax.add_patch(Circle((0.2, -1.05 - 0.45 * k), 0.12, facecolor=c, edgecolor="white", clip_on=False))
    ax.text(0.45, -1.05 - 0.45 * k, lab, va="center", fontsize=6.6, color=INK, clip_on=False)
ax.set_title("(a) lattice, motion, refuges", loc="left")

# ------------------------------------------------------------------ (b) s(a)
ax = axs[1]
a_w, r, w, gam, m_c = 12, 0.1, 0.2, 0.1, 8


def traj(s_mother, sbarV, crowd_from=None, m=12, T=60):
    s = [w * s_mother]
    for a in range(1, T + 1):
        ds = (r * sbarV if a <= a_w else 0.0)
        if crowd_from is not None and a >= crowd_from:
            ds -= gam * max(0, m - m_c)
        s.append(float(np.clip(s[-1] + ds, 0, 1)))
    return np.arange(T + 1), np.array(s)


ax.axvspan(0, a_w, color="#f1f0ec", zorder=0)
ax.text(a_w / 2, 1.04, "R1 window\n$a\\leq a_w$", ha="center", va="bottom", fontsize=6.8, color=INK2)
t, s1 = traj(1.0, 0.9, crowd_from=22)
ax.plot(t, s1, color=ORANGE, lw=2)
ax.text(24, 0.70, "early cohort: socialized,\nthen crowded (CR)", fontsize=6.6, color=INK, va="center")
t, s2 = traj(1.0, 0.9)
ax.plot(t[:23], s2[:23], color=BLUE, lw=2)
ax.text(38, 0.97, "if uncrowded:\nkeeps $s\\approx1$", fontsize=6.6, color=INK, va="top")
ax.plot(t[22:], s2[22:], color=BLUE, lw=1.2, ls=(0, (3, 2)))
t, s3 = traj(0.15, 0.08)
ax.plot(t, s3, color=AQUA, lw=2)
ax.text(26.5, 0.245, "late cohort, deteriorated\nmother: never socialized (BO)", fontsize=6.6, color=INK, va="bottom")
ax.axhline(0.2, color=MUTED, lw=0.8, ls=":")
ax.text(59.5, 0.19, "$s=0.2$", ha="right", va="top", fontsize=6.5, color=INK2)
ax.annotate("$s_{nb}=w\\,s_{\\rm mother}$", xy=(0.3, 0.2), xytext=(1.0, 0.36), fontsize=6.6,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("R2: $\\Delta s=r\\,\\bar s_V$", xy=(5, 0.6), xytext=(13.5, 0.52), fontsize=6.6,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(0, 60); ax.set_ylim(-0.02, 1.25)
ax.set_xlabel("age $a$ (weeks)"); ax.set_ylabel("social state $s$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_yticks([0, 0.5, 1])
ax.set_title("(b) social state over a life (illustrative)", loc="left")

# ------------------------------------------------------------------ (c) diagrama de fase esperado
ax = axs[2]
al = np.linspace(1.0, 8.5, 200)
kap = lambda n, a: 1.0 / (n + 1.0 / a)                 # n* = 1/kappa - 1/alpha
ax.fill_between(al, kap(10.0, al), kap(5.5, al), where=al >= 3.5, color="#e3eefb", zorder=0)
ax.plot(al, kap(10.0, al), color=BLUE, lw=1)
ax.plot(al, kap(5.5, al), color=BLUE, lw=1)
ax.axvline(3.5, color=BLUE, lw=0.8, ls=(0, (3, 2)))
ax.text(7.3, 0.125, "Calhoun-like\nband", ha="center", va="center", fontsize=7, color=INK)
ax.text(6.4, 0.03, "pools only, no isolated\nclass ($n^*\\gg m_c$)", ha="center", fontsize=6.4, color=INK2)
ax.text(6.4, 0.255, "homogeneous, no pools\n($n^*\\lesssim m_c/2$)", ha="center", fontsize=6.4, color=INK2)
ax.text(1.1, 0.03, "weak\naggregation", ha="left", va="bottom", fontsize=6.4, color=INK2)
pts = [(4, 0.15, 0.97), (6, 0.15, 0.95), (4, 0.10, 0.93), (4, 0.20, 0.23), (4, 0.0, 0.00),
       (3, 0.15, 0.70), (5, 0.15, 1.00), (4, 0.125, 0.97), (4, 0.175, 0.67)]
for (a, k, f) in pts:
    ok = f >= 0.8
    ax.plot(a, k, "o", ms=6.5, mfc=BLUE if ok else "white", mec=BLUE, mew=1.3, zorder=5, clip_on=False)
    left = (a == 4)
    if a == 3:
        ax.text(a - 0.12, k - 0.012, f"{f:.2f}", fontsize=6.2, color=INK, va="top", ha="right", zorder=6)
    else:
        ax.text(a - 0.25 if left else a, k + (0.0 if left else 0.012), f"{f:.2f}", fontsize=6.2, color=INK,
                va="center" if left else "bottom", ha="right" if left else "center", zorder=6)
ax.annotate("C0", xy=(4.05, 0.153), xytext=(4.9, 0.205), fontsize=7, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlim(1, 8.5); ax.set_ylim(-0.012, 0.31)
ax.set_xlabel("affinity attraction $\\alpha$"); ax.set_ylabel("aversion of the deteriorated $\\kappa$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(c) expected for C0 (no refuges)", loc="left")

fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUT}/model_schematic.pdf", bbox_inches="tight")
if os.environ.get("U25_PREVIEW_PNG"):
    fig.savefig(os.environ["U25_PREVIEW_PNG"], dpi=200, bbox_inches="tight")
print("ok")
