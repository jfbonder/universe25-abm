"""Figura esquemática del modelo, versión a una columna para la versión PRE
(manuscript/pre/figs/model_schematic_pre.pdf; ancho \\columnwidth = 8.6 cm).

Adapta scripts/figures/fig_model_schematic.py (figura a dos columnas de la versión extendida, que no se toca):
(a) grilla: movimiento con afinidad, pool hacinado y refugios de capacidad 2, con el peso de movimiento
    descompuesto por factor (base, AV, R7) al costado;
(b) trayectorias ilustrativas de s(a) bajo R1 + R2 con herencia parcial: iteración determinista de las
    ecuaciones del modelo con entornos fijos, no una simulación (mismos valores que la figura extendida).
El panel (c) de la extendida (diagrama de fase esperado para C0, sin refugios) no se incluye: es específico de C0
y la versión PRE remite a la extendida para C0.
Paleta: los mismos tres slots categóricos que la figura extendida (validados all-pairs con el validador de la
skill dataviz: CVD dE 9.2; el aqua tiene contraste < 3:1 y lleva rotulado directo y textura en los refugios).
Uso: python3 scripts/figures/fig_model_schematic_pre.py   (U25_PREVIEW_PNG=ruta.png guarda además un PNG)"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3"
FS, FS_SMALL = 7.0, 6.5
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS + 0.5, "pdf.fonttype": 42,
                     "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})

ROOT = Path(__file__).resolve().parents[2]   # raíz del repo
OUT = ROOT / "manuscript" / "pre" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

COL_IN = 8.6 / 2.54                           # \columnwidth de REVTeX (PRE), en pulgadas
fig = plt.figure(figsize=(COL_IN, 3.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.3, 1.0], width_ratios=[1.0, 0.93],
              left=0.115, right=0.975, top=0.952, bottom=0.10, hspace=0.50, wspace=0.04)

# ------------------------------------------------------------------ (a) grilla
ax = ax_a = fig.add_subplot(gs[0, 0])
G = 7
ax.set_xlim(-0.05, G + 0.05); ax.set_ylim(-0.05, G + 0.05); ax.set_aspect("equal"); ax.axis("off")
refuges = {(0, 6), (1, 2), (5, 5), (6, 1), (2, 5), (6, 4), (0, 0)}
for i in range(G):
    for j in range(G):
        if (i, j) in refuges:
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor="#eef7f3", edgecolor=AQUA, hatch="////",
                                   linewidth=0.0, alpha=0.9))
        ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor=GRID, linewidth=0.5))


def dots(cell, n, color, r=0.15):
    """n individuos en la celda `cell`, en una subgrilla k x k."""
    i, j = cell
    k = max(2, int(np.ceil(np.sqrt(n))))
    rr = min(r, 0.42 / k)
    pts = [(i + (a + 0.5) / k, j + (b + 0.5) / k) for a in range(k) for b in range(k)][:n]
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), rr, facecolor=color, edgecolor="white", linewidth=0.5, zorder=3))


dots((4, 2), 11, ORANGE)            # pool hacinado (m > m_c)
dots((5, 2), 3, ORANGE); dots((4, 1), 2, ORANGE)
dots((1, 2), 1, AQUA); dots((5, 5), 2, AQUA); dots((6, 1), 1, AQUA); dots((2, 5), 1, AQUA)
dots((1, 4), 3, BLUE); dots((2, 3), 2, BLUE); dots((3, 4), 1, BLUE)
# focal y su vecindario de Moore
fx, fy = 2.5, 3.5
ax.add_patch(Rectangle((1, 2), 3, 3, fill=False, edgecolor=INK, linewidth=0.9, linestyle=(0, (3, 2)), zorder=4))
ax.add_patch(Circle((fx, fy), 0.13, facecolor=BLUE, edgecolor=INK, linewidth=0.9, zorder=5))
for (tx, ty, lw) in ((1.5, 4.5, 2.0), (3.5, 4.5, 0.9), (2.5, 2.5, 0.55), (1.5, 2.5, 0.55), (3.5, 3.5, 0.75)):
    ax.add_patch(FancyArrowPatch((fx, fy), (tx, ty), arrowstyle="-|>", mutation_scale=6, linewidth=lw,
                                 color=INK2, shrinkA=5, shrinkB=5, zorder=4))
ax.set_title("(a) lattice, motion, refuges", loc="left", pad=3)

# columna derecha de (a): rótulos, leyenda directa y peso de movimiento por factor
axr = fig.add_subplot(gs[0, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
# conectores desde los rótulos hacia el pool y un refugio (coordenadas de datos de ax -> axr)
con = dict(arrowstyle="-", color=INK2, lw=0.55)
axr.annotate("pool, $m>m_c$", xy=(4.75, 2.55), xycoords=ax.transData, xytext=(0.06, 0.45),
             textcoords=axr.transAxes, fontsize=FS_SMALL, va="center", ha="left", arrowprops=con)
axr.annotate("refuge, $c_R=2$", xy=(6.0, 5.5), xycoords=ax.transData, xytext=(0.06, 0.93),
             textcoords=axr.transAxes, fontsize=FS_SMALL, va="center", ha="left", arrowprops=con)
axr.text(0.06, 0.69, "Moore neighbourhood;\narrow width $\\propto W_i(y)$", fontsize=FS_SMALL - 0.3,
         va="center", ha="left", color=INK2, linespacing=1.15)
for k, (c, lab) in enumerate(((BLUE, "socialized"), (ORANGE, "deteriorated, crowded"),
                              (AQUA, "deteriorated, isolated"))):
    y = 0.245 - 0.115 * k
    axr.add_patch(Circle((0.10, y), 0.035, facecolor=c, edgecolor="white", transform=axr.transAxes,
                         clip_on=False))
    axr.text(0.17, y, lab, va="center", fontsize=FS_SMALL, color=INK, transform=axr.transAxes)

# peso de movimiento debajo de (a): cada factor como un trozo de texto, con su regla rotulada debajo
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
pieces = ((r"$W_i(y)=[1+\alpha A_i(y)]$", "affinity (core model)"),
          (r"$\;\times\;e^{-\kappa(1-s_i)^{p}n_y}$", "aversion (AV)"),
          (r"$\;\times\;[1+\pi(1-b_i)^{q}\,\mathbf{1}_{\mathcal{R}}(y)]$", "refuge preference (R7)"))
y0_a = ax_a.get_position().y0          # borde inferior real de la grilla (aspecto 1)
y_eq, y_lab = y0_a - 0.048, y0_a - 0.090
txts = [fig.text(0, y_eq, tex, fontsize=FS, ha="left", va="center", color=INK) for tex, _ in pieces]
fig.canvas.draw()
widths = [t.get_window_extent(renderer).width / fig.bbox.width for t in txts]
x = 0.5 * (gs.left + gs.right) - 0.5 * sum(widths)          # centrado en el ancho de los paneles
for t, wdt, (_, lab) in zip(txts, widths, pieces):
    t.set_x(x)
    xc = x + 0.5 * wdt + (0.012 if lab != pieces[0][1] else 0.0)   # el \times inicial corre el centro
    fig.text(xc, y_lab, lab, fontsize=FS_SMALL - 0.3, ha="center", va="center", color=INK2)
    x += wdt

# ------------------------------------------------------------------ (b) s(a)
ax = fig.add_subplot(gs[1, :])
a_w, r, w, gam, m_c = 12, 0.1, 0.2, 0.1, 8


def traj(s_mother, sbarV, crowd_from=None, m=12, T=60):
    """Ecs. (social) + (aprendizaje) con entorno fijo: aprendizaje r*sbarV sólo si a <= a_w (R1, R2),
    daño gamma*(m-m_c)_+ desde la edad crowd_from; s_nb = w*s_madre (s_0 = 0)."""
    s = [w * s_mother]
    for a in range(1, T + 1):
        ds = (r * sbarV if a <= a_w else 0.0)
        if crowd_from is not None and a >= crowd_from:
            ds -= gam * max(0, m - m_c)
        s.append(float(np.clip(s[-1] + ds, 0, 1)))
    return np.arange(T + 1), np.array(s)


ax.axvspan(0, a_w, color="#f1f0ec", zorder=0, lw=0)
ax.text(a_w / 2, 1.03, "R1 window\n$a\\leq a_w$", ha="center", va="bottom", fontsize=FS_SMALL, color=INK2,
        linespacing=1.1)
t, s1 = traj(1.0, 0.9, crowd_from=22)
ax.plot(t, s1, color=ORANGE, lw=1.8)
ax.text(26.0, 0.76, "early cohort: socialized,\nthen crowded", fontsize=FS_SMALL, color=INK, va="center")
t, s2 = traj(1.0, 0.9)
ax.plot(t[:23], s2[:23], color=BLUE, lw=1.8)
ax.plot(t[22:], s2[22:], color=BLUE, lw=1.1, ls=(0, (3, 2)))
ax.text(59.5, 1.03, "if uncrowded: $s\\approx1$", fontsize=FS_SMALL, color=INK, va="bottom", ha="right")
t, s3 = traj(0.15, 0.08)
ax.plot(t, s3, color=AQUA, lw=1.8)
ax.annotate("late cohort, deteriorated mother:\nnever socialized", xy=(31.0, 0.135), xytext=(27.0, 0.29),
            fontsize=FS_SMALL, color=INK, va="bottom", linespacing=1.1,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.55))
ax.axhline(0.2, color=MUTED, lw=0.7, ls=":")
ax.text(59.5, 0.215, "$s=0.2$", ha="right", va="bottom", fontsize=FS_SMALL - 0.3, color=INK2)
ax.annotate("$s_{\\rm nb}=w\\,s_{\\rm mother}$", xy=(0.3, 0.2), xytext=(2.6, 0.27), fontsize=FS_SMALL,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.55))
ax.annotate("R2: $\\Delta s=r\\,\\bar s_V$", xy=(3.6, 0.53), xytext=(6.8, 0.48), fontsize=FS_SMALL,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.55))
ax.text(a_w + 1.5, 1.03, "$a>a_w$: ratchet, $\\dot s\\leq0$", ha="left", va="bottom", fontsize=FS_SMALL,
        color=INK2)
ax.set_xlim(0, 60); ax.set_ylim(-0.02, 1.27)
ax.set_xlabel("age $a$ (weeks)", labelpad=1.5); ax.set_ylabel("social state $s$", labelpad=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_yticks([0, 0.5, 1]); ax.set_xticks([0, 12, 24, 36, 48, 60])
ax.tick_params(length=2.5, pad=1.5)
ax.set_title("(b) social state over a life (illustrative)", loc="left", pad=3)

fig.savefig(OUT / "model_schematic_pre.pdf")
if os.environ.get("U25_PREVIEW_PNG"):
    fig.savefig(os.environ["U25_PREVIEW_PNG"], dpi=300)
print(f"ok: {OUT / 'model_schematic_pre.pdf'}")
