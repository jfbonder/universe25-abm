"""Criterio de linaje (Tabla tab:lineage) con los parámetros de C1 y N* del balance con r = γ.

    python3 scripts/analysis/mf_lineage_v2.py

Valor esperado de un proceso de ramificación SIN aprendizaje de vecinos fuera del linaje: un linaje que arranca con
s0 acumula prod_g R0(s_g) descendientes antes de llegar a la viabilidad s_v (R0(s_v) = 1), con
  R0(s) = R0 · s^3 (padres igualmente deteriorados: concepción ∝ s_f s_m, supervivencia ∝ s_f) o
  R0(s) = R0 · s^2 en la primera generación si la pareja es sana (s_m = 1).
Mapas de la competencia entre generaciones:
  sin R2 (herencia total, recuperación individual en la ventana): s_{g+1} = s_g + Δ, Δ = r a_w (aditivo);
  con R2 (λ = 1, herencia w, aprendizaje de vecinos con competencia φ S): s_{g+1} = (w + φ Λ) s_g, Λ = r a_w.
"""
import numpy as np
from scipy import optimize, stats

G, mc = 900, 8


def D(nu, xi=None):
    j = np.arange(0, 400)
    pmf = stats.poisson.pmf(j, nu) if xi is None else stats.nbinom.pmf(j, xi, xi / (xi + nu))
    return float(np.sum(np.maximum(1 + j - mc, 0) * pmf))


def surv(amax, k=0.15, A=600):
    a = np.arange(0, A + 1)
    h = 1 / (1 + np.exp(-k * (a - amax)))
    return np.cumprod(1 - h)


def R0_of(amax, fert_hi, p0=0.3, L=6.55, q0=0.95):
    S = surv(amax)
    return 0.5 * p0 * L * q0 * S[6:fert_hi + 1].sum()


def lineage(R0, s0, step, healthy=False, gmax=200):
    sv = R0 ** (-1 / 3)
    s, prod, g = s0, 1.0, 0
    while s < sv and g < gmax:
        e = 2 if (healthy and g == 0) else 3
        prod *= R0 * s ** e
        s = step(s)
        g += 1
    return prod if s >= sv else 0.0, g


if __name__ == "__main__":
    print("balance D(nu*) = r/gamma con r = gamma = 0,1 (C1):")
    nus = optimize.brentq(lambda n: D(n) - 1.0, 1, 20)
    print(f"  Poisson nu* = {nus:.2f}, N* = {G * nus:.0f};", end=" ")
    for xi in (5, 2, 1):
        print(f"NB xi={xi}: N* = {G * optimize.brentq(lambda n: D(n, xi) - 1.0, 0.5, 30):.0f}", end="; ")
    print(f"\n  capacidad K = m_c|G| = {mc * G}")
    R0b, R0l = R0_of(60, 50), R0_of(120, 80)
    print(f"R0 base = {R0b:.1f} (s_v = {R0b ** (-1/3):.3f}); R0 longevo = {R0l:.1f} (s_v = {R0l ** (-1/3):.3f};"
          f" con pareja sana R0 s^2 = 1 en s = {R0l ** -0.5:.3f})")
    s0s = (0.02, 0.05, 0.10, 0.20)
    rows = []
    for lab, R0, step in (
            ("base, sin R2, Δ=0,12 (r=0,01, a_w=12)", R0b, lambda s: s + 0.12),
            ("C1 sin R2, Δ=0,12", R0l, lambda s: s + 0.12),
            ("C1 sin R2, Δ=0,36 (r=0,03)", R0l, lambda s: s + 0.36),
            ("C1 con R2, w+φΛ = 1,4 (φ=1)", R0l, lambda s: 1.4 * s),
            ("C1 con R2, w+φΛ = 0,8 (φ=0,5)", R0l, lambda s: 0.8 * s)):
        for healthy in (False, True):
            vals = [lineage(R0, s0, step, healthy) for s0 in s0s]
            rows.append((lab + (", pareja sana" if healthy else ""), vals))
    print("\nlinaje: prod_g R0(s_g) hasta s_v (0 = no llega nunca); entre paréntesis, generaciones")
    for lab, vals in rows:
        print(f"  {lab:48s} " + "  ".join(f"{v:9.2g} ({g:2d})" for v, g in vals))
    print(f"\nprimera generación con pareja sana, R0 longevo y s0 = 0,1: {R0l * 0.01:.2f}")
