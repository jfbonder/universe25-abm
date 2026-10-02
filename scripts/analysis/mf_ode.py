"""
Campo medio 2D (N, S) del modelo Universe 25 (cierre de primer momento, s^3) y análisis:
puntos fijos, Jacobiano, integral primera, efecto del recorte S in [0,1] (ciclo límite),
profundidad del valle y umbral de Allee.

Ocupación local: Poisson(lambda) (alpha=0, sin agregación) o Binomial Negativa de media lambda
e índice de dispersión 1 + lambda/kappa (agregación por camadas / afinidad).
"""
import numpy as np
from scipy.stats import poisson, nbinom
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---- parámetros del ABM ----
G2 = 900; K = 8; r = 0.01; d = 0.10; p0 = 0.30; q0 = 0.95
Lbar = 6.55
k_m = 0.15; a_life = 60
a = np.arange(0, 200); h = 1 / (1 + np.exp(-k_m * (a - a_life)))
Surv = np.cumprod(1 - h)
LE = 1 + Surv.sum()                        # esperanza de vida ~ 44.8 semanas
mu = 1.0 / LE                              # tasa de muerte efectiva (población estacionaria)
phi = Surv[5:50].sum() / LE                # fracción de la población en edad reproductiva (estacionaria) ~0.85
phiF = phiM = 0.5 * phi


def D_pois(lam):
    x = np.arange(0, 400)
    return np.sum(np.maximum(1 + x - K, 0) * poisson.pmf(x, lam))


def D_nb(lam, kappa):
    """Binomial negativa de media lam y 'size' kappa (var = lam + lam^2/kappa)."""
    if np.isinf(kappa):
        return D_pois(lam)
    x = np.arange(0, 600)
    p = kappa / (kappa + lam)
    return np.sum(np.maximum(1 + x - K, 0) * nbinom.pmf(x, kappa, p))


def make_rhs(kappa=np.inf, closure=3):
    lam_grid = np.linspace(0, 20, 801)
    Dg = np.array([D_nb(l, kappa) for l in lam_grid])

    def D(lam):
        return np.interp(lam, lam_grid, Dg)

    def b(N):  # nacimientos per cápita (semana^-1) con s=1
        lam = N / G2
        return phiF * p0 * Lbar * q0 * (1 - np.exp(-phiM * lam))

    def rhs(t, z, clamp=True):
        N, S = z
        dN = N * (b(N) * S ** closure - mu)
        dS = r - d * D(N / G2)
        if clamp:
            if S >= 1 and dS > 0: dS = 0.0
            if S <= 0 and dS < 0: dS = 0.0
        return [dN, dS]

    return rhs, D, b


def fixed_point(D, b):
    lam_s = brentq(lambda l: D(l) - r / d, 0.01, 20)
    Ns = lam_s * G2
    Ss = (mu / b(Ns)) ** (1 / 3)
    return Ns, Ss


def allee(b):
    """N_A tal que b(N_A)*1 = mu (umbral de Allee determinista con S=1)."""
    return brentq(lambda N: b(N) - mu, 1e-3, 1e5)


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    print(f"LE={LE:.2f} sem, mu={mu:.4f}, phi_repro={phi:.3f}, phiF={phiF:.3f}")
    for kappa, nombre in [(np.inf, 'Poisson'), (5.0, 'NB kappa=5'), (2.0, 'NB kappa=2'), (1.0, 'NB kappa=1')]:
        rhs, D, b = make_rhs(kappa)
        Ns, Ss = fixed_point(D, b)
        NA = allee(b)
        # Jacobiano en el punto fijo
        eps = 1e-3
        dDdl = (D(Ns / G2 + eps) - D(Ns / G2 - eps)) / (2 * eps)
        J = np.array([[0.0, 3 * Ns * b(Ns) * Ss ** 2], [-d * dDdl / G2, 0.0]])
        ev = np.linalg.eigvals(J)
        print(f"\n[{nombre}] N*={Ns:.0f}  S*={Ss:.3f}  N_Allee={NA:.1f}  autovalores={ev}  "
              f"periodo lineal={2*np.pi/abs(ev[0].imag):.1f} sem")
        # integración desde (N0=80, S0=1) con recorte
        T = 3000
        sol = solve_ivp(lambda t, z: rhs(t, z, True), (0, T), [80, 1.0], max_step=0.25, dense_output=True, rtol=1e-8, atol=1e-10)
        N, S = sol.y
        from scipy.signal import find_peaks
        pk, _ = find_peaks(N, prominence=100); tr, _ = find_peaks(-N, prominence=100)
        print("  picos N:", np.round(N[pk][:6]).astype(int), " valles N:", np.round(N[tr][:6], 2))
        print("  tiempos de pico:", np.round(sol.t[pk][:6]).astype(int), " S_min:", S.min().round(4), "S_max:", S.max().round(4))
        if kappa == 5.0:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(sol.t, N); ax[0].set_xlabel('semana'); ax[0].set_ylabel('N'); ax[0].set_title(f'ODE 2D ({nombre}), N(t)')
            ax[1].plot(sol.t, S); ax[1].set_xlabel('semana'); ax[1].set_ylabel('S'); ax[1].set_title('S(t)')
            # retrato de fases con y sin recorte
            ax[2].plot(N, S, lw=0.8, label='con recorte S∈[0,1]')
            sol2 = solve_ivp(lambda t, z: rhs(t, z, False), (0, 600), [80, 1.0], max_step=0.25, rtol=1e-8, atol=1e-10)
            ax[2].plot(sol2.y[0], sol2.y[1], lw=0.8, ls='--', label='sin recorte (órbita neutra)')
            ax[2].plot([Ns], [Ss], 'ko'); ax[2].set_xscale('log'); ax[2].set_xlabel('N (log)'); ax[2].set_ylabel('S'); ax[2].legend()
            ax[2].set_title('plano de fases; punto = equilibrio (centro)')
            plt.tight_layout(); plt.savefig('fig_ode2d.png', dpi=120)
    # verificación numérica de la integral primera para el caso Poisson sin recorte
    rhs, D, b = make_rhs(np.inf)
    sol2 = solve_ivp(lambda t, z: rhs(t, z, False), (0, 400), [2000, 0.6], max_step=0.1, rtol=1e-10, atol=1e-12)
    N, S = sol2.y
    # f(S) = b(N)S^3 - mu depende de N via b(N) -> no es exactamente separable salvo b constante (densidad alta).
    # Con b ~ const (1-exp(-phiM lam) ~ 1), H = b S^4/4 - mu S - ∫ (r - d D(N/G2))/N dN
    from scipy.integrate import quad
    bc = phiF * p0 * Lbar * q0
    def Hfun(N, S):
        I = quad(lambda n: (r - d * D(n / G2)) / n, 1000, N)[0]
        return bc * S ** 4 / 4 - mu * S - I
    Hs = np.array([Hfun(n, s) for n, s in zip(N[::50], S[::50])])
    print("\nIntegral primera (aprox. b=cte) a lo largo de la órbita: media %.5f, desvío %.2e (rel %.1e)" % (Hs.mean(), Hs.std(), Hs.std() / abs(Hs.mean())))
