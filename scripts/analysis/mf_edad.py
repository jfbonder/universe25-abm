"""
Campo medio determinista estructurado por edad (tiempo discreto semanal), cierre de primer momento
por cohorte: n_a(t) = número esperado de individuos de edad a, s_a(t) = estado social medio de la cohorte.
Ocupación local Poisson(N/|G|) (o binomial negativa), fertilidad p0 * s_f * s_m, supervivencia neonatal q0*s_f,
herencia s_cria = s_madre (promedio ponderado por fecundidad), mortalidad logística por edad.
Opcional: fracción 'refugio' rho de individuos que no sufren hacinamiento (heterogeneidad espacial mínima).
"""
import numpy as np
from mf_ode import D_nb, G2, K, r, d, p0, q0, Lbar, h as hazard_tab

AMAX = 130


def run(T=2000, N0=80, kappa=np.inf, rho=0.0, s0=1.0, age0=4, G2_=G2):
    lam_grid = np.linspace(0, 40, 1601)
    Dg = np.array([D_nb(l, kappa) for l in lam_grid])
    D = lambda lam: np.interp(lam, lam_grid, Dg)
    n = np.zeros(AMAX + 1); s = np.full(AMAX + 1, s0)   # cohortes 'expuestas'
    n[age0] = N0 * (1 - rho)
    nR = np.zeros(AMAX + 1); sR = np.full(AMAX + 1, s0)  # cohortes 'refugio' (sin hacinamiento)
    nR[age0] = N0 * rho
    rep = np.zeros(AMAX + 1, bool); rep[6:51] = True
    out = []
    for t in range(T + 1):
        N = n.sum() + nR.sum()
        S = ((n * s).sum() + (nR * sR).sum()) / N if N > 0 else 0.0
        out.append((t, N, S))
        if N < 1e-6:
            break
        lam = N / G2_
        # estado social
        s = np.clip(s + r - d * D(lam), 0, 1)
        sR = np.clip(sR + r, 0, 1)
        # reproducción (machos y hembras en la misma celda; mezcla de ambos grupos en proporciones)
        nM = 0.5 * ((n * rep).sum() + (nR * rep).sum())
        if nM > 0:
            sM = 0.5 * ((n * s * rep).sum() + (nR * sR * rep).sum()) / nM
            pmate = 1 - np.exp(-nM / G2_)
            bE = 0.5 * n * rep * pmate * p0 * s * sM * Lbar * q0 * s      # nacidos vivos por cohorte expuesta
            bRf = 0.5 * nR * rep * pmate * p0 * sR * sM * Lbar * q0 * sR
            B = bE.sum() + bRf.sum()
            s_new = ((bE * s).sum() + (bRf * sR).sum()) / B if B > 0 else 0.0
        else:
            B = 0.0; s_new = 0.0
        # envejecimiento + mortalidad
        surv = 1 - hazard_tab[:AMAX + 1]
        n = np.r_[0.0, (n * surv)[:-1]]; s = np.r_[s_new, s[:-1]]
        nR = np.r_[0.0, (nR * surv)[:-1]]; sR = np.r_[s_new, sR[:-1]]
        n[1] = B * (1 - rho) * surv[0]; nR[1] = B * rho * surv[0]
        s[1] = s_new; sR[1] = s_new
    return np.array(out)


if __name__ == '__main__':
    from scipy.signal import find_peaks
    for kappa in [np.inf, 2.0]:
        for rho in [0.0, 0.02, 0.05, 0.1]:
            o = run(kappa=kappa, rho=rho)
            N = o[:, 1]; S = o[:, 2]
            pk, _ = find_peaks(N, prominence=50)
            ext = o[-1, 0] if N[-1] < 1e-6 else np.nan
            print(f"kappa={kappa} rho={rho}: N_max={N.max():.0f} t_pico={np.argmax(N)} S_min={S.min():.4f} "
                  f"picos={np.round(N[pk][:6]).astype(int)} t_picos={pk[:6]} extincion(N<1e-6)={ext}  N_min tras 1er pico={N[np.argmax(N):].min():.3g}")
