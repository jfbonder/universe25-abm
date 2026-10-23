"""Ecuación cinética corregida (R2.2) del ABM bien mezclado (jump_prob=1, jump_mode="uniform", sin refugios).

Estado: F[a, i] = densidad por celda (individuos / |G|) de edad a (edad al momento de la actualización de s de
la semana) y estado s = i/M, i = 0..M (ambos sexos; por simetría la mitad son hembras). Paso semanal, en el orden
exacto de Universe.step:

 1. Posiciones uniformes: el individuo ve k ~ Poisson(nu) otros en su celda y K' ~ Poisson(8 nu) otros en el resto
    del bloque 3x3 (independiente de k). K = k + K'.
 2. s' = Pi_[0,1](s + rho - gamma (k + 1 - m_c)_+),  rho = r 1[a en ventana] [(1-lambda) + lambda Ybar_K],
    Ybar_K = media de K sorteos i.i.d. de la marginal de s (pre-actualización; 0 si K = 0).
    modo "exact": ley exacta de Ybar_K (potencias de convolución por FFT);  modo "closure": Ybar_K := S(1-e^{-9nu})
    (cierre de rigor.md §7.1).  Fuera de la ventana rho se multiplica por plast_outside_rec.
 3. Reproducción con los s' (hembras y machos en edad fértil): la hembra encuentra macho fértil en su celda con
    prob. 1-(1-phi)^k (phi = fracción de machos fértiles en la población); el macho elegido tiene tipo pre ~
    marginal de los machos fértiles y comparte la celda (mismo k: mismo daño) y el bloque (mismo shift de R2,
    aproximación: se ignora la diferencia O(1/K) por la autoexclusión).  pair="shared" (límite verdadero) o
    pair="indep" (rigor.md §7.1 paso 2: s_m post ~ marginal de los machos, independiente del k de la hembra, y
    encuentro 1-e^{-nu_M}).  Crías esperadas = p0 Lbar q0 s_f'^2 s_m', con s_cría = w s_f' + (1-w) s0.
 4. Envejecimiento y muerte con h(a+1) (las crías hacen el primer sorteo a edad 1 en la misma semana).

Los desplazamientos de s no enteros en unidades de la grilla se reparten linealmente entre los dos nodos vecinos
(conserva la media).  Elegir M con gamma*M y r*M enteros (C1: M = 200 -> 20 y 20; SL: r M = 2).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import poisson
from u25.params import Params, preset


def params_C1(**kw):
    return preset("calhoun_C1", **kw)


def params_SL(**kw):
    return Params(**kw)


class Kinetic:
    def __init__(self, p: Params, M: int = 200, A: int | None = None, mode: str = "exact", pair: str = "shared",
                 nu0: float = 400 / 900, age0: int | None = None, block: float = 8.0, tol: float = 1e-13,
                 F0: np.ndarray | None = None):
        self.p = p
        self.M = M
        self.mode = mode
        self.pair = pair
        self.block = block
        self.tol = tol
        self.A = A if A is not None else int(p.a_max + 80)
        a = np.arange(self.A + 2)
        self.h = 1.0 / (1.0 + np.exp(-p.k_mort * (a - p.a_max)))
        self.sgrid = np.arange(M + 1) / M
        self.gM = p.gamma * M
        self.rM = p.rec * M
        self.Lbar = float(np.dot(p.litter_sizes, p.litter_probs))
        # ventana de plasticidad (edades al actualizar: 1..)
        win = (a >= p.plast_age_min) & (a <= p.plast_age_max)
        if not (p.plast_outside_rec != 1.0 and (p.plast_age_min > 0 or np.isfinite(p.plast_age_max))):
            win[:] = True
        self.win = win[: self.A + 1]
        self.out_fac = p.plast_outside_rec
        self.rep = ((a >= p.repro_age_min) & (a <= p.repro_age_max))[: self.A + 1]
        assert p.mating_rule == "random" and p.mate_radius == 0 and p.refractory_weeks == 0
        assert p.social_rec_source == "moore" and p.social_rec_isolated == "zero" and p.rec_threshold == 0
        assert p.defeat_gamma == 0 and p.social_hazard == 0 and p.social_mortality_factor == 0 and p.mu_bg == 0
        assert p.competence_age is None and p.nest_period == 0
        if F0 is None:
            F0 = np.zeros((self.A + 1, M + 1))
            F0[p.initial_age if age0 is None else age0, int(round(p.initial_s * M))] = nu0
        self.F = F0.copy()
        self.t = 0
        self.hist = []

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _split(pos, w, n):
        """Reparte pesos w en posiciones fraccionarias pos (0..n-1) linealmente."""
        out = np.zeros(n)
        pos = np.clip(pos, 0, n - 1)
        lo = np.floor(pos).astype(np.int64)
        fr = pos - lo
        hi = np.minimum(lo + 1, n - 1)
        np.add.at(out, lo, w * (1 - fr))
        np.add.at(out, hi, w * fr)
        return out

    def _shift(self, X, d):
        """Desplaza (con proyección a [0, M]) las filas de X (..., M+1) en d nodos enteros."""
        M = self.M
        if d == 0:
            return X.copy()
        Y = np.zeros_like(X)
        if d > 0:
            if d >= M:
                Y[..., M] = X.sum(-1)
                return Y
            Y[..., d:M] = X[..., : M - d]
            Y[..., M] = X[..., M - d:].sum(-1)
        else:
            d = -d
            if d >= M:
                Y[..., 0] = X.sum(-1)
                return Y
            Y[..., 1: M + 1 - d] = X[..., d + 1:]
            Y[..., 0] = X[..., : d + 1].sum(-1)
        return Y

    def _shift_frac(self, X, x):
        """Desplazamiento fraccionario x >= 0 o < 0 (reparto lineal entre floor y ceil)."""
        lo = math.floor(x)
        fr = x - lo
        Y = self._shift(X, lo) * (1 - fr)
        if fr > 1e-15:
            Y += self._shift(X, lo + 1) * fr
        return Y

    def _Q_shift(self, ps, nu):
        """W[k, j]: prob. conjunta de k otros en la celda y del desplazamiento de recuperación j (nodos enteros,
        j = 0..ceil(rM)) para los individuos en la ventana. Devuelve (kv, Pk, W, jv)."""
        p = self.p
        lam = p.social_rec_lambda
        kmax = int(poisson.ppf(1 - self.tol, nu)) + 2 if nu > 0 else 1
        kv = np.arange(kmax + 1)
        Pk = poisson.pmf(kv, nu)
        nj = int(math.ceil(self.rM)) + 2
        jv = np.arange(nj)
        if lam == 0 or self.rM == 0:
            W = np.zeros((kmax + 1, nj))
            W[:, :] = Pk[:, None] * self._split(np.array([self.rM]), np.array([1.0]), nj)[None, :]
            return kv, Pk, W, jv
        lb = self.block * nu
        Kpmax = int(poisson.ppf(1 - self.tol, lb)) + 2 if lb > 0 else 0
        Kp = np.arange(Kpmax + 1)
        PKp = poisson.pmf(Kp, lb)
        Kmax = kmax + Kpmax
        Q = np.zeros((Kmax + 1, nj))
        Q[0] = self._split(np.array([self.rM * (1 - lam)]), np.array([1.0]), nj)
        S = float(np.dot(ps, self.sgrid))
        if self.mode == "closure":
            ybar = S * (1 - math.exp(-(self.block + 1) * nu))
            row = self._split(np.array([self.rM * ((1 - lam) + lam * ybar)]), np.array([1.0]), nj)
            Q[:] = row[None, :]
        else:
            M = self.M
            nfft = 1 << int(math.ceil(math.log2(Kmax * M + 2)))
            P = np.fft.rfft(ps, nfft)
            # potencias en el dominio de Fourier (con renormalización por seguridad numérica)
            Pk_ = np.ones_like(P)
            for K in range(1, Kmax + 1):
                Pk_ = Pk_ * P
                dist = np.fft.irfft(Pk_, nfft)[: K * M + 1]
                dist = np.maximum(dist, 0.0)
                tot = dist.sum()
                if tot <= 0:
                    continue
                dist /= tot
                pos = self.rM * ((1 - lam) + lam * np.arange(K * M + 1) / (K * M))
                keep = dist > 1e-16
                Q[K] = self._split(pos[keep], dist[keep], nj)
        # W[k, j] = Pk[k] sum_K' PKp[K'] Q[k + K', j]
        W = np.zeros((kmax + 1, nj))
        for k in range(kmax + 1):
            W[k] = Pk[k] * (PKp[:, None] * Q[k: k + Kpmax + 1]).sum(0)
        return kv, Pk, W, jv

    # ------------------------------------------------------------------ paso
    def step(self):
        p = self.p
        F = self.F
        M = self.M
        nu = F.sum()
        rec = {}
        rec["t"] = self.t + 1
        rec["nu_pre"] = nu
        if nu <= 0:
            self.t += 1
            return rec
        ps = F.sum(0) / nu
        rec["theta"] = 1.0 - ps[0]                    # fracción de vecinos con s > 0 (los vecinos ~ marginal)
        # Φ: s > 0, o s = 0 con edad <= a_w (juveniles latentes); θ_Φ = φ/ν
        aw = p.plast_age_max if (np.isfinite(p.plast_age_max) and p.plast_outside_rec == 0) else np.inf
        lat = F[1: int(min(aw, self.A)) + 1, 0].sum() if np.isfinite(aw) else F[:, 0].sum()
        rec["phi"] = float(F[:, 1:].sum() + lat)
        rec["theta_phi"] = rec["phi"] / nu
        # competencia media por edad: crías (a = 1) y salida de la ventana (a = a_w + 1)
        for aa, nm in ((1, "S_a1"), (13, "S_a13")):
            m_ = F[aa].sum()
            rec[nm] = float(np.dot(F[aa], self.sgrid) / m_) if m_ > 0 else np.nan
        rec["S"] = float(np.dot(ps, self.sgrid))
        juvmask = (np.arange(self.A + 1) >= 1) & (np.arange(self.A + 1) <= min(p.plast_age_max, 10 ** 6))
        Fj = F[juvmask].sum(0)
        rec["S_juv"] = float(np.dot(Fj, self.sgrid) / Fj.sum()) if Fj.sum() > 0 else np.nan
        kv, Pk, W, jv = self._Q_shift(ps, nu)
        dk = self.gM * np.maximum(0, kv + 1 - p.m_c)          # daño en nodos
        win = self.win
        Fw, Fo = F[win], F[~win]
        repw, repo = self.rep[win], self.rep[~win]
        phi = 0.5 * F[self.rep].sum() / nu
        # machos fértiles: s pre (marginal)
        Fm = 0.5 * F[self.rep]
        newF = np.zeros_like(F)
        Gb = np.zeros(M + 1)                                   # densidad de madres post (ponderada) por s
        sg = self.sgrid
        out_rec = self.out_fac
        # para "indep": acumulamos la post-distribución de los machos fértiles y la de las hembras sin macho
        Fpost_rep = np.zeros(M + 1)
        zero = np.zeros(M + 1)
        for ik, k in enumerate(kv):
            d = dk[ik]
            pm = 1.0 - (1.0 - phi) ** k
            Yo_rep_sum = zero
            if Fo.shape[0] and out_rec == 0:
                Yo = self._shift_frac(Fo, -d)
                newF[~win] += Pk[ik] * Yo
                Yo_rep_sum = Yo[repo].sum(0)
            for ij, j in enumerate(jv):
                wkj = W[ik, ij]
                if wkj < 1e-16:
                    continue
                Yw = self._shift_frac(Fw, j - d)
                newF[win] += wkj * Yw
                if Fo.shape[0] and out_rec != 0:
                    Yo = self._shift_frac(Fo, out_rec * j - d)
                    newF[~win] += wkj * Yo
                    Yo_rep_sum = Yo[repo].sum(0)
                Ysum_rep = Yw[repw].sum(0) + Yo_rep_sum
                # densidad post (ambos sexos) de los fértiles que vivieron (k, j); hembras = 1/2
                if self.pair == "shared":
                    tot = Ysum_rep.sum()
                    sm = float(np.dot(Ysum_rep, sg) / tot) if tot > 0 else 0.0
                    Gb += wkj * pm * sm * 0.5 * Ysum_rep
                else:
                    Fpost_rep += wkj * Ysum_rep
        if self.pair == "indep":
            tot = Fpost_rep.sum()
            Sm = float(np.dot(Fpost_rep, sg) / tot) if tot > 0 else 0.0
            nuM = 0.5 * tot
            Gb = 0.5 * Fpost_rep * Sm * (1 - math.exp(-nuM))
        # nacimientos
        wts = Gb * p.p0 * self.Lbar * sg * np.minimum(p.neonatal_survival * sg, 1.0)
        births = wts.sum()
        nbpos = (p.inherit_weight * sg + (1 - p.inherit_weight) * p.s_newborn_base) * M
        nb = self._split(nbpos, wts, M + 1)
        self.last_nb = nb / nb.sum() if nb.sum() > 0 else None
        rec["births"] = births
        post = newF
        rec["S_post"] = float(np.dot(post.sum(0), sg) / post.sum()) if post.sum() > 0 else np.nan
        rec["frac0_post"] = float(post.sum(0)[0] / post.sum()) if post.sum() > 0 else np.nan
        Frep = post[self.rep].sum(0)
        rec["S_rep"] = float(np.dot(Frep, sg) / Frep.sum()) if Frep.sum() > 0 else np.nan
        rec["frac_rep_pos"] = float(1 - Frep[0] / Frep.sum()) if Frep.sum() > 0 else np.nan
        # envejecimiento y muerte
        F2 = np.zeros_like(F)
        F2[2:] = post[1:-1] * (1 - self.h[2: self.A + 1])[:, None]
        F2[1] = nb * (1 - self.h[1])
        self.F = F2
        self.t += 1
        rec["nu"] = F2.sum()
        Fs = F2.sum(0)
        nz = np.flatnonzero(Fs > 1e-300 * 0 + 1e-30)
        rec["max_s"] = float(sg[nz[-1]]) if nz.size else 0.0
        return rec

    def R1_frozen(self, g0=None, horizon=None):
        """R̂_1 (dicotomia.md, Teorema 2'): hijas esperadas a lo largo de la vida de UNA hembra nacida ahora (edad 1 en
        la próxima actualización) con ley de s inicial g0 (default: s = 1), en el ambiente congelado x_t = self.F
        (misma ley de (k, K'), de Ybar y de los machos en todas las semanas). Cuenta crías al nacer (antes del sorteo
        de muerte a edad 1), hembras = 1/2."""
        p = self.p; F = self.F; M = self.M; sg = self.sgrid
        nu = F.sum()
        if nu <= 0:
            return 0.0
        ps = F.sum(0) / nu
        kv, Pk, W, jv = self._Q_shift(ps, nu)
        dk = self.gM * np.maximum(0, kv + 1 - p.m_c)
        phi = 0.5 * F[self.rep].sum() / nu
        reps = np.flatnonzero(self.rep)
        # media del s post de los machos fértiles para cada (k, j)
        sm = np.zeros(W.shape)
        for ik in range(len(kv)):
            for ij, j in enumerate(jv):
                if W[ik, ij] < 1e-16:
                    continue
                tot = 0.0; acc = 0.0
                for a in reps:
                    if F[a].sum() <= 0:
                        continue
                    D = (j if self.win[a] else self.out_fac * j) - dk[ik]
                    Y = self._shift_frac(F[a], D)
                    tot += Y.sum(); acc += np.dot(Y, sg)
                sm[ik, ij] = acc / tot if tot > 0 else 0.0
        pm = 1.0 - (1.0 - phi) ** kv
        fert = p.p0 * self.Lbar * sg * np.minimum(p.neonatal_survival * sg, 1.0)   # por s'_f (sin s'_m)
        g = np.zeros(M + 1)
        if g0 is None:
            g[M] = 1.0
        else:
            g = np.asarray(g0, float).copy()
        dau = 0.0
        H = horizon or self.A
        for a in range(1, H):
            if g.sum() < 1e-14:
                break
            gn = np.zeros(M + 1)
            for ik in range(len(kv)):
                for ij, j in enumerate(jv):
                    w_ = W[ik, ij]
                    if w_ < 1e-16:
                        continue
                    D = (j if self.win[a] else self.out_fac * j) - dk[ik]
                    Y = self._shift_frac(g[None, :], D)[0]
                    gn += w_ * Y
                    if self.rep[a]:
                        dau += 0.5 * w_ * pm[ik] * sm[ik, ij] * np.dot(Y, fert)
            g = gn * (1 - self.h[a + 1])
        return dau

    def run(self, T, stop_nu=1e-12, verbose=False):
        for _ in range(T):
            r = self.step()
            self.hist.append(r)
            if verbose and self.t % 50 == 0:
                print(self.t, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if r.get("nu", 0) < stop_nu:
                break
        import pandas as pd
        return pd.DataFrame(self.hist)
