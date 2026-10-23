"""Fase 7a: ecuación cinética extendida (subclase de kinetic.Kinetic).

Agrega, sin cambiar el caso por defecto (pair="shared" reproduce Kinetic.step):
  * pair="global": la hembra siempre encuentra macho; su s' es la ley post-actualización de los machos
    reproductivos de toda la población (límite de apareamiento global).
  * pair="moore": padre uniforme entre los machos reproductivos del bloque 3x3 de la hembra. En el límite de
    Poisson: k_M ~ Bin(k, phi) machos en la celda (comparten k y el desplazamiento de R2: misma ley que la hembra)
    y K'_M ~ Poi(8 nu phi) en las otras 8 celdas (independientes: ley marginal post de los machos reproductivos).
    P(padre de la celda) = E[k_M/(k_M+K'_M); k_M+K'_M>=1], P(padre de otra celda) = E[K'_M/(k_M+K'_M); ...].
  * removal=dict(t_r, f, s_thr, period): al final de la semana t_r (y cada `period` semanas si period>0) se retira
    la fracción f de los adultos (edad >= repro_age_min) con s < s_thr.
  * mu_bg: mortalidad de fondo semanal (como u25: p = 1-(1-p_edad)(1-mu_bg)).
  * registro de phi_eps = masa con s > eps para eps en EPS (y la masa Phi con juveniles latentes).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import poisson, binom
from kinetic import Kinetic, params_C1, params_SL

EPS = (0.0, 0.01, 0.05, 0.243)


class KineticF7(Kinetic):
    def __init__(self, p, removal=None, **kw):
        mu = p.mu_bg
        if mu:
            p = p.replace(mu_bg=0.0)       # Kinetic exige mu_bg = 0; se aplica aquí
        rad = p.mate_radius
        if rad:
            p = p.replace(mate_radius=0)
        super().__init__(p, **kw)
        self.mu_bg = mu
        self.removal = removal
        self.removed = []

    def _moore_probs(self, kv, phi, nu):
        """Psame[k], Pother[k] para el apareamiento en el bloque."""
        mu = 8.0 * nu * phi
        Kmax = int(poisson.ppf(1 - 1e-14, mu)) + 2 if mu > 0 else 0
        Kp = np.arange(Kmax + 1)
        PK = poisson.pmf(Kp, mu)
        Ps = np.zeros(len(kv)); Po = np.zeros(len(kv))
        for i, k in enumerate(kv):
            km = np.arange(k + 1)
            Pkm = binom.pmf(km, k, phi)
            tot = km[:, None] + Kp[None, :]
            W = Pkm[:, None] * PK[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                fs = np.where(tot > 0, km[:, None] / np.maximum(tot, 1), 0.0)
                fo = np.where(tot > 0, Kp[None, :] / np.maximum(tot, 1), 0.0)
            Ps[i] = float((W * fs).sum()); Po[i] = float((W * fo).sum())
        return Ps, Po

    def step(self):
        p = self.p
        F = self.F
        M = self.M
        nu = F.sum()
        rec = {"t": self.t + 1, "nu_pre": nu}
        if nu <= 0:
            self.t += 1
            return rec
        ps = F.sum(0) / nu
        sg = self.sgrid
        rec["theta"] = 1.0 - ps[0]
        for e in EPS:
            rec[f"phi_{e:g}"] = float(F[:, sg > e + 1e-12].sum())
        aw = p.plast_age_max if (np.isfinite(p.plast_age_max) and p.plast_outside_rec == 0) else np.inf
        lat = F[1: int(min(aw, self.A)) + 1, 0].sum() if np.isfinite(aw) else F[:, 0].sum()
        rec["phi"] = float(F[:, 1:].sum() + lat)
        rec["S"] = float(np.dot(ps, sg))
        aw_ = p.plast_age_max if np.isfinite(p.plast_age_max) else 12
        ages = np.arange(F.shape[0])
        pot = p.rec * np.maximum(aw_ - ages + 1, 0)
        le = ages <= p.repro_age_max
        Fle = F[le]
        rec["psi_hat"] = float((Fle * (sg[None, :] + pot[le][:, None])).sum())
        rec["psi_hat_M"] = 0.5 * rec["psi_hat"]
        Fa = F[p.repro_age_min:].sum(0)
        rec["frac_adult_sv"] = float(Fa[sg > 0.243].sum() / Fa.sum()) if Fa.sum() > 0 else np.nan
        kv, Pk, W, jv = self._Q_shift(ps, nu)
        dk = self.gM * np.maximum(0, kv + 1 - p.m_c)
        rec["D"] = float(np.dot(Pk, np.maximum(0, kv + 1 - p.m_c)))
        rec["H"] = float(Pk[kv + 1 <= p.m_c].sum())
        win = self.win
        Fw, Fo = F[win], F[~win]
        repw, repo = self.rep[win], self.rep[~win]
        phi = 0.5 * F[self.rep].sum() / nu
        newF = np.zeros_like(F)
        Gb = np.zeros(M + 1)
        Gb_oth = np.zeros(M + 1)
        out_rec = self.out_fac
        Fpost_rep = np.zeros(M + 1)
        zero = np.zeros(M + 1)
        if self.pair == "moore":
            Psame, Pother = self._moore_probs(kv, phi, nu)
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
                if self.pair in ("shared", "moore"):
                    tot = Ysum_rep.sum()
                    sm = float(np.dot(Ysum_rep, sg) / tot) if tot > 0 else 0.0
                    if self.pair == "shared":
                        Gb += wkj * pm * sm * 0.5 * Ysum_rep
                    else:
                        Gb += wkj * Psame[ik] * sm * 0.5 * Ysum_rep
                        Gb_oth += wkj * Pother[ik] * 0.5 * Ysum_rep
                Fpost_rep += wkj * Ysum_rep
        tot = Fpost_rep.sum()
        Sm = float(np.dot(Fpost_rep, sg) / tot) if tot > 0 else 0.0
        if self.pair == "indep":
            Gb = 0.5 * Fpost_rep * Sm * (1 - math.exp(-0.5 * tot))
        elif self.pair == "global":
            Gb = 0.5 * Fpost_rep * Sm
        elif self.pair == "moore":
            Gb = Gb + Sm * Gb_oth
        wts = Gb * p.p0 * self.Lbar * sg * np.minimum(p.neonatal_survival * sg, 1.0)
        births = wts.sum()
        nbpos = (p.inherit_weight * sg + (1 - p.inherit_weight) * p.s_newborn_base) * M
        nb = self._split(nbpos, wts, M + 1)
        rec["births"] = births
        post = newF
        surv = (1 - self.h[2: self.A + 1]) * (1 - self.mu_bg)
        F2 = np.zeros_like(F)
        F2[2:] = post[1:-1] * surv[:, None]
        F2[1] = nb * (1 - self.h[1]) * (1 - self.mu_bg)
        self.t += 1
        rem = self.removal
        if rem is not None:
            tr, f, per = rem["t_r"], rem["f"], rem.get("period", 0)
            ts = rem.get("t_stop")
            if (self.t == tr or (per > 0 and self.t > tr and (self.t - tr) % per == 0)) and (ts is None or self.t <= ts):
                mask = sg < rem.get("s_thr", 0.243) - 1e-12
                a0 = p.repro_age_min
                before = F2[a0:, mask].sum()
                F2[a0:, mask] *= (1 - f)
                self.removed.append((self.t, float(before * f)))
        self.F = F2
        rec["nu"] = F2.sum()
        return rec


def classify(df, T):
    """Clase del límite fluido: 'esteril' / 'allee' / 'fijo' / 'ciclo' (como bifurc.py)."""
    nu = df.nu.to_numpy(); t = df.t.to_numpy()
    out = dict(T_end=int(t[-1]), nu_peak=float(nu.max()), t_peak=int(t[nu.argmax()]))
    if nu[-1] < 1e-9:
        i = np.flatnonzero(nu < 1e-4)[0]
        S = float(df.S.iloc[i])
        out.update(cls="esteril" if S < 0.05 else "allee", S_at_low=S, t_ext_fluid=int(t[-1]),
                   t_fl_L30=int(t[np.flatnonzero((t > out["t_peak"]) & (900 * nu < 0.5))[0]]))
    else:
        late = df[df.t > t[-1] - 300]
        a = (late.nu.max() - late.nu.min()) / late.nu.mean()
        out.update(cls="fijo" if a < 0.02 else "ciclo", nu_min_late=float(late.nu.min()),
                   nu_mean_late=float(late.nu.mean()), nu_max_late=float(late.nu.max()), S_late=float(late.S.mean()))
    return out
