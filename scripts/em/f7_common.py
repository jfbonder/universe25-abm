"""Fase 7a (físico estadístico): infraestructura común.

Subclases del ABM (u25 sin modificar) con:
  * apareamiento: "cell" (u25, mate_radius=0), "moore" (u25, mate_radius=1, padre uniforme entre los machos
    reproductivos del bloque 3x3) y "global" (NUEVO, sólo aquí: padre uniforme entre todos los machos
    reproductivos de la arena; ver pedido_programador.md);
  * cadena truncada: redondeo estocástico rho_h de s tras cada
    actualización y en las crías, tope de edad A (muerte segura al cumplir A);
  * registro de la última semana de cría (madres que concibieron y padres que engendraron) -> observables
    operativos;
  * observables operativos semanales (E4): theta_eps, theta_R, D_J, H_J, m*_J, Phi_eps, ...;
  * retiro de una fracción f de los adultos no reproductores en la semana t_r (efecto hidra, E4/P2).

Con opts por defecto (mate="cell", h=0, A=None, sin retiro) las trayectorias son idénticas bit a bit a las de
u25.model.Universe / WMUniverse (verificado en check_identical()).
"""
from __future__ import annotations
import os, sys, zlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root (u25)
from u25.model import Universe
from u25.params import Params, preset
from u25 import kernels as K
from wm_abm import WMUniverse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GROUP_DIR = {"f7_e1": "adult_recovery", "f7_mate": "mate_search", "f7_op": "operational_indicators",
             "f7_hyd": "hydra_removal", "f7_mubg": "background_mortality", "f7_p3": "juvenile_protection"}


def group_dir(group):
    """results_em/<English name> of a sweep group ("f7_e1" -> adult_recovery, "f7_mate" -> mate_search)."""
    return os.path.join(ROOT, "results_em", GROUP_DIR.get(group, group))


S_V = 0.243          # s_v de C1 (transmission.tex, Def. 4.3)
EPS_LIST = (0.0, 0.01, 0.05, S_V)
BRED_WINDOW = 4      # "reproductivamente activo": concibió (hembra) o engendró (macho) en las últimas 4 semanas


def stoch_round(s, h, rng):
    if h <= 0:
        return s
    y = np.clip(s, 0.0, 1.0) / h
    lo = np.floor(y)
    fr = y - lo
    up = rng.random(s.shape[0]) < fr
    return np.clip((lo + up) * h, 0.0, 1.0)


class F7Mixin:
    AGENT_FIELDS = Universe.AGENT_FIELDS + ("last_bred",)

    def __init__(self, params, rng=None, opts=None):
        self.opts = dict(mate="cell", h=0.0, A=None, a_w_obs=12, log_every=1, log_dense_until=400,
                         log_sparse_every=5, removal=None, obs=True)
        if opts:
            self.opts.update(opts)
        super().__init__(params, rng=rng)
        if self.opts["h"] > 0:
            self.s = stoch_round(self.s, self.opts["h"], self.rng)
        self.last_bred = np.full(self.N, -10 ** 6, np.int64)
        self.obs_log = []
        self.removed_log = []

    # ------------------------------------------------------------- cadena truncada
    def update_social_state(self, m):
        if self.opts["obs"]:
            t = self.t
            if t <= self.opts["log_dense_until"] or t % self.opts["log_sparse_every"] == 0:
                d = op_observables(self, m)
                d["t"] = t
                self.obs_log.append(d)
        Universe.update_social_state(self, m)
        if self.opts["h"] > 0:
            self.s = stoch_round(self.s, self.opts["h"], self.rng)

    def social_delta(self, m):
        """P3: opts["manip"] = dict(kind="theta1" | "nodamage", t1, t2). Durante t1 <= t <= t2 los juveniles
        en la ventana aprenden de demostradores perfectos (s̄_V := 1; "theta1") o no sufren daño ("nodamage")."""
        d = Universe.social_delta(self, m)
        mp = self.opts.get("manip")
        if mp is None or not (mp["t1"] <= self.t <= mp["t2"]):
            return d
        p = self.p
        juv = (self.age >= p.plast_age_min) & (self.age <= p.plast_age_max)
        if not juv.any():
            return d
        lam = p.social_rec_lambda
        if mp["kind"] == "theta1":
            rec = p.rec * ((1.0 - lam) + lam * 1.0)
            det = p.gamma * np.maximum(0, m - p.m_c)
            d = np.where(juv, rec - det, d)
        elif mp["kind"] == "nodamage":
            rec = p.rec * ((1.0 - lam) + lam * self.social_learning_sbar())
            d = np.where(juv, rec, d)
        return d

    def death_probability(self):
        pa = super().death_probability()
        A = self.opts["A"]
        if A is not None:
            pa = np.where(self.age >= A, 1.0, pa)
        return pa

    # ------------------------------------------------------------- reproducción (copia de Universe.reproduce
    # con: apareamiento global opcional, redondeo de las crías y registro de last_bred)
    def reproduce(self, cell):
        p = self.p
        G2 = p.grid_size ** 2
        self.births = 0
        self.conceptions = 0
        self.pups = 0
        self.n_births_relocated = 0
        self.births_block_step = None
        rep = self.is_reproductive()
        fem = np.flatnonzero(rep & ~self.male & (self.refr == 0))
        mal = np.flatnonzero(rep & self.male)
        if fem.size == 0 or mal.size == 0:
            return
        mc = cell[mal]
        if p.mating_rule == "dominant":
            order = np.lexsort((-self.s[mal], mc))
        else:
            order = np.argsort(mc, kind="stable")
        mal_s = mal[order]
        cnt = np.bincount(mc, minlength=G2)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        fc = cell[fem]
        mate = self.opts["mate"]
        if mate == "global":
            fathers = mal[self.rng.integers(mal.size, size=fem.size)]
        elif p.mate_radius == 0:
            has = cnt[fc] > 0
            fem = fem[has]
            fc = fc[has]
            if fem.size == 0:
                return
            fathers = self.choose_fathers(fem, fc, mal_s, start, cnt)
        else:
            fem, fathers = self._fathers_moore(fem, cnt, start, mal_s)
            if fem.size == 0:
                return
        prob = p.p0 * self.s[fem] * self.s[fathers]
        conc = self.rng.random(fem.size) < prob
        mothers = fem[conc]
        self.last_bred[mothers] = self.t
        self.last_bred[fathers[conc]] = self.t
        first = mothers[~self.conceived[mothers] & (self.birth_t[mothers] >= 1)
                        & (self.age[mothers] <= self.COHORT_CONC_AGE)]
        if first.size:
            bw, cnt_ = np.unique(self.birth_t[first], return_counts=True)
            for b, c in zip(bw.tolist(), cnt_.tolist()):
                self.coh_f_conc[b] = self.coh_f_conc.get(b, 0) + c
        self.conceived[mothers] = True
        self.conceptions = int(mothers.size)
        if mothers.size == 0:
            return
        if p.refractory_weeks > 0:
            self.refr[mothers] = p.refractory_weeks
        L = self.rng.choice(self._lit, size=mothers.size, p=self._litp)
        self.pups = int(L.sum())
        q = np.clip(p.neonatal_survival * self.s[mothers], 0.0, 1.0)
        surv = self.rng.binomial(L, q)
        nb = int(surv.sum())
        self.births = nb
        if nb == 0:
            return
        mrep = np.repeat(mothers, surv)
        B = p.obs_blocks
        G = p.grid_size
        blk = (self.x[mrep] * B // G) * B + (self.y[mrep] * B // G)
        bb = np.bincount(blk, minlength=B * B)
        self.births_block += bb
        self.births_block_step = bb
        new_ids = self.next_id + np.arange(nb, dtype=np.int64)
        self.next_id += nb
        self._ensure_alive_capacity(self.next_id)
        self._alive_id[new_ids] = True
        self.id = np.concatenate([self.id, new_ids])
        self.male = np.concatenate([self.male, self.rng.random(nb) < p.p_male])
        nbx, nby = self.x[mrep], self.y[mrep]
        nbcell = self._cell[mrep]
        if self._hard == "strict" and self.refuge is not None:
            nbx, nby, nbcell = self._place_newborns(mrep, nbx, nby, nbcell)
        self.x = np.concatenate([self.x, nbx])
        self.y = np.concatenate([self.y, nby])
        self.age = np.concatenate([self.age, np.zeros(nb, np.int64)])
        snb = self.newborn_s(mrep)
        if self.opts["h"] > 0:
            snb = stoch_round(snb, self.opts["h"], self.rng)
        self.s = np.concatenate([self.s, snb])
        self.refr = np.concatenate([self.refr, np.zeros(nb, np.int64)])
        self._m = np.concatenate([self._m, self._m[mrep]])
        self._cell = np.concatenate([self._cell, nbcell])
        self.cap = np.concatenate([self.cap, np.full(nb, np.inf)])
        self.iso_run = np.concatenate([self.iso_run, np.zeros(nb, np.int64)])
        self.conceived = np.concatenate([self.conceived, np.zeros(nb, bool)])
        self.h_mat = np.concatenate([self.h_mat, self.s[mrep]])
        self.birth_t = np.concatenate([self.birth_t, np.full(nb, self.t, np.int64)])
        self.last_bred = np.concatenate([self.last_bred, np.full(nb, -10 ** 6, np.int64)])
        nf_new = int((~self.male[-nb:]).sum())
        if nf_new:
            self.coh_f_born[self.t] = self.coh_f_born.get(self.t, 0) + nf_new

    # ------------------------------------------------------------- retiro (hidra)
    def remove_mask(self, die):
        nd = int(die.sum())
        if nd == 0:
            return 0
        self._alive_id[self.id[die]] = False
        keep = ~die
        newidx = np.cumsum(keep) - 1
        newidx[die] = -1
        for name in self.AGENT_FIELDS:
            setattr(self, name, getattr(self, name)[keep])
        if self._pa.size:
            self._pa, self._pb, self._pF = K.remap_pairs(self._pa, self._pb, self._pF, newidx)
        return nd

    def step(self):
        super().step()
        rem = self.opts["removal"]
        if rem is not None and self.N:
            t_r, f, rule = rem["t_r"], rem["f"], rem.get("rule", "s")
            period = rem.get("period", 0)
            t_stop = rem.get("t_stop")
            if (self.t == t_r or (period > 0 and self.t > t_r and (self.t - t_r) % period == 0)) and \
                    (t_stop is None or self.t <= t_stop):
                adult = self.age >= self.p.repro_age_min
                if rule == "s":            # no reproductor por estado: s < s_v
                    nonrep = adult & (self.s < rem.get("s_thr", S_V))
                else:                      # operativo: no crió en las últimas BRED_WINDOW semanas
                    nonrep = adult & (self.last_bred < self.t - rem.get("window", 8))
                cand = np.flatnonzero(nonrep)
                rr = np.random.default_rng([7, self.t, int(self.N)])
                k = int(np.floor(f * cand.size + rr.random()))      # redondeo estocástico de f*n
                die = np.zeros(self.N, bool)
                if k > 0:
                    die[rr.choice(cand, size=min(k, cand.size), replace=False)] = True
                nd = self.remove_mask(die)
                self.removed_log.append((self.t, nd, int(cand.size), int(self.N)))


class F7Spatial(F7Mixin, Universe):
    pass


class F7WM(F7Mixin, WMUniverse):
    measure_theta = False

    def update_social_state(self, m):
        return F7Mixin.update_social_state(self, m)


# ----------------------------------------------------------------------------- observables operativos
def op_observables(U, m):
    """Antes de actualizar s (s de la semana anterior, posiciones de esta semana). m = ocupación post-movimiento."""
    p = U.p
    G = p.grid_size
    cell = U._cell
    s = U.s
    n = s.size
    out = dict(N=n)
    if n == 0:
        return out
    a_w = U.opts["a_w_obs"]
    juv = (U.age >= 1) & (U.age <= a_w)
    adult = U.age >= p.repro_age_min
    repage = (U.age >= p.repro_age_min) & (U.age <= p.repro_age_max)
    bred = repage & (U.last_bred >= U.t - BRED_WINDOW)
    Mc = np.bincount(cell, minlength=G * G).reshape(G, G).astype(np.float64)
    MV = U._moore_sum(Mc).ravel()[cell] - 1.0
    nj = int(juv.sum())
    out["n_juv"] = nj
    out["n_adult"] = int(adult.sum())
    # masas
    for e in EPS_LIST:
        out[f"Phi_{e:g}"] = int((s > e).sum())
    out["Phi_lat"] = int(((s > 0) | juv).sum())                 # Φ (s>0 o juvenil latente)
    out["S_mean"] = float(s.mean())
    # competencia potencial: s + r (a_w - a + 1)^+, edades <= a_rep
    aw = p.plast_age_max if np.isfinite(p.plast_age_max) else a_w
    shat = s + p.rec * np.maximum(aw - U.age + 1, 0)
    le = U.age <= p.repro_age_max
    out["Psi_hat"] = float(shat[le].sum())
    out["Psi_hat_M"] = float(shat[le & U.male].sum())
    out["frac_adult_sv"] = float((s[adult] > S_V).mean()) if adult.any() else np.nan
    out["frac_rep_bred"] = float(bred.sum() / max(repage.sum(), 1))
    fr = repage & ~U.male
    out["frac_fem_bred"] = float((bred & fr).sum() / max(fr.sum(), 1))
    out["mstar_all"] = float(m.mean() - 1.0)
    if nj:
        mj = m[juv].astype(float)
        out["mstar_J"] = float(mj.mean() - 1.0)                    # hacinamiento medio experimentado (Lloyd)
        out["D_J"] = float(np.maximum(mj - p.m_c, 0).mean())        # exceso medio sobre m_c
        out["H_J"] = float((mj <= p.m_c).mean())                    # fracción de juveniles en bolsones sanos
        ok = juv & (MV > 0.5)
        if ok.any():
            for e in (0.0, 0.05, S_V):
                Pc = np.bincount(cell, weights=(s > e).astype(float), minlength=G * G).reshape(G, G)
                PV = U._moore_sum(Pc).ravel()[cell] - (s > e)
                out[f"theta_{e:g}"] = float(np.mean(PV[ok] / MV[ok]))
            # theta_R: fracción de los vecinos en edad reproductiva que criaron en las últimas 4 semanas
            Rc = np.bincount(cell, weights=repage.astype(float), minlength=G * G).reshape(G, G)
            Bc = np.bincount(cell, weights=bred.astype(float), minlength=G * G).reshape(G, G)
            RV = U._moore_sum(Rc).ravel()[cell] - repage
            BV = U._moore_sum(Bc).ravel()[cell] - bred
            okr = juv & (RV > 0.5)
            out["theta_R"] = float(np.mean(BV[okr] / RV[okr])) if okr.any() else np.nan
            out["frac_juv_iso"] = float(1 - ok.sum() / nj)
    return out


# ----------------------------------------------------------------------------- configuraciones
def make_params(model, L, wm, refuges=None, T=2000, cpc=40.0, **over):
    """model in {"C1","SL"}. refuges None -> C1: según wm (espacial con refugios = preset; b.m. sin refugios)."""
    n_half = int(round(200 * L * L / 900))
    if model == "C1":
        base = dict(preset("calhoun_C1").to_dict())
        if refuges is None:
            refuges = not wm
        if not refuges:
            base.update(refuge_frac=0.0, refuge_pref=0.0, refuge_hard=False)
    elif model == "SL":
        base = Params().to_dict()
        if refuges:
            base.update(refuge_frac=0.3, refuge_capacity=2, refuge_seed=0, refuge_pref=100.0, refuge_hard="strict")
    else:
        raise KeyError(model)
    base.update(grid_size=L, n_males=n_half, n_females=n_half, T=T, max_pop=int(cpc * L * L))
    if wm:
        base.update(jump_prob=1.0, jump_mode="uniform")
    base.update(over)
    return Params.from_dict(base)


def build(model, L, wm, seed_key, k, T=2000, refuges=None, opts=None, **over):
    mate = (opts or {}).get("mate", "cell")
    if mate == "moore":
        over = dict(over, mate_radius=1)
    p = make_params(model, L, wm, refuges=refuges, T=T, **over)
    cfg = zlib.crc32(seed_key.encode())
    rng = np.random.default_rng([20261005, cfg, k])
    cls = F7WM if wm else F7Spatial
    return cls(p, rng=rng, opts=opts)


def run_and_pack(U, T=None):
    se, su = U.run(T)
    B = se["births"].to_numpy()
    t = se["t"].to_numpy()
    nz = np.flatnonzero(B > 0)
    out = dict(extinct=bool(su["extinct"]), exploded=bool(su["exploded"]),
               t_ext=float(su["t_ext"]) if su["extinct"] else np.nan, t_end=int(su["t_end"]),
               N_max=int(su["N_max"]), t_peak=int(su["t_peak"]),
               t_last=int(t[nz[-1]]) if nz.size else -1, N0=int(su["N0"]))
    # series compactas: denso hasta 400, luego cada 5
    keep = (t <= 400) | (t % 5 == 0) | (t == t[-1])
    out["s_t"] = t[keep].astype(np.int32)
    out["s_N"] = se["N"].to_numpy()[keep].astype(np.int32)
    out["s_B"] = B[keep].astype(np.int32)
    out["s_S"] = se["S_mean"].to_numpy()[keep].astype(np.float32)
    if U.obs_log:
        keys = sorted(set().union(*[d.keys() for d in U.obs_log]))
        for c in keys:
            out[f"o_{c}"] = np.array([d.get(c, np.nan) for d in U.obs_log], np.float32)
    if U.removed_log:
        out["removed"] = np.array(U.removed_log, np.int64)
    return out


def save_npz(fn, d):
    tmp = fn + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, fn)


def check_identical(T=120, seed=5):
    """Con opciones neutras las trayectorias coinciden bit a bit con Universe / WMUniverse."""
    res = []
    for wm in (True, False):
        p = make_params("C1", 15, wm, T=T)
        a = (WMUniverse if wm else Universe)(p, rng=np.random.default_rng(seed))
        if wm:
            a.measure_theta = False
        sa, _ = a.run()
        b = (F7WM if wm else F7Spatial)(p, rng=np.random.default_rng(seed))
        sb, _ = b.run()
        res.append((wm, np.array_equal(sa["N"].to_numpy(), sb["N"].to_numpy()) and np.allclose(a.s, b.s)))
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(check_identical(), f"{time.time()-t0:.1f}s")
