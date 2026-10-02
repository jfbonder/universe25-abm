"""Reimplementación eficiente del modelo del notebook original (ver u25/reference.py para el puerto fiel).

Estado de agentes en struct-of-arrays (id, sexo, x, y, edad, s).  Orden de un paso
(idéntico al notebook, `Universe.step`):

  1. movimiento simultáneo en el vecindario de Moore (incluye quedarse) con pesos
     w_i(y) = 1 + alpha * sum_{j en y, j != i} F_ij   (snapshot pre-movimiento);  t += 1
  2. ocupación post-movimiento
  3. afinidades: pares a distancia de Chebyshev <= 1 se refuerzan, el resto decae, poda < eps
  4. estado social: s += rec - gamma (m - m_c)_+ , recorte a [0,1]   (m = ocupación de la celda)
  5. reproducción (edades previas al envejecimiento, s ya actualizado)
  6. envejecimiento (incluye neonatos -> edad 1)
  7. mortalidad logística en la edad (también se aplica a los neonatos), se borran sus afinidades

Afinidades: almacén disperso ordenado por clave (id_lo << 32 | id_hi) con decaimiento
perezoso: se guarda (F, t_ultimo_refuerzo) y el valor efectivo es F (1-delta)^(t - t0).
Esto es *exactamente* equivalente al decaimiento paso a paso del notebook, incluida la poda
(el decaimiento es monótono, la poda se hace cada paso con el valor efectivo).

Observación clave para el movimiento: los pares (i, j) con j en el vecindario de Moore de i
en el instante del movimiento t+1 son exactamente los pares "activos" reforzados en el paso t
(posiciones sin cambio entre ambos momentos, salvo nacimientos con F=0 y muertes).  Por eso
el campo A_i(y) = sum_j F_ij se arma con la lista de pares activos del paso anterior, sin
búsquedas en el diccionario.

Reglas de la Fase 2, todas con default neutro:
  R1 ventana (plast_*), techo de competencia (competence_age), R2 recuperación social
  (social_rec_lambda, social_rec_source, rec_threshold), R3 saltos (jump_prob, jump_mode),
  R4 retiro (alpha_s_mode, crowd_aversion), R5 derrota (defeat_gamma), founders_same_cell.
  Con los defaults la trayectoria es idéntica bit a bit a la de la Fase 1 (test de huella).

v2, también con default neutro: capacidad de refugios estrictamente dura
(refuge_hard = True/"strict"; la regla de v1 es "legacy"), mortalidad de fondo (mu_bg), período de nido
(nest_period), movimiento asincrónico (async_move) y edades iniciales distribuidas (initial_age_dist).

Puntos de extensión: subclasear `Universe` y redefinir alguno de
  alpha_individual, movement_weights, social_delta, choose_fathers, death_probability,
  newborn_s, extra_stats
(la clase debe estar definida a nivel de módulo para poder usarla con multiprocessing).
"""
from __future__ import annotations

import time as _time

import numpy as np

from .params import Params
from . import kernels as K

_MASK32 = np.int64(0xFFFFFFFF)


class AffinityStore:
    """Diccionario disperso de afinidades con decaimiento perezoso.

    Se guarda, por par (clave id_lo << 32 | id_hi, arreglo ordenado), el valor F al último
    refuerzo y el paso t0 de ese refuerzo; el valor al final del paso t es F (1-delta)^(t-t0).
    La poda (F efectivo < eps, o individuo muerto) se hace en la fusión lineal del paso siguiente
    (`kernels.merge_reinforce`); `count` ignora lógicamente lo ya podable, de modo que el conteo
    coincide exactamente con len(affinity) del notebook al final de cada paso.
    """

    def __init__(self, eta: float, delta: float, eps: float):
        self.eta = eta
        self.delta = delta
        self.eps = eps
        self.keys = np.empty(0, np.int64)
        self.val = np.empty(0, np.float64)
        self.t0 = np.empty(0, np.int64)
        self._powtab = np.ones(1)
        self._cur = None     # buffers (keys, val, t0) en uso
        self._spare = None   # buffers libres para la próxima fusión

    def __len__(self):
        return self.keys.shape[0]

    def _pow(self, t):
        if self._powtab.shape[0] < t + 2:
            n = max(t + 2, 2 * self._powtab.shape[0], 1024)
            self._powtab = np.power(1.0 - self.delta, np.arange(n, dtype=np.float64))
        return self._powtab

    def effective(self, t: int) -> np.ndarray:
        """Valores al final del paso t (sin aplicar poda)."""
        return self.val * self._pow(t)[t - self.t0]

    def as_dict(self, t: int, alive_id: np.ndarray) -> dict:
        """{(id_lo, id_hi): F} al final del paso t, con la poda aplicada (para tests)."""
        eff = self.effective(t)
        lo = self.keys >> 32
        hi = self.keys & _MASK32
        keep = (eff >= self.eps) & alive_id[lo] & alive_id[hi]
        return {(int(a), int(b)): float(v) for a, b, v in zip(lo[keep], hi[keep], eff[keep])}

    def reinforce(self, q: np.ndarray, t: int, alive_id: np.ndarray) -> np.ndarray:
        """Refuerza los pares activos q (claves únicas ordenadas) en el paso t. Devuelve F nuevo.

        Usa doble buffer para no asignar memoria nueva en cada paso (el costo dominante es
        ancho de banda de memoria / page faults, no aritmética).
        """
        powtab = self._pow(t)
        need = len(self) + q.shape[0]
        b = self._spare
        if b is None or b[0].shape[0] < need:
            cap = int(need * 1.5) + 1024
            b = (np.empty(cap, np.int64), np.empty(cap, np.float64), np.empty(cap, np.int64))
        F = np.empty(q.shape[0], np.float64)
        n = K.merge_reinforce_into(self.keys, self.val, self.t0, q, t, self.eta, self.eps,
                                   powtab, alive_id, b[0], b[1], b[2], F)
        old = self._cur
        self._cur = b
        self._spare = old
        self.keys, self.val, self.t0 = b[0][:n], b[1][:n], b[2][:n]
        return F

    def count(self, t: int, alive_id: np.ndarray):
        """(n, F medio) de afinidades vigentes al final del paso t."""
        if len(self) == 0:
            return 0, 0.0
        n, s = K.count_effective(self.keys, self.val, self.t0, t, self.eps, self._pow(t), alive_id)
        return int(n), (s / n if n else 0.0)


class Universe:
    def __init__(self, params: Params | None = None, rng: np.random.Generator | None = None):
        p = params if params is not None else Params()
        p.validate()
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(p.seed)
        G = p.grid_size
        n0 = p.n_males + p.n_females
        self.id = np.arange(n0, dtype=np.int64)
        self.male = np.concatenate([np.ones(p.n_males, bool), np.zeros(p.n_females, bool)])
        if p.founders_same_cell:
            self.x = np.full(n0, G // 2, np.int64)
            self.y = np.full(n0, G // 2, np.int64)
        else:
            xy = self.rng.integers(G, size=(n0, 2))
            self.x = xy[:, 0].astype(np.int64)
            self.y = xy[:, 1].astype(np.int64)
        self.age = np.full(n0, p.initial_age, np.int64)
        if p.initial_age_dist == "uniform" and n0:
            # condición inicial no sincrónica (v2): edades uniformes en [initial_age, initial_age_hi]
            self.age = self.rng.integers(p.initial_age, p.initial_age_hi + 1, size=n0).astype(np.int64)
        self.s = np.full(n0, float(p.initial_s))
        self.refr = np.zeros(n0, np.int64)
        self.cap = np.full(n0, np.inf)           # techo de competencia (competence_age)
        self.iso_run = np.zeros(n0, np.int64)    # semanas consecutivas con m_i <= obs_iso_m
        self.conceived = np.zeros(n0, bool)      # ever_conceived (O11b)
        self.h_mat = np.ones(n0)                 # s de la madre al nacer (fundadores: 1)
        self.birth_t = -self.age.copy()          # semana de nacimiento (fundadores: -edad)
        self.coh_f_born = {}                     # semana de nacimiento -> nº de hembras nacidas
        self.coh_f_conc = {}                     # ... -> nº que concibió alguna vez con edad <= 48
        self.next_id = n0
        self._alive_id = np.ones(max(1024, 2 * n0), bool)
        self.store = AffinityStore(p.eta, p.delta, p.eps)
        self._pa = np.empty(0, np.int64)   # pares activos del último paso (índices de agente)
        self._pb = np.empty(0, np.int64)
        self._pF = np.empty(0, np.float64)
        self.t = 0
        self.births = 0
        self.deaths = 0
        self.conceptions = 0
        self._m = np.zeros(n0, np.int64)   # ocupación post-movimiento de la celda de cada agente
        self._lit = np.asarray(p.litter_sizes, np.int64)
        self._litp = np.asarray(p.litter_probs, np.float64)
        self._nbuf = None
        self._cell = np.zeros(n0, np.int64)       # celda post-movimiento
        self.n_jumps = 0
        self.n_defeated = 0
        self.n_refuge_rejected = 0
        self.n_births_relocated = 0               # crías ubicadas fuera de un refugio lleno (strict)
        self.n_capacity_unresolved = 0            # excesos sin celda libre posible (sólo si todo es refugio)
        self.n_zero_weight = 0                    # filas de pesos todas nulas (el agente se queda)
        self.tot = dict(refuge_rejected=0, births_relocated=0, zero_weight=0)   # acumulados de la corrida
        self._hard = p.refuge_hard_mode           # 'off' | 'strict' | 'legacy'
        self.pups = 0
        self.deaths_social_exp = 0.0
        self.obs = {}                             # observables medidos dentro del paso
        B = p.obs_blocks
        self.births_block = np.zeros(B * B, np.int64)
        self.births_block_step = None
        self.refuge = None                       # máscara de celdas refugio (G*G) o None
        if p.refuge_frac > 0:
            nr = int(round(p.refuge_frac * G * G))
            rr = np.random.default_rng(p.refuge_seed)
            self.refuge = np.zeros(G * G, bool)
            self.refuge[rr.choice(G * G, nr, replace=False)] = True
            if self._hard == "strict" and n0:
                self._place_within_capacity()

    AGENT_FIELDS = ("id", "male", "x", "y", "age", "s", "refr", "cap", "iso_run", "_m", "_cell",
                    "conceived", "birth_t", "h_mat")
    COHORT_CONC_AGE = 48   # O11b: fracción evaluada al cumplir 48 semanas (o al final de la corrida)

    @classmethod
    def from_agents(cls, params: Params, rng: np.random.Generator, male, age, s, x=None, y=None,
                    cap=None, h_mat=None):
        """Universo nuevo (grilla vacía, sin afinidades) con los agentes dados. Si x/y son None
        se ubican todos en la celda central (o al azar si params.founders_same_cell es False)."""
        n = len(male)
        p2 = params.replace(n_males=0, n_females=0)
        U = cls(p2, rng=rng)
        G = p2.grid_size
        U.id = np.arange(n, dtype=np.int64)
        U.male = np.asarray(male, bool).copy()
        U.age = np.asarray(age, np.int64).copy()
        U.s = np.asarray(s, np.float64).copy()
        if x is None:
            if params.founders_same_cell:
                x = np.full(n, G // 2)
                y = np.full(n, G // 2)
            else:
                xy = rng.integers(G, size=(n, 2))
                x, y = xy[:, 0], xy[:, 1]
        U.x = np.asarray(x, np.int64).copy()
        U.y = np.asarray(y, np.int64).copy()
        U.refr = np.zeros(n, np.int64)
        U.cap = np.full(n, np.inf) if cap is None else np.asarray(cap, float).copy()
        U.iso_run = np.zeros(n, np.int64)
        U.conceived = np.zeros(n, bool)
        U.h_mat = np.ones(n) if h_mat is None else np.asarray(h_mat, float).copy()
        U.birth_t = -U.age.copy()
        U.coh_f_born, U.coh_f_conc = {}, {}
        U._m = np.zeros(n, np.int64)
        U._cell = np.zeros(n, np.int64)
        U.next_id = n
        U._alive_id = np.ones(max(1024, 2 * n), bool)
        if U.refuge is not None and U._hard == "strict" and n:
            U._place_within_capacity()
        return U

    # ------------------------------------------------------------------ utilidades
    @property
    def N(self) -> int:
        return self.id.shape[0]

    def _ensure_alive_capacity(self, n):
        if n > self._alive_id.shape[0]:
            new = np.zeros(max(n, 2 * self._alive_id.shape[0]), bool)
            new[: self._alive_id.shape[0]] = self._alive_id
            self._alive_id = new

    def is_reproductive(self) -> np.ndarray:
        return (self.age >= self.p.repro_age_min) & (self.age <= self.p.repro_age_max)

    def density_map(self) -> np.ndarray:
        G = self.p.grid_size
        return np.bincount(self.x * G + self.y, minlength=G * G).reshape(G, G)

    def _nest_mask(self):
        """Crías en el nido (edad <= nest_period): inmóviles y sin contribución al hacinamiento. None si
        nest_period = 0."""
        if self.p.nest_period <= 0:
            return None
        return self.age <= self.p.nest_period

    def _crowd_occupancy(self, nest=None):
        """Ocupación por celda que cuenta para el hacinamiento (sin las crías en el nido)."""
        G = self.p.grid_size
        c = self.x * G + self.y
        if nest is None:
            return np.bincount(c, minlength=G * G)
        return np.bincount(c[~nest], minlength=G * G)

    # ------------------------------------------------------------------ capacidad de los refugios
    def _cells_out_of_refuge(self, src: np.ndarray) -> np.ndarray:
        """Para cada celda de `src`, una celda no refugio uniforme del anillo de Chebyshev más chico
        (1, 2, ...) que tenga alguna. Consume len(src) uniformes. -1 si no hay ninguna (todo es refugio)."""
        p = self.p
        G = p.grid_size
        src = np.asarray(src, np.int64)
        out = np.full(src.size, -1, np.int64)
        if src.size == 0:
            return out
        u = self.rng.random(src.size)
        sx, sy = src // G, src % G
        todo = np.arange(src.size)
        r = 1
        while todo.size and r <= G:
            offs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if max(abs(dx), abs(dy)) == r]
            ox = np.array([o[0] for o in offs], np.int64)
            oy = np.array([o[1] for o in offs], np.int64)
            nx = sx[todo, None] + ox[None, :]
            ny = sy[todo, None] + oy[None, :]
            if p.periodic:
                valid = np.ones(nx.shape, bool)
                nx %= G
                ny %= G
            else:
                valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
                nx = np.clip(nx, 0, G - 1)
                ny = np.clip(ny, 0, G - 1)
            c = nx * G + ny
            ok = valid & ~self.refuge[c]
            k = ok.sum(1)
            has = k > 0
            if has.any():
                h = np.flatnonzero(has)
                pick = np.minimum((u[todo[h]] * k[h]).astype(np.int64), k[h] - 1)
                col = (np.cumsum(ok[h], 1) <= pick[:, None]).sum(1)
                out[todo[h]] = c[h, col]
            todo = todo[~has]
            r += 1
        return out

    def _relocate(self, idx: np.ndarray):
        """Mueve a los agentes idx (en refugios llenos) a una celda no refugio cercana."""
        G = self.p.grid_size
        dest = self._cells_out_of_refuge(self.x[idx] * G + self.y[idx])
        ok = dest >= 0
        self.n_capacity_unresolved += int((~ok).sum())
        self.x[idx[ok]] = dest[ok] // G
        self.y[idx[ok]] = dest[ok] % G

    def _place_within_capacity(self):
        """Capacidad estricta en la condición inicial: en cada refugio con más ocupantes que la capacidad
        se quedan `capacidad` (sorteados) y el resto pasa a una celda vecina no refugio."""
        G = self.p.grid_size
        cap = self.p.refuge_capacity
        cell = self.x * G + self.y
        occ = np.bincount(cell, minlength=G * G)
        over = self.refuge[cell] & (occ[cell] > cap)
        if not over.any():
            return
        idx = np.flatnonzero(over)
        idx = idx[self.rng.permutation(idx.size)]
        c = cell[idx]
        o = np.argsort(c, kind="stable")
        cs = c[o]
        first = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
        rank = np.arange(cs.size) - np.repeat(first, np.diff(np.r_[first, cs.size]))
        out = np.sort(idx[o][rank >= cap])
        if out.size:
            self._relocate(out)

    # ------------------------------------------------------------------ movimiento
    def alpha_individual(self) -> np.ndarray | float:
        """alpha_i (escalar si es constante). Punto de extensión (1)."""
        p = self.p
        if p.alpha_s_mode == "const":
            return p.alpha
        s = self.s
        if p.alpha_s_mode == "power":
            f = s ** p.alpha_s_power
        elif p.alpha_s_mode == "threshold":
            f = (s >= p.alpha_s_threshold).astype(np.float64)
        else:  # sigmoid
            f = 1.0 / (1.0 + np.exp(-(s - p.alpha_s_threshold) / p.alpha_s_width))
        return p.alpha_low + (p.alpha - p.alpha_low) * f

    def _neighbor_cells(self):
        p = self.p
        G = p.grid_size
        nx = self.x[:, None] + K.DX[None, :]
        ny = self.y[:, None] + K.DY[None, :]
        if p.periodic:
            valid = np.ones(nx.shape, bool)
            nx %= G
            ny %= G
        else:
            valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
            nx = np.clip(nx, 0, G - 1)
            ny = np.clip(ny, 0, G - 1)
        return nx * G + ny, valid

    def movement_weights(self, A: np.ndarray, nbcell: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Pesos N x 9 (columna d = desplazamiento (DX[d], DY[d])). Puntos de extensión (1) y (3).

        A[i, d] = sum de afinidades de i con los agentes de la celda vecina d.
        """
        p = self.p
        a = self.alpha_individual()
        a2 = a if np.isscalar(a) else a[:, None]
        if p.weight_form == "linear":
            W = 1.0 + a2 * A
            if np.any(np.asarray(a) < 0):
                np.maximum(W, p.weight_floor, out=W)
        else:
            W = np.exp(a2 * A)
        if p.crowd_aversion > 0 or p.territory_strength > 0:
            G = p.grid_size
            if p.crowd_aversion > 0:
                occ = self._crowd_occupancy(self._nest_mask())
                n_y = occ[nbcell].astype(np.float64)
                n_y[:, 4] -= 1.0  # sin contarse a sí mismo
                c = p.crowd_aversion * (1.0 - self.s) ** p.crowd_power
                W *= np.exp(-c[:, None] * n_y)
            if p.territory_strength > 0:
                rm = self.male & self.is_reproductive()
                occm = np.bincount(self.x[rm] * G + self.y[rm], minlength=G * G)
                M_y = occm[nbcell[rm]].astype(np.float64)
                M_y[:, 4] -= 1.0
                W[rm] *= np.exp(-p.territory_strength * M_y)
        if self.refuge is not None:
            G = p.grid_size
            occ = np.bincount(self.x * G + self.y, minlength=G * G)
            n_y = occ[nbcell]
            n_y[:, 4] -= 1
            W[self.refuge[nbcell] & (n_y >= p.refuge_capacity)] = 0.0
        W[~valid] = 0.0
        if self.refuge is not None and p.refuge_pref > 0:
            base = self.s if p.refuge_pref_basis == "current" else np.clip(self.h_mat, 0.0, 1.0)
            f = 1.0 + p.refuge_pref * (1.0 - base) ** p.refuge_pref_power
            W = np.where(self.refuge[nbcell], W * f[:, None], W)
        return W

    def move(self, do_jump: bool = True):
        n = self.N
        self.n_jumps = 0
        self.n_refuge_rejected = 0
        self.n_zero_weight = 0
        if n == 0:
            return
        p = self.p
        x_pre, y_pre = self.x, self.y
        if p.async_move:
            self._move_async()
        else:
            A = K.accumulate_affinity_field(self._pa, self._pb, self._pF, self.x, self.y, n,
                                            p.grid_size, p.periodic)
            nbcell, valid = self._neighbor_cells()
            W = self.movement_weights(A, nbcell, valid)
            d = K.sample_rows(W, self.rng.random(n))
            self.n_zero_weight = int((W.sum(1) <= 0).sum())
            nest = self._nest_mask()
            if nest is not None:
                d[nest] = 4                      # crías en el nido: inmóviles
            self.x = self.x + K.DX[d]
            self.y = self.y + K.DY[d]
            if p.periodic:
                self.x %= p.grid_size
                self.y %= p.grid_size
        if p.jump_prob > 0 and do_jump:
            self.jump(x_pre, y_pre)
        if self.refuge is not None:
            if self._hard == "strict":
                self._enforce_refuge_capacity_strict(x_pre, y_pre)
            elif self._hard == "legacy":
                self._enforce_refuge_capacity(x_pre, y_pre)
        self.tot["refuge_rejected"] += self.n_refuge_rejected
        self.tot["zero_weight"] += self.n_zero_weight

    def _move_async(self):
        """Actualización asincrónica (async_move): orden aleatorio, uno por vez, con ocupación, afinidades y
        refugios llenos según las posiciones actuales. Las afinidades son exactas: se toman del almacén para
        todos los pares a distancia <= 2 al inicio del subpaso (los únicos que pueden ser vecinos durante el
        barrido). Consume permutation(N) + random(N)."""
        p = self.p
        n = self.N
        G = p.grid_size
        a, b = K.build_pairs_r2(self.x, self.y, G, p.periodic)
        lo, hi, keys = K.radix_sort_pairs(a, b, n, self.id)
        st = self.store
        F = np.zeros(keys.shape[0])
        if len(st) and keys.size:
            idx = np.minimum(np.searchsorted(st.keys, keys), len(st) - 1)
            found = st.keys[idx] == keys
            fi = idx[found]
            v = st.val[fi] * st._pow(self.t)[self.t - st.t0[fi]]
            v[v < st.eps] = 0.0
            F[found] = v
        keep = F > 0
        lo, hi, F = lo[keep], hi[keep], F[keep]
        src = np.concatenate([lo, hi])
        dst = np.concatenate([hi, lo])
        Fv = np.concatenate([F, F])
        o = np.argsort(src, kind="stable")
        dst, Fv = dst[o], Fv[o]
        ptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
        al = self.alpha_individual()
        al = np.full(n, float(al)) if np.isscalar(al) else np.asarray(al, np.float64)
        crowd = (p.crowd_aversion * (1.0 - self.s) ** p.crowd_power if p.crowd_aversion > 0
                 else np.zeros(n))
        nest = self._nest_mask()
        mobile = np.ones(n, bool) if nest is None else ~nest
        cell = self.x * G + self.y
        occ = np.bincount(cell, minlength=G * G).astype(np.int64)
        occ_c = np.bincount(cell[mobile], minlength=G * G).astype(np.int64)
        rm = self.male & self.is_reproductive()
        occm = np.bincount(cell[rm], minlength=G * G).astype(np.int64)
        has_ref = self.refuge is not None
        refuge = self.refuge if has_ref else np.zeros(G * G, bool)
        if has_ref and p.refuge_pref > 0:
            base = self.s if p.refuge_pref_basis == "current" else np.clip(self.h_mat, 0.0, 1.0)
            rfac = 1.0 + p.refuge_pref * (1.0 - base) ** p.refuge_pref_power
        else:
            rfac = np.ones(n)
        order = self.rng.permutation(n).astype(np.int64)
        u = self.rng.random(n)
        x = self.x.copy()
        y = self.y.copy()
        nz = K.async_move_kernel(order, u, x, y, G, p.periodic, ptr, dst, Fv, al, p.weight_form == "linear",
                                 bool(np.any(al < 0)), p.weight_floor, crowd, occ_c, p.territory_strength, rm,
                                 occm, refuge, p.refuge_capacity, has_ref, rfac, mobile, occ)
        self.n_zero_weight = int(nz)
        self.x, self.y = x, y

    def _enforce_refuge_capacity_strict(self, xp, yp):
        """Capacidad dura de verdad (refuge_hard='strict', v2). Los que no se movieron tienen
        prioridad. Los que entran se admiten en un orden sorteado una vez (permutación, mismo consumo de RNG
        que 'legacy') hasta completar la capacidad que dejan los residentes; los rechazados vuelven a su celda
        anterior y desde ese momento cuentan como residentes de ella, lo que puede desplazar a un admitido;
        se itera hasta que no hay rechazos (punto fijo; converge porque los rechazos son monótonos).
        Si al inicio del subpaso se cumple ocupación <= capacidad, también se cumple al final."""
        G = self.p.grid_size
        cap = self.p.refuge_capacity
        cell = self.x * G + self.y
        moved = (self.x != xp) | (self.y != yp)
        ent = np.flatnonzero(self.refuge[cell] & moved)
        self.n_refuge_rejected = 0
        if ent.size == 0:
            return
        ent = ent[self.rng.permutation(ent.size)]          # prioridad entre los que entran
        total = 0
        while True:
            cell = self.x * G + self.y
            inr = self.refuge[cell]
            moved = (self.x != xp) | (self.y != yp)
            cnt = np.bincount(cell[inr & ~moved], minlength=G * G)
            ent = ent[(inr & moved)[ent]]                  # conserva el orden sorteado
            if ent.size == 0:
                break
            c = cell[ent]
            o = np.argsort(c, kind="stable")
            cs = c[o]
            first = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
            rank = np.arange(cs.size) - np.repeat(first, np.diff(np.r_[first, cs.size]))
            rej = ent[o][rank >= (cap - cnt[cs])]
            if rej.size == 0:
                break
            self.x[rej] = xp[rej]
            self.y[rej] = yp[rej]
            total += int(rej.size)
        self.n_refuge_rejected = total

    def _enforce_refuge_capacity(self, xp, yp):
        """Capacidad "dura" de v1 (refuge_hard='legacy'): los que ya estaban tienen prioridad; los que entran,
        en orden aleatorio, hasta llenar la capacidad; los rechazados vuelven a su celda anterior. Una sola
        pasada: un rechazado que vuelve a un refugio del que salió no se cuenta, ni las crías.
        Equivalente vectorizado (mismo consumo de RNG) del prototipo rules_fis.RefugePrefHard."""
        G = self.p.grid_size
        cap = self.p.refuge_capacity
        cell = self.x * G + self.y
        inr = self.refuge[cell]
        moved = (self.x != xp) | (self.y != yp)
        cnt = np.bincount(cell[inr & ~moved], minlength=G * G)
        ent = np.flatnonzero(inr & moved)
        if ent.size == 0:
            return
        ent = ent[self.rng.permutation(ent.size)]
        c = cell[ent]
        o = np.argsort(c, kind="stable")            # agrupa por celda conservando el orden sorteado
        cs = c[o]
        first = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
        rank = np.arange(cs.size) - np.repeat(first, np.diff(np.r_[first, cs.size]))
        rej = ent[o][rank >= (cap - cnt[cs])]
        if rej.size:
            self.x[rej] = xp[rej]
            self.y[rej] = yp[rej]
        self.n_refuge_rejected = int(rej.size)

    def _pairs_current(self):
        """Pares vecinos (Chebyshev <= 1) en las posiciones actuales con su afinidad vigente (fin de t),
        sin reforzar. Se usa en los subpasos de movimiento 2..k."""
        p = self.p
        a, b, _, _ = K.build_pairs(self.x, self.y, p.grid_size, p.periodic)
        lo, hi, keys = K.radix_sort_pairs(a, b, self.N, self.id)
        st = self.store
        F = np.zeros(keys.shape[0])
        if len(st) and keys.size:
            idx = np.minimum(np.searchsorted(st.keys, keys), len(st) - 1)
            found = st.keys[idx] == keys
            fi = idx[found]
            v = st.val[fi] * st._pow(self.t)[self.t - st.t0[fi]]
            v[v < st.eps] = 0.0
            F[found] = v
        self._pa, self._pb, self._pF = lo, hi, F

    def jump(self, x_pre, y_pre):
        """R3: saltos de largo alcance tras el paso local (destinos según posiciones pre-movimiento)."""
        p = self.p
        n = self.N
        G = p.grid_size
        J = np.flatnonzero(self.rng.random(n) < p.jump_prob)
        nest = self._nest_mask()
        if nest is not None:
            J = J[~nest[J]]                     # las crías en el nido no saltan
        self.n_jumps = int(J.size)
        if J.size == 0:
            return
        if p.jump_mode == "uniform":
            c = self.rng.integers(G * G, size=J.size)
            self.x[J] = c // G
            self.y[J] = c % G
            return
        if n < 2:
            return
        j = self.rng.integers(n - 1, size=J.size)
        j += (j >= J)                       # uniforme entre los otros
        if p.jump_mode == "affinity":
            jaff = self._affinity_partners(J)
            j = np.where(jaff >= 0, jaff, j)
        self.x[J] = x_pre[j]
        self.y[J] = y_pre[j]

    def _store_pairs_idx(self, t):
        """Pares vigentes del almacén como (índice_lo, índice_hi, F) de agentes vivos."""
        st = self.store
        if len(st) == 0 or self.N == 0:
            e = np.empty(0, np.int64)
            return e, e, np.empty(0)
        eff = st.effective(t)
        lo = st.keys >> 32
        hi = st.keys & _MASK32
        ok = (eff >= st.eps) & self._alive_id[lo] & self._alive_id[hi]
        lo, hi, eff = lo[ok], hi[ok], eff[ok]
        ilo = np.searchsorted(self.id, lo)
        ihi = np.searchsorted(self.id, hi)
        ilo = np.minimum(ilo, self.N - 1)
        ihi = np.minimum(ihi, self.N - 1)
        ok = (self.id[ilo] == lo) & (self.id[ihi] == hi)
        return ilo[ok], ihi[ok], eff[ok]

    def _affinity_partners(self, J):
        """Para cada i en J, un conocido j con prob. ∝ F_ij (-1 si no tiene). Consume len(J) uniformes."""
        u = self.rng.random(J.size)
        out = np.full(J.size, -1, np.int64)
        ilo, ihi, F = self._store_pairs_idx(self.t)
        if F.size == 0:
            return out
        pos = np.full(self.N, -1, np.int64)
        pos[J] = np.arange(J.size)
        a = pos[ilo] >= 0
        b = pos[ihi] >= 0
        owner = np.concatenate([pos[ilo[a]], pos[ihi[b]]])
        other = np.concatenate([ihi[a], ilo[b]])
        w = np.concatenate([F[a], F[b]])
        if owner.size == 0:
            return out
        o = np.argsort(owner, kind="stable")
        owner, other, w = owner[o], other[o], w[o]
        C = np.cumsum(w)
        cnt = np.bincount(owner, minlength=J.size)
        end = np.cumsum(cnt)
        start = end - cnt
        has = cnt > 0
        base = np.where(start > 0, C[np.maximum(start - 1, 0)], 0.0)
        tot = C[np.maximum(end - 1, 0)] - base
        k = np.searchsorted(C, base + u * tot, side="right")
        k = np.clip(k, start, np.maximum(end - 1, 0))
        out[has] = other[k[has]]
        return out

    # ------------------------------------------------------------------ afinidades
    def update_affinities(self):
        p = self.p
        a, b, cell, occ = K.build_pairs(self.x, self.y, p.grid_size, p.periodic)
        lo, hi, keys = K.radix_sort_pairs(a, b, self.N, self.id)
        F = self.store.reinforce(keys, self.t, self._alive_id)
        self._pa = lo
        self._pb = hi
        self._pF = F
        self.n_active_pairs = keys.shape[0]
        return cell, occ

    # ------------------------------------------------------------------ estado social
    def _moore_sum(self, A2):
        """Suma sobre el vecindario de Moore (incluida la celda) de un campo G x G."""
        if self.p.periodic:
            out = np.zeros_like(A2)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    out += np.roll(np.roll(A2, dx, 0), dy, 1)
            return out
        P = np.pad(A2, 1)
        G = A2.shape[0]
        out = np.zeros_like(A2)
        for dx in range(3):
            for dy in range(3):
                out += P[dx:dx + G, dy:dy + G]
        return out

    def social_learning_sbar(self) -> np.ndarray:
        """R2: s medio de los modelos sociales de cada i (vecinos de Moore o conocidos)."""
        p = self.p
        n = self.N
        s = self.s
        if p.social_rec_source == "moore":
            G = p.grid_size
            cell = self._cell
            Sc = np.bincount(cell, weights=s, minlength=G * G).reshape(G, G)
            Mc = np.bincount(cell, minlength=G * G).reshape(G, G).astype(np.float64)
            SV = self._moore_sum(Sc).ravel()
            MV = self._moore_sum(Mc).ravel()
            num = SV[cell] - s
            den = MV[cell] - 1.0
        else:
            ilo, ihi, F = self._store_pairs_idx(self.t)
            num = np.bincount(ilo, weights=F * s[ihi], minlength=n) + \
                np.bincount(ihi, weights=F * s[ilo], minlength=n)
            den = np.bincount(ilo, weights=F, minlength=n) + np.bincount(ihi, weights=F, minlength=n)
        iso = s if p.social_rec_isolated == "self" else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            sbar = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), iso)
        return np.clip(sbar, 0.0, 1.0)

    def dominant_males(self, cell):
        """R5: máscara de machos adultos y de los dominantes de su celda (mayor s; empates al azar)."""
        p = self.p
        am = np.flatnonzero(self.male & (self.age >= p.repro_age_min))
        dom = np.zeros(self.N, bool)
        if am.size:
            u = self.rng.random(am.size)
            o = np.lexsort((u, -self.s[am], cell[am]))
            sc = cell[am][o]
            first = np.ones(o.size, bool)
            first[1:] = sc[1:] != sc[:-1]
            dom[am[o[first]]] = True
        adult_m = np.zeros(self.N, bool)
        adult_m[am] = True
        return adult_m, dom

    def social_delta(self, m: np.ndarray) -> np.ndarray:
        """ds_i dado m_i (ocupación post-movimiento). Puntos de extensión (2), R2, R2', R5."""
        p = self.p
        rec = np.full(m.shape[0], p.rec)
        det = p.gamma * np.maximum(0, m - p.m_c)
        if p.social_rec_lambda > 0:
            rec = rec * ((1.0 - p.social_rec_lambda) + p.social_rec_lambda * self.social_learning_sbar())
        if p.rec_threshold > 0:
            rec = np.where(self.s >= p.rec_threshold, rec, 0.0)
        self.n_defeated = 0
        if p.defeat_gamma > 0:
            adult_m, dom = self.dominant_males(self._cell)
            lose = adult_m & ~dom
            self.n_defeated = int(lose.sum())
            det = det + p.defeat_gamma * lose
        if (p.plast_outside_rec != 1.0 or p.plast_outside_det != 1.0) and \
                (p.plast_age_min > 0 or np.isfinite(p.plast_age_max)):
            inwin = (self.age >= p.plast_age_min) & (self.age <= p.plast_age_max)
            rec = np.where(inwin, rec, rec * p.plast_outside_rec)
            det = np.where(inwin, det, det * p.plast_outside_det)
        return rec - det

    def update_social_state(self, m: np.ndarray):
        self.s = np.clip(self.s + self.social_delta(m), 0.0, 1.0)
        a_w = self.p.competence_age
        if a_w is not None:
            fr = (self.age >= a_w) & np.isinf(self.cap)
            self.cap[fr] = self.s[fr]
            np.minimum(self.s, self.cap, out=self.s)

    # ------------------------------------------------------------------ reproducción
    def choose_fathers(self, fem, fc, mal_s, start, cnt):
        """Padre para cada hembra fem (en celda fc). mal_s: machos reproductivos ordenados por celda."""
        if self.p.mating_rule == "random":
            k = cnt[fc]
            pick = start[fc] + (self.rng.random(fem.shape[0]) * k).astype(np.int64)
            pick = np.minimum(pick, start[fc] + k - 1)
            return mal_s[pick]
        # dominante: macho de mayor s en la celda (mal_s ordenado por celda y -s)
        return mal_s[start[fc]]

    def _fathers_moore(self, fem, cnt, start, mal_s):
        """mate_radius=1: padre entre los machos reproductivos del vecindario de Moore de la hembra
        (uniforme, o el de mayor s con mating_rule='dominant')."""
        G = self.p.grid_size
        nx = self.x[fem][:, None] + K.DX[None, :]
        ny = self.y[fem][:, None] + K.DY[None, :]
        if self.p.periodic:
            valid = np.ones(nx.shape, bool)
            nx %= G
            ny %= G
        else:
            valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
            nx = np.clip(nx, 0, G - 1)
            ny = np.clip(ny, 0, G - 1)
        nb = nx * G + ny
        C = np.where(valid, cnt[nb], 0)
        tot = C.sum(1)
        has = tot > 0
        fem, nb, C, tot = fem[has], nb[has], C[has], tot[has]
        if fem.size == 0:
            return fem, fem
        if self.p.mating_rule == "random":
            u = (self.rng.random(fem.size) * tot).astype(np.int64)
            u = np.minimum(u, tot - 1)
            cc = np.cumsum(C, 1)
            d = (cc <= u[:, None]).sum(1)
            prev = np.where(d > 0, cc[np.arange(fem.size), np.maximum(d - 1, 0)], 0)
            k = u - np.where(d > 0, prev, 0)
            ch = nb[np.arange(fem.size), d]
            return fem, mal_s[start[ch] + k]
        # dominante: mal_s ordenado por celda y -s -> el primero de cada celda; máximo sobre las 9
        cand = np.where(C > 0, mal_s[np.minimum(start[nb], mal_s.size - 1)], -1)
        sc = np.where(cand >= 0, self.s[np.maximum(cand, 0)], -np.inf)
        return fem, cand[np.arange(fem.size), np.argmax(sc, 1)]

    def newborn_s(self, mothers: np.ndarray) -> np.ndarray:
        p = self.p
        return p.inherit_weight * self.s[mothers] + (1.0 - p.inherit_weight) * p.s_newborn_base

    def reproduce(self, cell: np.ndarray):
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
        if p.mate_radius == 0:
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
        self.s = np.concatenate([self.s, self.newborn_s(mrep)])
        self.refr = np.concatenate([self.refr, np.zeros(nb, np.int64)])
        self._m = np.concatenate([self._m, self._m[mrep]])
        self._cell = np.concatenate([self._cell, nbcell])
        self.cap = np.concatenate([self.cap, np.full(nb, np.inf)])
        self.iso_run = np.concatenate([self.iso_run, np.zeros(nb, np.int64)])
        self.conceived = np.concatenate([self.conceived, np.zeros(nb, bool)])
        self.h_mat = np.concatenate([self.h_mat, self.s[mrep]])
        self.birth_t = np.concatenate([self.birth_t, np.full(nb, self.t, np.int64)])
        nf_new = int((~self.male[-nb:]).sum())
        if nf_new:
            self.coh_f_born[self.t] = self.coh_f_born.get(self.t, 0) + nf_new

    def _place_newborns(self, mrep, nbx, nby, nbcell):
        """refuge_hard='strict': las crías que dejarían un refugio por encima de la capacidad (contando a
        todos los presentes) se ubican en una celda vecina no refugio, la misma para todas las crías
        excedentes de una madre. Entre las crías de un mismo refugio, cuáles se quedan se sortea."""
        G = self.p.grid_size
        cap = self.p.refuge_capacity
        bc = nbx * G + nby
        inr = self.refuge[bc]
        if not inr.any():
            return nbx, nby, nbcell
        occ0 = np.bincount(self.x * G + self.y, minlength=G * G)
        idx = np.flatnonzero(inr)
        idx = idx[self.rng.permutation(idx.size)]
        c = bc[idx]
        o = np.argsort(c, kind="stable")
        cs = c[o]
        first = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
        rank = np.arange(cs.size) - np.repeat(first, np.diff(np.r_[first, cs.size]))
        over = np.sort(idx[o][rank >= (cap - occ0[cs])])
        if over.size == 0:
            return nbx, nby, nbcell
        mo, inv = np.unique(mrep[over], return_inverse=True)
        dest = self._cells_out_of_refuge(self.x[mo] * G + self.y[mo])[inv]
        ok = dest >= 0
        self.n_capacity_unresolved += int((~ok).sum())
        nbx, nby, nbcell = nbx.copy(), nby.copy(), nbcell.copy()
        nbx[over[ok]] = dest[ok] // G
        nby[over[ok]] = dest[ok] % G
        nbcell[over[ok]] = dest[ok]
        self.n_births_relocated = int(ok.sum())
        return nbx, nby, nbcell

    # ------------------------------------------------------------------ mortalidad
    def death_probability(self) -> np.ndarray:
        """Punto de extensión (4)."""
        p = self.p
        mu = p.social_mortality_factor
        if mu == 0:
            pa = 1.0 / (1.0 + np.exp(-p.k_mort * (self.age - p.a_max)))
        elif p.social_mortality_form == "notebook":
            # forma original (signo invertido para a < a_max); sólo para comparación
            k = p.k_mort * (1.0 + mu * (1.0 - self.s))
            pa = 1.0 / (1.0 + np.exp(-k * (self.age - p.a_max)))
        else:
            a_eff = p.a_max * (1.0 - mu * (1.0 - self.s))
            pa = 1.0 / (1.0 + np.exp(-p.k_mort * (self.age - a_eff)))
        if p.social_hazard > 0:
            hs = np.clip(p.social_hazard * (1.0 - self.s) ** p.social_hazard_power, 0.0, 1.0)
            pa = 1.0 - (1.0 - pa) * (1.0 - hs)
        if p.mu_bg > 0:
            pa = 1.0 - (1.0 - pa) * (1.0 - p.mu_bg)
        return pa

    def _age_only_death_probability(self):
        """Probabilidad de muerte no social: edad y, si mu_bg > 0, mortalidad de fondo (atribución de A4)."""
        p = self.p
        pa = 1.0 / (1.0 + np.exp(-p.k_mort * (self.age - p.a_max)))
        if p.mu_bg > 0:
            pa = 1.0 - (1.0 - pa) * (1.0 - p.mu_bg)
        return pa

    def adult_deaths(self):
        n = self.N
        if n == 0:
            self.deaths = 0
            return
        pd_ = self.death_probability()
        die = self.rng.random(n) < pd_
        if self.p.social_mortality_factor != 0 or self.p.social_hazard > 0:
            # atribución: sum sobre los muertos de p_social/p_total, p_social = p_total - p_edad
            pt = pd_[die]
            pa = self._age_only_death_probability()[die]
            self.deaths_social_exp = float(np.sum(np.clip(pt - pa, 0, None) / np.maximum(pt, 1e-300)))
        else:
            self.deaths_social_exp = 0.0
        nd = int(die.sum())
        self.deaths = nd
        if nd == 0:
            return
        self._alive_id[self.id[die]] = False
        keep = ~die
        newidx = np.cumsum(keep) - 1
        newidx[die] = -1
        for name in self.AGENT_FIELDS:
            setattr(self, name, getattr(self, name)[keep])
        if self._pa.size:
            self._pa, self._pb, self._pF = K.remap_pairs(self._pa, self._pb, self._pF, newidx)

    # ------------------------------------------------------------------ paso
    def step(self):
        k = self.p.move_substeps
        for sub in range(k):
            if sub > 0 and not self.p.async_move:
                self._pairs_current()
            self.move(do_jump=(sub == k - 1))
        self.t += 1
        cell, occ = self.update_affinities()
        nest = self._nest_mask()
        if nest is not None and nest.any():
            # las crías en el nido no cuentan para el hacinamiento de nadie; la propia es la de los no-crías + 1
            occ_e = np.bincount(cell[~nest], minlength=self.p.grid_size ** 2)
            self._m = occ_e[cell] + nest
        else:
            self._m = occ[cell]
        self._cell = cell
        self.update_social_state(self._m)
        self._measure_post_move()
        self.reproduce(cell)
        self.tot["births_relocated"] += self.n_births_relocated
        self.age += 1
        if self.p.refractory_weeks > 0:
            np.maximum(self.refr - 1, 0, out=self.refr)
        self.adult_deaths()
        self.n_aff, self.mean_aff = self.store.count(self.t, self._alive_id)

    def _measure_post_move(self):
        """Observables de la calibración medidos tras moverse y actualizar s (antes de nacer/morir)."""
        p = self.p
        m = self._m
        n = m.shape[0]
        self.iso_run = np.where(m <= p.obs_iso_m, self.iso_run + 1, 0)
        o = {}
        keys = ("mstar", "f_over", "corr_s_m", "frac_withdrawn", "frac_crowded_low", "phi_BO")
        if n:
            o["mstar"] = float(m.mean() - 1.0)            # crowding de Lloyd (experimentado)
            o["f_over"] = float((m > p.m_c).mean())
            adult = self.age >= p.repro_age_min
            na = int(adult.sum())
            low = adult & (self.s < p.obs_s_low)
            if na:
                o["frac_withdrawn"] = float((low & (m <= p.obs_iso_m)).sum() / na)
                o["frac_crowded_low"] = float((low & (m > p.m_c)).sum() / na)
                o["phi_BO"] = float((low & (self.iso_run >= p.obs_iso_weeks)).sum() / na)
                for k in self.CROWD_FIXED_M:          # clase hacinada con umbral fijo (v2)
                    o[f"frac_crowded_low_m{k}"] = float((low & (m > k)).sum() / na)
            else:
                o["frac_withdrawn"] = o["frac_crowded_low"] = o["phi_BO"] = np.nan
                for k in self.CROWD_FIXED_M:
                    o[f"frac_crowded_low_m{k}"] = np.nan
            if na >= 10:
                sa, ma = self.s[adult], m[adult].astype(float)
                o["corr_s_m"] = float(np.corrcoef(sa, ma)[0, 1]) if sa.std() > 0 and ma.std() > 0 else np.nan
            else:
                o["corr_s_m"] = np.nan
        else:
            o = {k: np.nan for k in keys}
            for k in self.CROWD_FIXED_M:
                o[f"frac_crowded_low_m{k}"] = np.nan
        # autocorrelación lag-1 de m_i por agente (para calibrar tau_F): corr(m_i(t), m_i(t-1))
        o["m_autocorr1"] = np.nan
        pid, pm = getattr(self, "_prev_ids", None), getattr(self, "_prev_m", None)
        if pid is not None and pid.size and n:
            j = np.minimum(np.searchsorted(pid, self.id), pid.size - 1)
            ok = pid[j] == self.id
            if ok.sum() >= 10:
                a, b = m[ok].astype(float), pm[j[ok]].astype(float)
                if a.std() > 0 and b.std() > 0:
                    o["m_autocorr1"] = float(np.corrcoef(a, b)[0, 1])
        self._prev_ids, self._prev_m = self.id.copy(), m.copy()
        self.obs = o

    O13B_OFFSETS = (0, 10, 20)
    CROWD_FIXED_M = (6, 8, 10, 12)   # umbrales fijos m > k de la clase hacinada (columnas frac_crowded_low_m{k})

    def _o13b_classes(self):
        p = self.p
        adult = self.age >= p.repro_age_min
        low = adult & (self.s < p.obs_s_low)
        iso = low & (self.iso_run >= p.obs_iso_weeks)
        cr = low & (self._m > p.m_c) if self._m.shape[0] == self.N else np.zeros(self.N, bool)
        return self.birth_t[iso].copy(), self.birth_t[cr].copy()

    def _track_o13b(self):
        """Snapshots de birth_t de las clases ISO y CR en t_pk, t_pk+10 y t_pk+20 (t_pk corriente)."""
        st = self._o13b
        if self.N > st["Nmax"]:
            st.update(Nmax=self.N, tpk=self.t, snaps={})
        off = self.t - st["tpk"]
        if off in self.O13B_OFFSETS and self.N:
            st["snaps"][off] = self._o13b_classes()

    def o13b_summary(self) -> dict:
        """Valores de O13b (criterio O13b) a partir de los snapshots."""
        from scipy.stats import mannwhitneyu
        st = getattr(self, "_o13b", None)
        out = dict(o13b_t_star=np.nan, o13b_n_iso=0, o13b_n_cr=0, o13b_med_bt_iso=np.nan,
                   o13b_med_bt_cr=np.nan, o13b_delta_b=np.nan, o13b_p_mw=np.nan, o13b_f_late=np.nan)
        if not st or st["tpk"] <= 0:
            return out
        order = [10, 0, 20]
        pick = None
        for off in order:
            if off in st["snaps"]:
                a, b = st["snaps"][off]
                if a.size >= 20 and b.size >= 20:
                    pick = off
                    break
        if pick is None:
            snap = st["snaps"].get(10) or st["snaps"].get(0)
            if snap is not None:
                out.update(o13b_n_iso=int(snap[0].size), o13b_n_cr=int(snap[1].size))
            return out
        a, b = st["snaps"][pick]
        tpk = st["tpk"]
        out.update(o13b_t_star=float(tpk + pick), o13b_n_iso=int(a.size), o13b_n_cr=int(b.size),
                   o13b_med_bt_iso=float(np.median(a)), o13b_med_bt_cr=float(np.median(b)),
                   o13b_iqr_bt_iso=float(np.subtract(*np.percentile(a, [75, 25]))),
                   o13b_iqr_bt_cr=float(np.subtract(*np.percentile(b, [75, 25]))),
                   o13b_delta_b=float((np.median(a) - np.median(b)) / tpk),
                   o13b_p_mw=float(mannwhitneyu(a, b, alternative="greater").pvalue),
                   o13b_f_late=float((a >= 0.8 * tpk).mean()))
        return out

    def extra_stats(self) -> dict:
        """Punto de extensión: columnas adicionales para la serie temporal."""
        return {}

    # ------------------------------------------------------------------ corrida
    def run(self, T: int | None = None, record_every: int = 1, verbose: bool = False, callback=None):
        """Corre hasta T o extinción. callback(U) se llama después de cada paso (p.ej. snapshots)."""
        from .stats import Recorder, summarize
        T = self.p.T if T is None else T
        rec = Recorder()
        self.n_aff, self.mean_aff, self.n_active_pairs = 0, 0.0, 0
        rec.record(self)
        t0 = _time.perf_counter()
        exploded = False
        self._o13b = dict(Nmax=self.N, tpk=0, snaps={})
        while self.t < T and self.N > 0:
            self.step()
            self._track_o13b()
            if callback is not None:
                callback(self)
            if self.N > self.p.max_pop:
                exploded = True
                rec.record(self)
                break
            if self.N == 0 or self.t % record_every == 0 or self.t == T:
                rec.record(self)
            if verbose and self.t % 50 == 0:
                print(f"t={self.t} N={self.N} S={self.s.mean() if self.N else 0:.3f} aff={self.n_aff}")
        wall = _time.perf_counter() - t0
        series = rec.to_frame()
        # cohortes (O11b): por semana de nacimiento t, hembras nacidas y cuántas concibieron (edad <= 48)
        tt = series["t"].to_numpy()
        series["coh_f_born"] = [self.coh_f_born.get(int(x), 0) for x in tt]
        series["coh_f_ever_conceived"] = [self.coh_f_conc.get(int(x), 0) for x in tt]
        summary = summarize(series, T, self)
        summary.update(self.o13b_summary())
        summary.update(n_refuge_rejected_tot=self.tot["refuge_rejected"],
                       n_births_relocated_tot=self.tot["births_relocated"],
                       n_zero_weight_tot=self.tot["zero_weight"],
                       n_capacity_unresolved=self.n_capacity_unresolved)
        summary["exploded"] = exploded
        summary["wall_time_s"] = wall
        return series, summary


def simulate(params: Params | None = None, T: int | None = None, seed=None,
             model_cls=Universe, record_every: int = 1, verbose: bool = False, callback=None):
    """Atajo: corre una realización y devuelve (series DataFrame, summary dict)."""
    p = params if params is not None else Params()
    rng = np.random.default_rng(seed if seed is not None else p.seed)
    U = model_cls(p, rng=rng)
    return U.run(T, record_every=record_every, verbose=verbose, callback=callback)
