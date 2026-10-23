"""ABM bien mezclado (jump_prob=1, jump_mode="uniform") con el paquete u25.

Clase WMUniverse: subclase de u25.model.Universe que omite el almacén de afinidades y el movimiento local.
Con saltos uniformes para todos cada semana, la posición final es uniforme e independiente de la posición
previa y de las afinidades (el paso local se sobrescribe; con refugios los rechazados vuelven a la celda
PREVIA al movimiento, que tampoco depende del paso local). Para conservar el consumo del RNG
(y por lo tanto trayectorias idénticas bit a bit a las de Universe) se consume rng.random(n) como haría
sample_rows. Validación: check_identical() compara con Universe para la misma semilla.

También mide θ(t): para los juveniles en la ventana de plasticidad (edad en [plast_age_min, plast_age_max]
al momento de la actualización de s), la fracción de los OTROS individuos de su bloque 3x3 con s>0 (s de la
semana anterior, el que usa R2), y la media s̄_V. Se puede usar también con el Universe espacial
(clase ThetaUniverse).
"""
from __future__ import annotations
import numpy as np
from u25.model import Universe
from u25.params import preset, Params
from u25 import kernels as K

A_W = 12  # ventana de C1 (plast_age_max); para SL se usa la misma ventana de edades como convención


def theta_obs(U, a_lo=1, a_hi=12, eps=0.0):
    """θ y s̄_V de los juveniles (edad a_lo..a_hi) con los s actuales (pre-actualización) y _cell actual."""
    p = U.p
    G = p.grid_size
    cell = U._cell
    s = U.s
    pos = (s > eps).astype(np.float64)
    Sc = np.bincount(cell, weights=s, minlength=G * G).reshape(G, G)
    Pc = np.bincount(cell, weights=pos, minlength=G * G).reshape(G, G)
    Mc = np.bincount(cell, minlength=G * G).reshape(G, G).astype(np.float64)
    SV = U._moore_sum(Sc).ravel()[cell] - s
    PV = U._moore_sum(Pc).ravel()[cell] - pos
    MV = U._moore_sum(Mc).ravel()[cell] - 1.0
    juv = (U.age >= a_lo) & (U.age <= a_hi)
    ok = juv & (MV > 0.5)
    out = dict(n_juv=int(juv.sum()), n_juv_nb=int(ok.sum()))
    if ok.any():
        out["theta"] = float(np.mean(PV[ok] / MV[ok]))            # media de fracciones individuales
        out["theta_ratio"] = float(PV[ok].sum() / MV[ok].sum())    # fracción agregada
        out["sbarV"] = float(np.mean(SV[ok] / MV[ok]))
        out["sbarV0"] = float(np.sum(SV[ok] / MV[ok]) / max(juv.sum(), 1))  # con 0 para aislados (lo que usa R2)
        out["K_mean"] = float(MV[ok].mean())
        out["frac_iso"] = float(1 - ok.sum() / max(juv.sum(), 1))
    else:
        out.update(theta=np.nan, theta_ratio=np.nan, sbarV=np.nan, sbarV0=np.nan, K_mean=np.nan,
                   frac_iso=np.nan if not juv.any() else 1.0)
    # población entera
    n = s.size
    out["frac_pos_all"] = float(pos.mean()) if n else np.nan
    # Φ: s > 0, o s = 0 con edad <= a_w (juvenil latente)
    out["n_phi"] = int(((s > eps) | ((U.age >= a_lo) & (U.age <= a_hi))).sum()) if n else 0
    out["S_mean"] = float(s.mean()) if n else np.nan
    out["max_s"] = float(s.max()) if n else np.nan
    juvs = s[juv]
    out["max_s_juv"] = float(juvs.max()) if juvs.size else np.nan
    return out


class _ThetaMixin:
    """Mide θ justo antes de la actualización de s (social_delta usa self.s y self._cell actuales)."""
    theta_log: list
    theta_every = 1

    def update_social_state(self, m):
        if not hasattr(self, "theta_log"):
            self.theta_log = []
        if self.t % self.theta_every == 0:
            a_hi = self.p.plast_age_max if np.isfinite(self.p.plast_age_max) else A_W
            d = theta_obs(self, 1, int(a_hi))
            d["t"] = self.t
            d["N"] = self.N
            self.theta_log.append(d)
        return super().update_social_state(m)


class ThetaUniverse(_ThetaMixin, Universe):
    pass


class WMUniverse(_ThetaMixin, Universe):
    """Variante bien mezclada exacta en ley (y bit a bit) sin el costo de las afinidades."""
    measure_theta = True

    def __init__(self, params, rng=None):
        assert params.jump_prob == 1.0 and params.jump_mode == "uniform" and params.move_substeps == 1
        assert not params.async_move and params.nest_period == 0
        super().__init__(params, rng=rng)

    def move(self, do_jump=True):
        n = self.N
        self.n_jumps = 0
        self.n_refuge_rejected = 0
        self.n_zero_weight = 0
        if n == 0:
            return
        x_pre, y_pre = self.x, self.y
        self.rng.random(n)            # consumo de sample_rows (paso local, sobrescrito por el salto)
        self.x = self.x.copy()
        self.y = self.y.copy()
        self.jump(x_pre, y_pre)
        if self.refuge is not None:
            if self._hard == "strict":
                self._enforce_refuge_capacity_strict(x_pre, y_pre)
            elif self._hard == "legacy":
                self._enforce_refuge_capacity(x_pre, y_pre)
        self.tot["refuge_rejected"] += self.n_refuge_rejected

    def update_affinities(self):
        G = self.p.grid_size
        cell = self.x * G + self.y
        occ = np.bincount(cell, minlength=G * G)
        self.n_active_pairs = 0
        return cell, occ

    def update_social_state(self, m):
        if self.measure_theta:
            return super().update_social_state(m)
        return Universe.update_social_state(self, m)


# ----------------------------------------------------------------------------- configuraciones
SL_BASE = dict()   # control corta vida = defaults del paquete (notebook, α = 1.4, a_max = 60, a_rep = 50)


def make_params(model: str, L: int, refuges: bool, wm: bool = True, T: int = 600, cap_per_cell: float = None,
                **over) -> Params:
    """model in {"C1", "SL"}. Densidad inicial fija 400/900 por celda; max_pop escalado con L^2."""
    n_half = int(round(200 * L * L / 900))
    if model == "C1":
        base = dict(preset("calhoun_C1").to_dict())
        if not refuges:
            base.update(refuge_frac=0.0, refuge_pref=0.0, refuge_hard=False)
        cpc = 15000 / 900 if cap_per_cell is None else cap_per_cell
    elif model == "SL":
        base = Params().to_dict()
        if refuges:   # mismos refugios de C1 (capacidad dura, R7) sobre el control
            base.update(refuge_frac=0.3, refuge_capacity=2, refuge_seed=0, refuge_pref=100.0, refuge_hard="strict")
        cpc = 60000 / 900 if cap_per_cell is None else cap_per_cell
    else:
        raise KeyError(model)
    base.update(grid_size=L, n_males=n_half, n_females=n_half, T=T, max_pop=int(cpc * L * L))
    if wm:
        base.update(jump_prob=1.0, jump_mode="uniform")
    base.update(over)
    return Params.from_dict(base)


def check_identical(model="C1", L=20, refuges=True, T=120, seed=3):
    p = make_params(model, L, refuges, T=T)
    a = Universe(p, rng=np.random.default_rng(seed)); sa, _ = a.run()
    b = WMUniverse(p, rng=np.random.default_rng(seed)); sb, _ = b.run()
    na, nb = sa["N"].to_numpy(), sb["N"].to_numpy()
    same = na.shape == nb.shape and np.array_equal(na, nb) and np.allclose(a.s, b.s) if a.N == b.N else False
    return same, na[-5:], nb[-5:]


if __name__ == "__main__":
    import time
    for model, ref in (("C1", True), ("C1", False), ("SL", False), ("SL", True)):
        t0 = time.time()
        print(model, ref, check_identical(model, 20, ref, T=150), f"{time.time()-t0:.1f}s", flush=True)
