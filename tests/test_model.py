import numpy as np
import pandas as pd
import pytest

from u25 import Params, Universe, simulate, run_ensemble
from u25 import kernels as K
from u25.reference import notebook_update_affinities, ReferenceUniverse

SMALL = dict(grid_size=12, n_males=30, n_females=30)


# ---------------------------------------------------------------- reproducibilidad
def test_reproducible_same_seed():
    p = Params(**SMALL)
    s1, m1 = simulate(p, T=60, seed=7)
    s2, m2 = simulate(p, T=60, seed=7)
    pd.testing.assert_frame_equal(s1, s2)
    m1.pop("wall_time_s"), m2.pop("wall_time_s")
    assert m1 == m2 or all((a == b) or (np.isnan(a) and np.isnan(b)) for a, b in zip(m1.values(), m2.values()))


def test_different_seeds_differ():
    p = Params(**SMALL)
    s1, _ = simulate(p, T=40, seed=1)
    s2, _ = simulate(p, T=40, seed=2)
    assert not s1["N"].equals(s2["N"])


# ---------------------------------------------------------------- casos límite
def test_no_females_goes_extinct_without_births():
    p = Params(n_males=50, n_females=0)
    s, m = simulate(p, T=400, seed=3)
    assert s["births"].sum() == 0
    assert (np.diff(s["N"]) <= 0).all()
    assert m["extinct"] and not m["censored"]
    assert np.isfinite(m["t_ext"])


def test_p0_zero_no_births():
    s, m = simulate(Params(p0=0.0, **SMALL), T=50, seed=1)
    assert s["births"].sum() == 0


class _SocialMonitor(Universe):
    min_ds = np.inf

    def update_social_state(self, m):
        old = self.s.copy()
        super().update_social_state(m)
        assert np.all(self.s >= 0) and np.all(self.s <= 1)
        _SocialMonitor.min_ds = min(_SocialMonitor.min_ds, float(np.min(self.s - old)) if old.size else 0)


def test_gamma_zero_social_never_decreases():
    _SocialMonitor.min_ds = np.inf
    p = Params(gamma=0.0, initial_s=0.3, **SMALL)
    U = _SocialMonitor(p, rng=np.random.default_rng(5))
    U.run(35)  # gamma=0 => crecimiento explosivo; T corto
    assert _SocialMonitor.min_ds >= 0.0


def test_social_bounds_and_positions_in_grid():
    p = Params(grid_size=8, n_males=60, n_females=60)
    U = Universe(p, rng=np.random.default_rng(0))
    for _ in range(40):
        U.step()
        if U.N == 0:
            break
        assert U.x.min() >= 0 and U.x.max() < 8 and U.y.min() >= 0 and U.y.max() < 8
        assert U.s.min() >= 0 and U.s.max() <= 1
        assert np.all(np.diff(U.id) > 0)  # invariante usado por radix_sort_pairs


def test_alpha_zero_uniform_moves_interior():
    p = Params(alpha=0.0, grid_size=11, n_males=1, n_females=0)
    U = Universe(p, rng=np.random.default_rng(0))
    U.x[:] = 5
    U.y[:] = 5
    nb, valid = U._neighbor_cells()
    W = U.movement_weights(np.zeros((1, 9)), nb, valid)
    W = np.repeat(W, 90000, axis=0)
    d = K.sample_rows(W, np.random.default_rng(1).random(W.shape[0]))
    counts = np.bincount(d, minlength=9)
    exp = W.shape[0] / 9
    chi2 = ((counts - exp) ** 2 / exp).sum()
    assert chi2 < 30  # 8 g.l.; p ~ 2e-4


def test_corner_never_leaves_grid():
    p = Params(grid_size=5, n_males=1, n_females=0)
    U = Universe(p, rng=np.random.default_rng(0))
    U.x[:] = 0
    U.y[:] = 0
    nb, valid = U._neighbor_cells()
    W = np.repeat(U.movement_weights(np.zeros((1, 9)), nb, valid), 5000, axis=0)
    d = K.sample_rows(W, np.random.default_rng(1).random(5000))
    assert set(np.unique(d)) == {4, 5, 7, 8}


# ---------------------------------------------------------------- afinidades: exactitud vs notebook
class _Shadow(Universe):
    """Mantiene en paralelo el diccionario del notebook sobre la misma trayectoria."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shadow = {}
        self.max_err_A = 0.0
        self.max_err_F = 0.0

    def movement_weights(self, A, nbcell, valid):
        # campo A por fuerza bruta desde el diccionario del notebook
        G = self.p.grid_size
        occ = {}
        for i in range(self.N):
            occ.setdefault((int(self.x[i]), int(self.y[i])), []).append(i)
        Ab = np.zeros_like(A)
        for i in range(self.N):
            for d in range(9):
                c = (int(self.x[i] + K.DX[d]), int(self.y[i] + K.DY[d]))
                for j in occ.get(c, []):
                    if j == i:
                        continue
                    key = tuple(sorted((int(self.id[i]), int(self.id[j]))))
                    Ab[i, d] += self.shadow.get(key, 0.0)
        self.max_err_A = max(self.max_err_A, float(np.abs(Ab - A).max()) if A.size else 0.0)
        return super().movement_weights(A, nbcell, valid)

    def update_affinities(self):
        out = super().update_affinities()
        occ = {}
        for i in range(self.N):
            occ.setdefault((int(self.x[i]), int(self.y[i])), []).append(int(self.id[i]))
        notebook_update_affinities(self.shadow, occ, self.p)
        return out

    def adult_deaths(self):
        before = set(self.id.tolist())
        super().adult_deaths()
        dead = before - set(self.id.tolist())
        for key in list(self.shadow):
            if key[0] in dead or key[1] in dead:
                del self.shadow[key]

    def step(self):
        super().step()
        d = self.store.as_dict(self.t, self._alive_id)
        assert set(d) == set(self.shadow), (len(d), len(self.shadow))
        assert self.n_aff == len(self.shadow)
        if d:
            err = max(abs(d[k] - self.shadow[k]) for k in d)
            self.max_err_F = max(self.max_err_F, err)


@pytest.mark.parametrize("kw", [
    dict(grid_size=10, n_males=25, n_females=25, alpha=2.0),
    dict(grid_size=10, n_males=25, n_females=25, alpha=1.4, delta=0.3, eps=1e-3, eta=0.5),
])
def test_affinities_exact_vs_notebook(kw):
    U = _Shadow(Params(**kw), rng=np.random.default_rng(11))
    for _ in range(45):
        U.step()
        if U.N == 0:
            break
    assert U.max_err_F < 1e-10
    assert U.max_err_A < 1e-9


# ---------------------------------------------------------------- extensiones
@pytest.mark.parametrize("kw", [
    dict(alpha_s_mode="power", alpha_low=-1.0, alpha_s_power=2.0),
    dict(alpha_s_mode="sigmoid", weight_form="exp"),
    dict(alpha_s_mode="threshold", alpha_low=0.0),
    dict(crowd_aversion=0.5),
    dict(territory_strength=1.0, mating_rule="dominant", refractory_weeks=3),
    dict(plast_age_min=0, plast_age_max=10, plast_outside_det=0.0, plast_outside_rec=0.0),
    dict(social_hazard=0.05, social_mortality_factor=1.0),
    dict(inherit_weight=0.0, s_newborn_base=1.0),
    dict(periodic=True),
])
def test_extensions_run(kw):
    s, m = simulate(Params(**SMALL, **kw), T=40, seed=2)
    assert len(s) >= 2


def test_plasticity_window_freezes_adults():
    p = Params(**SMALL, plast_age_min=0, plast_age_max=5, plast_outside_det=0.0,
               plast_outside_rec=0.0, initial_s=0.5, initial_age=10)
    U = Universe(p, rng=np.random.default_rng(0))
    s0 = U.s.copy()
    U.update_social_state(np.full(U.N, 50))
    assert np.allclose(U.s, s0)


def test_inherit_weight_zero_newborn_full_s():
    p = Params(**SMALL, inherit_weight=0.0, s_newborn_base=1.0, initial_s=0.6, gamma=0.0, rec=0.0)
    s, _ = simulate(p, T=12, seed=4)
    # sólo fundadores (s=0.6) y neonatos (s=1)
    assert s["s_q90"].iloc[-1] == pytest.approx(1.0) or s["births"].sum() == 0


# ---------------------------------------------------------------- referencia (estadístico)
def test_matches_reference_statistically():
    p = Params(grid_size=8, n_males=15, n_females=15)
    T = 30
    fast = [simulate(p, T=T, seed=s)[0]["N"].to_numpy() for s in range(12)]
    ref = [np.array([h["N"] for h in ReferenceUniverse(p, seed=s).run(T)]) for s in range(12)]
    f = np.array([x[-1] if len(x) == T + 1 else 0 for x in fast], float)
    r = np.array([x[-1] if len(x) == T else 0 for x in ref], float)
    se = np.sqrt(f.var(ddof=1) / len(f) + r.var(ddof=1) / len(r))
    assert abs(f.mean() - r.mean()) < 4 * se + 1e-9, (f.mean(), r.mean(), se)


# ---------------------------------------------------------------- ensamble
def test_ensemble_parallel_equals_serial(tmp_path):
    p = Params(**SMALL)
    d1 = run_ensemble(p, n_reps=3, T=25, base_seed=9, n_jobs=1, out=tmp_path / "a")
    d2 = run_ensemble(p, n_reps=3, T=25, base_seed=9, n_jobs=2, out=tmp_path / "b")
    cols = ["N_max", "t_peak", "total_births", "max_aff"]
    pd.testing.assert_frame_equal(d1[cols], d2[cols])
    assert (tmp_path / "a" / "summary.csv").exists()
    assert (tmp_path / "a" / "params.json").exists()
    assert any((tmp_path / "a").glob("series.*"))


def test_ensemble_sweep_points(tmp_path):
    pl = [Params(**SMALL, alpha=a) for a in (0.0, 3.0)]
    df = run_ensemble(pl, n_reps=2, T=15, base_seed=0, n_jobs=2, out=None, save_series=False)
    assert sorted(df["alpha"].unique()) == [0.0, 3.0]
    assert len(df) == 4


def test_radix_sort_matches_argsort():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.integers(0, 6, n)
    y = rng.integers(0, 6, n)
    ids = np.sort(rng.choice(10 ** 6, n, replace=False)).astype(np.int64)
    a, b, _, _ = K.build_pairs(x, y, 6, False)
    lo, hi, keys = K.radix_sort_pairs(a, b, n, ids)
    ref = np.sort((np.minimum(ids[a], ids[b]) << 32) | np.maximum(ids[a], ids[b]))
    assert np.array_equal(keys, ref)
    # todos los pares a distancia de Chebyshev <= 1, una sola vez
    d = np.maximum(np.abs(x[:, None] - x[None, :]), np.abs(y[:, None] - y[None, :]))
    assert len(keys) == int(((d <= 1).sum() - n) // 2)
