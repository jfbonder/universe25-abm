"""Tests de la v2: R7 estrictamente dura, O13p/O17p en u25, O10/O11 redefinidos, flags nuevos (mu_bg, nest_period,
async_move, edades iniciales), opciones del evaluador (t′_pk, umbral fijo de clase hacinada) y re-evaluación de
barridos."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from u25 import Params, Universe, simulate
from u25 import kernels as K
from u25 import objectives as OB
from u25.params import preset

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(grid_size=12, n_males=30, n_females=30)


def _fp(s):
    import hashlib
    cols = ['N', 'births', 'deaths', 'S_mean', 'n_aff', 'mean_aff', 'occ_var', 's_q50']
    a = np.concatenate([s[c].to_numpy(float) for c in cols])
    return hashlib.sha256(np.round(a, 12).tobytes()).hexdigest()[:16]


# =========================================================================== capacidad de R7
class CapacityProbe(Universe):
    """Registra la ocupación máxima de los refugios después de cada subpaso de movimiento, inmediatamente después
    de los nacimientos (reproduce(), antes de las muertes) y al final del paso."""

    def __init__(self, *a, **k):
        self.max_move = self.max_birth = self.max_end = 0
        super().__init__(*a, **k)
        self.max_init = self._refuge_max()

    def _refuge_max(self):
        if self.refuge is None or self.N == 0:
            return 0
        G = self.p.grid_size
        return int(np.bincount(self.x * G + self.y, minlength=G * G)[self.refuge].max(initial=0))

    def move(self, do_jump=True):
        super().move(do_jump)
        self.max_move = max(self.max_move, self._refuge_max())

    def reproduce(self, cell):
        super().reproduce(cell)
        self.max_birth = max(self.max_birth, self._refuge_max())

    def step(self):
        super().step()
        self.max_end = max(self.max_end, self._refuge_max())


@pytest.mark.parametrize("name,kw", [
    ("calhoun_C1", {}),
    ("calhoun_C1p", {}),
    ("calhoun_C2", {}),
    ("calhoun_C1", dict(move_substeps=2, jump_prob=0.05)),
    ("calhoun_C1", dict(nest_period=3)),
    ("calhoun_C1", dict(async_move=True)),
    ("calhoun_C1", dict(initial_age_dist="uniform", initial_age_hi=40, mu_bg=0.003, refuge_capacity=1)),
])
def test_refuge_occupancy_never_exceeds_capacity(name, kw):
    """Invariante de R7 'strict': ocupación <= capacidad en todo refugio, siempre (condición inicial, después
    de cada subpaso de movimiento, justo después de los nacimientos y al final del paso)."""
    p = preset(name, **kw)
    assert p.refuge_hard_mode == "strict"
    U = CapacityProbe(p, rng=np.random.default_rng(11))
    U.run(160)
    cap = p.refuge_capacity
    assert U.max_init <= cap and U.max_move <= cap and U.max_birth <= cap and U.max_end <= cap, \
        (U.max_init, U.max_move, U.max_birth, U.max_end)
    if cap >= 2:                     # con capacidad 1 no hay apareamiento dentro de un refugio
        assert U.tot["births_relocated"] > 0                       # en C1 nacen crías en refugios llenos
    assert U.n_capacity_unresolved == 0


def test_legacy_rule_is_v1_and_violates_capacity():
    """'legacy' es la regla de v1 (huella bit a bit en test_phase2) y deja refugios sobrecargados."""
    U = CapacityProbe(preset("calhoun_C1", refuge_hard="legacy"), rng=np.random.default_rng(11))
    U.run(160)
    assert U.max_move > 2 and U.max_birth > 2 and U.max_end > 2
    assert U.tot["births_relocated"] == 0


def test_refuge_hard_modes_and_parsing():
    from u25.params import parse_value
    from u25.sweep import _cast
    for v, mode in ((False, "off"), (True, "strict"), ("strict", "strict"), ("legacy", "legacy"),
                    ("True", "strict"), ("false", "off"), ("off", "off")):
        assert Params(refuge_hard=v).refuge_hard_mode == mode
    with pytest.raises(ValueError):
        Params(refuge_hard="duro").validate()
    p = preset("calhoun_C1")
    assert parse_value(p, "refuge_hard", "legacy") == "legacy" and parse_value(p, "refuge_hard", "1") is True
    assert _cast(Params(), "refuge_hard", "legacy") == "legacy" and _cast(p, "refuge_hard", True) is True
    assert _cast(Params(), "refuge_hard", "True") is True and _cast(p, "refuge_hard", np.bool_(False)) is False


def _toy(cap=1, n=3):
    """Grilla 5x5 con refugios en (1,1) y (1,3) (refuge_seed elegido) y n agentes."""
    p = Params(grid_size=5, n_males=0, n_females=n, refuge_frac=2 / 25, refuge_capacity=cap,
               refuge_hard="strict")
    for seed in range(200):
        U = Universe(p.replace(refuge_seed=seed), rng=np.random.default_rng(0))
        r = np.flatnonzero(U.refuge)
        if len(r) == 2 and abs(r[0] // 5 - r[1] // 5) + abs(r[0] % 5 - r[1] % 5) >= 2:
            return U, r
    raise RuntimeError


def test_strict_rejected_returning_resident_keeps_its_place():
    """A sale del refugio R1 hacia R2 (lleno por B, residente) y es rechazado; mientras tanto C entró a R1.
    Con 'strict' A vuelve a R1 como residente y C es rechazado (vuelve a su celda); con una sola pasada ('legacy')
    R1 quedaba con A y C, por encima de la capacidad 1."""
    U, (r1, r2) = _toy(cap=1, n=3)
    G = 5
    c_out = next(c for c in range(G * G) if not U.refuge[c] and max(abs(c // G - r1 // G), abs(c % G - r1 % G)) == 1)
    xp = np.array([r1 // G, r2 // G, c_out // G])      # antes: A en R1, B en R2, C afuera
    yp = np.array([r1 % G, r2 % G, c_out % G])
    U.x = np.array([r2 // G, r2 // G, r1 // G])        # después del muestreo: A -> R2, B se queda, C -> R1
    U.y = np.array([r2 % G, r2 % G, r1 % G])
    U._enforce_refuge_capacity_strict(xp, yp)
    cells = U.x * G + U.y
    assert cells.tolist() == [r1, r2, c_out]
    assert U.n_refuge_rejected == 2
    # la regla de v1 deja a A y C juntos en R1
    U.x = np.array([r2 // G, r2 // G, r1 // G])
    U.y = np.array([r2 % G, r2 % G, r1 % G])
    U._enforce_refuge_capacity(xp, yp)
    assert np.bincount(U.x * G + U.y, minlength=G * G)[r1] == 2


def test_strict_newborns_in_full_refuge_go_to_a_non_refuge_neighbour():
    p = Params(grid_size=6, n_males=1, n_females=1, initial_age=10, p0=1.0, neonatal_survival=1.0,
               refuge_frac=0.2, refuge_capacity=2, refuge_hard="strict")
    U = Universe(p, rng=np.random.default_rng(3))
    G = 6
    rc = int(np.flatnonzero(U.refuge)[0])
    U.x[:] = rc // G
    U.y[:] = rc % G                                     # macho y hembra llenan el refugio (capacidad 2)
    cell = U.x * G + U.y
    U._m = np.full(2, 2, np.int64)
    U._cell = cell
    U.reproduce(cell)
    assert U.births > 0
    nb = U.birth_t == U.t
    nbc = U.x[nb] * G + U.y[nb]
    assert not U.refuge[nbc].any()                      # ninguna cría quedó en un refugio
    assert np.unique(nbc).size == 1                     # una sola celda por madre
    assert max(abs(nbc[0] // G - rc // G), abs(nbc[0] % G - rc % G)) == 1
    assert U.n_births_relocated == U.births
    assert (U._cell[nb] == nbc).all()


def test_strict_initial_placement_and_transplant_within_capacity():
    p = preset("calhoun_C1", founders_same_cell=True, n_males=4, n_females=4)
    U = Universe(p, rng=np.random.default_rng(0))
    G = p.grid_size
    occ = np.bincount(U.x * G + U.y, minlength=G * G)
    assert occ[U.refuge].max(initial=0) <= 2
    rc = int(np.flatnonzero(U.refuge)[0])
    V = Universe.from_agents(p, np.random.default_rng(0), male=[1, 0, 1, 0, 1], age=[10] * 5, s=[1.0] * 5,
                             x=[rc // G] * 5, y=[rc % G] * 5)
    occ = np.bincount(V.x * G + V.y, minlength=G * G)
    assert occ[rc] == 2 and occ[V.refuge].max() <= 2 and V.N == 5


def test_strict_presets_fingerprints():
    """Huellas de regresión de los presets con la capacidad estricta (las de v1 están en test_phase2, con
    refuge_hard='legacy')."""
    for name, fp in (("calhoun_C1", "a8249d1bc00a849e"), ("calhoun_C1p", "da4484fbd2dd8093"),
                     ("calhoun_C2", "472f12aa511e71e9")):
        s, _ = simulate(preset(name), T=120, seed=7)
        assert _fp(s) == fp, name


# =========================================================================== flags nuevos
def test_new_flags_neutral_by_default():
    base, _ = simulate(Params(**SMALL), T=40, seed=3)
    for kw in (dict(mu_bg=0.0), dict(nest_period=0), dict(async_move=False), dict(initial_age_dist="fixed")):
        s, _ = simulate(Params(**SMALL, **kw), T=40, seed=3)
        assert _fp(s) == _fp(base), kw


def test_mu_bg_background_mortality():
    p = Params(**SMALL, p0=0.0, a_max=1e3, mu_bg=0.05)
    U = Universe(p, rng=np.random.default_rng(0))
    deaths = []
    for _ in range(40):
        n = U.N
        U.step()
        deaths.append((U.deaths, n))
    d, n = np.array(deaths).sum(0)
    assert abs(d / n - 0.05) < 0.015
    # la mortalidad de fondo no se atribuye a la social (A4)
    p2 = Params(**SMALL, p0=0.0, a_max=1e3, mu_bg=0.2, social_hazard=1e-9)
    U = Universe(p2, rng=np.random.default_rng(1))
    tot = 0.0
    for _ in range(10):
        U.step()
        tot += U.deaths_social_exp
    assert tot < 1e-3


class NestProbe(Universe):
    """Comprueba, dentro del paso, que la m de daño excluye a las crías en el nido."""
    checked = 0

    def update_social_state(self, m):
        nest = self._nest_mask()
        G = self.p.grid_size
        exp = np.bincount(self._cell[~nest], minlength=G * G)[self._cell] + nest
        assert (m == exp).all()
        NestProbe.checked += int(nest.sum())
        super().update_social_state(m)


def test_nest_period_pups_immobile_and_not_crowding():
    p = preset("calhoun_C1", nest_period=3)
    U = NestProbe(p, rng=np.random.default_rng(2))
    G = p.grid_size
    seen = 0
    for _ in range(14):
        pos0 = dict(zip(U.id.tolist(), (U.x * G + U.y).tolist()))
        age0 = dict(zip(U.id.tolist(), U.age.tolist()))
        U.step()
        for i, c in zip(U.id.tolist(), (U.x * G + U.y).tolist()):
            if i in pos0 and 1 <= age0[i] <= 3:
                assert c == pos0[i]                        # en el nido no se mueven
                seen += 1
    assert seen > 100 and NestProbe.checked > 100


def test_initial_ages_uniform():
    p = Params(**SMALL, initial_age_dist="uniform", initial_age=4, initial_age_hi=30)
    U = Universe(p, rng=np.random.default_rng(0))
    assert U.age.min() >= 4 and U.age.max() <= 30 and np.unique(U.age).size > 10
    assert (U.birth_t == -U.age).all()


def test_async_move_kernel_matches_sync_weights_for_a_single_mover():
    """Con un solo agente móvil, el barrido asincrónico elige con los mismos pesos que el sincrónico (afinidades
    del almacén a distancia <= 2, aversión, refugios llenos y preferencia R7 incluidas)."""
    p = preset("calhoun_C1", n_males=60, n_females=60, grid_size=8)
    U = Universe(p, rng=np.random.default_rng(5))
    for _ in range(6):
        U.step()
    n = U.N
    A = K.accumulate_affinity_field(U._pa, U._pb, U._pF, U.x, U.y, n, p.grid_size, p.periodic)
    nb, valid = U._neighbor_cells()
    W = U.movement_weights(A, nb, valid)
    G = p.grid_size
    a, b = K.build_pairs_r2(U.x, U.y, G, p.periodic)
    lo, hi, keys = K.radix_sort_pairs(a, b, n, U.id)
    st = U.store
    idx = np.minimum(np.searchsorted(st.keys, keys), len(st) - 1)
    found = st.keys[idx] == keys
    F = np.zeros(keys.size)
    F[found] = st.val[idx[found]] * st._pow(U.t)[U.t - st.t0[idx[found]]]
    F[F < st.eps] = 0
    src, dst, Fv = np.r_[lo, hi], np.r_[hi, lo], np.r_[F, F]
    o = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
    cell = U.x * G + U.y
    crowd = p.crowd_aversion * (1 - U.s) ** p.crowd_power
    rfac = 1 + p.refuge_pref * (1 - U.s) ** p.refuge_pref_power
    rm = U.male & U.is_reproductive()
    for i in range(0, n, max(1, n // 25)):
        for u in (0.05, 0.37, 0.71, 0.97):
            mob = np.zeros(n, bool)
            mob[i] = True
            x, y = U.x.copy(), U.y.copy()
            uu = np.full(n, u)
            K.async_move_kernel(np.arange(n), uu, x, y, G, p.periodic, ptr, dst[o], Fv[o], np.full(n, p.alpha),
                                True, False, p.weight_floor, crowd, np.bincount(cell, minlength=G * G),
                                0.0, rm, np.bincount(cell[rm], minlength=G * G), U.refuge, p.refuge_capacity, True,
                                rfac, mob, np.bincount(cell, minlength=G * G))
            d = K.sample_rows(W[i:i + 1], np.array([u]))[0]
            assert (x[i], y[i]) == (U.x[i] + K.DX[d], U.y[i] + K.DY[d])


def test_async_move_runs_one_cell_per_substep():
    p = preset("calhoun_C1", async_move=True, n_males=80, n_females=80, grid_size=12)
    U = Universe(p, rng=np.random.default_rng(1))
    for _ in range(8):
        x0, y0 = U.x.copy(), U.y.copy()
        ids0 = U.id.copy()
        U.move()
        assert max(np.abs(U.x - x0).max(), np.abs(U.y - y0).max()) <= 1
        U.x, U.y = x0, y0
        U.step()
        assert ids0.size  # noqa
    s, _ = simulate(p, T=40, seed=2)
    assert s["N"].iloc[-1] > 0


# =========================================================================== O13p y O17p
def syn(N, births=None, **cols):
    N = np.asarray(N, float)
    n = N.size
    t = np.arange(n)
    births = np.zeros(n) if births is None else np.asarray(births, float)
    d = dict(t=t, N=N, births=births, conceptions=births / 6.0, N_f=N / 2, N_m=N / 2, N_repro_f=N / 4,
             deaths=np.zeros(n), S_mean=np.full(n, 0.8), mean_age=np.full(n, 20.0), occ_var=N / 900 * 3,
             occ_frac_cells=np.full(n, 0.5))
    d.update(cols)
    return pd.DataFrame(d)


def test_O13p_synthetic_states():
    N = np.r_[np.linspace(400, 2000, 41), np.linspace(2000, 100, 60)]
    p = Params()
    for bo, cr, st in ((0.15, 0.2, "PASS"), (0.07, 0.2, "MARG"), (0.02, 0.2, "FAIL"), (0.15, 0.05, "FAIL")):
        S = syn(N, phi_BO=np.full(N.size, bo), frac_crowded_low=np.full(N.size, cr), N_adult=np.full(N.size, 100))
        assert OB.eval_O13p(S, p, OB.characteristic_times(S, p)).status == st
    S = syn(N, phi_BO=np.full(N.size, 0.2), frac_crowded_low=np.full(N.size, 0.2), N_adult=np.full(N.size, 5))
    assert OB.eval_O13p(S, p, OB.characteristic_times(S, p)).status == "NA"      # < 20 adultos
    assert OB.eval_O13p(syn(N), p, OB.characteristic_times(syn(N), p)).status == "NA"


def test_O13p_equals_v1_f2b_model():
    """O13p de u25 = obj_O13p de scripts/production/f2b_model.extras (fuente de la v1)."""
    mat = ROOT / "scripts" / "production"
    if not (mat / "f2b_model.py").exists():
        pytest.skip("f2b_model no disponible")
    sys.path.insert(0, str(mat))
    try:
        from f2b_model import extras
    finally:
        sys.path.remove(str(mat))
    for name, seed in (("calhoun_C1", 1), ("calhoun_C1p", 2), ("calhoun_C0", 3), ("calhoun_C1", 4)):
        p = preset(name)
        S, summ = simulate(p, T=110, seed=seed)
        r = OB.evaluate_run(S, p, summ)
        assert r["O13p"].status == extras(S, p, summ)["obj_O13p"], name


def test_O17p_strict_na_and_err():
    N = np.r_[np.linspace(400, 2000, 41), np.linspace(2000, 0, 60)]
    S = syn(N)
    r = OB.evaluate_run(S, Params())
    assert r["O17p"].status == "FAIL"                     # NA en O5..O13p no aprueba
    fake = {k: OB.Result("PASS") for k in OB.ESSENTIAL + ["O13p"]}
    fake.update({k: OB.Result("CLEAR") for k in OB.ANTI})
    fake["_times"] = dict(N0=400.0)
    assert OB.o17p_status(fake).status == "PASS"
    for k, st, exp in (("O13p", "NA", "FAIL"), ("O13", "NA", "FAIL"), ("O1", "NA", "PASS"), ("O2", "NA", "PASS"),
                       ("O5", "ERR", "ERR"), ("A2", "NA", "FAIL"), ("A1", "TRIGGERED", "FAIL"),
                       ("O13p", "MARG", "FAIL")):
        f2 = dict(fake)
        f2[k] = OB.Result(st)
        assert OB.o17p_status(f2).status == exp, (k, st)
    f2 = dict(fake, O1=OB.Result("NA"), _times=dict(N0=8.0))  # con N0 <= 20 O1 se evalúa: NA no alcanza
    assert OB.o17p_status(f2).status == "FAIL"


def test_criterion_exception_gives_ERR_not_pass(monkeypatch):
    def boom(*a, **k):
        raise ValueError("roto")
    monkeypatch.setattr(OB, "eval_O13p", boom)
    s, summ = simulate(preset("calhoun_C1"), T=60, seed=1)
    r = OB.evaluate_run(s, preset("calhoun_C1"), summ)
    assert r["O13p"].status == "ERR" and r["O17p"].status == "ERR"
    monkeypatch.setattr(OB, "eval_O5", boom)
    r = OB.evaluate_run(s, preset("calhoun_C1"), summ)
    assert r["calhoun_like"].status == "ERR"


def test_o17p_from_summary():
    ok = {f"obj_{k}": "PASS" for k in OB.ESSENTIAL + ["O13p"]}
    ok.update({f"obj_{k}": "CLEAR" for k in OB.ANTI})
    ok.update(obj_O1="NA", obj_O2="NA", obj_O3="NA", N0=400)
    rows = [ok, dict(ok, obj_O13p="NA"), dict(ok, obj_O13p="NA:ValueError"), dict(ok, obj_O11="NA"),
            dict(ok, obj_O13p=""), dict(ok, obj_A2="TRIGGERED"), dict(ok, N0=8), dict(ok, obj_O7="ERR"),
            dict(ok, obj_O13p=np.nan)]
    df = pd.DataFrame(rows)
    assert OB.o17p_from_summary(df).tolist() == ["PASS", "FAIL", "ERR", "FAIL", "FAIL", "FAIL", "FAIL", "ERR",
                                                 "FAIL"]
    assert OB.o17p_from_summary(df.drop(columns="obj_O13p")).tolist() == ["FAIL"] * 7 + ["ERR", "FAIL"]
    # lectura por defecto de pandas ('NA' -> NaN): O1–O3 siguen admitidos como NA
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    assert OB.o17p_from_summary(pd.read_csv(buf)).tolist()[:2] == ["PASS", "FAIL"]


# =========================================================================== O10 / O11 v2
def test_O10_uses_adults_not_dilution():
    """Colonia en boom sin deterioro: S_mean cae por dilución (crías con s baja) pero S_adult no -> O10 de v1
    PASS, O10 v2 FAIL."""
    N = np.r_[np.linspace(400, 5000, 41), np.linspace(5000, 100, 60)]
    S = syn(N, S_mean=np.where(np.arange(N.size) >= 15, 0.35, 0.9), S_adult=np.full(N.size, 0.8))
    ct = OB.characteristic_times(S, Params())
    assert OB.eval_O10_v1(S, Params(), ct).status == "PASS"
    assert OB.eval_O10(S, Params(), ct).status == "FAIL"
    # caída transitoria recuperada antes del pico: FAIL
    sad = np.where((np.arange(N.size) >= 10) & (np.arange(N.size) < 20), 0.3, 0.8)
    S = syn(N, S_adult=sad)
    assert OB.eval_O10(S, Params(), OB.characteristic_times(S, Params())).status == "FAIL"
    # O10b: fase B funcional (S_adult >= 0.5 hasta 0.5 t_pk)
    for ts, st in ((25, "PASS"), (14, "MARG"), (8, "FAIL")):
        S = syn(N, S_adult=np.where(np.arange(N.size) >= ts, 0.2, 0.9))
        ct = OB.characteristic_times(S, Params())
        assert OB.eval_O10(S, Params(), ct).status == "PASS"
        assert OB.eval_O10b(S, Params(), ct).status == st, ts


def test_O11_no_vacuous_pass():
    n = 120
    t = np.arange(n)
    N = np.r_[np.linspace(400, 2600, 50), np.full(n - 50, 2600.0)]
    B = np.where(t < 30, 100.0, 0.0)
    sm = np.clip(0.5 - 0.012 * t, 0.05, None)
    # cohortes tardías con >= 30 maduros y s < 0.2 -> PASS
    S = syn(N, births=B, s_mat=sm, n_mat=np.where((t >= 6) & (t < 36), 50.0, 0.0))
    r = OB.eval_O11(S, Params(), OB.characteristic_times(S, Params()))
    assert r.status == "PASS", r
    # ventana tardía vacía (nadie madura) -> NA, no PASS (v1: PASS)
    nm = np.where((t >= 6) & (t < 20), 50.0, 0.0)
    S = syn(N, births=B, s_mat=np.where(nm > 0, sm, np.nan), n_mat=nm)
    ct = OB.characteristic_times(S, Params())
    assert OB.eval_O11(S, Params(), ct).status == "NA"
    # cohortes tardías que se socializan (s > 0.2) con tendencia decreciente -> MARG
    S = syn(N, births=B, s_mat=np.clip(0.9 - 0.005 * t, 0.3, None), n_mat=np.where((t >= 6) & (t < 36), 50.0, 0.0))
    assert OB.eval_O11(S, Params(), OB.characteristic_times(S, Params())).status == "MARG"


def test_O10_O11_discriminate_gamma0():
    """Modelo nulo gamma = 0 (sin daño por hacinamiento): O10 y O11 de v1 pasaban; los de v2 no."""
    p = preset("calhoun_C1")
    S, summ = simulate(p, T=130, seed=1001)
    r = OB.evaluate_run(S, p, summ)
    assert r["O10"].status == "PASS" and r["O11"].status == "PASS"
    p0 = preset("calhoun_C1", gamma=0.0, max_pop=6000)
    S, summ = simulate(p0, T=130, seed=1001)
    r = OB.evaluate_run(S, p0, summ)
    ct = OB.characteristic_times(S, p0)
    assert OB.eval_O10_v1(S, p0, ct).status == "PASS" and OB.eval_O11_v1(S, p0, ct).status == "PASS"
    assert r["O10"].status == "FAIL" and r["O11"].status != "PASS"


# =========================================================================== opciones del evaluador
def test_peak_time_options():
    t = np.arange(150)
    N = np.r_[np.linspace(400, 2000, 31), np.full(50, 2000.0), np.linspace(1999, 0, 69)]
    N[60] = 2010                                         # argmax en el medio de la meseta
    S = syn(N)
    p = Params()
    assert OB.characteristic_times(S, p)["t_pk"] == 60
    assert OB.characteristic_times(S, p, peak_time="plateau_start")["t_pk"] == 30
    ct = OB.characteristic_times(S, p, peak_time="plateau_end")
    assert ct["t_pk"] == 81 and ct["N_pk"] == 2010 and ct["t_pk_argmax"] == 60
    assert OB.characteristic_times(S, p, peak_time="plateau_mid")["t_pk"] == 55
    r0 = OB.evaluate_run(S, p)
    r1 = OB.evaluate_run(S, p, peak_time="plateau_start")
    assert r1["_times"]["t_pk"] == 30 and r0["_times"]["t_pk"] == 60
    r2 = OB.evaluate_run(S, p, peak_time="plateau_end")
    assert r2["O8b"].value < r0["O8b"].value              # t_pk + 0.31 t_pk cae fuera de la meseta
    with pytest.raises(ValueError):
        OB.characteristic_times(S, p, peak_time="medio")


def test_crowd_class_threshold_columns():
    p = preset("calhoun_C1", m_c=12)
    S, summ = simulate(p, T=90, seed=3)
    for k in (6, 8, 10, 12):
        assert f"frac_crowded_low_m{k}" in S
    a, b = S["frac_crowded_low"].to_numpy(), S["frac_crowded_low_m12"].to_numpy()
    assert np.allclose(a[1:], b[1:], equal_nan=True)     # m > m_c con m_c = 12
    assert (S["frac_crowded_low_m8"].fillna(0) >= S["frac_crowded_low_m12"].fillna(0)).all()
    assert OB.crowd_column(p, None) == "frac_crowded_low" and OB.crowd_column(p, 12) == "frac_crowded_low"
    assert OB.crowd_column(p, 8) == "frac_crowded_low_m8"
    r = OB.evaluate_run(S, p, summ, crowd_class_threshold=8)
    assert r["O13p"].detail == "frac_crowded_low_m8"
    r = OB.evaluate_run(S, p, summ, crowd_class_threshold=7)   # sin columna -> NA, nunca PASS
    assert r["O13p"].status == "NA" and r["O17p"].status == "FAIL"


# =========================================================================== re-evaluación
def test_sweep_writes_O13p_O17p_and_reeval_reproduces(tmp_path):
    from u25.sweep import run_sweep
    from u25.reeval import reeval_dir
    d = pd.DataFrame(dict(point=[0, 1], refuge_hard=["strict", "legacy"]))
    out = tmp_path / "src"
    base = preset("calhoun_C1", n_males=60, n_females=60, grid_size=12)
    run_sweep(d, 2, out, T=60, base_seed=5, n_jobs=1, base=base, quiet=True)
    s = pd.read_csv(out / "summary.csv", keep_default_na=False)
    assert {"obj_O13p", "obj_O17p", "obj_O10b"} <= set(s.columns)
    meta = json.load(open(out / "meta.json"))
    assert meta["u25_version"] == 2 and meta["refuge_hard_modes"] == ["legacy", "strict"]
    info = reeval_dir(out, tmp_path / "re", n_jobs=1)
    chk = info["check"]
    assert chk["n_rows"] == 4 and chk["differing_non_objective_columns"] == {} and not info["legacy_r7"]
    assert chk["objective_status_changes"] == {}
    # un barrido "v1" (sin u25_version) con refuge_hard=True se re-corre como 'legacy'
    v1 = tmp_path / "v1"
    run_sweep(pd.DataFrame(dict(point=[0], refuge_hard=["legacy"])), 2, v1, T=60, base_seed=5, n_jobs=1,
              base=base, quiet=True)
    m = json.load(open(v1 / "meta.json"))
    m.pop("u25_version")
    json.dump(m, open(v1 / "meta.json", "w"))
    pl = json.load(open(v1 / "params.json"))
    pl[0]["refuge_hard"] = True
    json.dump(pl, open(v1 / "params.json", "w"))
    info = reeval_dir(v1, tmp_path / "re1", n_jobs=1)
    assert info["legacy_r7"]
    num = [c for c in info["check"]["differing_non_objective_columns"] if c != "refuge_hard"]
    assert num == []
    with pytest.raises(ValueError):
        reeval_dir(out, out)


# =========================================================================== O5 con t_B99 y O4-v2
def test_O5_birth_ref_B99():
    """Cola rala de crías hasta el pico: con 'last' (v1) t_B0 cae justo después del pico y O5 pasa casi por
    construcción; con t′_pk (inicio de la meseta) 'last' falla y 'B99' (99 % de los nacimientos) pasa."""
    n = 200
    t = np.arange(n)
    N = np.r_[np.linspace(400, 2000, 31), np.full(40, 2000.0), np.linspace(2000, 0, n - 71)]
    N[70] = 2005                                          # argmax al final de la meseta (= última cría)
    B = np.where(t < 30, 100.0, 0.0)
    B[[40, 50, 60, 70]] = 5.0                             # cola rala: < 1 % de los nacimientos
    S = syn(N, births=B)
    p = Params()
    ct = OB.characteristic_times(S, p)
    assert OB.eval_O5(S, p, ct).status == "PASS"
    assert ct["t_B99"] == 29
    ctp = OB.characteristic_times(S, p, peak_time="plateau_start")
    assert OB.eval_O5(S, p, ctp, "last").status != "PASS"
    assert OB.eval_O5(S, p, ctp, "B99").status == "PASS"
    r = OB.evaluate_run(S, p, peak_time="plateau_start", o5_birth_ref="B99")
    assert r["O5"].status == "PASS" and "t_B99" in r["O5"].detail
    assert OB.evaluate_run(S, p)["O5"] == OB.evaluate_run(S, p, o5_birth_ref="last")["O5"]   # default v1


def test_O4_v2_free_space():
    p = Params()
    K = p.grid_size ** 2 * p.m_c
    for rho, f0, v1, v2 in ((0.36, 0.17, "PASS", "PASS"), (0.72, 0.03, "PASS", "FAIL"), (0.5, 0.07, "PASS", "MARG"),
                            (0.85, 0.5, "MARG", "MARG"), (0.85, 0.05, "MARG", "FAIL"), (1.1, 0.5, "FAIL", "FAIL")):
        N = np.r_[np.linspace(100, rho * K, 50), np.linspace(rho * K, 10, 50)]
        S = syn(N, occ_frac_cells=np.full(N.size, 1 - f0))
        ct = OB.characteristic_times(S, p)
        assert OB.eval_O4(S, p, ct, {}).status == v1
        assert OB.eval_O4(S, p, ct, {}, "v2").status == v2, (rho, f0)
    S = syn(np.linspace(100, 2000, 50), occ_frac_cells=np.full(50, 0.5))
    ct = OB.characteristic_times(S, p)
    assert OB.eval_O4(S, p, ct, {"exploded": True}, "v2").status == "FAIL"
    assert OB.eval_O4(S.drop(columns="occ_frac_cells"), p, ct, {}, "v2").status == "NA"


def test_eval_options_in_sweep_and_defaults_unchanged(tmp_path):
    from u25.sweep import run_sweep
    base = preset("calhoun_C1", n_males=60, n_females=60, grid_size=12)
    d = pd.DataFrame(dict(point=[0]))
    run_sweep(d, 2, tmp_path / "v1", T=70, base_seed=3, n_jobs=1, base=base, quiet=True)
    run_sweep(d, 2, tmp_path / "v1b", T=70, base_seed=3, n_jobs=1, base=base, quiet=True,
              eval_kw=dict(o5_birth_ref="last", o4_version="v1", peak_time="argmax"))
    kw = dict(o5_birth_ref="B99", o4_version="v2", peak_time="plateau_start")
    run_sweep(d, 2, tmp_path / "v2", T=70, base_seed=3, n_jobs=1, base=base, quiet=True, eval_kw=kw)
    a, b = (pd.read_csv(tmp_path / x / "summary.csv", keep_default_na=False) for x in ("v1", "v1b"))
    obj = [c for c in a.columns if c.startswith("obj_")]
    assert a[obj].equals(b[obj])                           # pasar los defaults explícitos no cambia nada
    assert json.load(open(tmp_path / "v2" / "meta.json"))["eval_kw"] == kw
    c = pd.read_csv(tmp_path / "v2" / "summary.csv", keep_default_na=False)
    assert c.drop(columns=obj + ["wall_time_s"]).equals(a.drop(columns=obj + ["wall_time_s"]))
    with pytest.raises(ValueError):
        run_sweep(d, 1, tmp_path / "bad", T=10, base=base, quiet=True, eval_kw=dict(o4_version="v3"))
    from u25.__main__ import main
    assert main(["sweep", "--design", str(tmp_path / "v1" / "design.csv"), "--reps", "1", "--T", "30",
                 "--out", str(tmp_path / "cli"), "--preset", "calhoun_C1", "--n-jobs", "1", "--quiet",
                 "--o5-birth-ref", "B99", "--o4-version", "v2", "--peak-time", "plateau_start"]) == 0
    assert json.load(open(tmp_path / "cli" / "sweep.json"))["eval_kw"] == kw
