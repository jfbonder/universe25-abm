"""Tests de la Fase 2a: reglas nuevas, observables, transplante y evaluador de objetivos."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from u25 import Params, Universe, simulate
from u25.params import preset
from u25.stats import bimodality_coefficient, gini
from u25 import objectives as OB
from u25.experiments import transplant_experiment, transplant_trial

SMALL = dict(grid_size=12, n_males=30, n_females=30)


def _fp(s):
    cols = ['N', 'births', 'deaths', 'S_mean', 'n_aff', 'mean_aff', 'occ_var', 's_q50']
    a = np.concatenate([s[c].to_numpy(float) for c in cols])
    return hashlib.sha256(np.round(a, 12).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- línea base bit a bit
@pytest.mark.parametrize("kw,T,seed,expected", [
    ({}, 150, 123, "8815b1b0111fffd7"),
    (SMALL, 80, 5, "f79fd85d10a1365e"),
    (dict(SMALL, alpha_s_mode='power', crowd_aversion=0.3, plast_age_max=12, plast_outside_rec=0.0,
          mating_rule='dominant', refractory_weeks=2), 80, 5, "62cf267140e43a12"),
])
def test_baseline_bit_identical_to_phase1(kw, T, seed, expected):
    """Huellas tomadas con el código de la Fase 1: los defaults neutros no cambian nada."""
    s, _ = simulate(Params(**kw), T=T, seed=seed)
    assert _fp(s) == expected


# ---------------------------------------------------------------- mortalidad social: signo
def test_social_mortality_sign_fixed():
    U = Universe(Params(n_males=0, n_females=151), rng=np.random.default_rng(0))
    U.age = np.arange(151)
    U.s = np.zeros(151)
    base = U.death_probability().copy()
    U.p = U.p.replace(social_mortality_factor=0.5)               # forma corregida ("shift")
    fixed = U.death_probability()
    assert np.all(fixed >= base - 1e-15)
    assert np.all(fixed[: int(U.p.a_max)] > base[: int(U.p.a_max)])  # jóvenes deteriorados mueren MÁS
    U.p = U.p.replace(social_mortality_form="notebook")          # forma original: signo invertido
    nb = U.death_probability()
    assert np.all(nb[:59] < base[:59])
    # con s = 1 no hay efecto
    U.s = np.ones(151)
    U.p = U.p.replace(social_mortality_form="shift")
    assert np.allclose(U.death_probability(), base)


def test_social_death_attribution_column():
    s, _ = simulate(Params(**SMALL, social_hazard=0.2), T=40, seed=1)
    assert (s["deaths_social_exp"] <= s["deaths"] + 1e-9).all()
    assert s["deaths_social_exp"].sum() > 0
    s0, _ = simulate(Params(**SMALL), T=20, seed=1)
    assert (s0["deaths_social_exp"] == 0).all()


# ---------------------------------------------------------------- R2 / R2'
def _universe_at(cells, s, age=10, **kw):
    """Universo con agentes en celdas dadas [(x,y), ...] y s dados."""
    n = len(cells)
    p = Params(grid_size=6, n_males=0, n_females=n, **kw)
    U = Universe(p, rng=np.random.default_rng(0))
    U.x = np.array([c[0] for c in cells], np.int64)
    U.y = np.array([c[1] for c in cells], np.int64)
    U.s = np.asarray(s, float)
    U.age = np.full(n, age, np.int64)
    U._cell = U.x * p.grid_size + U.y
    return U


def test_R2_moore_sbar_bruteforce():
    cells = [(0, 0), (0, 1), (1, 1), (3, 3), (5, 5), (4, 4)]
    s = [0.2, 0.4, 0.9, 0.1, 0.7, 0.3]
    U = _universe_at(cells, s, social_rec_lambda=1.0)
    sb = U.social_learning_sbar()
    for i, (x, y) in enumerate(cells):
        nb = [s[j] for j, (a, b) in enumerate(cells) if j != i and max(abs(a - x), abs(b - y)) <= 1]
        exp = np.mean(nb) if nb else 0.0
        assert sb[i] == pytest.approx(exp)


def test_R2_lambda1_all_deteriorated_absorbing():
    p = Params(**SMALL, social_rec_lambda=1.0, initial_s=0.0, gamma=0.0)
    U = Universe(p, rng=np.random.default_rng(2))
    for _ in range(15):
        U.step()
    assert np.all(U.s == 0.0)  # s ≡ 0 es absorbente con lambda=1
    # sin R2 (lambda=0) se recupera
    U2 = Universe(p.replace(social_rec_lambda=0.0), rng=np.random.default_rng(2))
    for _ in range(15):
        U2.step()
    assert U2.s.max() > 0.1


def test_R2_isolated_modes():
    U = _universe_at([(0, 0), (5, 5)], [0.5, 0.5], social_rec_lambda=1.0, gamma=0.0)
    ds = U.social_delta(np.ones(2, np.int64))
    assert np.allclose(ds, 0.0)                         # aislado no aprende
    U.p = U.p.replace(social_rec_isolated="self")
    ds = U.social_delta(np.ones(2, np.int64))
    assert np.allclose(ds, U.p.rec * 0.5)


def test_R2_affinity_source_runs_and_bounded():
    s, _ = simulate(Params(**SMALL, social_rec_lambda=0.8, social_rec_source="affinity"), T=30, seed=3)
    assert s["S_mean"].dropna().between(0, 1).all()


def test_rec_threshold():
    U = _universe_at([(0, 0), (5, 5)], [0.1, 0.5], rec_threshold=0.3, gamma=0.0)
    ds = U.social_delta(np.ones(2, np.int64))
    assert ds[0] == 0.0 and ds[1] == pytest.approx(U.p.rec)


# ---------------------------------------------------------------- techo de competencia
def test_competence_age_ceiling():
    p = Params(**SMALL, competence_age=6, initial_age=8, initial_s=0.5, gamma=0.0, p0=0.0)
    U = Universe(p, rng=np.random.default_rng(0))
    for _ in range(10):
        U.step()
    assert np.allclose(U.s, 0.51)   # techo congelado en el primer paso (0.5 + rec)


# ---------------------------------------------------------------- R3 saltos
@pytest.mark.parametrize("mode", ["social", "affinity"])
def test_jumps_land_on_pre_move_cells(mode):
    p = Params(**SMALL, jump_prob=1.0, jump_mode=mode, p0=0.0)
    U = Universe(p, rng=np.random.default_rng(1))
    for _ in range(3):
        U.step()
    pre = set(zip(U.x.tolist(), U.y.tolist()))
    U.move()
    assert U.n_jumps == U.N
    assert set(zip(U.x.tolist(), U.y.tolist())) <= pre


def test_jumps_uniform_and_zero_prob_consumes_no_rng():
    s, _ = simulate(Params(**SMALL, jump_prob=0.3, jump_mode="uniform"), T=20, seed=2)
    assert s["n_jumps"].iloc[1:].sum() > 0
    a, _ = simulate(Params(**SMALL, jump_prob=0.0, jump_mode="affinity"), T=20, seed=2)
    b, _ = simulate(Params(**SMALL), T=20, seed=2)
    assert _fp(a) == _fp(b)


# ---------------------------------------------------------------- R5 derrota
def test_defeat_gamma_only_non_dominant_males():
    p = Params(grid_size=6, n_males=3, n_females=1, defeat_gamma=0.2, gamma=0.0, rec=0.0)
    U = Universe(p, rng=np.random.default_rng(0))
    U.x[:] = 2
    U.y[:] = 2
    U.x[2] = 5   # macho solo en otra celda: es dominante de su celda
    U.age[:] = 10
    U.s = np.array([0.9, 0.5, 0.4, 0.3])
    U._cell = U.x * 6 + U.y
    ds = U.social_delta(np.ones(4, np.int64))
    assert ds[0] == 0 and ds[1] == pytest.approx(-0.2) and ds[2] == 0 and ds[3] == 0


def test_founders_same_cell():
    U = Universe(Params(founders_same_cell=True), rng=np.random.default_rng(0))
    assert (U.x == 15).all() and (U.y == 15).all()


def test_presets():
    p = preset("longevo")
    assert p.a_max == 120 and p.repro_age_max == 80
    assert preset("longevo", alpha=2.0).alpha == 2.0
    with pytest.raises(KeyError):
        preset("nada")


# ---------------------------------------------------------------- observables
def test_observable_columns_and_ranges():
    s, m = simulate(Params(**SMALL), T=40, seed=4)
    for c in ["h", "ref_frac", "mstar", "f_over", "corr_s_m", "frac_withdrawn", "frac_crowded_low",
              "bimod_s", "s_mat", "n_mat", "age_min", "births_block_0", "births_block_15", "I_D",
              "phi_BO", "pups", "deaths_social_exp", "S_adult"]:
        assert c in s.columns, c
    assert s["h"].between(0, 1).all()
    assert (s[[f"births_block_{k}" for k in range(16)]].sum(axis=1) == s["births"]).all()
    assert (s["pups"] >= s["births"]).all()
    for k in ["R_rebound", "n_rebounds10", "t_B0", "A_H", "rho_ref", "mstar_at_peak", "gini_births_blocks"]:
        assert k in m


def test_bimodality_and_gini():
    rng = np.random.default_rng(0)
    bi = np.concatenate([rng.normal(0, .05, 500), rng.normal(1, .05, 500)])
    assert bimodality_coefficient(bi) > 5 / 9 > bimodality_coefficient(rng.normal(0, 1, 2000))
    assert gini(np.ones(16)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(15), 1.0]) == pytest.approx(15 / 16)


# ---------------------------------------------------------------- transplante
def test_transplant_small():
    df = transplant_experiment(Params(**SMALL), n_reps=2, T_main=60, T_after=40, n_jobs=1)
    assert {"B", "D"} <= set(df.phase)
    assert df["success"].dropna().isin([0.0, 1.0]).all()
    assert df.attrs["O12"].status in ("PASS", "MARG", "FAIL", "NA")


def test_transplant_trial_healthy_founds():
    snap = dict(t=0, N=8, male=np.r_[np.ones(4, bool), np.zeros(4, bool)], age=np.full(8, 10),
                s=np.ones(8), cap=np.full(8, np.inf))
    r = transplant_trial(Params(), snap, np.random.default_rng(0), T_after=60)
    assert r["success"] == 1.0
    snap["s"] = np.zeros(8)
    r = transplant_trial(Params(), snap, np.random.default_rng(0), T_after=60)
    assert r["success"] == 0.0


# ---------------------------------------------------------------- evaluador: tests sintéticos
def syn(N, births=None, conceptions=None, **cols):
    N = np.asarray(N, float)
    n = N.size
    t = np.arange(n)
    if births is None:
        births = np.zeros(n)
    births = np.asarray(births, float)
    if conceptions is None:
        conceptions = births / 6.0
    d = dict(t=t, N=N, births=births, conceptions=conceptions, N_f=N / 2, N_m=N / 2,
             N_repro_f=N / 4, deaths=np.zeros(n), S_mean=np.full(n, 0.8), mean_age=np.full(n, 20.0),
             occ_var=N / 900 * 3, occ_frac_cells=np.full(n, 0.5))
    d.update(cols)
    return pd.DataFrame(d)


def test_synthetic_logistic_triggers_A1_fails_O5_O6():
    t = np.arange(400)
    N = 7000 / (1 + np.exp(-0.2 * (t - 40))) + 60 * np.sin(t / 5.0) * np.exp(-t / 100)
    S = syn(N, births=np.full(400, 50.0))
    r = OB.evaluate_run(S, Params())
    assert r["A1"].status == "TRIGGERED"
    assert r["O5"].status == "FAIL"
    assert r["O6"].status == "FAIL"


def test_synthetic_hump_passes_O5_O6():
    t = np.arange(200)
    N = np.where(t <= 50, 20 * np.exp(0.11 * t), 20 * np.exp(0.11 * 50) * np.exp(-0.03 * (t - 50)))
    N[-5:] = 0
    B = np.where(t < 48, 0.3 * N, 0.0)
    S = syn(N, births=B)
    r = OB.evaluate_run(S, Params())
    assert r["O5"].status == "PASS", r["O5"]
    assert r["O6"].status == "PASS", r["O6"]
    assert r["A2"].status == "CLEAR"
    assert r["O7"].status == "PASS"


def test_synthetic_damped_oscillation_fails_O6_triggers_A2():
    t = np.arange(800)
    N = 3000 + 2000 * np.exp(-t / 400) * np.sin(2 * np.pi * t / 100)
    S = syn(N, births=0.1 * N)
    r = OB.evaluate_run(S, Params())
    assert r["O6"].status == "FAIL"
    assert r["A2"].status == "TRIGGERED"


def test_missing_columns_give_NA_not_errors():
    S = pd.DataFrame(dict(t=np.arange(50), N=np.linspace(400, 1000, 50)))
    r = OB.evaluate_run(S, Params())
    for k in OB.RUN_OBJECTIVES + OB.ANTI:
        assert r[k].status in ("PASS", "MARG", "FAIL", "NA", "TRIGGERED", "CLEAR")
        assert "error" not in r[k].detail, (k, r[k])
    for k in ("O5", "O9", "O11", "O13", "O15"):
        assert r[k].status == "NA", (k, r[k])


def test_marg_at_band_edges():
    p = Params()
    K = p.grid_size ** 2 * p.m_c
    # O4: rho_pk = 0.9 -> MARG ; 0.8 -> PASS ; 1.01 -> FAIL
    for rho, st in ((0.9, "MARG"), (0.8, "PASS"), (1.01, "FAIL")):
        N = np.r_[np.linspace(100, rho * K, 50), np.linspace(rho * K, 10, 50)]
        ct = OB.characteristic_times(syn(N), p)
        assert OB.eval_O4(None, p, ct, {}).status == st
    # O1 con N0 <= 20: r1 = t_lag/t_pk = 0.07 -> MARG; 0.2 -> PASS; N0 > 20 -> NA
    for tlag, st in ((7, "MARG"), (20, "PASS")):
        N = np.r_[np.full(tlag, 8.0), np.linspace(16, 1000, 100 - tlag)]
        ct = OB.characteristic_times(syn(N), p)
        assert OB.eval_O1(None, p, ct).status == st
    ct = OB.characteristic_times(syn(np.linspace(400, 1000, 30)), p)
    assert OB.eval_O1(None, p, ct).status == "NA"
    # O10 (v2, sobre S_adult) y O10 de v1 (S_mean): t_s05 = t_pk + 1 -> MARG ; t_pk - 5 -> PASS
    N = np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 10, 40)]
    for ts, st in ((41, "MARG"), (35, "PASS"), (50, "FAIL")):
        Sm = np.where(np.arange(81) >= ts, 0.3, 0.9)
        S = syn(N, S_mean=Sm, S_adult=Sm)
        ct = OB.characteristic_times(S, p)
        assert OB.eval_O10(S, p, ct).status == st
        assert OB.eval_O10_v1(S, p, ct).status == st


def test_evaluate_ensemble_real_runs():
    from u25 import run_ensemble
    import tempfile
    d = tempfile.mkdtemp()
    df = run_ensemble(Params(**SMALL), n_reps=2, T=60, base_seed=0, n_jobs=1, out=d)
    ser = pd.read_parquet(d + "/series.parquet")
    tab, per = OB.evaluate_ensemble(ser, df, Params(**SMALL))
    assert {"O17", "O7_KM", "A1"} <= set(tab.code)
    assert len(per) == 2


def test_kaplan_meier_simple():
    P, lo, hi = OB.kaplan_meier_pext([10, 20, np.nan, np.nan], 100)
    assert P == pytest.approx(0.5)
    assert lo <= P <= hi


# ---------------------------------------------------------------- cambios tras verificar contra Calhoun 1973
def test_O8b_slow_initial_decline():
    p = Params()
    N = np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 850, 30), np.linspace(850, 0, 30)]
    ct = OB.characteristic_times(syn(N), p)
    r = OB.eval_O8b(None, p, ct)          # t_pk=40, t=52: N ≈ 0.94 N_pk
    assert r.status == "PASS" and r.value > 0.8
    N2 = np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 0, 8), np.zeros(5)]
    ct = OB.characteristic_times(syn(N2), p)
    assert OB.eval_O8b(None, p, ct).value == 0.0    # extinta antes de t: nu = 0 -> FAIL
    N3 = np.r_[np.linspace(10, 1000, 41), np.full(5, 990.0)]    # censurada antes de t
    assert OB.eval_O8b(None, p, OB.characteristic_times(syn(N3), p)).status == "NA"


def test_O15b_and_RB():
    p = Params()
    t = np.arange(120)
    N = np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 10, 79)]
    B = np.where(t < 45, 20.0, 0.0)
    C = np.where(t < 70, 5.0, 0.0)                  # concepciones persisten tras la última cría
    S = syn(N, births=B, conceptions=C)
    ct = OB.characteristic_times(S, p)
    r = OB.eval_O15b(S, p, ct)
    assert r.status == "PASS" and r.value == pytest.approx((70 - 45) / 40)
    S2 = syn(N, births=B, conceptions=np.where(t < 45, 5.0, 0.0))
    assert OB.eval_O15b(S2, p, OB.characteristic_times(S2, p)).status == "FAIL"
    blocks = {f"births_block_{k}": B * (8.0 if k == 0 else 1.0) for k in range(16)}
    S3 = syn(N, births=B, conceptions=C, **blocks)
    rb = OB.eval_RB(S3, p, OB.characteristic_times(S3, p))
    assert rb.value == pytest.approx(8.0) and rb.status == "PASS"


def test_O11b_ever_conceived():
    p = Params()
    N = np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 10, 40)]
    born = np.where((np.arange(81) >= 33) & (np.arange(81) <= 40), 10, 0)
    for conc, st in ((2, "PASS"), (4, "MARG"), (8, "FAIL")):
        S = syn(N, coh_f_born=born, coh_f_ever_conceived=np.where(born > 0, conc, 0))
        assert OB.eval_O11b(S, p, OB.characteristic_times(S, p)).status == st
    assert OB.eval_O11b(syn(N), p, OB.characteristic_times(syn(N), p)).status == "NA"


def test_ever_conceived_observable_consistent():
    s, _ = simulate(Params(**SMALL), T=60, seed=2)
    assert (s["coh_f_ever_conceived"] <= s["coh_f_born"]).all()
    assert s["coh_f_born"].sum() > 0 and s["coh_f_ever_conceived"].sum() > 0


def test_founders_preset():
    p = preset("founders")
    assert (p.n_males, p.n_females, p.initial_age, p.founders_same_cell) == (4, 4, 7, True)
    U = Universe(p, rng=np.random.default_rng(0))
    assert U.N == 8 and (U.age == 7).all()
    r = OB.evaluate_run(simulate(p, T=60, seed=1)[0], p)
    assert r["O1"].status != "NA"          # O1 se evalúa con N0 <= 20


def test_transplant_tD_175_early_and_mixed():
    df = transplant_experiment(Params(**SMALL), n_reps=2, T_main=60, T_after=30, n_jobs=1)
    assert set(df.phase) == {"B", "D", "MIX_M", "MIX_F"}
    d = df[df.phase == "D"]
    for _, r in d.iterrows():
        exp = r.t_pk + max(4, round(0.1 * r.t_pk)) if r.early_D else round(1.75 * r.t_pk)
        assert r.t_snap == exp
    assert df.attrs["O12b"].status in ("PASS", "FAIL", "NA")
    assert OB.mixed_transplant_status([0, 0, 1, 0]).status == "PASS"
    assert OB.mixed_transplant_status([1, 1, 0]).status == "FAIL"


def test_mixed_trial_healthy_partners_found_only_if_D_recover():
    from u25.experiments import mixed_trial
    snap = dict(t=0, N=4, male=np.ones(4, bool), age=np.full(4, 10), s=np.zeros(4), cap=np.full(4, np.inf))
    r = mixed_trial(Params(), snap, np.random.default_rng(0), True, T_after=40)
    assert r["success"] == 0.0 and r["n_M"] == 4 and r["n_F"] == 4


# ---------------------------------------------------------------- Fase 2b: C0, subpasos, refugios
def test_calhoun_C0_regression_fingerprint():
    p = preset("calhoun_C0")
    assert (p.alpha, p.rec, p.social_rec_lambda, p.max_pop, p.n_males, p.n_females) == (4.0, 0.1, 1.0, 15000, 200, 200)
    s, _ = simulate(p, T=120, seed=7)
    assert _fp(s) == "3da3353ea899ac7d"


def test_move_substeps_regression_and_neutral():
    s, _ = simulate(preset("calhoun_C0", move_substeps=2), T=60, seed=7)
    assert _fp(s) == "e91eee85af410a52"
    a, _ = simulate(Params(**SMALL, move_substeps=1), T=30, seed=3)
    b, _ = simulate(Params(**SMALL), T=30, seed=3)
    assert _fp(a) == _fp(b)


def test_move_substeps_displacement_and_pairs_exact():
    p = Params(**SMALL, move_substeps=3, p0=0.0)
    U = Universe(p, rng=np.random.default_rng(0))
    for _ in range(4):
        U.step()
    x0, y0 = U.x.copy(), U.y.copy()
    U.step()
    dmax = np.maximum(np.abs(U.x - x0), np.abs(U.y - y0)).max()
    assert dmax <= 3
    # los pares de un subpaso son exactamente los vecinos actuales, con la afinidad del almacén
    U._pairs_current()
    d = U.store.as_dict(U.t, U._alive_id)
    for i, j, f in zip(U._pa[:200], U._pb[:200], U._pF[:200]):
        assert max(abs(U.x[i] - U.x[j]), abs(U.y[i] - U.y[j])) <= 1
        key = (int(min(U.id[i], U.id[j])), int(max(U.id[i], U.id[j])))
        assert f == pytest.approx(d.get(key, 0.0))


def test_refuge_capacity_blocks_occupied_refuges():
    p = Params(grid_size=6, n_males=2, n_females=0, refuge_frac=1.0, refuge_capacity=1)
    U = Universe(p, rng=np.random.default_rng(0))
    assert U.refuge.all()
    U.x[:] = [2, 2]
    U.y[:] = [2, 3]
    nb, valid = U._neighbor_cells()
    W = U.movement_weights(np.zeros((2, 9)), nb, valid)
    # agente 0 (2,2): no puede entrar a (2,3) (d=5) porque ya hay otro; sí quedarse (d=4)
    assert W[0, 5] == 0 and W[0, 4] > 0 and W[1, 3] == 0
    s, _ = simulate(preset("founders", refuge_frac=0.1), T=60, seed=7)
    assert _fp(s) == "cb6cb05790e67238" and "frac_in_refuge" in s


# ---------------------------------------------------------------- C1: R7 (refuge_pref + capacidad dura), mate_radius, O13b
def test_calhoun_C1_fingerprint_and_C0_unchanged():
    p = preset("calhoun_C1")
    assert (p.refuge_frac, p.refuge_capacity, p.refuge_pref, p.refuge_hard_mode) == (0.3, 2, 100.0, "strict")
    s, _ = simulate(p, T=120, seed=7)
    assert _fp(s) == "a8249d1bc00a849e"          # v2: capacidad estrictamente dura
    s, _ = simulate(preset("calhoun_C1", refuge_hard="legacy"), T=120, seed=7)
    assert _fp(s) == "d4ad46e33cf300a9"          # huella de v1 (commit 253b3fc), bit a bit
    s, _ = simulate(preset("calhoun_C0"), T=120, seed=7)
    assert _fp(s) == "3da3353ea899ac7d"


def test_refuge_hard_capacity_respected_after_move():
    # con una sola celda refugio y muchos que quieren entrar, entra a lo sumo `capacidad`
    p2 = Params(grid_size=5, n_males=0, n_females=9, refuge_frac=0.04, refuge_seed=1, refuge_pref=1e6,
                refuge_hard=True, refuge_capacity=2, initial_s=0.0)
    U = Universe(p2, rng=np.random.default_rng(0))
    rc = int(np.flatnonzero(U.refuge)[0])
    rx, ry = rc // 5, rc % 5
    nbrs = [(rx + dx, ry + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and 0 <= rx + dx < 5 and 0 <= ry + dy < 5]
    U.x = np.array([c[0] for c in nbrs][:9], np.int64)
    U.y = np.array([c[1] for c in nbrs][:9], np.int64)
    n = U.x.size
    for name in ("id", "male", "age", "s", "refr", "cap", "iso_run", "_m", "_cell", "conceived", "birth_t"):
        setattr(U, name, getattr(U, name)[:n])
    U.move()
    assert (U.x * 5 + U.y == rc).sum() == 2
    assert U.n_refuge_rejected == n - 2


def test_mate_radius():
    p = Params(grid_size=6, n_males=1, n_females=1, initial_age=10, p0=1.0)
    for radius, expect in ((0, 0), (1, 1)):
        U = Universe(p.replace(mate_radius=radius), rng=np.random.default_rng(0))
        U.x[:] = [2, 3]
        U.y[:] = [2, 3]
        cell = U.x * 6 + U.y
        U._m = np.ones(2, np.int64)
        U._cell = cell
        U.reproduce(cell)
        assert U.conceptions == expect
    # radio 0 es neutro (bit a bit) y radio 1 corre
    a, _ = simulate(Params(**SMALL, mate_radius=0), T=30, seed=3)
    b, _ = simulate(Params(**SMALL), T=30, seed=3)
    assert _fp(a) == _fp(b)
    s, _ = simulate(preset("founders_longevo", mate_radius=1), T=80, seed=7)
    assert _fp(s) == "13eda72e9a7cf90a"


def test_mate_radius_dominant():
    p = Params(grid_size=6, n_males=2, n_females=1, initial_age=10, p0=1.0, mate_radius=1,
               mating_rule="dominant")
    U = Universe(p, rng=np.random.default_rng(0))
    U.x[:] = [1, 3, 2]
    U.y[:] = [1, 3, 2]
    U.s = np.array([0.5, 0.9, 1.0])
    cell = U.x * 6 + U.y
    fem, fathers = U._fathers_moore(np.array([2]), np.bincount(cell[:2], minlength=36),
                                    np.r_[0, np.cumsum(np.bincount(cell[:2], minlength=36))[:-1]],
                                    np.array([0, 1]))
    assert fathers[0] == 1


def test_O13b_from_summary():
    p = Params()
    S = syn(np.r_[np.linspace(10, 1000, 41), np.linspace(1000, 10, 40)])
    ct = OB.characteristic_times(S, p)
    base = dict(o13b_n_iso=50, o13b_n_cr=60, o13b_t_star=50.0)
    assert OB.eval_O13b(S, p, ct, {**base, "o13b_delta_b": 0.2, "o13b_p_mw": 0.001, "o13b_f_late": 0.7}).status == "PASS"
    assert OB.eval_O13b(S, p, ct, {**base, "o13b_delta_b": 0.2, "o13b_p_mw": 0.001, "o13b_f_late": 0.3}).status == "MARG"
    assert OB.eval_O13b(S, p, ct, {**base, "o13b_delta_b": -0.1, "o13b_p_mw": 0.9, "o13b_f_late": 0.0}).status == "FAIL"
    assert OB.eval_O13b(S, p, ct, {**base, "o13b_delta_b": np.nan}).status == "NA"
    assert OB.eval_O13b(S, p, ct, {}).status == "NA"


def test_O13b_summary_keys_from_model():
    s, m = simulate(preset("calhoun_C1"), T=120, seed=7)
    for k in ("o13b_n_iso", "o13b_n_cr", "o13b_delta_b", "o13b_p_mw", "o13b_f_late"):
        assert k in m
    U = Universe(Params(**SMALL), rng=np.random.default_rng(0))
    assert (U.birth_t == -Params().initial_age).all()   # fundadores: -edad inicial


def test_calhoun_C2_maternal_basis_fingerprint():
    p = preset("calhoun_C2")
    assert (p.refuge_pref_basis, p.refuge_pref, p.refuge_pref_power) == ("maternal", 1000.0, 1.0)
    s, _ = simulate(p, T=120, seed=7)
    assert _fp(s) == "472f12aa511e71e9"          # v2 (strict)
    s, _ = simulate(p.replace(refuge_hard="legacy"), T=120, seed=7)
    assert _fp(s) == "866e1cb78550308b"          # v1


def test_h_mat_is_mothers_s_at_birth():
    U = Universe(preset("calhoun_C2"), rng=np.random.default_rng(1))
    assert (U.h_mat == 1.0).all()
    for _ in range(12):
        U.step()
    nb = U.birth_t >= 1
    assert nb.any() and (U.h_mat[nb] <= 1).all() and (U.h_mat[~nb] == 1).all()
    # con s_newborn_base = 0, s_nb = inherit_weight * h_mat al nacer (antes de actualizar s)
    s0, _ = simulate(preset("calhoun_C2", refuge_pref_basis="current"), T=40, seed=1)
    s1, _ = simulate(preset("calhoun_C1", refuge_pref=1000.0, refuge_pref_power=1.0), T=40, seed=1)
    assert _fp(s0) == _fp(s1)


def test_O6_detects_slow_regrowth_still_rising_at_censoring():
    """Punto ciego detectado en f2b_base (C1 rep 59): colapso a ~0.5 % de N_pk y rebrote
    lento hasta ~12 % de N_pk que sigue subiendo al censurar. find_peaks no lo ve (no hay caída posterior);
    el conteo por subidas desde el mínimo corriente sí."""
    t = np.arange(600)
    N = np.where(t <= 50, 400 + 46 * t, 0.0)
    N = np.where((t > 50) & (t <= 150), 2700 * np.exp(-(t - 50) / 18.0), N)
    N = np.where(t > 150, 13 + 0.7 * (t - 150), N)       # sube hasta ~330 (0.12 N_pk) en t = 599
    B = np.where(t < 45, 100.0, np.where(t > 300, 3.0, 0.0))
    S = syn(N, births=B)
    r = OB.evaluate_run(S, Params())
    assert r["O6"].status == "FAIL", r["O6"]
    assert "n_reb=1" in r["O6"].detail
    # caída monótona sin rebote: sigue pasando
    N2 = np.where(t <= 50, 400 + 46 * t, 2700 * np.exp(-(t - 50) / 60.0))
    r2 = OB.evaluate_run(syn(N2, births=np.where(t < 45, 100.0, 0.0)), Params())
    assert r2["O6"].status == "PASS", r2["O6"]


def test_calhoun_C1p_fingerprint():
    p = preset("calhoun_C1p")
    assert p == preset("calhoun_C1", crowd_aversion=0.0)
    s, _ = simulate(p, T=120, seed=7)
    assert _fp(s) == "da4484fbd2dd8093"          # v2 (strict)
    s, _ = simulate(p.replace(refuge_hard="legacy"), T=120, seed=7)
    assert _fp(s) == "b37566d372952dd0"          # v1
