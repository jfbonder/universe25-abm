"""Tests de la v3: R1 en la configuración de C1 y el borde a = a_w, O10 v2 sin PASS en vacío, o17p_from_summary
sin N0, O11 medido a la edad a_w y los insumos de `make figs`."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from u25 import Params, Universe, simulate
from u25 import objectives as OB
from u25.params import preset
from u25.stats import aw_obs_age

ROOT = Path(__file__).resolve().parents[1]


# =========================================================================== R1
class SocialProbe(Universe):
    """Registra, en cada actualización del estado social, cuántos agentes con edad > a_w subieron su s y las
    edades (al momento de la actualización) en las que hubo alguna suba."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_old = self.n_old_up = 0
        self.up_ages: set[int] = set()

    def update_social_state(self, m):
        s0, age = self.s.copy(), self.age.copy()
        super().update_social_state(m)
        up = self.s > s0 + 1e-15
        old = age > self.p.plast_age_max
        self.n_old += int(old.sum())
        self.n_old_up += int((up & old).sum())
        self.up_ages.update(np.unique(age[up]).tolist())


@pytest.mark.parametrize("name", ["calhoun_C1", "calhoun_C1p"])
def test_R1_no_rise_of_s_after_window_in_C1(name):
    """R1 pura en la configuración de producción (daño por hacinamiento activo, R2 social, AV, R7): ningún agente
    con edad > a_w aumenta s; la recuperación sólo ocurre en las edades 1..a_w."""
    p = preset(name)
    assert np.isfinite(p.plast_age_max) and p.plast_outside_rec == 0.0
    U = SocialProbe(p, rng=np.random.default_rng(5))
    U.run(200)
    assert U.n_old > 10_000                        # hubo muchos agentes-semana fuera de la ventana
    assert U.n_old_up == 0
    assert max(U.up_ages) == int(p.plast_age_max) and min(U.up_ages) >= 1


def test_R1_window_edge_exactly_aw_recovery_updates():
    """Borde a = a_w: una cría (edad 1 en su primera actualización, como en reproduce()) recibe recuperación en
    exactamente a_w actualizaciones (edades 1..a_w), es decir Λ = r·a_w; ninguna después."""
    a_w, r = 12, 0.01
    p = Params(grid_size=5, n_males=0, n_females=1, gamma=0.0, rec=r, social_rec_lambda=0.0,
               plast_age_max=float(a_w), plast_outside_rec=0.0, alpha=0.0, a_max=200.0)
    U = SocialProbe.from_agents(p, np.random.default_rng(0), male=[False], age=[1], s=[0.3])
    hist = []
    for _ in range(30):
        U.step()
        hist.append(float(U.s[0]))
    rises = np.flatnonzero(np.diff(np.r_[0.3, hist]) > 1e-15)
    assert rises.size == a_w and rises.tolist() == list(range(a_w))     # las primeras a_w actualizaciones
    assert U.up_ages == set(range(1, a_w + 1))
    assert hist[a_w - 1] == pytest.approx(0.3 + r * a_w)                 # Λ = r a_w
    assert hist[-1] == pytest.approx(hist[a_w - 1])                       # congelado después de a_w


# =========================================================================== O10 y O17p sin N0
def _syn(N, **cols):
    N = np.asarray(N, float)
    n = N.size
    d = dict(t=np.arange(n), N=N, births=np.r_[np.full(30, 50.0), np.zeros(n - 30)], N_adult=N / 2,
             occ_frac_cells=np.full(n, 0.5))
    d.update(cols)
    return pd.DataFrame(d)


def test_O10_nan_at_peak_is_not_a_vacuous_pass():
    p = preset("calhoun_C1")
    N = np.r_[np.linspace(400, 2000, 41), np.full(20, 2000), np.linspace(2000, 0, 40)]
    n = N.size
    sa = np.where(np.arange(n) >= 20, 0.3, 0.9)
    S = _syn(N, S_adult=sa)
    ct = OB.characteristic_times(S, p, peak_time="plateau_start")
    i = int(ct["t_pk"])
    assert OB.eval_O10(S, p, ct).status == "PASS"
    sa_nan = sa.copy()
    sa_nan[i] = np.nan                                     # S_adult(t_pk) indefinida
    S = _syn(N, S_adult=sa_nan)
    r = OB.eval_O10(S, p, OB.characteristic_times(S, p, peak_time="plateau_start"))
    assert r.status == "NA", r
    # se recupera antes del pico y S_adult(t_pk) = NaN: tampoco PASS (antes: PASS en vacío)
    sa2 = np.where((np.arange(n) >= 20) & (np.arange(n) < i - 1), 0.3, 0.9)
    sa2[i] = np.nan
    S = _syn(N, S_adult=sa2)
    r = OB.eval_O10(S, p, OB.characteristic_times(S, p, peak_time="plateau_start"))
    assert r.status != "PASS", r
    # en O17p un NA de O10 cuenta como FAIL
    res = OB.evaluate_run(S, p, {}, peak_time="plateau_start")
    assert res["O10"].status != "PASS" and res["O17p"].status != "PASS"


def test_o17p_from_summary_without_N0_counts_O1_O3_NA_as_fail():
    row = {f"obj_{k}": "PASS" for k in OB.ESSENTIAL + ["O13p"]}
    row.update({f"obj_{k}": "CLEAR" for k in OB.ANTI})
    df = pd.DataFrame([row, {**row, "obj_O1": "NA"}])
    assert OB.o17p_from_summary(df.assign(N0=100)).tolist() == ["PASS", "PASS"]
    assert OB.o17p_from_summary(df).tolist() == ["PASS", "FAIL"]


# =========================================================================== O11 a la edad a_w
def test_aw_obs_age():
    assert aw_obs_age(preset("calhoun_C1")) == 12
    assert aw_obs_age(Params()) == 12                                    # sin ventana finita: la de C1
    assert aw_obs_age(preset("calhoun_C1", plast_age_max=24.0)) == 24


def test_series_s_aw_is_s_after_the_window():
    """s_aw / n_aw: agentes con edad registrada a_w + 1 (ya recibieron las a_w actualizaciones de la ventana)."""
    p = preset("calhoun_C1")
    S, _ = simulate(p, T=60, seed=2)
    assert {"s_aw", "n_aw"} <= set(S.columns)
    tf = 13 - p.initial_age                                            # los fundadores pasan por a_w + 1 en tf
    assert int(S.n_aw[S.t == tf].iloc[0]) == p.n_males + p.n_females
    assert (S.n_aw[(S.t <= 12) & (S.t != tf)] == 0).all() and S.n_aw[S.t > 12].max() > 0   # crías: t > a_w
    U = Universe(p, rng=np.random.default_rng(2))
    for _ in range(40):
        U.step()
    from u25.stats import Recorder
    R = Recorder()
    R.record(U)
    m = U.age == 13
    assert R.rows[0]["n_aw"] == int(m.sum())
    if m.any():
        assert R.rows[0]["s_aw"] == pytest.approx(float(U.s[m].mean()))


def test_O11_age_option_default_unchanged_and_aw():
    p = preset("calhoun_C1")
    S, summ = simulate(p, T=200, seed=3)
    kw = dict(peak_time="plateau_start", o5_birth_ref="B99", o4_version="v2", crowd_class_threshold=8)
    r0 = OB.evaluate_run(S, p, summ, **kw)
    r6 = OB.evaluate_run(S, p, summ, o11_age="mature6", **kw)
    rw = OB.evaluate_run(S, p, summ, o11_age="aw", **kw)
    assert r0["O11"] == r6["O11"]                                          # default = v2
    assert rw["O11"].status == "PASS" and "edad=12" in rw["O11"].detail
    assert rw["O11"].value < r6["O11"].value                               # s sigue cayendo hasta a_w en C1
    with pytest.raises(ValueError):
        OB.validate_eval_kw(dict(o11_age="a12"))
    assert OB.validate_eval_kw(dict(o11_age="aw")) == dict(o11_age="aw")
    # sin las columnas s_aw/n_aw: NA (no PASS)
    assert OB.evaluate_run(S.drop(columns=["s_aw", "n_aw"]), p, summ, o11_age="aw", **kw)["O11"].status == "NA"


def test_O11_aw_synthetic_windows():
    """Con "aw" las cohortes tardías son las nacidas en [t_B50, t_B0] observadas a_w semanas después."""
    n = 140
    t = np.arange(n)
    N = np.r_[np.linspace(400, 2600, 50), np.full(n - 50, 2600.0)]
    B = np.where(t < 30, 100.0, 0.0)
    p = Params(plast_age_max=12.0)
    sm = np.clip(0.5 - 0.012 * t, 0.05, None)
    # cohortes que llegan a a_w en [t_B50 + 12, t_B0 + 12] con s < 0.2: PASS; s_mat (edad 6) alta: FAIL/MARG
    S = pd.DataFrame(dict(t=t, N=N, births=B, s_aw=np.where((t >= 12) & (t < 42), sm, np.nan),
                          n_aw=np.where((t >= 12) & (t < 42), 50.0, 0.0),
                          s_mat=np.full(n, 0.6), n_mat=np.where((t >= 6) & (t < 36), 50.0, 0.0)))
    ct = OB.characteristic_times(S, p)
    assert OB.eval_O11(S, p, ct, "aw").status == "PASS"
    assert OB.eval_O11(S, p, ct, "mature6").status != "PASS"
    # cohortes tardías que no llegan a a_w (ventana [t_B50 + 12, ...] vacía): NA
    S2 = S.assign(n_aw=np.where((t >= 12) & (t < 20), 50.0, 0.0))
    assert OB.eval_O11(S2, p, OB.characteristic_times(S2, p), "aw").status == "NA"


# =========================================================================== make figs
def test_fig_inputs_series_versioned_and_complete():
    """La serie reducida de la v2 que leen fig_trajectories, fig_cohorts y numbers_extra está versionada (no la
    excluye .gitignore) y tiene las columnas que esas funciones leen."""
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import fig_inputs as FI
    assert ("results_v2", "f2b_base") in FI.SERIES_STAGES
    f = ROOT / "results_v2" / "f2b_base" / "series_figs.parquet"
    assert f.exists() and f.stat().st_size < 6e6
    cols = set(pd.read_parquet(f).columns)
    assert {"point", "rep", "t", "N", "births", "S_mean", "S_adult", "mean_age"} <= cols
    ign = subprocess.run(["git", "check-ignore", "-q", str(f.relative_to(ROOT))], cwd=ROOT)
    assert ign.returncode == 1                                                # no ignorado


def test_figs_results_does_not_write_numbers_on_failure():
    """figs_results.py: con alguna salida fallida no reescribe results_numbers.json."""
    src = (ROOT / "scripts" / "figures" / "figs_results.py").read_text()
    main = src[src.index('if __name__ == "__main__":'):]
    i_fail, i_write = main.index("if fallas:"), main.index("NUMS.write_text")
    assert i_fail < i_write and "sys.exit(1)" in main[i_fail:i_write]
