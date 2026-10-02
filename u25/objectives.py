"""Evaluador automático de los objetivos O1–O17 y anti-objetivos A1–A4.

Especificación: la de los criterios de la Fase 2a, verificada contra Calhoun 1973 (O12 en t_D = 1.75 t_pk con
fallback early-D, O12b, ventana nueva de O11, O11b, O8b, O15b como (D) y R_B informativo en O14); los criterios
vigentes se describen en la versión extendida del manuscrito (manuscript/extended/sections/observations.tex y
appendix_results.tex). Las secciones citadas abajo (§0.3, §O6, etc.) remiten a la numeración de esa especificación.
Las elecciones de implementación, donde la especificación es ambigua, están marcadas "[impl]".

Estados
-------
Cada criterio por corrida devuelve un `Result(status, value, detail)`:
* objetivos: status ∈ {PASS, MARG, FAIL, NA, ERR};
* anti-objetivos: status ∈ {TRIGGERED, CLEAR, NA, ERR}.

Si falta una columna, el criterio da NA. Si un criterio lanza una excepción, da ERR (v2; antes NA), y ERR nunca
cuenta como aprobado: `calhoun_like` (O17_run) y O17p pasan a ERR.

v2
--
* O13p (aislados persistentes, antes en scripts/production/f2b_model.py) y O17p (resultado primario) se calculan
  acá: `eval_O13p`, `o17p_status` (por corrida) y `o17p_from_summary` (sobre las columnas obj_* de un
  summary.csv). O17p es estricto: exige PASS en todos los (E), salvo O1–O3, que pueden ser NA sólo si N0 > 20
  (exclusión declarada); cualquier otro NA cuenta como FAIL y cualquier ERR da ERR.
* O10 se evalúa sobre la s media de los adultos (S_adult) y O10b (D, nuevo) pide una fase B funcional; O11 ya no
  aprueba en vacío: la condición sobre las cohortes tardías (nacidas en la segunda mitad de la natalidad)
  necesita >= 30 animales maduros, si no O11 es NA.
* Opciones de re-puntuación: `peak_time` ("argmax" = t_pk; "plateau_start" = t′_pk del manuscrito, primera
  semana con N >= (1 - plateau_tol) N_pk; "plateau_end", última semana en esa banda; "plateau_mid") y
  `crowd_class_threshold` k (clase hacinada con m > k fijo en O13/O13p, columnas frac_crowded_low_m{k}).

Uso
---
    from u25.objectives import evaluate_run, evaluate_ensemble, transplant_status
    res = evaluate_run(series, params, summary)          # {"O1": Result, ..., "O13p": ..., "O17p": ...}
    res = evaluate_run(series, params, summary, peak_time="plateau_start", crowd_class_threshold=8)
    tab, per_run = evaluate_ensemble(series_df, summary_df, params_or_list,
                                     transplant_df=None, control_gini=None)
    df["obj_O17p"] = o17p_from_summary(df)               # sobre un summary.csv (columnas obj_*)
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.signal import find_peaks

from .params import Params
from .stats import aw_obs_age, gini

PASS, MARG, FAIL, NA, ERR = "PASS", "MARG", "FAIL", "NA", "ERR"
TRIG, CLEAR = "TRIGGERED", "CLEAR"

ESSENTIAL = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O10", "O11", "O13"]  # (E) por corrida (O12 aparte)
DESIRABLE = ["O8", "O8b", "O9", "O10b", "O11b", "O13b", "O14", "O15", "O15b"]
EXPLORATORY = ["O16"]
INFORMATIVE = ["R_B", "G_B"]          # no puntúan
RUN_OBJECTIVES = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O8b", "O9", "O10", "O10b", "O11", "O11b",
                  "O13", "O13p", "O13b", "O14", "O15", "O15b", "O16"]
O17P_NA_OK = ("O1", "O2", "O3")      # NA admitido en O17p sólo si N0 > 20 (latencia y fase B impuestas)
PEAK_TIMES = ("argmax", "plateau_start", "plateau_end", "plateau_mid")
COHORT_MIN_N = 30                     # O11: animales maduros mínimos en las cohortes tardías
ANTI = ["A1", "A2", "A3", "A4"]


class Result(NamedTuple):
    status: str
    value: float = float("nan")
    detail: str = ""


def _nan(x) -> bool:
    try:
        return x is None or (isinstance(x, (float, np.floating)) and math.isnan(x))
    except TypeError:
        return False


def _col(S: pd.DataFrame, name: str):
    return S[name].to_numpy(float) if name in S.columns else None


# ---------------------------------------------------------------- preprocesamiento (§0.2–0.3)
def centered_ma(x, w: int = 5) -> np.ndarray:
    """Media móvil centrada de w pasos con ventana truncada en los bordes."""
    x = np.asarray(x, float)
    n = x.size
    h = w // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def back_sum(x, w: int = 4) -> np.ndarray:
    """Suma hacia atrás de w pasos: sum_{u=t-w+1..t} x(u) (truncada al inicio)."""
    x = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = x.size
    lo = np.clip(np.arange(n) - w + 1, 0, n)
    return c[np.arange(n) + 1] - c[lo]


def _first(mask, t):
    i = np.flatnonzero(mask)
    return (float(t[i[0]]), int(i[0])) if i.size else (np.nan, -1)


O11_AGES = ("mature6", "aw")
EVAL_OPTIONS = dict(peak_time=PEAK_TIMES, o5_birth_ref=("last", "B99"), o4_version=("v1", "v2"), o11_age=O11_AGES)


def validate_eval_kw(kw: dict | None) -> dict:
    """Valida las opciones de evaluate_run (para fallar antes de un barrido y no con ERR en cada corrida)."""
    kw = dict(kw or {})
    allowed = set(EVAL_OPTIONS) | {"crowd_class_threshold", "plateau_tol"}
    bad = set(kw) - allowed
    if bad:
        raise ValueError(f"opciones del evaluador desconocidas: {sorted(bad)}")
    for k, vals in EVAL_OPTIONS.items():
        if k in kw and kw[k] not in vals:
            raise ValueError(f"{k}={kw[k]!r} inválido; opciones {vals}")
    if kw.get("crowd_class_threshold") is not None:
        kw["crowd_class_threshold"] = int(kw["crowd_class_threshold"])
    return kw


def peak_index(N, how: str = "argmax", tol: float = 0.02) -> int:
    """Índice del instante de pico: argmax de N, o el inicio / fin / centro de la meseta del pico (semanas
    con N >= (1 - tol) N_pk; primera y última de ellas). Ver `PEAK_TIMES`."""
    N = np.asarray(N, float)
    i = int(np.argmax(N))
    if how == "argmax":
        return i
    if how not in PEAK_TIMES:
        raise ValueError(f"peak_time inválido: {how!r}; opciones {PEAK_TIMES}")
    band = np.flatnonzero(N >= (1.0 - tol) * N[i])
    a, b = int(band[0]), int(band[-1])
    return {"plateau_start": a, "plateau_end": b, "plateau_mid": (a + b) // 2}[how]


def characteristic_times(S: pd.DataFrame, params: Params, peak_time: str = "argmax",
                         plateau_tol: float = 0.02) -> dict:
    """Instantes característicos (§0.3). `peak_time` elige el instante de pico t_pk (default: argmax de N);
    N_pk es siempre el máximo de N. t_pk_argmax, t_plateau_start y t_plateau_end se guardan siempre."""
    t = S["t"].to_numpy(float)
    N = S["N"].to_numpy(float)
    births = _col(S, "births")
    concs = _col(S, "conceptions")
    ct = dict(t=t, N=N)
    ct["Ns"] = centered_ma(N, 5)
    ct["B4"] = back_sum(births) if births is not None else None
    ct["C4"] = back_sum(concs) if concs is not None else None
    ct["Lbar"] = float(np.dot(params.litter_sizes, params.litter_probs))
    ct["K"] = params.grid_size ** 2 * params.m_c
    alive = np.flatnonzero(N > 0)
    ct["i_end"] = int(alive[-1]) if alive.size else 0
    ct["t_end"] = float(t[ct["i_end"]])
    ct["extinct"] = bool(N[-1] == 0)
    ct["N0"] = float(N[0])
    i_max = int(np.argmax(N))
    i_pk = peak_index(N, peak_time, plateau_tol)
    ct.update(i_pk=i_pk, t_pk=float(t[i_pk]), N_pk=float(N[i_max]), t_pk_argmax=float(t[i_max]),
              t_plateau_start=float(t[peak_index(N, "plateau_start", plateau_tol)]),
              t_plateau_end=float(t[peak_index(N, "plateau_end", plateau_tol)]), peak_time=peak_time)
    ct["t_lag"], ct["i_lag"] = _first(N >= 2 * N[0], t)
    if births is not None:
        ct["t_A"], _ = _first((births > 0) & (t > 0), t)
    else:
        ct["t_A"] = np.nan
    # natalidad
    for key, X in (("B", ct["B4"]), ("C", ct["C4"])):
        ct[f"t_{key}max"] = ct[f"i_{key}max"] = np.nan
        ct[f"t_{key}0"] = ct[f"t_{key}0_last"] = ct[f"t_{key}0_first"] = np.nan
        if X is None or X.max() <= 0:
            continue
        ib = int(np.argmax(X))
        thr = 0.01 * X.max()
        ct[f"t_{key}max"], ct[f"i_{key}max"] = float(t[ib]), ib
        above = np.flatnonzero(X >= thr)
        k = int(above[-1]) + 1
        ct[f"t_{key}0_last"] = float(t[k]) if k < t.size else np.nan
        low = X < thr
        f = np.flatnonzero(low[ib:])
        ct[f"t_{key}0_first"] = float(t[ib + f[0]]) if f.size else np.nan
        # sostenido 26 semanas o hasta la extinción (§0.3)
        tb0 = np.nan
        for i in range(ib, ct["i_end"] + 1):
            if not low[i]:
                continue
            j = min(i + 26, ct["i_end"])
            if low[i:j + 1].all():
                tb0 = float(t[i])
                break
        ct[f"t_{key}0"] = tb0
    S_mean = _col(S, "S_mean")
    ct["t_s05"] = _first(np.nan_to_num(S_mean, nan=1.0) < 0.5, t)[0] if S_mean is not None else np.nan
    S_ad = _col(S, "S_adult")
    ct["t_sad05"] = _first(np.nan_to_num(S_ad, nan=1.0) < 0.5, t)[0] if S_ad is not None else np.nan
    if births is not None and births.sum() > 0:      # mediana (O11 v2) y 99 % (O5, o5_birth_ref="B99")
        cb = np.cumsum(births)
        ct["t_B50"] = float(t[int(np.searchsorted(cb, 0.5 * cb[-1]))])
        ct["t_B99"] = float(t[int(np.searchsorted(cb, 0.99 * cb[-1]))])
    else:
        ct["t_B50"] = ct["t_B99"] = np.nan
    # ventanas
    if np.isfinite(ct["t_lag"]):
        t1 = max(ct["t_lag"], 1.0)
    else:
        t1 = ct["t_A"]
    ct["t1"] = t1
    ct["t2"] = _first((t > t1) & (N >= 0.3 * ct["N_pk"]), t)[0] if np.isfinite(t1) else np.nan
    ct["t3"] = _first(N >= 0.5 * ct["N_pk"], t)[0]
    ct["t4"] = _first(N >= 0.95 * ct["N_pk"], t)[0]
    return ct


def _at(ct, x, tt):
    """Valor de la serie x en el instante tt (o NaN)."""
    if x is None or _nan(tt):
        return np.nan
    i = np.searchsorted(ct["t"], tt)
    return float(x[i]) if i < len(x) and ct["t"][i] == tt else np.nan


# ---------------------------------------------------------------- objetivos por corrida
def eval_O1(S, p, ct):
    if ct["N0"] > 20:
        return Result(NA, np.nan, f"N0={ct['N0']:.0f} > 20 (latencia impuesta)")
    if _nan(ct["t_lag"]) or ct["t_pk"] <= 0:
        return Result(FAIL, np.nan, "t_lag NaN")
    r1 = ct["t_lag"] / ct["t_pk"]
    if 0.10 <= r1 <= 0.35:
        st = PASS
    elif 0.05 <= r1 < 0.10 or 0.35 < r1 <= 0.50:
        st = MARG
    else:
        st = FAIL
    return Result(st, r1, f"t_lag={ct['t_lag']:.0f} t_A={ct['t_A']}")


def _growth_fit(ct):
    t, N = ct["t"], ct["N"]
    if _nan(ct["t1"]) or _nan(ct["t2"]):
        return None
    m = (t >= ct["t1"]) & (t <= ct["t2"]) & (N > 0)
    if m.sum() < 4:
        return None
    n_dup = math.log2(_at(ct, N, ct["t2"]) / max(_at(ct, N, ct["t1"]), 1))
    res = _st.linregress(t[m], np.log(N[m]))
    return dict(R2=res.rvalue ** 2, r_B=res.slope, n_dup=n_dup, npts=int(m.sum()))


def eval_O2(S, p, ct):
    g = _growth_fit(ct)
    if g is None:
        return Result(NA, np.nan, "W_B con < 4 puntos")
    if g["n_dup"] < 1.5:
        return Result(NA, g["R2"], f"n_dup={g['n_dup']:.2f} < 1.5")
    st = PASS if g["R2"] >= 0.90 else MARG if g["R2"] >= 0.80 else FAIL
    Td = math.log(2) / g["r_B"] if g["r_B"] > 0 else np.nan
    return Result(st, g["R2"], f"r_B={g['r_B']:.3f} Td_B={Td:.1f} n_dup={g['n_dup']:.2f}")


def eval_O3(S, p, ct, o2):
    if o2.status == NA:
        return Result(NA, np.nan, "O2 NA")
    t3, t4 = ct["t3"], ct["t4"]
    if _nan(t3) or _nan(t4) or t4 <= t3:
        return Result(NA, np.nan, "t4 <= t3")
    g = _growth_fit(ct)
    rC = math.log(_at(ct, ct["N"], t4) / _at(ct, ct["N"], t3)) / (t4 - t3)
    rho = g["r_B"] / rC if rC > 0 else np.inf
    dC = (t4 - t3) / ct["t_pk"]
    if rho >= 2 and dC >= 0.10:
        st = PASS
    elif rho >= 1.5 and dC >= 0.05:   # [impl] ambas condiciones al menos marginales
        st = MARG
    else:
        st = FAIL
    return Result(st, rho, f"r_C={rC:.3f} d_C={dC:.3f}")


def eval_O4(S, p, ct, summary, version: str = "v1"):
    """O4. v1: rho_pk = N_pk/K (K = m_c L^2, "capacidad de umbral de daño") <= 0.8 PASS, <= 1 MARG.
    v2 (especificación §O4-v2): además espacio libre en el pico,
    f_0 = 1 - occ_frac_cells(t_pk). PASS si rho <= 0.8 y f_0 >= 0.10; MARG si rho <= 0.8 y 0.05 <= f_0 < 0.10, o si
    0.8 < rho <= 1 y f_0 >= 0.10; FAIL en otro caso o si la corrida explotó. NA si falta occ_frac_cells."""
    rho = ct["N_pk"] / ct["K"]
    if version == "v1":
        if summary.get("exploded", False) or rho > 1.0:
            st = FAIL
        elif rho <= 0.8:
            st = PASS
        else:
            st = MARG
        return Result(st, rho, "pico trivial (<0.05 K)" if rho < 0.05 else "")
    if version != "v2":
        raise ValueError(f"o4_version inválida: {version!r} (v1 | v2)")
    if summary.get("exploded", False):
        return Result(FAIL, rho, "exploded")
    if "occ_frac_cells" not in S:
        return Result(NA, rho, "falta occ_frac_cells")
    f0 = 1.0 - float(S["occ_frac_cells"].iloc[ct["i_pk"]])
    if rho <= 0.8 and f0 >= 0.10:
        st = PASS
    elif (rho <= 0.8 and 0.05 <= f0 < 0.10) or (0.8 < rho <= 1.0 and f0 >= 0.10):
        st = MARG
    else:
        st = FAIL
    return Result(st, rho, f"f0={f0:.3f}")


def eval_O5(S, p, ct, birth_ref: str = "last"):
    """O5. Fin de la natalidad: "last" (v1) = t_B0, cese sostenido de los nacimientos; "B99" (v2,
    especificación v2 §6) = t_B99, semana en que se acumula el 99 % de los nacimientos (la cola rala de las últimas
    crías no la mueve). Delta5 = (t_fin - t_pk)/t_pk, nu5 = N(t_fin)/N_pk; PASS si nu5 >= 0.7 y Delta5 <= 0.5, MARG si
    nu5 >= 0.5 y Delta5 <= 1. La versión v2 primaria combina "B99" con peak_time="plateau_start"."""
    if birth_ref not in ("last", "B99"):
        raise ValueError(f"o5_birth_ref inválido: {birth_ref!r} (last | B99)")
    tB0 = ct["t_B0"] if birth_ref == "last" else ct["t_B99"]
    if ct["B4"] is None:
        return Result(NA, np.nan, "falta births")
    if _nan(tB0):
        return Result(FAIL, np.nan, "natalidad no muere sostenidamente" if birth_ref == "last" else "sin nacimientos")
    if ct["t_Bmax"] > ct["t_pk"]:
        return Result(FAIL, np.nan, "t_Bmax > t_pk")
    d5 = (tB0 - ct["t_pk"]) / ct["t_pk"]
    nu5 = _at(ct, ct["N"], tB0) / ct["N_pk"]
    if nu5 >= 0.7 and d5 <= 0.5:
        st = PASS
    elif nu5 >= 0.5 and d5 <= 1.0:   # [impl] ambas al menos marginales
        st = MARG
    else:
        st = FAIL
    return Result(st, nu5, f"Delta5={d5:.3f}" + ("" if birth_ref == "last" else f" t_B99={tB0:.0f}"))


def _rises(x, amp) -> int:
    """Nº de subidas de x mayores que amp desde el mínimo corriente; tras cada una se exige volver a
    caer amp desde el máximo antes de contar otra (histéresis)."""
    cnt, up, ref = 0, False, x[0]
    for v in x:
        if not up:
            ref = min(ref, v)
            if v - ref >= amp:
                cnt += 1
                up, ref = True, v
        else:
            ref = max(ref, v)
            if ref - v >= amp:
                up, ref = False, v
    return cnt


def n_rebounds(ct) -> int:
    """Rebotes post-pico de Ñ (§O6a): picos con prominencia >= 0.10 N_pk (find_peaks), y además
    [corrección Fase 3] subidas >= 0.10 N_pk desde el mínimo corriente aunque no estén seguidas de una
    caída (rebote en curso al censurar) o cuya base derecha sea alta. find_peaks sólo no ve un rebrote
    lento que sigue subiendo en t_end: la prominencia de un pico se mide contra la base más alta de los
    dos lados, y el extremo final no es un pico. Se devuelve el máximo de ambos conteos."""
    m = ct["t"] > ct["t_pk"]
    m &= ct["t"] <= ct["t_end"]
    x = ct["Ns"][m]
    if x.size < 3:
        return 0
    amp = 0.10 * ct["N_pk"]
    pk, _ = find_peaks(x, prominence=amp)
    return int(max(pk.size, _rises(x, amp)))


def hysteresis_AH(ct):
    if ct["B4"] is None:
        return np.nan
    t, N = ct["t"], ct["N"]
    b = ct["B4"] / (4 * np.maximum(N, 1))
    ratios = []
    post = np.flatnonzero(t > ct["t_pk"])
    if post.size == 0:
        return np.nan
    for lv in (0.2, 0.35, 0.5):
        up = np.flatnonzero((t <= ct["t_pk"]) & (N >= lv * ct["N_pk"]))
        if up.size == 0:
            continue
        bu = b[up[0]]
        if bu <= 0:
            continue
        x = N[post] / ct["N_pk"]
        near = post[np.abs(x - lv) <= 0.05]
        if near.size:
            bd = b[near].mean()
        else:
            bd = b[post[np.argmin(np.abs(x - lv))]]
        ratios.append(bd / bu)
    return float(np.mean(ratios)) if ratios else np.nan


def eval_O6(S, p, ct):
    nreb = n_rebounds(ct)
    AH = hysteresis_AH(ct)
    ct["n_reb"] = nreb
    if _nan(AH):
        return Result(NA, nreb, "A_H NA")
    if nreb == 0 and AH <= 0.10:
        st = PASS
    elif nreb == 0 and AH <= 0.25:
        st = MARG
    else:
        st = FAIL
    return Result(st, AH, f"n_reb={nreb}")


def eval_O7(S, p, ct):
    return Result(PASS if ct["extinct"] else FAIL, ct["t_end"] if ct["extinct"] else np.nan,
                  "" if ct["extinct"] else "censurada")


def eval_O8(S, p, ct):
    if not ct["extinct"] or _nan(ct["t_B0"]):
        return Result(NA, np.nan, "no extinta o t_B0 NaN")
    age = _col(S, "mean_age")
    if age is None:
        return Result(NA, np.nan, "falta mean_age")
    T_ext = ct["t_end"] + 1 if ct["extinct"] else np.nan  # primer t con N=0
    rho8 = (T_ext - ct["t_pk"]) / ct["t_pk"]
    t = ct["t"]
    m = (t >= ct["t_B0"] + 4) & (t <= T_ext - 4) & np.isfinite(age)
    if m.sum() < 8:
        return Result(NA, rho8, "< 8 puntos para beta8")
    b8 = _st.linregress(t[m], age[m]).slope
    if rho8 >= 1.0 and b8 >= 0.5:
        st = PASS
    elif rho8 >= 0.5 and b8 >= 0.3:  # [impl] ambas al menos marginales
        st = MARG
    else:
        st = FAIL
    return Result(st, rho8, f"beta8={b8:.3f}")


def eval_O8b(S, p, ct):
    """nu8b = N(t_pk + round(0.31 t_pk)) / N_pk (declive inicial lento)."""
    tt = ct["t_pk"] + round(0.31 * ct["t_pk"])
    if tt > ct["t_end"]:
        if ct["extinct"]:
            nu = 0.0
        else:
            return Result(NA, np.nan, f"t={tt:.0f} > t_end (censurada)")
    else:
        nu = _at(ct, ct["N"], tt) / ct["N_pk"]
        if _nan(nu):
            return Result(NA, np.nan, "t no registrado")
    st = PASS if nu >= 0.80 else MARG if nu >= 0.60 else FAIL
    return Result(st, nu, f"t={tt:.0f}")


def eval_O9(S, p, ct):
    amin = _col(S, "age_min")
    if amin is None or "N_f" not in S:
        return Result(NA, np.nan, "falta age_min")
    t, N = ct["t"], ct["N"]
    idx = np.flatnonzero((t > ct["t_pk"]) & (N <= 0.05 * ct["N_pk"]) & (N > 0))
    if idx.size == 0:
        return Result(NA, np.nan, "N no cae a 5% con vivos")
    k = idx[0]
    t9 = t[k]
    f9 = S["N_f"].iloc[k] / N[k]
    if _nan(ct["t_B0"]):
        return Result(FAIL, f9, "t_B0 NaN")
    old = amin[k] >= 0.8 * (t9 - ct["t_B0"])
    st = PASS if (old and f9 > 0.5) else MARG if old else FAIL
    return Result(st, f9, f"age_min={amin[k]:.0f} t9-tB0={t9 - ct['t_B0']:.0f}")


def eval_O10_v1(S, p, ct):
    """O10 de v1: t(S_mean < 0.5) < t_pk. S_mean incluye a las crías, que nacen con s <= w s_f: pasa por
    dilución en cualquier colonia que crece rápido, también con gamma = 0. Sólo comparación."""
    ts = ct["t_s05"]
    if "S_mean" not in S:
        return Result(NA, np.nan, "falta S_mean")
    if _nan(ts):
        return Result(FAIL, np.nan, "S nunca < 0.5")
    d = (ct["t_pk"] - ts) / ct["t_pk"] if ct["t_pk"] > 0 else np.nan
    if ts < ct["t_pk"]:
        st = PASS
    elif abs(ts - ct["t_pk"]) <= 2:
        st = MARG
    else:
        st = FAIL
    return Result(st, d, f"t_s05={ts:.0f}")


def eval_O10(S, p, ct):
    """O10 (E, v2): muerte de la organización social antes del pico, medida en los ADULTOS (edad >= a_min):
    t_ad = primera semana con S_adult < 0.5. PASS si t_ad < t_pk y además S_adult(t_pk) < 0.5 (no es una caída
    transitoria); MARG si |t_ad - t_pk| <= 2 (sin la PASS anterior); FAIL en otro caso (t_ad > t_pk + 2,
    S_adult nunca baja de 0.5, o se recuperó antes del pico); NA si t_ad < t_pk pero S_adult(t_pk) es NaN (sin
    adultos en t_pk: no aprueba en vacío). Valor: (t_pk - t_ad)/t_pk."""
    if "S_adult" not in S:
        return Result(NA, np.nan, "falta S_adult")
    ts = ct["t_sad05"]
    if _nan(ts):
        return Result(FAIL, np.nan, "S_adult nunca < 0.5")
    sad_pk = _at(ct, _col(S, "S_adult"), ct["t_pk"])
    d = (ct["t_pk"] - ts) / ct["t_pk"] if ct["t_pk"] > 0 else np.nan
    det = f"t_ad={ts:.0f} S_adult(t_pk)={sad_pk:.2f}"
    if ts < ct["t_pk"] and _nan(sad_pk):
        # sin S_adult en t_pk no se puede verificar que la caída no sea transitoria -> NA (en O17p,
        # FAIL), nunca PASS en vacío
        return Result(NA, d, det + " (S_adult(t_pk) indefinida)")
    if ts < ct["t_pk"] and sad_pk < 0.5:
        st = PASS
    elif abs(ts - ct["t_pk"]) <= 2:
        st = MARG
    else:
        st = FAIL
    return Result(st, d, det)


def eval_O10b(S, p, ct):
    """O10b (D, v2): fase B funcional, S_adult >= 0.5 durante el crecimiento hasta 0.5 t_pk.
    rho10b = t_ad / t_pk; PASS si >= 0.5, MARG si >= 0.3, FAIL si no."""
    if "S_adult" not in S:
        return Result(NA, np.nan, "falta S_adult")
    if ct["t_pk"] <= 0:
        return Result(NA, np.nan, "t_pk = 0")
    ts = ct["t_sad05"]
    r = np.inf if _nan(ts) else ts / ct["t_pk"]
    st = PASS if r >= 0.5 else MARG if r >= 0.3 else FAIL
    return Result(st, r, f"t_ad={ts}")


def mann_kendall(y):
    """Test de Mann–Kendall (sin corrección por empates). Devuelve (S, z, p bilateral)."""
    y = np.asarray(y, float)
    n = y.size
    if n < 3:
        return np.nan, np.nan, np.nan
    d = np.sign(y[None, :] - y[:, None])
    Sst = float(np.triu(d, 1).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (Sst - np.sign(Sst)) / math.sqrt(var) if Sst != 0 else 0.0
    return Sst, z, float(2 * (1 - _st.norm.cdf(abs(z))))


def eval_O11_v1(S, p, ct):
    """O11 de v1 (sólo comparación): cohortes post nacidas en [0.8 t_pk, t_B0]; con la ventana vacía la
    condición se daba por cumplida."""
    proxy = False
    if "s_mat" in S and "n_mat" in S:
        sm, nm = _col(S, "s_mat"), _col(S, "n_mat")
    elif "s_q50" in S:
        sm, nm, proxy = _col(S, "s_q50"), np.full(len(S), 5.0), True
    else:
        return Result(NA, np.nan, "falta s_mat")
    t = ct["t"]
    t_hi = ct["t_B0"] if not _nan(ct["t_B0"]) else ct["t_end"]
    t_lo = ct["t_lag"] if not _nan(ct["t_lag"]) else 0.0
    m = (t >= t_lo) & (t <= t_hi) & (nm >= 5) & np.isfinite(sm)
    trend_ok = None
    zz = np.nan
    if m.sum() >= 3:
        Sst, zz, pv = mann_kendall(sm[m])
        trend_ok = bool(Sst < 0 and pv < 0.05)
    # cohortes NACIDAS en [0.8 t_pk, t_B0] -> maduran en [0.8 t_pk + a_min, t_B0 + a_min]
    a0 = p.repro_age_min
    hi = ct["t_B0"] + a0 if not _nan(ct["t_B0"]) else ct["t_pk"] + 26
    post = (t >= 0.8 * ct["t_pk"] + a0) & (t <= hi) & (nm > 0) & np.isfinite(sm)
    if post.any():
        spost = float(np.average(sm[post], weights=nm[post]))
        post_ok = spost < 0.2
    else:
        # [impl] sin cohortes que maduren en la ventana: no nació nadie -> se cumple
        spost, post_ok = np.nan, (True if trend_ok is not None else None)
    if trend_ok is None and post_ok is None:
        return Result(NA, np.nan, "sin cohortes")
    k = int(bool(trend_ok)) + int(bool(post_ok))
    st = PASS if k == 2 else MARG if k == 1 else FAIL
    return Result(st, spost, f"MK z={zz:.2f}" + (" proxy=s_q50" if proxy else ""))


def eval_O11(S, p, ct, age: str = "mature6"):
    """O11 (E, v2): fracaso progresivo de la socialización de las cohortes.
    (i) tendencia: Mann–Kendall decreciente (p < 0.05) de s_mat (s de los que cumplen a_min) en [t_lag, t_B0];
    (ii) cohortes tardías: las nacidas en la segunda mitad de la natalidad, [t_B50, t_B0] (t_B50 = semana en que
        los nacimientos acumulados llegan a la mitad; t_B0 o t_end si la natalidad no cesa), que maduran en
        [t_B50 + a_min, t_B0 + a_min]; s̄_mat,tarde (pesada por n_mat) < 0.2.
    (ii) necesita >= COHORT_MIN_N (30) animales maduros; si no, O11 es NA (no aprueba en vacío; en O17p un NA
    de O11 cuenta como FAIL). PASS si (i) y (ii); MARG si sólo una; FAIL si ninguna. Valor: s̄_mat,tarde.

    age: "mature6" (default v2) mide s a la edad a_min = 6 (columnas s_mat/n_mat),
    todavía dentro de la ventana R1; "aw" la mide al cierre de la ventana (columnas s_aw/n_aw: s después de las
    a_w actualizaciones de las edades 1..a_w, con a_w = plast_age_max, o 12 si no hay ventana finita; ver
    `stats.aw_obs_age`). Con "aw" las mismas cohortes de nacimiento se observan a_w semanas después de nacer:
    la tendencia se toma en [t_lag, t_B0] + (a_w - a_min) y las cohortes tardías en [t_B50 + a_w, t_B0 + a_w].
    Umbrales (0.2, p < 0.05, >= 30 animales) sin cambios."""
    if age not in O11_AGES:
        raise ValueError(f"o11_age inválida: {age!r} {O11_AGES}")
    a0 = p.repro_age_min
    if age == "aw":
        sc, nc, lag = "s_aw", "n_aw", aw_obs_age(p)
    else:
        sc, nc, lag = "s_mat", "n_mat", a0
    if sc not in S or nc not in S:
        return Result(NA, np.nan, f"falta {sc}")
    sm, nm = _col(S, sc), _col(S, nc)
    t = ct["t"]
    t_hi = ct["t_B0"] if not _nan(ct["t_B0"]) else ct["t_end"]
    t_lo = ct["t_lag"] if not _nan(ct["t_lag"]) else 0.0
    sh = lag - a0                     # 0 con "mature6": idéntico a v2
    m = (t >= t_lo + sh) & (t <= t_hi + sh) & (nm >= 5) & np.isfinite(sm)
    trend_ok, zz = False, np.nan
    if m.sum() >= 3:
        Sst, zz, pv = mann_kendall(sm[m])
        trend_ok = bool(Sst < 0 and pv < 0.05)
    if _nan(ct["t_B50"]):
        return Result(NA, np.nan, "sin nacimientos")
    late = (t >= ct["t_B50"] + lag) & (t <= t_hi + lag) & (nm > 0) & np.isfinite(sm)
    n_late = float(nm[late].sum())
    det = f"MK z={zz:.2f} n_tarde={n_late:.0f} t_B50={ct['t_B50']:.0f}" + (f" edad={lag}" if age == "aw" else "")
    if n_late < COHORT_MIN_N:
        return Result(NA, np.nan, det + f" (< {COHORT_MIN_N} maduros en las cohortes tardías)")
    s_late = float(np.average(sm[late], weights=nm[late]))
    k = int(trend_ok) + int(s_late < 0.2)
    st = PASS if k == 2 else MARG if k == 1 else FAIL
    return Result(st, s_late, det)


def eval_O11b(S, p, ct):
    """Fracción de hembras nacidas en [t_pk-7, t_pk] que concibieron alguna vez (hasta 48 sem.)."""
    if "coh_f_born" not in S or "coh_f_ever_conceived" not in S:
        return Result(NA, np.nan, "falta ever_conceived")
    t = ct["t"]
    m = (t >= ct["t_pk"] - 7) & (t <= ct["t_pk"])
    nb = float(S["coh_f_born"].to_numpy(float)[m].sum())
    if nb <= 0:
        return Result(NA, np.nan, "sin hembras nacidas en la ventana")
    f = float(S["coh_f_ever_conceived"].to_numpy(float)[m].sum()) / nb
    st = PASS if f <= 0.30 else MARG if f <= 0.50 else FAIL
    return Result(st, f, f"n_f={nb:.0f}")


def crowd_column(p, crowd_class_threshold=None) -> str:
    """Columna de la clase hacinada: frac_crowded_low (m > m_c) o, con umbral fijo k, frac_crowded_low_m{k}
    (si k == m_c ambas coinciden)."""
    k = crowd_class_threshold
    if k is None or (p is not None and int(k) == int(p.m_c)):
        return "frac_crowded_low"
    return f"frac_crowded_low_m{int(k)}"


def eval_O13(S, p, ct, crowd_col="frac_crowded_low"):
    need = ("frac_withdrawn", crowd_col)
    if any(c not in S for c in need):
        return Result(NA, np.nan, "faltan columnas")
    t = ct["t"]
    fw, fc = _col(S, "frac_withdrawn"), _col(S, crowd_col)
    nad = _col(S, "N_adult")
    best, ok_any, marg_any = np.nan, False, False
    for tt in (ct["t_pk"], ct["t_pk"] + 10, ct["t_pk"] + 20):
        i = np.searchsorted(t, tt)
        if i >= len(t) or t[i] != tt:
            continue
        if nad is not None and nad[i] < 20:
            continue
        if not (np.isfinite(fw[i]) and np.isfinite(fc[i])):
            continue
        best = np.nanmax([best, min(fw[i], fc[i])])
        if fw[i] >= 0.10 and fc[i] >= 0.10:
            ok_any = True
        elif 0.05 <= fw[i] < 0.10 and fc[i] >= 0.10:
            marg_any = True
    csm = _col(S, "corr_s_m")
    sign = False
    if csm is not None and not _nan(ct["t3"]) and not _nan(ct["t4"]):
        a = csm[(t >= ct["t3"]) & (t <= ct["t4"])]
        b = csm[(t >= ct["t_pk"] + 10) & (t <= ct["t_pk"] + 30)]
        if np.isfinite(a).any() and np.isfinite(b).any():
            sign = bool(np.nanmean(a) < -0.1 and np.nanmean(b) >= 0)
    if _nan(best) and not sign:
        return Result(NA, np.nan, "sin instantes con >= 20 adultos")
    st = PASS if ok_any else MARG if (marg_any or sign) else FAIL
    return Result(st, best, f"cambio de signo C_sm={sign}")


def eval_O13p(S, p, ct, crowd_col="frac_crowded_low"):
    """O13p (E en O17p): O13 con la clase aislada PERSISTENTE (phi_BO: s < 0.2 y m <= 2 sostenido 4 semanas)
    en lugar de la instantánea. En cada t de {t_pk, t_pk+10, t_pk+20} registrado
    y con >= 20 adultos: PASS si phi_BO >= 0.10 y Φ_CR >= 0.10 en alguno; MARG si 0.05 <= phi_BO < 0.10 y
    Φ_CR >= 0.10 en alguno; FAIL si no; NA si no hay ningún instante evaluable. Valor: max_t min(phi_BO, Φ_CR).
    Equivale a scripts/production/f2b_model.extras (obj_O13p) de v1."""
    if "phi_BO" not in S or crowd_col not in S:
        return Result(NA, np.nan, "faltan columnas phi_BO / " + crowd_col)
    t = ct["t"]
    bo, fc, nad = _col(S, "phi_BO"), _col(S, crowd_col), _col(S, "N_adult")
    seen = ok_any = marg_any = False
    best = np.nan
    for tt in (ct["t_pk"], ct["t_pk"] + 10, ct["t_pk"] + 20):
        i = int(np.searchsorted(t, tt))
        if i >= len(t) or t[i] != tt:
            continue
        if nad is not None and nad[i] < 20:
            continue
        if not (np.isfinite(bo[i]) and np.isfinite(fc[i])):
            continue
        seen = True
        best = np.nanmax([best, min(bo[i], fc[i])])
        if bo[i] >= 0.10 and fc[i] >= 0.10:
            ok_any = True
        elif 0.05 <= bo[i] < 0.10 and fc[i] >= 0.10:
            marg_any = True
    if not seen:
        return Result(NA, np.nan, "sin instantes con >= 20 adultos")
    return Result(PASS if ok_any else MARG if marg_any else FAIL, best, crowd_col)


def o17p_status(res: dict) -> Result:
    """O17p (resultado primario, estricto): PASS sólo si todos los (E) por corrida dan PASS (O1–O3 pueden ser
    NA si N0 > 20), O13p da PASS y ningún anti-objetivo se dispara ni es NA. Cualquier ERR -> ERR."""
    codes = ESSENTIAL + ["O13p"] + ANTI
    errs = [k for k in codes if k in res and res[k].status == ERR]
    if errs:
        return Result(ERR, np.nan, "ERR en " + ",".join(errs))
    N0 = res.get("_times", {}).get("N0", np.inf)
    bad = []
    for k in ESSENTIAL + ["O13p"]:
        st = res[k].status if k in res else NA
        if st == PASS or (st == NA and k in O17P_NA_OK and N0 > 20):
            continue
        bad.append(f"{k}={st}")
    for k in ANTI:
        st = res[k].status if k in res else NA
        if st != CLEAR:
            bad.append(f"{k}={st}")
    return Result(FAIL if bad else PASS, float(not bad), " ".join(bad))


def o17p_from_summary(df: pd.DataFrame, n0_col: str = "N0") -> pd.Series:
    """O17p estricto sobre las columnas obj_* de un summary.csv (una fila por corrida): 'PASS', 'FAIL' o 'ERR'.
    Usa obj_O1..obj_O13 (E), obj_O13p y obj_A1..A4. Una columna faltante, un NA (salvo O1–O3 con N0 > 20; sin
    columna N0, también los de O1–O3),
    o un valor vacío cuentan como FAIL; un 'ERR' o 'NA:<Error>' (excepción atrapada en v1) da ERR. Un NaN (lo
    que deja pandas al leer 'NA' sin keep_default_na=False) se trata como NA."""
    n = len(df)
    ok = np.ones(n, bool)
    err = np.zeros(n, bool)
    # sin la columna N0 no se puede verificar la exclusión de O1–O3: sus NA cuentan como FAIL
    N0 = pd.to_numeric(df[n0_col], errors="coerce").to_numpy(float) if n0_col in df else np.zeros(n)
    for k in ESSENTIAL + ["O13p"] + ANTI:
        c = f"obj_{k}"
        if c not in df:
            ok[:] = False
            continue
        v = df[c].astype(object).where(df[c].notna(), NA).astype(str).str.strip()   # NaN (lectura por
        err |= ((v == ERR) | v.str.startswith("NA:")).to_numpy()                       # defecto de "NA") -> NA
        if k in ANTI:
            ok &= (v == CLEAR).to_numpy()
        elif k in O17P_NA_OK:
            ok &= ((v == PASS) | ((v == NA) & (N0 > 20))).to_numpy()
        else:
            ok &= (v == PASS).to_numpy()
    return pd.Series(np.where(err, ERR, np.where(ok, PASS, FAIL)), index=df.index, name="obj_O17p")


def eval_O13b(S, p, ct, summary):
    """Aislados persistentes (ISO) nacidos más tarde que los hacinados (CR). Usa las claves o13b_* del
    resumen (snapshots de birth_t que toma el modelo en t_pk, t_pk+10, t_pk+20)."""
    if "o13b_n_iso" not in summary:
        return Result(NA, np.nan, "falta birth_t (o13b_*)")
    nI, nC = summary.get("o13b_n_iso", 0), summary.get("o13b_n_cr", 0)
    db = summary.get("o13b_delta_b", np.nan)
    if _nan(db):
        return Result(NA, np.nan, f"clases < 20 (ISO={nI}, CR={nC})")
    pv, fl = summary.get("o13b_p_mw", np.nan), summary.get("o13b_f_late", np.nan)
    det = f"p_MW={pv:.3g} f_late={fl:.2f} n_ISO={nI} n_CR={nC} t*={summary.get('o13b_t_star')}"
    if db <= 0 or not pv < 0.05:
        return Result(FAIL, db, det)
    if db >= 0.1 and fl >= 0.5:
        return Result(PASS, db, det)
    if fl < 0.5:
        return Result(MARG, db, det)
    return Result(FAIL, db, det)   # [impl] 0 < Δb < 0.1 con p < 0.05 y f_late >= 0.5: no cubierto -> FAIL


def eval_O14(S, p, ct):
    if "occ_var" not in S or "occ_frac_cells" not in S:
        return Result(NA, np.nan, "faltan columnas")
    i = ct["i_pk"]
    lam = ct["N_pk"] / p.grid_size ** 2
    var = float(S["occ_var"].iloc[i])
    ID = var / lam if lam > 0 else np.nan
    f0 = 1 - float(S["occ_frac_cells"].iloc[i])
    f0_ok = f0 >= 2 * math.exp(-lam) + 0.02
    if ID >= 2 and f0_ok:
        st = PASS
    elif ID >= 1.3:
        st = MARG
    else:
        st = FAIL
    mstar = lam + var / lam - 1 if lam > 0 else np.nan
    return Result(st, ID, f"m*={mstar:.1f} f0={f0:.3f} f0_Poisson={math.exp(-lam):.3f}")


def eval_O15(S, p, ct):
    if ct["B4"] is None or ct["C4"] is None:
        return Result(NA, np.nan, "faltan births/conceptions")
    t = ct["t"]
    C4, B4 = ct["C4"], ct["B4"]
    ok = C4 >= 3
    IM = np.full(len(t), np.nan)
    IM[ok] = 1 - B4[ok] / (ct["Lbar"] * C4[ok])
    t_hi = ct["t_B0"] if not _nan(ct["t_B0"]) else ct["t_end"]
    t_lo = ct["t_lag"] if not _nan(ct["t_lag"]) else 0.0
    m1 = (t >= t_lo) & (t <= t_hi) & np.isfinite(IM)
    if m1.sum() < 5:
        return Result(NA, np.nan, "< 5 puntos")
    rho = _st.spearmanr(t[m1], IM[m1]).statistic
    c1 = bool(np.isfinite(rho) and rho >= 0.5)
    c2, med = False, np.nan
    if not _nan(ct["t_B0"]):
        tc = ct["t_C0_last"] if not _nan(ct["t_C0_last"]) else ct["t_end"]
        m2 = (t >= ct["t_B0"]) & (t <= tc) & np.isfinite(IM)
        if m2.any():
            med = float(np.median(IM[m2]))
            c2 = med >= 0.9
    k = int(c1) + int(c2)
    st = PASS if k == 2 else MARG if k == 1 else FAIL
    o15b = np.nan
    if not _nan(ct["t_C0_last"]) and not _nan(ct["t_B0_last"]):
        o15b = (ct["t_C0_last"] - ct["t_B0_last"]) / ct["t_pk"]
    return Result(st, med, f"rho={rho:.2f} O15b={o15b:.3f}")


def eval_O15b(S, p, ct):
    """rho15b = (t_C0_last - t_B0_last)/t_pk: concepciones persisten tras la última cría viable."""
    if ct["B4"] is None or ct["C4"] is None:
        return Result(NA, np.nan, "faltan births/conceptions")
    if _nan(ct["t_C0_last"]) or _nan(ct["t_B0_last"]):
        return Result(NA, np.nan, "natalidad/concepciones no cesan")
    r = (ct["t_C0_last"] - ct["t_B0_last"]) / ct["t_pk"]
    st = PASS if r >= 0.1 else MARG if r > 0 else FAIL
    return Result(st, r, "")


def eval_RB(S, p, ct):
    """R_B = max/min (min forzado >= 1) de nacimientos acumulados por bloque en [t_lag, t_pk]. Informativo."""
    bcols = [c for c in S.columns if c.startswith("births_block_")]
    if not bcols or _nan(ct["t_lag"]):
        return Result(NA, np.nan, "faltan births_block_k o t_lag")
    m = (ct["t"] >= ct["t_lag"]) & (ct["t"] <= ct["t_pk"])
    v = S.loc[m, bcols].to_numpy(float).sum(0)
    rb = float(v.max() / max(v.min(), 1.0))
    return Result(PASS if rb >= 4 else FAIL, rb, "informativo (Calhoun ≈ 8.5)")


def eval_O16(S, p, ct):
    if not (p.defeat_gamma > 0 or p.territory_strength > 0):
        return Result(NA, np.nan, "sin agresión en el modelo")
    m, f = _col(S, "low_s_m_adult"), _col(S, "low_s_f_adult")
    if m is None or f is None:
        return Result(NA, np.nan, "faltan columnas")
    sel = ct["t"] >= ct["t_pk"]
    mm, ff = np.nanmean(m[sel]), np.nanmean(f[sel])
    if _nan(mm) or _nan(ff):
        return Result(NA, np.nan, "")
    return Result(PASS if mm > ff else FAIL, mm - ff, f"low_s M={mm:.2f} F={ff:.2f} (X, no puntúa)")


# ---------------------------------------------------------------- anti-objetivos
def eval_A1(S, p, ct):
    t = ct["t"]
    m = (t >= ct["t_pk"]) & (t <= ct["t_end"])
    pplat = float(np.mean(np.abs(ct["Ns"][m] / ct["N_pk"] - 1) <= 0.10)) if m.any() else 0.0
    trig = pplat >= 0.8 and (ct["t_end"] - ct["t_pk"]) >= 100 and ct["N_pk"] / ct["K"] >= 0.8
    return Result(TRIG if trig else CLEAR, pplat, "")


def eval_A2(S, p, ct):
    if ct["B4"] is None:
        return Result(NA, np.nan, "falta births")
    nreb = ct.get("n_reb", n_rebounds(ct))
    B4, t = ct["B4"], ct["t"]
    regrow = False
    if not _nan(ct["t_B0_first"]):
        regrow = bool(np.any((t > ct["t_B0_first"]) & (B4 >= 0.10 * B4.max())))
    trig = regrow or nreb >= 2
    return Result(TRIG if trig else CLEAR, nreb, f"rebrote_natalidad={regrow}")


def eval_A3(S, p, ct):
    if ct["B4"] is None or "N_repro_f" not in S:
        return Result(NA, np.nan, "faltan columnas")
    t = ct["t"]
    nf = _col(S, "N_repro_f")
    bf = np.where(nf > 0, ct["B4"] / (4 * np.maximum(nf, 1)), np.nan)
    if _nan(ct["t1"]) or _nan(ct["t2"]):
        return Result(NA, np.nan, "W_B indefinida")
    wb = (t >= ct["t1"]) & (t <= ct["t2"]) & np.isfinite(bf)
    if not wb.any():
        return Result(NA, np.nan, "W_B vacía")
    bB = float(np.median(bf[wb]))
    low = np.nan_to_num(bf, nan=0.0) < 0.5 * bB
    th = np.nan
    for i in np.flatnonzero(t > ct["t1"]):
        if i + 4 <= len(t) and low[i:i + 4].all():
            th = float(t[i])
            break
    if _nan(th):
        trig = ct["extinct"]  # [impl] nunca cae a la mitad y aun así se extinguió
    else:
        trig = th > ct["t_pk"] + 2
    return Result(TRIG if trig else CLEAR, th, f"b_B={bB:.3f}")


def eval_A4(S, p, ct):
    if not (p.social_hazard > 0 or p.social_mortality_factor > 0):
        return Result(CLEAR, 0.0, "NA estructural (sin mortalidad social)")
    ds, d = _col(S, "deaths_social_exp"), _col(S, "deaths")
    if ds is None or d is None:
        return Result(NA, np.nan, "falta deaths_social_exp")
    m = (ct["t"] >= ct["t_pk"]) & (ct["t"] <= ct["t_end"] + 1)
    tot = d[m].sum()
    phi = float(ds[m].sum() / tot) if tot > 0 else 0.0
    return Result(TRIG if phi >= 0.5 else CLEAR, phi, "")


# ---------------------------------------------------------------- corrida y ensamble
def evaluate_run(series: pd.DataFrame, params: Params | None = None, summary: dict | None = None,
                 peak_time: str = "argmax", crowd_class_threshold: int | None = None,
                 plateau_tol: float = 0.02, o5_birth_ref: str = "last", o4_version: str = "v1",
                 o11_age: str = "mature6") -> dict:
    """Evalúa O1–O16, O13p y A1–A4 sobre una corrida y arma calhoun_like (O17_run, regla de v1: NA cuenta
    como aprobado) y O17p (estricto, ver `o17p_status`). Devuelve {código: Result}.
    Una columna faltante da NA; una excepción dentro de un criterio da ERR (nunca PASS).
    peak_time / plateau_tol: instante de pico usado por todos los criterios normalizados por t_pk (default
    argmax; "plateau_start" es el t′_pk del manuscrito). crowd_class_threshold: umbral fijo k de la clase
    hacinada (m > k) en O13 y O13p, independiente de m_c (None = m_c, como en v1). O13b usa los snapshots
    del modelo (t_pk = argmax, m > m_c) y no cambia con estas opciones. o5_birth_ref: "last" (v1, t_B0) | "B99"
    (t_B99); la versión v2 primaria de O5 es o5_birth_ref="B99" con peak_time="plateau_start". o4_version: "v1" |
    "v2" (además f_0 >= 0.10 en t_pk). o11_age: "mature6" (default v2: s a la edad a_min = 6) | "aw" (s al cierre
    de la ventana R1, a la edad a_w), ver `eval_O11`."""
    p = params or Params()
    summary = summary or {}
    S = series.sort_values("t").reset_index(drop=True)
    ct = characteristic_times(S, p, peak_time=peak_time, plateau_tol=plateau_tol)
    ccol = crowd_column(p, crowd_class_threshold)
    out = {}

    def safe(code, f, *a):
        try:
            out[code] = f(*a)
        except Exception as e:  # noqa: BLE001 - un criterio roto no aprueba: ERR explícito
            out[code] = Result(ERR, np.nan, f"error: {type(e).__name__}: {e}")

    safe("O1", eval_O1, S, p, ct)
    safe("O2", eval_O2, S, p, ct)
    safe("O3", eval_O3, S, p, ct, out["O2"])
    safe("O4", eval_O4, S, p, ct, summary, o4_version)
    safe("O5", eval_O5, S, p, ct, o5_birth_ref)
    safe("O6", eval_O6, S, p, ct)
    safe("O7", eval_O7, S, p, ct)
    safe("O8", eval_O8, S, p, ct)
    safe("O8b", eval_O8b, S, p, ct)
    safe("O9", eval_O9, S, p, ct)
    safe("O10", eval_O10, S, p, ct)
    safe("O10b", eval_O10b, S, p, ct)
    safe("O11", eval_O11, S, p, ct, o11_age)
    safe("O11b", eval_O11b, S, p, ct)
    safe("O13", eval_O13, S, p, ct, ccol)
    safe("O13p", eval_O13p, S, p, ct, ccol)
    safe("O13b", eval_O13b, S, p, ct, summary)
    safe("O14", eval_O14, S, p, ct)
    safe("O15", eval_O15, S, p, ct)
    safe("O15b", eval_O15b, S, p, ct)
    safe("R_B", eval_RB, S, p, ct)
    safe("O16", eval_O16, S, p, ct)
    for a, f in (("A1", eval_A1), ("A2", eval_A2), ("A3", eval_A3), ("A4", eval_A4)):
        safe(a, f, S, p, ct)
    if any(out[k].status == ERR for k in ESSENTIAL + ANTI):
        out["calhoun_like"] = Result(ERR, np.nan, "ERR en un criterio")
    else:
        calhoun = all(out[k].status in (PASS, NA) for k in ESSENTIAL) and \
            not any(out[k].status == TRIG for k in ANTI)
        out["calhoun_like"] = Result(PASS if calhoun else FAIL, float(calhoun), "")
    # G_B (complemento de O14): Gini de nacimientos por bloque en [t_lag, t_pk]
    bcols = [c for c in S.columns if c.startswith("births_block_")]
    gb = np.nan
    if bcols and not _nan(ct["t_lag"]):
        m = (ct["t"] >= ct["t_lag"]) & (ct["t"] <= ct["t_pk"])
        gb = gini(S.loc[m, bcols].to_numpy(float).sum(0))
    out["G_B"] = Result(NA if _nan(gb) else PASS, gb, "informativo")
    out["_times"] = {k: ct[k] for k in ("t_pk", "N_pk", "t_lag", "t_A", "t_Bmax", "t_B0", "t_B0_last",
                                       "t_C0_last", "t_s05", "t_sad05", "t_B50", "t_B99", "t1", "t2", "t3", "t4",
                                       "t_end", "extinct", "N0", "t_pk_argmax", "t_plateau_start",
                                       "t_plateau_end")}
    out["O17p"] = o17p_status(out)
    return out


def flatten(res: dict) -> dict:
    row = {}
    for k, v in res.items():
        if k == "_times":
            row.update(v)
        else:
            row[k] = v.status
            row[f"{k}_val"] = v.value
            row[f"{k}_detail"] = v.detail
    return row


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return cen - hw, cen + hw


def kaplan_meier_pext(t_ext, censored_T):
    """P_ext(T) = 1 - S_KM(T) con IC de Greenwood. t_ext: tiempos (NaN = censurada en censored_T)."""
    t_ext = np.asarray(t_ext, float)
    Tc = np.asarray(censored_T, float) * np.ones_like(t_ext)
    times = np.where(np.isfinite(t_ext), t_ext, Tc)
    event = np.isfinite(t_ext)
    Smax = 1.0
    var_sum = 0.0
    for u in np.unique(times[event]):
        at_risk = (times >= u).sum()
        d = ((times == u) & event).sum()
        if at_risk > 0:
            Smax *= 1 - d / at_risk
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
    se = Smax * math.sqrt(var_sum)
    P = 1 - Smax
    return P, max(0.0, P - 1.96 * se), min(1.0, P + 1.96 * se)


def transplant_status(PB_succ, PD_succ) -> Result:
    """O12 a partir de las listas de éxitos (0/1) de transplantes de fase B y D."""
    b = np.asarray([x for x in PB_succ if not _nan(x)], float)
    d = np.asarray([x for x in PD_succ if not _nan(x)], float)
    if b.size == 0 or d.size == 0:
        return Result(NA, np.nan, "sin transplantes")
    PB, PD = b.mean(), d.mean()
    pv = _st.fisher_exact([[int(d.sum()), int(d.size - d.sum())],
                           [int(b.sum()), int(b.size - b.sum())]], alternative="less").pvalue
    det = f"P_T(B)={PB:.2f} (n={b.size}) P_T(D)={PD:.2f} (n={d.size}) p={pv:.3g}"
    if PB < 0.5:
        return Result(NA, PD, "control B fracasa: " + det)
    if PD <= 0.10 and pv < 0.05:
        return Result(PASS, PD, det)
    if PD <= 0.25 and pv < 0.05:
        return Result(MARG, PD, det)
    return Result(FAIL, PD, det)


def mixed_transplant_status(succ) -> Result:
    """O12b: transplante mixto (D + parejas sanas). PASS si P_T(mixto) <= 0.25."""
    x = np.asarray([v for v in succ if not _nan(v)], float)
    if x.size == 0:
        return Result(NA, np.nan, "sin transplantes mixtos")
    P = float(x.mean())
    return Result(PASS if P <= 0.25 else FAIL, P, f"P_T(mixto)={P:.2f} (n={x.size})")


def evaluate_ensemble(series_df: pd.DataFrame, summary_df: pd.DataFrame | None = None,
                      params: Params | list | None = None, transplant_df: pd.DataFrame | None = None,
                      control_gini=None, **eval_kw):
    """Evalúa todas las réplicas (columnas `point`, `rep`) y agrega por punto.

    Devuelve (tabla, per_run).
    - tabla: una fila por (point, código) con n_PASS, n_MARG, n_FAIL, n_NA, f = PASS/(PASS+MARG+FAIL),
      IC de Wilson y veredicto (E: f >= 0.8; D: f >= 0.5; 'MARG' si el IC contiene el umbral).
      Anti-objetivos: f = fracción TRIGGERED.
      O7 de ensamble con Kaplan–Meier (P_ext(T)). O12 con `transplant_df` (columnas point opcional,
      phase ∈ {B, D}, success). O14 complemento si se da `control_gini` (G_B de corridas con alpha=0).
      O17 = fracción de corridas tipo Calhoun (más O12 si está disponible).
    - per_run: una fila por réplica con estado, valor y detalle de cada criterio.
    """
    if "point" not in series_df:
        series_df = series_df.assign(point=0)
    if summary_df is not None and "point" not in summary_df:
        summary_df = summary_df.assign(point=0)
    plist = params if isinstance(params, list) else None
    rows = []
    for (pt, rp), S in series_df.groupby(["point", "rep"], sort=True):
        summ = {}
        if summary_df is not None:
            sm = summary_df[(summary_df.point == pt) & (summary_df.rep == rp)]
            if not sm.empty:
                summ = sm.iloc[0].to_dict()
        p = plist[int(pt)] if plist is not None else (params or Params())
        r = flatten(evaluate_run(S, p, summ, **eval_kw))
        r.update(point=pt, rep=rp, T_run=float(summ.get("T", S["t"].max())))
        rows.append(r)
    per = pd.DataFrame(rows)
    tab = []
    for pt, g in per.groupby("point"):
        for k in RUN_OBJECTIVES + ["R_B"] + ANTI:
            v = g[k]
            if k in ANTI:
                nT, nC, nN = (v == TRIG).sum(), (v == CLEAR).sum(), (v == NA).sum()
                n = nT + nC
                lo, hi = wilson(nT, n)
                tab.append(dict(point=pt, code=k, cls="A", n_PASS=np.nan, n_MARG=np.nan, n_FAIL=np.nan,
                                n_TRIG=nT, n_NA=nN, n=n, f=nT / n if n else np.nan, ci_lo=lo, ci_hi=hi,
                                verdict=("TRIGGERED" if n and nT / n >= 0.5 else "CLEAR") if n else NA))
                continue
            nP, nM, nF, nN, nE = [(v == s).sum() for s in (PASS, MARG, FAIL, NA, ERR)]
            n = nP + nM + nF + nE                 # ERR cuenta en el denominador (nunca como PASS)
            lo, hi = wilson(nP, n)
            cls = "E" if k in ESSENTIAL else "D" if k in DESIRABLE else "I" if k in INFORMATIVE else "X"
            thr = 0.8 if cls == "E" else 0.5
            f = nP / n if n else np.nan
            if n == 0:
                verdict = NA
            elif f >= thr:
                verdict = PASS
            elif lo <= thr <= hi:
                verdict = MARG
            else:
                verdict = FAIL
            tab.append(dict(point=pt, code=k, cls=cls, n_PASS=nP, n_MARG=nM, n_FAIL=nF, n_TRIG=np.nan,
                            n_NA=nN, n_ERR=nE, n=n, f=f, ci_lo=lo, ci_hi=hi, verdict=verdict))
        # O7 por Kaplan–Meier
        te = np.where(g["extinct"].astype(bool), g["t_end"] + 1, np.nan)
        P, lo, hi = kaplan_meier_pext(te, g["T_run"].to_numpy())
        tab.append(dict(point=pt, code="O7_KM", cls="E", n=len(g), f=P, ci_lo=lo, ci_hi=hi,
                        verdict=PASS if P >= 0.9 else MARG if P >= 0.8 else FAIL))
        # O12
        o12 = None
        if transplant_df is not None:
            tr = transplant_df if "point" not in transplant_df else transplant_df[transplant_df.point == pt]
            o12 = transplant_status(tr.loc[tr.phase == "B", "success"], tr.loc[tr.phase == "D", "success"])
            if "early_D" in tr and tr.loc[tr.phase == "D", "early_D"].astype(bool).any():
                o12 = o12._replace(detail=o12.detail + f" early-D en {int(tr.loc[tr.phase == 'D', 'early_D'].astype(bool).sum())}")
            tab.append(dict(point=pt, code="O12", cls="E", n=len(tr), f=o12.value, verdict=o12.status,
                            detail=o12.detail))
            mx = tr[tr.phase.astype(str).str.startswith("MIX")]
            if len(mx):
                o12b = mixed_transplant_status(mx["success"])
                tab.append(dict(point=pt, code="O12b", cls="D", n=len(mx), f=o12b.value,
                                verdict=o12b.status, detail=o12b.detail))
        # O14 complemento (G_B vs control alpha=0)
        if control_gini is not None and g["G_B_val"].notna().any():
            gb = g["G_B_val"].dropna()
            cg = np.asarray(control_gini, float)
            cg = cg[np.isfinite(cg)]
            if cg.size:
                pv = _st.mannwhitneyu(gb, cg, alternative="greater").pvalue
                ok = (gb.median() - np.median(cg) >= 0.05) and pv < 0.05
                tab.append(dict(point=pt, code="O14_GB", cls="D", n=len(gb), f=float(gb.median()),
                                verdict=PASS if ok else FAIL,
                                detail=f"med G_B={gb.median():.3f} control={np.median(cg):.3f} p={pv:.3g}"))
        # O17
        cl = g["calhoun_like"] == PASS
        k = int(cl.sum())
        n = len(g)
        lo, hi = wilson(k, n)
        fC = k / n if n else np.nan
        if o12 is not None and o12.status not in (PASS, NA):
            fC_eff, note = 0.0, "O12 no se cumple en el punto"
        else:
            fC_eff, note = fC, "" if o12 is not None else "sin O12"
        tab.append(dict(point=pt, code="O17", cls="E", n=n, n_PASS=k, f=fC_eff, ci_lo=lo, ci_hi=hi,
                        verdict=PASS if fC_eff >= 0.8 else MARG if fC_eff >= 0.5 else FAIL, detail=note))
        # O17p (estricto; con O12 si está disponible, igual que O17)
        cp = g["O17p"]
        k, nE = int((cp == PASS).sum()), int((cp == ERR).sum())
        lo, hi = wilson(k, n)
        fP = 0.0 if (o12 is not None and o12.status not in (PASS, NA)) else (k / n if n else np.nan)
        tab.append(dict(point=pt, code="O17p", cls="E", n=n, n_PASS=k, n_ERR=nE, f=fP, ci_lo=lo, ci_hi=hi,
                        verdict=PASS if fP >= 0.8 else MARG if fP >= 0.5 else FAIL, detail=note))
    return pd.DataFrame(tab), per


def lhs_volume(tab: pd.DataFrame, thr: float = 0.8, n_boot: int = 2000, seed: int = 0):
    """O17 por diseño: fracción de puntos con f_Calhoun >= thr (IC bootstrap sobre puntos)."""
    f = tab.loc[tab.code == "O17", "f"].to_numpy(float)
    if f.size == 0:
        return np.nan, np.nan, np.nan
    ok = (f >= thr).astype(float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(ok, size=(n_boot, ok.size)).mean(1)
    return float(ok.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
