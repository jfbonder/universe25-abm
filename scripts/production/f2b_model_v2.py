"""Clase de modelo de la V2 (producción): `Universe` + observables y re-puntuaciones por corrida.

No cambia la dinámica ni consume números aleatorios. Sobre la serie a resolución completa (antes de que el runner
la compacte) agrega al resumen:

  * las columnas de la v1 (f2b_model.extras: x_rho_pk, x_f0_pk, x_BO4, x_CR, x_D_last, ...) y obj_O13p sólo si
    el evaluador de u25 no lo calcula (si lo calcula, la columna del evaluador manda: fuente única);
  * meseta: x_tpk2 = t'_pk (primera semana con N >= 0.98 N_max), x_tleave (última), x_tmid (centro), x_plateau;
  * re-puntuación con t'_pk (prefijo obj_tp_) y con el centro de la meseta (obj_tm_) de O4–O11, O13, O13p,
    A1–A4, O17_run y O17p: con la API de u25 si v2_api.json la declara (evaluador.tpk_kw), si no
    con un fallback que reemplaza t_pk (no N_pk) en characteristic_times;
  * clase hacinada con umbral FIJO m > m_fijo (8): obj_mf_O13, obj_mf_O13p, obj_mf_O17_run, obj_mf_O17p (y
    obj_tpmf_O17p, ambas variantes juntas), con evaluate_run(crowd_class_threshold=8) y la columna
    frac_crowded_low_m8 de u25 si existen; si no, con la serie propia fcr_mfix;
  * fracciones de clase por instante para re-puntuar sin simular con otro umbral θ: x_BO4_k, x_BOi_k,
    x_CR_k, x_CRf_k, x_nad_k para k = 0, 10, 20 semanas después de t_pk;
  * x_n_last1 (nacimientos de la última semana con nacimientos) y x_n_last4, x_t_s05ad (primera semana con
    S adulta < 0,5), x_tB90 (semana del 90 % de los nacimientos acumulados), x_sd_tlast no (es de ensamble);
  * serie: fcr_mfix, ref_young (fracción de los ocupantes de refugios con edad < a_w) y S_juv.

O17p (criterio primario): si u25 lo devuelve (evaluador.o17p_key) se usa ese; si no, O17p = O17_run PASS y O13p
PASS, con O13p NA -> FAIL (en la v1 el NA contaba como PASS y no había ningún NA).
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import u25.objectives as OBJ
from u25.model import Universe

from f2b_model import EXTRA_KEYS as KEYS_V1
from f2b_model import extras as extras_v1

API = json.load(open(Path(__file__).with_name("v2_api.json")))
EV = API.get("evaluador", {})
M_FIX = int(EV.get("m_fijo", 8))
OFFS = (0, 10, 20)
SERIES_V2 = ["fcr_mfix", "ref_young", "S_juv"]
VAR_CODES = ["O4", "O5", "O6", "O7", "O8", "O8b", "O9", "O10", "O11", "O13", "A1", "A2", "A3", "A4"]
ESS = list(OBJ.ESSENTIAL)
ANTI = list(OBJ.ANTI)


def _has_kw(fn, kw: dict | None) -> bool:
    if not kw:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return all(k in sig.parameters for k in kw) or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def plateau(t, N, tol=0.02):
    """t'_pk (primera semana con N >= (1-tol) N_max), última semana en la banda, centro y duración."""
    t = np.asarray(t, float); N = np.asarray(N, float)
    if not len(N) or N.max() <= 0:
        return dict(tpk2=np.nan, tleave=np.nan, tmid=np.nan, dur=np.nan)
    idx = np.flatnonzero(N >= (1 - tol) * N.max())
    a, b = float(t[idx[0]]), float(t[idx[-1]])
    return dict(tpk2=a, tleave=b, tmid=float(np.floor((a + b) / 2)), dur=b - a)


def _ct_with_tpk(tsel):
    """characteristic_times con t_pk = tsel (N_pk sigue siendo el máximo). Las ventanas t1–t4, t_s05, t_B0, …
    no dependen de t_pk en u25.objectives, así que basta reemplazar i_pk y t_pk."""
    orig = OBJ.characteristic_times

    def ct(S, p):
        c = orig(S, p)
        i = int(np.searchsorted(c["t"], tsel))
        i = min(max(i, 0), len(c["t"]) - 1)
        c.update(i_pk=i, t_pk=float(c["t"][i]))
        return c
    return ct


def _evaluate(series, p, summ, tsel=None, kw=None):
    """evaluate_run normal, con la opción de la API (kw) o con el fallback de t_pk = tsel."""
    if tsel is None:
        return OBJ.evaluate_run(series, p, summ)
    if kw and _has_kw(OBJ.evaluate_run, kw):
        return OBJ.evaluate_run(series, p, summ, **kw)
    orig = OBJ.characteristic_times
    OBJ.characteristic_times = _ct_with_tpk(tsel)
    try:
        return OBJ.evaluate_run(series, p, summ)
    finally:
        OBJ.characteristic_times = orig


def _o13p_from(S, t_pk, fc_col="frac_crowded_low", thr=0.10):
    """O13 persistente (mismo esquema que f2b_model.extras) en t_pk, +10, +20."""
    t = S["t"].to_numpy(float)
    bo4 = S["phi_BO"].to_numpy(float) if "phi_BO" in S else None
    fc = S[fc_col].to_numpy(float) if fc_col in S else None
    nad = S["N_adult"].to_numpy(float) if "N_adult" in S else None
    if bo4 is None or fc is None or not np.isfinite(t_pk):
        return "NA"
    seen = ok = marg = False
    for tt in (t_pk, t_pk + 10, t_pk + 20):
        i = int(np.searchsorted(t, tt))
        if i >= len(t) or t[i] != tt or (nad is not None and nad[i] < 20):
            continue
        if not (np.isfinite(bo4[i]) and np.isfinite(fc[i])):
            continue
        seen = True
        if bo4[i] >= thr and fc[i] >= thr:
            ok = True
        elif 0.5 * thr <= bo4[i] < thr and fc[i] >= thr:
            marg = True
    if not seen:
        return "NA"
    return "PASS" if ok else ("MARG" if marg else "FAIL")


def _o17(res: dict) -> str:
    calh = all(res[k].status in (OBJ.PASS, OBJ.NA) for k in ESS if k in res) and \
        not any(res[k].status == OBJ.TRIG for k in ANTI if k in res)
    return "PASS" if calh else "FAIL"


def _o17p(o17: str, o13p: str) -> str:
    return "PASS" if (o17 == "PASS" and o13p == "PASS") else "FAIL"


def primary_kw() -> dict:
    """kwargs del evaluador primario (v2_api.json: evaluador.primario)."""
    return {k: v for k, v in EV.get("primario", {}).items() if not k.startswith("_")}


def variant_kw(pref) -> dict:
    v = dict(EV["variantes"][pref])
    base = v.pop("_base", "primario")
    kw = primary_kw() if base == "primario" else {}
    kw.update(v)
    return kw


def evaluate_with(S, p, summ, kw: dict):
    """evaluate_run con las opciones kw; '_criterios_v1' reemplaza O10 y O11 por las versiones v1 y rehace
    O17_run y O17p (evaluador v1 completo)."""
    kw = dict(kw)
    v1 = bool(kw.pop("_criterios_v1", False))
    kw = {k: v for k, v in kw.items() if not k.startswith("_")}
    if hasattr(OBJ, "validate_eval_kw"):
        OBJ.validate_eval_kw(kw)
    r = OBJ.evaluate_run(S, p, summ, **kw)
    if v1 and hasattr(OBJ, "eval_O10_v1") and hasattr(OBJ, "eval_O11_v1"):
        Ss = S.sort_values("t").reset_index(drop=True)
        ckw = {k: kw[k] for k in ("peak_time", "plateau_tol") if k in kw}
        ct = OBJ.characteristic_times(Ss, p, **ckw)
        for code, fn in (("O10", OBJ.eval_O10_v1), ("O11", OBJ.eval_O11_v1)):
            try:
                r[code] = fn(Ss, p, ct)
            except Exception as e:  # noqa: BLE001
                r[code] = OBJ.Result(getattr(OBJ, "ERR", "NA"), np.nan, f"error: {e}")
        if hasattr(OBJ, "o17p_status"):
            r["O17p"] = OBJ.o17p_status(r)
    return r


def order_parameters(S, p, pl) -> dict:
    """Parámetros de orden continuos por corrida (especificación de la calibración)."""
    o = {}
    t = S["t"].to_numpy(float); N = S["N"].to_numpy(float)
    G2 = p.grid_size ** 2
    Npk = float(N.max()) if len(N) else np.nan
    fr = float(getattr(p, "refuge_frac", 0.0) or 0.0)
    KR = p.m_c * (1 - fr) * G2 + min(getattr(p, "refuge_capacity", 1), p.m_c) * fr * G2 if fr > 0 else p.m_c * G2
    o["x_rhoR_pk"] = Npk / KR if KR > 0 else np.nan

    def at(col, tt):
        if col not in S or not np.isfinite(tt):
            return np.nan
        i = int(np.searchsorted(t, tt))
        return float(S[col].iloc[i]) if i < len(t) and t[i] == tt else np.nan
    o["x_f0_tp"] = 1.0 - at("occ_frac_cells", pl["tpk2"]) if "occ_frac_cells" in S else np.nan
    o["x_h_tp"] = at("h", pl["tpk2"])
    if "h" in S:
        j = np.flatnonzero((t >= 3) & (S["h"].to_numpy(float) < 0.5))
        o["x_t_h"] = float(t[j[0]]) if j.size else np.nan
    if "f_over" in S:
        j = np.flatnonzero(np.nan_to_num(S["f_over"].to_numpy(float)) >= 0.05)
        o["x_t_on"] = float(t[j[0]]) if j.size else np.nan
    if "births" in S and S["births"].sum() > 0:
        cb = np.cumsum(S["births"].to_numpy(float))
        q = {qq: float(t[int(np.searchsorted(cb, qq * cb[-1]))]) for qq in (0.9, 0.99)}
        o["x_tB99"] = q[0.99]
        o["x_tauB"] = (q[0.99] - q[0.9]) / math.log(10)
    if "S_adult" in S and len(N) and N[0] > 0:
        j = np.flatnonzero(np.nan_to_num(S["S_adult"].to_numpy(float), nan=1.0) < 0.5)
        o["x_GB"] = float(np.log2(N[j[0]] / N[0])) if j.size and N[j[0]] > 0 else np.nan
    return o


def extras_v2(S, p, summ) -> dict:
    out = {}
    try:
        out.update(extras_v1(S, p, summ))
    except Exception as e:  # noqa: BLE001
        out.update({k: np.nan for k in KEYS_V1})
        out["obj_O13p"] = f"NA:{type(e).__name__}"
    t = S["t"].to_numpy(float)
    N = S["N"].to_numpy(float)
    if len(t) < 2:
        return out
    ipk = int(np.argmax(N)); tpk = float(t[ipk])
    pl = plateau(t, N)
    t0 = {"argmax": tpk, "plateau_start": pl["tpk2"], "plateau_end": pl["tleave"],
          "plateau_mid": pl["tmid"]}.get(primary_kw().get("peak_time", "argmax"), tpk)   # t_pk primario
    out.update(x_tpk2=pl["tpk2"], x_tleave=pl["tleave"], x_tmid=pl["tmid"], x_plateau=pl["dur"])
    # fracciones de clase por instante (re-puntuación con otro θ sin simular)
    crf = EV.get("m_fijo_col") if EV.get("m_fijo_col") in S else "fcr_mfix"     # clase hacinada m > 8 fijo
    cols = dict(BO4="phi_BO", BOi="frac_withdrawn", CR="frac_crowded_low", CRf=crf, nad="N_adult")
    for k in OFFS:
        i = int(np.searchsorted(t, t0 + k))
        hit = i < len(t) and t[i] == t0 + k
        for nm, c in cols.items():
            out[f"x_{nm}_{k}"] = float(S[c].iloc[i]) if (hit and c in S) else np.nan
    # natalidad: última semana, 90 % acumulado, S adulta < 0.5
    if "births" in S:
        B = S["births"].to_numpy(float)
        if (B > 0).any():
            il = int(np.flatnonzero(B > 0)[-1])
            out["x_n_last1"] = float(B[il])
            out["x_n_last4"] = float(B[(t > t[il] - 4) & (t <= t[il])].sum())
            cb = np.cumsum(B)
            out["x_tB90"] = float(t[int(np.searchsorted(cb, 0.9 * cb[-1]))])
            out["x_tB50"] = float(t[int(np.searchsorted(cb, 0.5 * cb[-1]))])
    if "S_adult" in S:
        sa = S["S_adult"].to_numpy(float)
        j = np.flatnonzero(np.nan_to_num(sa, nan=1.0) < 0.5)
        out["x_t_s05ad"] = float(t[j[0]]) if j.size else np.nan
    # Las columnas obj_* las escribe el barrido con el evaluador PRIMARIO (eval_kw = evaluador.primario): fuente
    # única. Acá: (a) parámetros de orden de la evaluación primaria; (b) las variantes como obj_<prefijo>_*.
    try:
        rp = evaluate_with(S, p, summ, primary_kw())
        out["x_Psi"] = float(rp["O13p"].value) if "O13p" in rp else np.nan        # Ψ = max_t min(Φ_pers, Φ_cr)
        tm = rp.get("_times", {})
        for k_ in ("t_pk", "t_B99", "t_B50", "t_sad05"):
            if k_ in tm:
                out[f"x_{k_}_prim"] = float(tm[k_]) if tm[k_] == tm[k_] else np.nan
    except Exception as e:  # noqa: BLE001
        out["x_Psi"] = np.nan
        out["x_err_primario"] = f"{type(e).__name__}: {e}"[:120]
    out.update(order_parameters(S, p, pl))
    for pref in EV.get("variantes", {}):
        kw = variant_kw(pref)
        try:
            r = evaluate_with(S, p, summ, kw)
        except Exception as e:  # noqa: BLE001 - p. ej. una opción que esta versión de u25 no tiene
            out[f"obj_{pref}_O17p"] = f"ERR:{type(e).__name__}"
            continue
        for c in VAR_CODES + ["O13p", "O17p", "O10b"]:
            if c in r:
                out[f"obj_{pref}_{c}"] = r[c].status
        out[f"obj_{pref}_O17_run"] = _o17(r)
        if "O17p" not in r:
            out[f"obj_{pref}_O17p"] = _o17p(out[f"obj_{pref}_O17_run"], str(out.get(f"obj_{pref}_O13p", "NA")))
    return out


class Extras2bV2:
    """Mixin: observables de serie (fcr_mfix, ref_young, S_juv) y columnas de resumen de la V2."""

    def _measure_post_move(self):
        super()._measure_post_move()
        p = self.p
        m = self._m
        n = m.shape[0]
        val = np.nan
        if n and n == self.N:
            adult = self.age >= p.repro_age_min
            na = int(adult.sum())
            if na:
                low = adult & (self.s < p.obs_s_low)
                val = float((low & (m > M_FIX)).sum() / na)
        self.obs["fcr_mfix"] = val

    def extra_stats(self) -> dict:
        d = dict(super().extra_stats())
        d["fcr_mfix"] = self.obs.get("fcr_mfix", np.nan) if self.t > 0 else np.nan
        n = self.N
        if n:
            juv = self.age < self.p.repro_age_min
            d["S_juv"] = float(self.s[juv].mean()) if juv.any() else np.nan
            ry = np.nan
            if getattr(self, "refuge", None) is not None:
                G = self.p.grid_size
                inr = self.refuge[self.x * G + self.y]
                if inr.any():
                    aw = self.p.plast_age_max if math.isfinite(self.p.plast_age_max) else 12
                    ry = float((self.age[inr] < aw).mean())
            d["ref_young"] = ry
        else:
            d["S_juv"] = d["ref_young"] = np.nan
        return d

    def run(self, T=None, record_every=1, verbose=False, callback=None):
        series, summ = super().run(T, record_every=record_every, verbose=verbose, callback=callback)
        try:
            summ.update(extras_v2(series.sort_values("t").reset_index(drop=True), self.p, summ))
        except Exception as e:  # noqa: BLE001 - nunca romper una corrida por un observable
            summ["obj_O17p"] = f"NA:{type(e).__name__}"
        return series, summ


class U2v2(Extras2bV2, Universe):
    pass
