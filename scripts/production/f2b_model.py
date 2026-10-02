"""Clases de modelo de la Fase 2b: `Universe` + columnas extra en el resumen por corrida.

No cambian la dinámica ni consumen números aleatorios: sólo calculan, sobre la serie a resolución completa
(antes de que el runner la compacte), observables que el resumen de u25 no trae:

  obj_O13p      O13 con la clase aislada *persistente* (phi_BO, 4 semanas) en lugar de la instantánea
                (mismo esquema que u25.objectives.eval_O13);
  x_rho_pk      N_max / K, K = L^2 m_c;              x_f0_pk   fracción de celdas vacías en t_pk;
  x_BO, x_BO4, x_CR   máximos en {t_pk, t_pk+10, t_pk+20} de frac_withdrawn, phi_BO y frac_crowded_low;
  x_in_ref_pk   fracción en refugios en t_pk;        x_S_adult_pk;
  x_rho8        (T_ext - t_pk)/t_pk;                 x_t_lastbirth, x_D_last = T_ext - t_lastbirth (Q4);
  x_n_last      nacimientos en las 10 semanas previas a la última cría;
  x_smat_B      s media de las cohortes que maduran en [t_lag, t_pk] (pesada por n_mat; Q6);
  x_smat_sen, x_smat_mk_p   pendiente de Sen y Mann–Kendall de s_mat en [t_lag, t_B0 + 6] (cascada).
"""
from __future__ import annotations

import math

import numpy as np

from u25.model import Universe

EXTRA_KEYS = ["obj_O13p", "x_rho_pk", "x_f0_pk", "x_BO", "x_BO4", "x_CR", "x_in_ref_pk", "x_S_adult_pk", "x_rho8",
              "x_t_lastbirth", "x_D_last", "x_n_last", "x_smat_B", "x_smat_sen", "x_smat_mk_p"]


def _mk(y):
    y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        return np.nan, np.nan
    s = 0.0
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / math.sqrt(var) if var > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    sl = [(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return float(np.median(sl)), float(p)


def extras(S, p, summ) -> dict:
    out = {k: np.nan for k in EXTRA_KEYS}
    out["obj_O13p"] = "NA"
    t = S["t"].to_numpy(float)
    N = S["N"].to_numpy(float)
    if len(t) < 2:
        return out
    ipk = int(np.argmax(N))
    tpk = t[ipk]
    K = p.grid_size ** 2 * p.m_c
    out["x_rho_pk"] = float(N[ipk] / K)

    def col(c):
        return S[c].to_numpy(float) if c in S else None

    occ = col("occ_frac_cells")
    if occ is not None:
        out["x_f0_pk"] = float(1 - occ[ipk])
    sa = col("S_adult")
    if sa is not None:
        out["x_S_adult_pk"] = float(sa[ipk])
    fr = col("frac_in_refuge")
    if fr is not None:
        out["x_in_ref_pk"] = float(fr[ipk])
    fw, fc, bo4, nad = col("frac_withdrawn"), col("frac_crowded_low"), col("phi_BO"), col("N_adult")
    bo, cr, b4 = [], [], []
    ok_any = marg_any = False
    seen = False
    for tt in (tpk, tpk + 10, tpk + 20):
        i = int(np.searchsorted(t, tt))
        if i >= len(t) or t[i] != tt:
            continue
        if fw is not None:
            bo.append(fw[i])
        if fc is not None:
            cr.append(fc[i])
        if bo4 is not None:
            b4.append(bo4[i])
        if bo4 is None or fc is None or (nad is not None and nad[i] < 20):
            continue
        if not (np.isfinite(bo4[i]) and np.isfinite(fc[i])):
            continue
        seen = True
        if bo4[i] >= 0.10 and fc[i] >= 0.10:
            ok_any = True
        elif 0.05 <= bo4[i] < 0.10 and fc[i] >= 0.10:
            marg_any = True
    out["x_BO"] = float(np.nanmax(bo)) if bo and np.isfinite(bo).any() else np.nan
    out["x_CR"] = float(np.nanmax(cr)) if cr and np.isfinite(cr).any() else np.nan
    out["x_BO4"] = float(np.nanmax(b4)) if b4 and np.isfinite(b4).any() else np.nan
    if seen:
        out["obj_O13p"] = "PASS" if ok_any else ("MARG" if marg_any else "FAIL")
    extinct = bool(N[-1] == 0)
    B = col("births")
    if B is not None and (B > 0).any():
        il = int(np.flatnonzero(B > 0)[-1])
        out["x_t_lastbirth"] = float(t[il])
        out["x_n_last"] = float(B[(t > t[il] - 10) & (t <= t[il])].sum())
        if extinct:
            out["x_D_last"] = float(t[-1] - t[il])
    if extinct and tpk > 0:
        out["x_rho8"] = float((t[-1] - tpk) / tpk)
    sm, nm = col("s_mat"), col("n_mat")
    if sm is not None and nm is not None:
        lag = np.flatnonzero(N >= 2 * N[0])
        t1 = t[lag[0]] if lag.size else 0.0
        m = (t >= t1) & (t <= tpk) & (nm > 0) & np.isfinite(sm)
        if m.any():
            out["x_smat_B"] = float(np.average(sm[m], weights=nm[m]))
        tb0 = summ.get("t_B0", np.nan)
        hi = (tb0 + 6) if (tb0 == tb0) else tpk + 26
        m2 = (t >= t1) & (t <= hi) & (nm > 0) & np.isfinite(sm)
        if m2.sum() >= 4:
            out["x_smat_sen"], out["x_smat_mk_p"] = _mk(sm[m2])
    return out


class Extras2b:
    """Mixin: agrega EXTRA_KEYS al resumen de Universe.run."""

    def run(self, T=None, record_every=1, verbose=False, callback=None):
        series, summ = super().run(T, record_every=record_every, verbose=verbose, callback=callback)
        try:
            summ.update(extras(series.sort_values("t").reset_index(drop=True), self.p, summ))
        except Exception as e:  # noqa: BLE001 - nunca romper una corrida por un observable
            summ.update({k: np.nan for k in EXTRA_KEYS})
            summ["obj_O13p"] = f"NA:{type(e).__name__}"
        return series, summ


class U2b(Extras2b, Universe):
    pass
