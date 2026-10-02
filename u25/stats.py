"""Series por paso y resumen escalar por corrida."""
from __future__ import annotations

import numpy as np
import pandas as pd

QS = (0.10, 0.25, 0.50, 0.75, 0.90)
AW_DEFAULT = 12        # a_w de C1; edad de medición de s_aw cuando no hay ventana R1 (nulos sin R1, longevo)


def aw_obs_age(p) -> int:
    """Edad a_w a la que se mide s_aw (O11 con o11_age="aw"): plast_age_max si la ventana R1 es finita; si no,
    AW_DEFAULT = 12 (la de C1), para comparar los nulos a la misma edad calendario."""
    a = p.plast_age_max
    return int(a) if np.isfinite(a) else AW_DEFAULT


class Recorder:
    def __init__(self):
        self.rows: list[dict] = []

    def record(self, U) -> None:
        p = U.p
        n = U.N
        G = p.grid_size
        row = {
            "t": U.t,
            "N": n,
            "N_m": int(U.male.sum()),
            "N_f": int(n - U.male.sum()),
            "births": U.births,
            "deaths": U.deaths,
            "conceptions": U.conceptions,
            "pups": U.pups,
            "n_jumps": U.n_jumps,
            "n_defeated": U.n_defeated,
            "deaths_social_exp": U.deaths_social_exp,
            "n_refuge_rejected": getattr(U, "n_refuge_rejected", 0),
            "n_births_relocated": getattr(U, "n_births_relocated", 0),
        }
        if n:
            s = U.s
            row["S_mean"] = float(s.mean())
            qv = np.quantile(s, QS)
            for q, v in zip(QS, qv):
                row[f"s_q{int(round(q * 100)):02d}"] = float(v)
            row["low_s_frac"] = float((s < 0.2).mean())
            rep = U.is_reproductive()
            row["N_repro_f"] = int((rep & ~U.male).sum())
            row["N_repro_m"] = int((rep & U.male).sum())
            row["S_repro_f"] = float(s[rep & ~U.male].mean()) if row["N_repro_f"] else np.nan
            row["mean_age"] = float(U.age.mean())
            occ = np.bincount(U.x * G + U.y, minlength=G * G)
            row["occ_max"] = int(occ.max())
            row["occ_var"] = float(occ.var())
            row["occ_frac_cells"] = float((occ > 0).mean())
            mo = occ.mean()
            row["I_D"] = float(occ.var() / mo) if mo > 0 else np.nan
            # --- observables de la calibración
            adult = U.age >= p.repro_age_min
            na = int(adult.sum())
            row["N_adult"] = na
            row["S_adult"] = float(s[adult].mean()) if na else np.nan
            rf = rep & ~U.male
            row["n_healthy_f"] = int((rf & (s > p.obs_s_v)).sum())
            row["h"] = row["n_healthy_f"] / row["N_repro_f"] if row["N_repro_f"] else 0.0
            row["ref_frac"] = float((s > p.obs_s_low).mean())
            low_a = adult & (s < p.obs_s_low)
            row["phi_w"] = float(low_a.sum() / na) if na else np.nan
            am, af = adult & U.male, adult & ~U.male
            row["low_s_m_adult"] = float((low_a & U.male).sum() / am.sum()) if am.any() else np.nan
            row["low_s_f_adult"] = float((low_a & ~U.male).sum() / af.sum()) if af.any() else np.nan
            row["bimod_s"] = bimodality_coefficient(s[adult]) if na >= 4 else np.nan
            mat = U.age == p.repro_age_min
            row["n_mat"] = int(mat.sum())
            row["s_mat"] = float(s[mat].mean()) if row["n_mat"] else np.nan
            # s al cierre de la ventana R1 (O11 con o11_age="aw"): agentes que ya recibieron sus a_w
            # actualizaciones dentro de la ventana (edades 1..a_w); al registrar, su edad es a_w + 1
            aw = U.age == aw_obs_age(p) + 1
            row["n_aw"] = int(aw.sum())
            row["s_aw"] = float(s[aw].mean()) if row["n_aw"] else np.nan
            row["age_min"] = int(U.age.min())
            if U.refuge is not None:
                inr = U.refuge[U.x * G + U.y]
                row["frac_in_refuge"] = float(inr.mean())
                row["S_in_refuge"] = float(s[inr].mean()) if inr.any() else np.nan
            # fracción de individuos que estuvieron hacinados (m > m_c) tras el movimiento
            row["crowded_frac"] = float((U._m > p.m_c).mean()) if U._m.shape[0] == n else np.nan
        else:
            row["S_mean"] = np.nan
            for q in QS:
                row[f"s_q{int(round(q * 100)):02d}"] = np.nan
            for k in ("low_s_frac", "S_repro_f", "mean_age", "occ_var", "occ_frac_cells", "crowded_frac",
                      "I_D", "S_adult", "ref_frac", "phi_w", "bimod_s", "s_mat", "s_aw", "age_min",
                      "low_s_m_adult", "low_s_f_adult"):
                row[k] = np.nan
            row["N_repro_f"] = row["N_repro_m"] = row["N_adult"] = row["n_healthy_f"] = row["n_mat"] = row["n_aw"] = 0
            row["h"] = 0.0
            row["occ_max"] = 0
        row["n_aff"] = int(U.n_aff)
        row["mean_aff"] = float(U.mean_aff)
        row["n_active_pairs"] = int(U.n_active_pairs)
        for k in ("mstar", "f_over", "corr_s_m", "frac_withdrawn", "frac_crowded_low", "phi_BO", "m_autocorr1"):
            row[k] = U.obs.get(k, np.nan) if U.t > 0 else np.nan
        for m in getattr(U, "CROWD_FIXED_M", ()):
            k = f"frac_crowded_low_m{m}"
            row[k] = U.obs.get(k, np.nan) if U.t > 0 else np.nan
        nb = p.obs_blocks ** 2
        bb = U.births_block_step
        for k in range(nb):
            row[f"births_block_{k}"] = int(bb[k]) if bb is not None else 0
        row.update(U.extra_stats())
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _first_time(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else np.nan


def bimodality_coefficient(x: np.ndarray) -> float:
    """Coeficiente de bimodalidad de Sarle con corrección muestral: BC = (g1^2+1)/(g2 + 3(n-1)^2/((n-2)(n-3))).
    BC > 5/9 ≈ 0.555 sugiere bimodalidad (uniforme = 5/9)."""
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return np.nan
    sd = x.std()
    if sd < 1e-12:
        return np.nan
    z = (x - x.mean()) / sd
    m3 = (z ** 3).mean()
    m4 = (z ** 4).mean()
    g1 = np.sqrt(n * (n - 1)) / (n - 2) * m3
    g2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 - 3) + 6)
    return float((g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def gini(v) -> float:
    v = np.sort(np.asarray(v, float))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float((2 * (i * v).sum() / (n * v.sum())) - (n + 1) / n)


def smooth(x, w: int = 4) -> np.ndarray:
    """Media móvil causal-centrada de ancho w (especificación de los criterios: 4 semanas)."""
    x = np.asarray(x, float)
    if x.size == 0 or w <= 1:
        return x.copy()
    k = np.ones(w) / w
    xp = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def rebound_R(N, i_pk: int, w: int = 4) -> float:
    """Rebote R = max_{t>=t_pk} [N(t) - min_{t_pk<=u<=t} N(u)] / N_pk  sobre N suavizada.
    R = 0 si la caída post-pico es monótona (Calhoun)."""
    Ns = smooth(N, w)
    tail = Ns[i_pk:]
    if tail.size == 0 or Ns[i_pk] <= 0:
        return np.nan
    run_min = np.minimum.accumulate(tail)
    return float((tail - run_min).max() / max(N[i_pk], 1))


def count_rebounds(N, i_pk: int, amp: float = 0.10, w: int = 4) -> int:
    """Nº de rebotes post-pico: subidas de N suavizada > amp*N_pk desde el mínimo corriente
    (se reinicia el mínimo tras cada rebote al caer de nuevo amp*N_pk desde el máximo)."""
    Ns = smooth(N, w)[i_pk:]
    if Ns.size == 0:
        return 0
    thr = amp * float(np.max(N))
    cnt, state, ref = 0, "down", Ns[0]
    for v in Ns:
        if state == "down":
            ref = min(ref, v)
            if v - ref > thr:
                cnt += 1
                state, ref = "up", v
        else:
            ref = max(ref, v)
            if ref - v > thr:
                state, ref = "down", v
    return cnt


def birth_stop_time(t, B, N, w: int = 4, frac: float = 0.01):
    """t_B0: primer t (posterior al máximo de B suavizada) desde el cual B suavizada < frac*max B
    para siempre. NaN si la natalidad no se detiene."""
    Bs = smooth(B, w)
    if Bs.size == 0 or Bs.max() <= 0:
        return np.nan, np.nan
    ib = int(np.argmax(Bs))
    thr = frac * Bs.max()
    above = np.flatnonzero(Bs[ib:] >= thr)
    k = ib + int(above[-1]) + 1
    if k >= Bs.size:
        return np.nan, np.nan
    return float(t[k]), float(N[k])


def hysteresis_AH(N, B, i_pk: int, w: int = 4, nbins: int = 10) -> float:
    """A_H: natalidad per cápita en la rama descendente / ascendente a igual N (bins de log N).
    NaN si no hay solapamiento."""
    N = np.asarray(N, float)
    b = smooth(np.asarray(B, float), w) / np.maximum(N, 1)
    up = np.arange(1, i_pk + 1)
    dn = np.arange(i_pk + 1, N.size)
    up = up[N[up] > 0]
    dn = dn[N[dn] > 0]
    if up.size < 2 or dn.size < 2:
        return np.nan
    lo = max(N[up].min(), N[dn].min())
    hi = min(N[up].max(), N[dn].max())
    if not hi > lo:
        return np.nan
    edges = np.geomspace(lo, hi, nbins + 1)
    ru, rd = [], []
    for a, c in zip(edges[:-1], edges[1:]):
        mu = (N[up] >= a) & (N[up] <= c)
        md = (N[dn] >= a) & (N[dn] <= c)
        if mu.any() and md.any():
            ru.append(b[up][mu].mean())
            rd.append(b[dn][md].mean())
    if not ru or np.sum(ru) <= 0:
        return np.nan
    return float(np.sum(rd) / np.sum(ru))


def summarize(series: pd.DataFrame, T: int, U=None) -> dict:
    t = series["t"].to_numpy()
    N = series["N"].to_numpy()
    S = series["S_mean"].to_numpy()
    i_pk = int(np.argmax(N))
    extinct = bool(N[-1] == 0)
    t_end = int(t[-1])
    t_ext = float(t_end) if extinct else np.nan
    alive = N > 0
    Sa = np.where(alive, S, np.nan)
    i_smin = int(np.nanargmin(Sa)) if alive.any() else 0
    out = {
        "N_max": int(N[i_pk]),
        "t_peak": int(t[i_pk]),
        "extinct": extinct,
        "censored": not extinct,
        "t_ext": t_ext,
        "t_end": t_end,
        "T": int(T),
        "N_final": int(N[-1]),
        "collapse_duration": (t_ext - t[i_pk]) if extinct else np.nan,
        # S_min sobre pasos con N > 0 (el notebook incluía el 0 artificial de la población extinta)
        "S_min": float(np.nanmin(Sa)) if alive.any() else np.nan,
        "t_S_min": int(t[i_smin]),
        "S_min_nb": float(np.nan_to_num(S, nan=0.0).min()),  # definición del notebook
        "S_at_peak": float(S[i_pk]),
        "S_final": float(Sa[alive][-1]) if alive.any() else np.nan,
        "t_S_below_0.5": _first_time(t, Sa < 0.5),
        "t_S_below_0.2": _first_time(t, Sa < 0.2),
        "max_aff": int(series["n_aff"].max()),
        "max_mean_aff": float(series["mean_aff"].max()),
        "max_occ": int(series["occ_max"].max()),
        "total_births": int(series["births"].sum()),
        "total_deaths": int(series["deaths"].sum()),
        "min_N_after_peak": int(N[i_pk:].min()),
    }
    # oscilaciones: nº de máximos locales "grandes" de N después del primer pico
    out["n_regrowths"] = count_regrowths(N)
    # ---- observables de la calibración / objetivos
    B = series["births"].to_numpy()
    out["R_rebound"] = rebound_R(N, i_pk)
    out["n_rebounds10"] = count_rebounds(N, i_pk)
    tB0, NB0 = birth_stop_time(t, B, N)
    out["t_B0"] = tB0
    out["NB0_over_Npk"] = NB0 / N[i_pk] if np.isfinite(NB0) and N[i_pk] > 0 else np.nan
    out["A_H"] = hysteresis_AH(N, B, i_pk)
    out["rho_ref"] = float(series["ref_frac"].iloc[i_pk]) if "ref_frac" in series else np.nan
    out["h_at_peak"] = float(series["h"].iloc[i_pk]) if "h" in series else np.nan
    out["mstar_at_peak"] = float(series["mstar"].iloc[i_pk]) if "mstar" in series else np.nan
    out["mstar_max"] = float(np.nanmax(series["mstar"])) if "mstar" in series and series["mstar"].notna().any() else np.nan
    out["f_over_at_peak"] = float(series["f_over"].iloc[i_pk]) if "f_over" in series else np.nan
    out["I_D_at_peak"] = float(series["I_D"].iloc[i_pk]) if "I_D" in series else np.nan
    # picos (misma regla que scripts/analysis/pipeline/peaks.find_cycles: serie cruda,
    # prominencia >= 10 % de N_max, distancia >= 40 semanas)
    from scipy.signal import find_peaks
    Nf = N.astype(float)
    pk, _ = find_peaks(Nf, prominence=max(0.10 * Nf.max(), 1.0), distance=40)
    out["N0"] = int(N[0])
    out["n_peaks"] = int(pk.size)
    out["period"] = float(np.mean(np.diff(t[pk]))) if pk.size > 1 else np.nan
    out["peak_times"] = ";".join(str(int(x)) for x in t[pk])
    out["peak_values"] = ";".join(str(int(x)) for x in N[pk])
    if "m_autocorr1" in series:
        out["m_autocorr1_mean"] = float(series["m_autocorr1"].mean()) if series["m_autocorr1"].notna().any() else np.nan
    if U is not None:
        out["gini_births_blocks"] = gini(U.births_block)
    return out


def count_regrowths(N: np.ndarray, rel: float = 0.5) -> int:
    """Nº de recuperaciones: veces que N cae por debajo de rel*max previo y vuelve a superar
    1/rel * (mínimo alcanzado).  Heurística simple para detectar régimen oscilatorio."""
    N = np.asarray(N, float)
    if N.size < 3:
        return 0
    count = 0
    state = "up"
    ref = N[0]
    for v in N:
        if state == "up":
            ref = max(ref, v)
            if v < rel * ref:
                state = "down"
                ref = v
        else:
            ref = min(ref, v)
            if ref > 0 and v > ref / rel:
                count += 1
                state = "up"
                ref = v
    return count
