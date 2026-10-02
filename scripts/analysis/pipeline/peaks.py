"""Picos y valles de N(t), período, drift entre picos (Mann-Kendall, Sen) y test del ensamble."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


def find_cycles(t, N, prominence_rel=0.10, distance=40):
    """Picos y valles con prominencia relativa al máximo global. Devuelve (idx_picos, idx_valles)."""
    N = np.asarray(N, float)
    prom = prominence_rel * N.max() if N.max() > 0 else 1
    pk, _ = find_peaks(N, prominence=prom, distance=distance)
    tr, _ = find_peaks(-N, prominence=prom, distance=distance)
    return pk, tr


def mann_kendall(x):
    """Test de tendencia de Mann-Kendall (con corrección por empates) y pendiente de Sen."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return dict(S=np.nan, z=np.nan, p=np.nan, sen=np.nan, n=n)
    S = 0.0
    for i in range(n - 1):
        S += np.sign(x[i + 1:] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = (S - np.sign(S)) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return dict(S=float(S), z=float(z), p=float(p), sen=float(np.median(slopes)), n=n)


def cycles_per_run(series: pd.DataFrame, skip_first=True, **kw):
    """Una fila por (point, rep): nº de picos, período, amplitud, drift de log(pico_k) (k>=2), MK, Sen."""
    rows = []
    for (pt, rep), g in series.groupby(["point", "rep"], sort=True):
        t = g["t"].to_numpy(); N = g["N"].to_numpy(float)
        pk, tr = find_cycles(t, N, **kw)
        row = dict(point=pt, rep=rep, n_peaks=len(pk), n_troughs=len(tr),
                   first_peak=N[pk[0]] if len(pk) else N.max(), t_first_peak=t[pk[0]] if len(pk) else t[np.argmax(N)],
                   period=float(np.mean(np.diff(t[pk]))) if len(pk) > 1 else np.nan,
                   mean_rest=float(N[pk[1:]].mean()) if len(pk) > 1 else np.nan,
                   trough_median=float(np.median(N[tr])) if len(tr) else np.nan,
                   amplitude_log=float(np.mean(np.log(N[pk][1:] / np.maximum(N[tr][:len(pk) - 1], 1)))) if len(pk) > 1 and len(tr) else np.nan,
                   slope_log=np.nan, mk_S=np.nan, mk_p=np.nan, sen=np.nan)
        peaks = N[pk][1:] if skip_first else N[pk]
        if len(peaks) >= 3:
            k = np.arange(len(peaks)); y = np.log(peaks)
            row["slope_log"] = float(np.polyfit(k, y, 1)[0])
            mk = mann_kendall(y); row.update(mk_S=mk["S"], mk_p=mk["p"], sen=mk["sen"])
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_drift_test(cyc: pd.DataFrame, min_peaks=4, B=2000, seed=0):
    """Test de drift a nivel ensamble: pendientes por corrida (Wilcoxon vs 0 y bootstrap de la media)
    y regresión agrupada con efectos fijos por corrida (within)."""
    d = cyc[(cyc["n_peaks"] >= min_peaks) & cyc["slope_log"].notna()]
    out = dict(n_runs=int(len(d)))
    if len(d) < 3:
        return out
    sl = d["slope_log"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(sl, len(sl)).mean() for _ in range(B)])
    out.update(mean_slope=float(sl.mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
               sd_slope=float(sl.std(ddof=1)), wilcoxon_p=float(stats.wilcoxon(sl).pvalue) if np.any(sl != 0) else np.nan,
               sign_pos=int((sl > 0).sum()), sign_neg=int((sl < 0).sum()),
               frac_mk_sig_neg=float(((d["mk_p"] < 0.05) & (d["mk_S"] < 0)).mean()),
               frac_mk_sig_pos=float(((d["mk_p"] < 0.05) & (d["mk_S"] > 0)).mean()),
               median_sen=float(d["sen"].median()))
    return out


def within_regression(series: pd.DataFrame, min_peaks=4, **kw):
    """Regresión log(pico_k) ~ k con efectos fijos por corrida (datos centrados por corrida).
    Devuelve pendiente, EE robusto por cluster (corrida) y p."""
    ks, ys, ids = [], [], []
    for (pt, rep), g in series.groupby(["point", "rep"]):
        N = g["N"].to_numpy(float); pk, _ = find_cycles(g["t"].to_numpy(), N, **kw)
        if len(pk) >= min_peaks:
            y = np.log(N[pk][1:]); k = np.arange(len(y), dtype=float)
            ks.append(k - k.mean()); ys.append(y - y.mean()); ids.append(np.full(len(y), len(ids)))
    if not ks:
        return dict(n_runs=0)
    k = np.concatenate(ks); y = np.concatenate(ys); cid = np.concatenate(ids)
    b = (k * y).sum() / (k * k).sum(); res = y - b * k
    # EE robusto por cluster
    num = sum(((k[cid == c] * res[cid == c]).sum()) ** 2 for c in np.unique(cid))
    se = np.sqrt(num) / (k * k).sum()
    return dict(n_runs=len(ks), slope=float(b), se=float(se), p=float(2 * stats.norm.sf(abs(b / se))) if se > 0 else np.nan)
