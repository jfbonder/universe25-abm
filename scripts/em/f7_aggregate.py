""" agrega las corridas .npz de un grupo de results_em/<grupo>/ en summary.csv.

Uso: python3 scripts/em/f7_aggregate.py e1 mate hyd mubg
Por configuración: n, extintas, P(extinción) a 300 / 500 / 1000 / 2000 semanas con IC de Wilson, mediana de T_ext
(IC bootstrap), mediana de t_last, tasa tardía (extinciones con t > 300 / tiempo en riesgo) con IC exacto de
Poisson, N_max, y el "fin de la natalidad" (natalidad < 5 % del máximo, mediana).
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT, group_dir


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_ci(k, T):
    lo = stats.chi2.ppf(0.025, 2 * k) / (2 * T) if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / (2 * T)
    return lo, hi


def boot_median(x, B=1000, seed=0):
    x = np.asarray(x, float)
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, size=(B, x.size)), axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def parse_cfg(cfg):
    d = {"cfg": cfg}
    for tok in cfg.split("_"):
        m = re.fullmatch(r"l([0-9.]+)", tok)
        if m: d["lam"] = float(m.group(1))
        m = re.fullmatch(r"o([0-9.]+)", tok)
        if m: d["out"] = float(m.group(1))
        m = re.fullmatch(r"L(\d+)", tok)
        if m: d["L"] = int(m.group(1))
        m = re.fullmatch(r"tr(\d+)", tok)
        if m: d["t_r"] = int(m.group(1))
        m = re.fullmatch(r"f([0-9.]+)", tok)
        if m: d["f"] = float(m.group(1))
        m = re.fullmatch(r"h([0-9.]+)", tok)
        if m: d["h"] = float(m.group(1))
        m = re.fullmatch(r"A(\d+|None)", tok)
        if m: d["A"] = m.group(1)
        m = re.fullmatch(r"per(\d+)", tok)
        if m: d["period"] = int(m.group(1))
    parts = cfg.split("_")
    d["model"] = parts[0]
    d["space"] = parts[1]
    if parts[1] in ("wm", "sp") and len(parts) > 2:
        d["variant"] = parts[2]
    return d


DESIGN_T = {"e1": 2000, "mate": 3000, "hyd": 1500, "mubg": 1000, "p3": 1500}


def summarize_dir(d, Tdes):
    fs = sorted(f for f in glob.glob(os.path.join(d, "seed*.npz")) if not f.endswith(".tmp.npz"))
    if not fs:
        return None
    te, tl, ext, tend, nmax, toff, expl = [], [], [], [], [], [], []
    for f in fs:
        try:
            z = np.load(f)
        except Exception:
            continue
        ext.append(bool(z["extinct"])); te.append(float(z["t_ext"])); tl.append(int(z["t_last"]))
        tend.append(int(z["t_end"])); nmax.append(int(z["N_max"])); expl.append(bool(z["exploded"]))
        B = z["s_B"].astype(float); t = z["s_t"]
        if B.max() > 0:
            ib = int(np.argmax(B))
            off = np.flatnonzero((np.arange(B.size) > ib) & (B < 0.05 * B.max()))
            toff.append(int(t[off[0]]) if off.size else np.nan)
        else:
            toff.append(np.nan)
    ext = np.array(ext); te = np.array(te); tend = np.array(tend)
    n = ext.size
    row = dict(n=n, n_ext=int(ext.sum()), n_expl=int(np.sum(expl)), T=Tdes,
               n_cens_early=int(np.sum(~ext & (tend < Tdes))))
    for H in (300, 500, 1000, 1500, 2000, 3000):
        # clasificable a horizonte H: extinta antes de H o seguida al menos hasta H
        classif = (ext & (te <= H)) | (tend >= H - 1)
        if H <= Tdes and classif.all():
            k = int(np.sum(ext & (te <= H)))
            row[f"P_ext_{H}"] = k / n
            row[f"P_ext_{H}_lo"], row[f"P_ext_{H}_hi"] = wilson(k, n)
    tex = te[ext]
    row["Text_med"] = float(np.median(tex)) if tex.size else np.nan
    row["Text_med_lo"], row["Text_med_hi"] = boot_median(tex)
    row["tlast_med"] = float(np.median(np.array(tl)[ext])) if tex.size else np.nan
    row["Nmax_med"] = float(np.median(nmax))
    row["t_off_med"] = float(np.nanmedian(toff)) if np.isfinite(toff).any() else np.nan
    # tasa tardía (t > 300)
    late_k = int(np.sum(ext & (te > 300)))
    risk = float(np.sum(np.clip(np.where(ext, te, tend) - 300, 0, None)))
    row["late_k"] = late_k; row["late_risk"] = risk
    row["late_rate"] = late_k / risk if risk > 0 else np.nan
    row["late_rate_lo"], row["late_rate_hi"] = poisson_ci(late_k, risk) if risk > 0 else (np.nan, np.nan)
    return row


def aggregate(group):
    base = group_dir(f"f7_{group}")
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        r = summarize_dir(d, DESIGN_T[group])
        if r is None:
            continue
        r.update(parse_cfg(os.path.basename(d)))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "summary.csv"), index=False)
    return df


if __name__ == "__main__":
    for g in sys.argv[1:] or ["e1", "mate", "hyd", "mubg"]:
        df = aggregate(g)
        print(g, len(df), "configuraciones")
