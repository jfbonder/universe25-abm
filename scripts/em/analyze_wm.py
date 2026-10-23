"""Análisis de la Tarea 1: tabla por (modelo, refugios, L) y ajustes de T_ext vs ln N.
Uso: python3 scripts/em/analyze_wm.py [results_em/wellmixed_size_scaling]"""
import glob, os, re, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load(base):
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "*_L*"))):
        m = re.match(r"(\w+?)_(ref|noref)_L(\d+)(.*)$", os.path.basename(d))
        if not m:
            continue
        for fn in glob.glob(os.path.join(d, "seed*.npz")):
            if fn.endswith(".tmp.npz"):
                continue
            z = np.load(fn)
            rows.append(dict(model=m.group(1), ref=m.group(2) == "ref", L=int(m.group(3)), tag=m.group(4),
                             seed=int(re.search(r"seed(\d+)", fn).group(1)), extinct=bool(z["extinct"]),
                             exploded=bool(z["exploded"]), t_ext=float(z["t_ext"]), t_end=float(z["t_end"]),
                             N_max=float(z["N_max"]), t_peak=float(z["t_peak"]), t_last=float(z["t_last"]),
                             n_last=float(z["n_last"]), N0=float(z["N0"])))
    return pd.DataFrame(rows)


def boot_ci(x, f=np.median, B=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan
    bs = [f(rng.choice(x, x.size)) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def table(df):
    out = []
    for (mo, rf, L, tag), g in df.groupby(["model", "ref", "L", "tag"]):
        te = g.t_ext[g.extinct]
        lo, hi = boot_ci(te)
        R = (g.t_ext - g.t_last)[g.extinct]
        out.append(dict(model=mo, ref=rf, L=L, tag=tag, n=len(g), ext=int(g.extinct.sum()), expl=int(g.exploded.sum()),
                        N0=int(g.N0.median()), Nmax_mean=g.N_max.mean(), Nmax_sd=g.N_max.std(),
                        Nmax_per_cell=g.N_max.mean() / L ** 2, t_peak=g.t_peak.median(),
                        Text_med=te.median(), Text_lo=lo, Text_hi=hi, Text_mean=te.mean(), Text_sd=te.std(),
                        Text_iqr=te.quantile(.75) - te.quantile(.25),
                        tlast_med=g.t_last[g.extinct].median(), tlast_sd=g.t_last[g.extinct].std(),
                        R_med=R.median(), R_mean=R.mean(), nlast_mean=g.n_last[g.extinct].mean()))
    return pd.DataFrame(out)


def fits(df, ycol="t_ext", xs=("lnN0",), min_L=10):
    """Ajuste por MCO de y = a + b ln N0 (todas las corridas, extintas) con IC bootstrap por grupo,
    y comparación con y = a + b sqrt(ln N0) y y = a + b N0 (lineal/exponencial ~ ln T = a + cN)."""
    res = {}
    for (mo, rf), g in df[df.extinct & (df.L >= min_L) & (df.tag == "")].groupby(["model", "ref"]):
        x = np.log(g.N0.to_numpy())
        y = g[ycol].to_numpy()
        out = {}
        for name, X in (("a+b lnN", x), ("a+b sqrt(lnN)", np.sqrt(x)), ("a+b N/1000", g.N0.to_numpy() / 1000)):
            A = np.c_[np.ones_like(X), X]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            # residuo sobre medianas por L (forma funcional) y bootstrap estratificado por L
            Ls = g.L.to_numpy()
            med = g.groupby("L")[ycol].median()
            Xm = {"a+b lnN": np.log(g.groupby("L").N0.median()), "a+b sqrt(lnN)": np.sqrt(np.log(g.groupby("L").N0.median())),
                  "a+b N/1000": g.groupby("L").N0.median() / 1000}[name]
            rms_med = float(np.sqrt(np.mean((med.to_numpy() - (coef[0] + coef[1] * Xm.to_numpy())) ** 2)))
            rng = np.random.default_rng(1)
            bs = []
            idx_by_L = [np.flatnonzero(Ls == L) for L in np.unique(Ls)]
            for _ in range(500):
                ii = np.concatenate([rng.choice(ix, ix.size) for ix in idx_by_L])
                c, *_ = np.linalg.lstsq(A[ii], y[ii], rcond=None)
                bs.append(c[1])
            out[name] = dict(a=float(coef[0]), b=float(coef[1]), b_lo=float(np.percentile(bs, 2.5)),
                             b_hi=float(np.percentile(bs, 97.5)), rms_medians=rms_med)
        res[f"{mo}_{'ref' if rf else 'noref'}"] = out
    return res


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_em", "wellmixed_size_scaling")
    df = load(base)
    tab = table(df)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string(index=False))
    for y in ("t_ext", "t_last"):
        print(f"\nAjustes de {y}:")
        print(json.dumps(fits(df, y), indent=1))
    tab.to_csv(os.path.join(base, "tabla_wm.csv"), index=False)
    df.to_csv(os.path.join(base, "corridas_wm.csv"), index=False)
