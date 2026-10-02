"""Análisis de supervivencia con censura a la derecha (tiempo de extinción de la colonia)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats, optimize


# ------------------------------------------------------------------ no paramétrico
def kaplan_meier(time, event, conf=0.95):
    """Estimador producto-límite con varianza de Greenwood e IC log-log. Devuelve DataFrame por tiempo de evento."""
    time = np.asarray(time, float); event = np.asarray(event, bool)
    ts = np.unique(time[event])
    rows = [dict(t=0.0, n_risk=len(time), d=0, S=1.0, var=0.0, lo=1.0, hi=1.0, H=0.0)]
    S = 1.0; gw = 0.0; H = 0.0
    z = stats.norm.ppf(0.5 + conf / 2)
    for t in ts:
        n = int((time >= t).sum()); d = int(((time == t) & event).sum())
        if n == 0:
            continue
        S *= 1 - d / n; H += d / n
        gw += d / (n * (n - d)) if n > d else np.inf
        if 0 < S < 1 and np.isfinite(gw):
            se = np.sqrt(gw) / abs(np.log(S))
            lo, hi = S ** np.exp(z * se), S ** np.exp(-z * se)
        else:
            lo = hi = S
        rows.append(dict(t=t, n_risk=n, d=d, S=S, var=S * S * gw, lo=lo, hi=hi, H=H))
    return pd.DataFrame(rows)


def km_at(km: pd.DataFrame, t):
    """S(t) escalonada."""
    idx = np.searchsorted(km["t"].to_numpy(), np.atleast_1d(t), side="right") - 1
    return km["S"].to_numpy()[idx]


def km_median(km: pd.DataFrame, q=0.5):
    """Mediana (cuantil q de t_ext) e IC por inversión de las bandas. NaN si S no cruza q."""
    def first_cross(col):
        m = km[col].to_numpy() <= q
        return float(km["t"].to_numpy()[np.argmax(m)]) if m.any() else np.nan
    # la banda inferior S_lo cruza q antes (cota inferior de la mediana); la superior S_hi, después
    return dict(median=first_cross("S"), lo=first_cross("lo"), hi=first_cross("hi"))


def nelson_aalen(time, event):
    km = kaplan_meier(time, event)
    return km[["t", "H"]]


def logrank(time, event, group):
    """Log-rank de k grupos. Devuelve chi2, df, p."""
    time = np.asarray(time, float); event = np.asarray(event, bool); group = np.asarray(group)
    gs = np.unique(group); k = len(gs)
    ts = np.unique(time[event])
    O = np.zeros(k); E = np.zeros(k); V = np.zeros((k, k))
    for t in ts:
        at = time >= t; d = ((time == t) & event)
        n = at.sum(); dt = d.sum()
        if n <= 1:
            continue
        ng = np.array([(at & (group == g)).sum() for g in gs], float)
        dg = np.array([(d & (group == g)).sum() for g in gs], float)
        e = dt * ng / n
        O += dg; E += e
        for i in range(k):
            for j in range(k):
                V[i, j] += dt * (ng[i] / n) * ((i == j) - ng[j] / n) * (n - dt) / (n - 1)
    diff = (O - E)[:-1]; Vr = V[:-1, :-1]
    try:
        chi2 = float(diff @ np.linalg.solve(Vr, diff))
    except np.linalg.LinAlgError:
        chi2 = float(diff @ np.linalg.pinv(Vr) @ diff)
    return dict(chi2=chi2, df=k - 1, p=float(stats.chi2.sf(chi2, k - 1)), O=O, E=E)


# ------------------------------------------------------------------ paramétrico con censura
def _loglik(dist, theta, x, ev):
    if dist == "exponential":
        lam = np.exp(theta[0])
        return np.sum(ev * np.log(lam) - lam * x)
    if dist == "weibull":
        k, lam = np.exp(theta)
        z = x / lam
        return np.sum(ev * (np.log(k / lam) + (k - 1) * np.log(np.maximum(z, 1e-300))) - z ** k)
    if dist == "lognormal":
        mu, sig = theta[0], np.exp(theta[1])
        lx = np.log(np.maximum(x, 1e-300))
        return np.sum(ev * stats.norm.logpdf(lx, mu, sig) - ev * lx + (1 - ev) * stats.norm.logsf(lx, mu, sig))
    raise ValueError(dist)


def fit_parametric(time, event, dist="weibull", t0=0.0):
    """MLE con censura a la derecha, condicionado a sobrevivir a t0 (se usan sólo tiempos > t0 y x = t - t0).
    Devuelve dict(params, loglik, aic, n, d, sf(t))."""
    time = np.asarray(time, float); event = np.asarray(event, bool)
    m = time > t0
    x = time[m] - t0; ev = event[m].astype(float)
    if ev.sum() == 0:
        return dict(dist=dist, params=None, loglik=np.nan, aic=np.nan, n=int(m.sum()), d=0, t0=t0)
    mean_x = x[ev > 0].mean()
    inits = {"exponential": [np.log(ev.sum() / x.sum())], "weibull": [0.0, np.log(mean_x)],
             "lognormal": [np.log(mean_x), 0.0]}[dist]
    res = optimize.minimize(lambda th: -_loglik(dist, th, x, ev), inits, method="Nelder-Mead",
                            options=dict(maxiter=5000, xatol=1e-8, fatol=1e-10))
    th = res.x
    if dist == "exponential":
        params = dict(rate=float(np.exp(th[0])))
        sf = lambda t: np.exp(-params["rate"] * np.maximum(t - t0, 0))
    elif dist == "weibull":
        params = dict(shape=float(np.exp(th[0])), scale=float(np.exp(th[1])))
        sf = lambda t: np.exp(-(np.maximum(t - t0, 0) / params["scale"]) ** params["shape"])
    else:
        params = dict(mu=float(th[0]), sigma=float(np.exp(th[1])))
        sf = lambda t: stats.norm.sf((np.log(np.maximum(t - t0, 1e-300)) - params["mu"]) / params["sigma"])
    ll = -res.fun; kpar = len(th)
    return dict(dist=dist, params=params, loglik=float(ll), aic=float(2 * kpar - 2 * ll), n=int(m.sum()),
                d=int(ev.sum()), t0=t0, sf=sf)


def lr_test(fit_null, fit_alt, df=1):
    """Razón de verosimilitud para modelos anidados (exponencial ⊂ Weibull)."""
    lr = 2 * (fit_alt["loglik"] - fit_null["loglik"])
    return dict(LR=float(lr), df=df, p=float(stats.chi2.sf(max(lr, 0.0), df)))


def gof_bootstrap(time, event, fit, B=500, seed=0):
    """Bondad de ajuste con censura: D = sup|S_KM - S_fit| sobre t>t0; p-valor por bootstrap paramétrico
    (simula tiempos del ajuste, censura en la misma T_max de cada corrida, reajusta)."""
    rng = np.random.default_rng(seed)
    time = np.asarray(time, float); event = np.asarray(event, bool)
    t0 = fit["t0"]; m = time > t0
    tt, ev = time[m], event[m]
    cens_T = np.where(ev, np.nan, tt)
    Tmax = np.nanmax(cens_T) if np.isfinite(cens_T).any() else tt.max()
    def D(tt_, ev_, f):
        km = kaplan_meier(tt_ - t0, ev_)
        return float(np.max(np.abs(km["S"].to_numpy() - f["sf"](km["t"].to_numpy() + t0))))
    d_obs = D(tt, ev, fit)
    dist = fit["dist"]; p = fit["params"]
    ds = []
    for _ in range(B):
        if dist == "exponential":
            x = rng.exponential(1 / p["rate"], len(tt))
        elif dist == "weibull":
            x = p["scale"] * rng.weibull(p["shape"], len(tt))
        else:
            x = np.exp(p["mu"] + p["sigma"] * rng.standard_normal(len(tt)))
        c = Tmax - t0
        ev_b = x <= c; x_b = np.minimum(x, c)
        if ev_b.sum() < 2:
            continue
        fb = fit_parametric(x_b + t0, ev_b, dist, t0=t0)
        ds.append(D(x_b + t0, ev_b, fb))
    ds = np.array(ds)
    return dict(D=d_obs, p=float((ds >= d_obs).mean()) if len(ds) else np.nan, B=len(ds))


def piecewise_hazard(time, event, cuts):
    """Hazard constante a trozos en intervalos definidos por cuts (lista de cortes). Devuelve tasas e IC
    (Poisson) y LR test de hazard constante vs a trozos."""
    time = np.asarray(time, float); event = np.asarray(event, bool)
    edges = [0.0] + list(cuts) + [np.inf]
    rows = []; ll_pw = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        expo = np.clip(np.minimum(time, b) - a, 0, None).sum()
        d = int((event & (time > a) & (time <= b)).sum())
        rate = d / expo if expo > 0 else np.nan
        lo, hi = (stats.chi2.ppf(0.025, 2 * d) / (2 * expo) if d > 0 else 0.0,
                  stats.chi2.ppf(0.975, 2 * d + 2) / (2 * expo)) if expo > 0 else (np.nan, np.nan)
        rows.append(dict(t_from=a, t_to=b, events=d, exposure=expo, rate=rate, lo=lo, hi=hi))
        if d > 0:
            ll_pw += d * np.log(rate) - rate * expo
    D = event.sum(); E = time.sum(); ll_c = D * np.log(D / E) - D if D > 0 else 0.0
    lr = 2 * (ll_pw - ll_c)
    return pd.DataFrame(rows), dict(LR=float(lr), df=len(cuts), p=float(stats.chi2.sf(max(lr, 0.0), len(cuts))))


def early_late_mixture(time, event, t_split):
    """Mezcla 'átomo temprano + cola': p_early = P(extinción antes de t_split) con IC Wilson, y hazard de la
    cola entre los que sobreviven a t_split."""
    time = np.asarray(time, float); event = np.asarray(event, bool)
    n = len(time); k = int((event & (time <= t_split)).sum())
    lo, hi = wilson(k, n)
    m = time > t_split
    d = int(event[m].sum()); expo = (time[m] - t_split).sum()
    return dict(p_early=k / n, lo=lo, hi=hi, n=n, tail_events=d, tail_rate=d / expo if expo > 0 else np.nan,
                tail_rate_lo=(stats.chi2.ppf(0.025, 2 * d) / (2 * expo) if d > 0 and expo > 0 else 0.0),
                tail_rate_hi=(stats.chi2.ppf(0.975, 2 * d + 2) / (2 * expo) if expo > 0 else np.nan))


def wilson(k, n, conf=0.95):
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.5 + conf / 2); p = k / n
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def survival_summary(time, event, t0_tail=None, t_split=None, probs_at=(300, 500, 1000, 2000), B_gof=300):
    """Resumen completo: KM, mediana, P_ext(T), mezcla temprana/tardía, ajustes paramétricos sobre la cola."""
    time = np.asarray(time, float); event = np.asarray(event, bool)
    km = kaplan_meier(time, event)
    out = dict(n=len(time), events=int(event.sum()), censored=int((~event).sum()), km=km, median=km_median(km))
    out["P_ext"] = {T: float(1 - km_at(km, T)[0]) for T in probs_at if T <= time.max()}
    if t_split is None:
        t_split = float(np.nanmedian(time[event])) * 1.5 if event.any() else np.nan
    out["t_split"] = t_split
    if np.isfinite(t_split):
        out["mixture"] = early_late_mixture(time, event, t_split)
    t0 = t_split if t0_tail is None else t0_tail
    fits = {}
    if np.isfinite(t0) and (event & (time > t0)).sum() >= 3:
        for dist in ("exponential", "weibull", "lognormal"):
            fits[dist] = fit_parametric(time, event, dist, t0=t0)
        out["lr_weibull_vs_exp"] = lr_test(fits["exponential"], fits["weibull"])
        if B_gof:
            out["gof"] = {d: gof_bootstrap(time, event, fits[d], B=B_gof) for d in fits}
    out["fits"] = fits
    return out
