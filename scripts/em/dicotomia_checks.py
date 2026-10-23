"""Chequeos numéricos sobre el mapa F (= ecuación cinética "exact",
pair="shared", con M = 1/h nodos de s y edades truncadas en A; el reparto lineal entre nodos es la esperanza del
redondeo estocástico ρ_h de el modelo truncado, así que coincide con F salvo la autoexclusión O(r/K) del
macho en el bloque).

Partes (argumento): orbit  -> (a)+(c): órbita de C1-trunc y SL-trunc: φ_t, t_0, t_0(δ), θ_t, cohortes, R̂_1(t), t_N
                    omega  -> (b): ω-límite de SL-trunc con r = 0,03 y 0,05 (ciclo): período, valle, ρ(DF^p)
                    sens   -> (e): sensibilidad a h y A
Uso: python3 scripts/em/dicotomia_checks.py orbit|omega|sens"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kinetic import Kinetic, params_C1, params_SL
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results_em", "kinetic")
EPS_STAR = {"C1": 2 / (1.8668 * 75), "SL": 2 / (1.8668 * 45)}
ETA_MAX = {k: (np.sqrt(2) - 1) ** 2 * v for k, v in EPS_STAR.items()}
LS = (30, 60, 90, 120, 240)


def orbit(model, M=100, A=None, T=400, r1_every=2, r1_until=200, extra=None):
    p = params_C1() if model == "C1" else params_SL()
    if extra:
        p = p.replace(**extra)
    A = A or (180 if p.a_max > 100 else 120)
    kin = Kinetic(p, M=M, A=A, mode="exact")
    rows = []
    for t in range(T):
        r1s = r1n = np.nan
        if r1_every and kin.t % r1_every == 0 and kin.t <= r1_until and kin.F.sum() > 1e-12:
            r1s = kin.R1_frozen()
            nbd = getattr(kin, "last_nb", None)
            r1n = kin.R1_frozen(nbd) if nbd is not None else np.nan
        rec = kin.step()
        rec["R1_s1"] = r1s; rec["R1_nb"] = r1n
        rows.append(rec)
        if kin.F.sum() < 1e-300:
            break
    return pd.DataFrame(rows)


def summarize_orbit(model, df):
    eta = ETA_MAX[model]
    out = dict(model=model, eps_star=EPS_STAR[model], eta_max=eta)
    t = df.t.to_numpy(); phi = df.phi.to_numpy(); B = df.births.to_numpy()
    def first(mask):
        i = np.flatnonzero(mask); return int(t[i[0]]) if i.size else None
    out["t_peak_nu"] = int(t[np.argmax(df.nu)]); out["nu_peak"] = float(df.nu.max())
    out["t_0"] = first(phi < eta)
    out["t_0_delta0.1"] = first(phi <= 0.1 * eta / 2)
    out["t_theta_below_2/3"] = first(df.theta.to_numpy() < 2 / 3)
    out["t_thetaphi_below_2/3"] = first(df.theta_phi.to_numpy() < 2 / 3)
    out["t_B_peak"] = int(t[np.argmax(B)])
    out["t_B_below_1pct"] = first((t > t[np.argmax(B)]) & (B < 0.01 * B.max()))
    out["t_N"] = {L: first((t > t[np.argmax(B)]) & (L * L * B <= 1)) for L in LS}
    out["t_nuG_below_half"] = {L: first((t > out["t_peak_nu"]) & (L * L * df.nu.to_numpy() < 0.5)) for L in LS}
    r = df.dropna(subset=["R1_s1"])
    out["R1_s1_first_below1"] = first_t(r, r.R1_s1 < 1)
    out["R1_nb_first_below1"] = first_t(r, r.R1_nb < 1)
    # tasa de decaimiento de B y φ en la fase final (ajuste log-lineal donde 1e-8 < B < 1e-3)
    m = (t > t[np.argmax(B)]) & (B > 1e-12) & (B < 1e-3)
    out["B_decay_rate"] = float(-np.polyfit(t[m], np.log(B[m]), 1)[0]) if m.sum() > 3 else None
    m = (t > t[np.argmax(B)]) & (phi > 1e-12) & (phi < 1e-2)
    out["phi_decay_rate"] = float(-np.polyfit(t[m], np.log(phi[m]), 1)[0]) if m.sum() > 3 else None
    out["phi_min"] = float(phi.min()); out["t_end"] = int(t[-1])
    return out


def first_t(r, mask):
    i = np.flatnonzero(mask.to_numpy()); return int(r.t.iloc[i[0]]) if i.size else None


# ------------------------------------------------------------------ (b) ω-límite y multiplicadores
def omega(model_extra, label, M=100, T=4000):
    p = params_SL(**model_extra)
    kin = Kinetic(p, M=M, A=120, mode="exact")
    hist = []
    for t in range(T):
        kin.step()
        if t >= T - 1200:
            hist.append(kin.F.copy())
    X = np.array(hist)
    nu = X.sum((1, 2))
    # período: p que minimiza ||x_{t+p} - x_t||_1 / ||x_t||_1 en la ventana final
    errs = {}
    for pp in range(2, 400):
        d = np.abs(X[-1] - X[-1 - pp]).sum() / X[-1].sum()
        errs[pp] = d
    pbest = min(errs, key=errs.get)
    res = dict(label=label, period=pbest, rel_return_err=errs[pbest], nu_min=float(nu.min()), nu_max=float(nu.max()),
               nu_mean=float(nu.mean()), phi_min=float(nu.min()), eps_star=EPS_STAR["SL"],
               valley_in_eps_star=float(nu.min() / EPS_STAR["SL"]))
    # ρ(DF^p) por iteración de potencias con diferencias finitas sobre x* = X[-1]
    xs = X[-1].copy()
    def Fp(x):
        k2 = Kinetic(p, M=M, A=120, mode="exact", F0=x)
        for _ in range(pbest):
            k2.step()
        return k2.F
    base = Fp(xs)
    rng = np.random.default_rng(0)
    v = rng.random(xs.shape) * (xs > 0); v /= np.abs(v).sum()
    lams = []
    for it in range(25):
        e = 1e-6 * xs.sum()
        w = (Fp(np.maximum(xs + e * v, 0)) - base) / e
        lam = np.abs(w).sum()
        lams.append(lam)
        v = w / lam
    res["rho_DFp_power"] = lams[-1]; res["rho_iter_tail"] = lams[-5:]
    res["rho_per_week"] = lams[-1] ** (1 / pbest)
    return res, nu


def main():
    part = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if part == "orbit":
        res = {}
        for model in ("C1", "SL"):
            t0 = time.time()
            df = orbit(model)
            df.to_csv(os.path.join(OUT, f"dicot_orbit_{model}.csv"), index=False)
            res[model] = summarize_orbit(model, df)
            print(model, f"{time.time()-t0:.0f}s", json.dumps(res[model]), flush=True)
        json.dump(res, open(os.path.join(OUT, "dicot_orbit.json"), "w"), indent=1)
    elif part == "omega":
        res = []
        for ex, lab in ((dict(rec=0.03), "SL r=0,03"), (dict(rec=0.05), "SL r=0,05"), (dict(rec=0.1), "SL r=0,1")):
            t0 = time.time()
            r, nu = omega(ex, lab)
            print(lab, f"{time.time()-t0:.0f}s", r, flush=True)
            res.append(r)
        json.dump(res, open(os.path.join(OUT, "dicot_omega.json"), "w"), indent=1, default=float)
    elif part == "sens":
        rows = []
        for model in ("C1", "SL"):
            for M in (50, 100, 200):
                for A in ((160, 180, 200) if model == "C1" else (100, 120, 140)):
                    if M != 100 and A != (180 if model == "C1" else 120):
                        continue
                    df = orbit(model, M=M, A=A, r1_every=0)
                    s = summarize_orbit(model, df)
                    rows.append(dict(model=model, h=1 / M, A=A, nu_peak=s["nu_peak"], t_peak=s["t_peak_nu"], t_0=s["t_0"],
                                     t_0d=s["t_0_delta0.1"], t_N30=s["t_N"][30], t_N240=s["t_N"][240],
                                     t_ext30=s["t_nuG_below_half"][30], B_rate=s["B_decay_rate"]))
                    print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "dicot_sens.csv"), index=False)


if __name__ == "__main__":
    main()
