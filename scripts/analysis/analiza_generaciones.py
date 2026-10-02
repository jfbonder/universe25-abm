"""Tabla por generaciones y cronología de fases a partir de la salida de generaciones.py."""
import sys, pickle
import numpy as np
import pandas as pd

CAL = dict(  # Calhoun 1973, en semanas desde la colonización (día/7)
    t_B1=104 / 7, t_endB=315 / 7, t_pk=560 / 7, t_last=600 / 7, t_lastconc=920 / 7, T_ext=1780 / 7)


def gen_table(runs, gmax=5):
    rows = []
    for r in runs:
        R = r["reg"]
        n = len(R["gen"])
        tot_b = (R["gen"] >= 1).sum()
        for g in range(0, gmax + 1):
            k = R["gen"] == g
            if g > 0 and not k.any():
                rows.append(dict(seed=r["seed"], g=g, n=0))
                continue
            f = k & ~R["male"]
            rows.append(dict(seed=r["seed"], g=g, n=int(k.sum()),
                             share=k.sum() / tot_b if g > 0 else np.nan,
                             t_birth_med=float(np.median(R["birth"][k])) if g > 0 else np.nan,
                             s_birth=float(np.nanmean(R["sbirth"][k])) if g > 0 else 1.0,
                             s6=float(np.nanmean(R["s6"][k])), s12=float(np.nanmean(R["s12"][k])),
                             s26=float(np.nanmean(R["s26"][k])),
                             f12_hi=float(np.mean(R["s12"][k][~np.isnan(R["s12"][k])] >= 0.5)) if (~np.isnan(R["s12"][k])).any() else np.nan,
                             dmg=float(R["dmg"][k].mean()) if "dmg" in R else np.nan,
                             pups_per_f=float(R["pups"][f].mean()) if f.any() else np.nan,
                             R_net=float(R["daught"][f].mean()) if f.any() else np.nan,
                             f_bred=float((R["pups"][f] > 0).mean()) if f.any() else np.nan))
    D = pd.DataFrame(rows)
    D["n_gb"] = D["n"] * D["g"]
    agg = D.groupby("g").agg(n=("n", "mean"), n_sd=("n", "std"), share=("share", "mean"),
                             t_birth_med=("t_birth_med", "median"), s_birth=("s_birth", "mean"),
                             s6=("s6", "mean"), s12=("s12", "mean"), s26=("s26", "mean"),
                             f12_hi=("f12_hi", "mean"), dmg=("dmg", "mean"),
                             pups_per_f=("pups_per_f", "mean"), R_net=("R_net", "mean"),
                             f_bred=("f_bred", "mean"))
    return agg


def extra(r):
    """Generación media de los nacimientos, g* (primera generación con R_net < 1) y la 'última mitad' de
    los nacidos (Calhoun: 'the last half of the population born')."""
    R = r["reg"]
    b = R["gen"] >= 1
    gmean = R["gen"][b].mean()
    gstar = np.nan
    for g in range(0, 12):
        f = (R["gen"] == g) & ~R["male"]
        if f.any() and R["daught"][f].mean() < 1:
            gstar = g
            break
    tb = R["birth"][b]
    t50 = np.median(tb)
    late = b & (R["birth"] >= t50)
    fl = late & ~R["male"]
    return dict(g_mean=gmean, g_star=gstar, lh_n=int(late.sum()), lh_s_birth=np.nanmean(R["sbirth"][late]),
                lh_s12=np.nanmean(R["s12"][late]), lh_f_bred=(R["pups"][fl] > 0).mean() if fl.any() else np.nan,
                lh_Rnet=R["daught"][fl].mean() if fl.any() else np.nan,
                share_F0_mothers=(R["gen"][b] == 1).mean())


def chrono(r):
    L = pd.DataFrame([{k: v for k, v in x.items() if k not in ("bg", "gen_alive")} for x in r["log"]])
    t = L.t.to_numpy(); N = L.N.to_numpy(); B = L.B.to_numpy(); Sad = L.Sad.to_numpy()
    Npk = N.max(); ipk = int(N.argmax()); tpk = t[ipk]
    inpl = np.flatnonzero(N >= 0.98 * Npk)
    tpk1, tpl_end = t[inpl[0]], t[inpl[-1]]
    cb = np.cumsum(B); tot = cb[-1]
    q = lambda x: t[np.searchsorted(cb, x * tot)]
    nz = np.flatnonzero(B > 0)
    i05 = np.flatnonzero(Sad < 0.5); i02 = np.flatnonzero(Sad < 0.2)
    t05 = t[i05[0]] if i05.size else np.nan
    # nacimientos de la última "cola": semanas con B>0 después de t90
    D = dict(t_B1=t[nz[0]], t_Bmax=t[int(B.argmax())], Bmax=int(B.max()), t50=q(0.5), t90=q(0.9), t99=q(0.99),
             t_last=t[nz[-1]], t_Sad05=t05, N_at_Sad05=N[i05[0]] / Npk if i05.size else np.nan,
             t_Sad02=t[i02[0]] if i02.size else np.nan,
             tpk1=tpk1, tpk=tpk, tpl_end=tpl_end, plateau=tpl_end - tpk1, T_ext=t[-1] if N[-1] == 0 else np.nan,
             Npk=Npk, births=int(tot))
    # crecimiento: tiempo de duplicación entre N0 y N0*2 y entre 0.25 y 0.5 Npk
    def tcross(x):
        j = np.flatnonzero(N >= x)
        return t[j[0]] if j.size else np.nan
    D["Td_first"] = tcross(2 * N[0]) - 0
    D["Td_mid"] = tcross(0.5 * Npk) - tcross(0.25 * Npk)
    D["Npk_over_N0"] = Npk / N[0]
    return D


if __name__ == "__main__":
    res = pickle.load(open(sys.argv[1], "rb"))
    tags = list(dict.fromkeys(r["tag"] for r in res))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for tag in tags:
        runs = [r for r in res if r["tag"] == tag]
        print(f"\n===== {tag}  (n={len(runs)})")
        print(gen_table(runs).round(3).to_string())
        E = pd.DataFrame([extra(r) for r in runs])
        print("extra (medianas):", E.median().round(3).to_dict())
        C = pd.DataFrame([chrono(r) for r in runs])
        med = C.median(numeric_only=True)
        print("cronología (medianas):")
        print(med.round(2).to_string())
        C["tpk_n"] = C.t90 / C.tpk
        print("t90/tpk", C.tpk_n.median().round(3), " t50/tpk", (C.t50 / C.tpk).median().round(3),
              " t_Sad05/tpk", (C.t_Sad05 / C.tpk).median().round(3), " t_last/tpk", (C.t_last / C.tpk).median().round(3),
              " t_last/tpk1", (C.t_last / C.tpk1).median().round(3), " (Text-tpk)/tpk", ((C.T_ext - C.tpk) / C.tpk).median().round(3))
