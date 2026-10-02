"""
ABM rápido (numpy) del modelo Universe 25 para alpha = 0 (movimiento aleatorio uniforme),
semánticamente idéntico al modelo original del notebook (preset `notebook` de u25) con ALPHA=0:
  move -> occupation -> social -> reproduce -> age -> deaths.

Modos de movimiento:
  'rw'  : paseo aleatorio en vecindario de Moore (incluye quedarse), bordes no periódicos
          (uniforme sobre las celdas válidas) -- es el modelo original con alpha=0.
  'mix' : cada individuo se reubica uniformemente en toda la grilla cada paso
          (campo medio cinético: ocupación exactamente Poisson(N/|G|), sin correlaciones espaciales
          madre-cría). Sirve para aislar el rol de la agregación por camadas.

Uso:  python abm_rapido.py   (corre un ejemplo y escribe un CSV en el directorio actual)
"""
import numpy as np

PAR = dict(
    G=30, K=8, r=0.01, d=0.10, p0=0.30, q0=0.95, a_min=6, a_max=50,
    L=np.array([3, 4, 5, 6, 7, 8, 9, 10]), pL=np.array([.03, .07, .15, .25, .23, .15, .08, .04]),
    a_life=60, k_mort=0.15, N0M=40, N0F=40, age0=4, s0=1.0,
)


def hazard(age, P):
    return 1.0 / (1.0 + np.exp(-P['k_mort'] * (age - P['a_life'])))


def run(T=500, seed=0, mode='rw', P=PAR, diag=False, stop_at_extinction=True):
    rng = np.random.default_rng(seed)
    G = P['G']; K = P['K']
    N0 = P['N0M'] + P['N0F']
    x = rng.integers(0, G, N0); y = rng.integers(0, G, N0)
    sex = np.r_[np.zeros(P['N0M'], bool), np.ones(P['N0F'], bool)]  # True = hembra
    age = np.full(N0, P['age0'], int)
    s = np.full(N0, P['s0'], float)
    out = []
    births = deaths = 0

    def stats(t, m, extra):
        N = len(s)
        rec = dict(time=t, population=N, births=births, deaths=deaths,
                   mean_age=age.mean() if N else 0.0, mean_social=s.mean() if N else 0.0,
                   low_social_fraction=(s < 0.2).mean() if N else 0.0,
                   mean_excess=np.maximum(m - K, 0).mean() if N else 0.0,   # D empírico por individuo
                   )
        rec.update(extra)
        return rec

    out.append(stats(0, np.ones(N0), dict(disp_index=np.nan, s3=np.nan, fert=np.nan, s_newborn=np.nan)))
    for t in range(1, T + 1):
        N = len(s)
        if N == 0:
            out.append(stats(t, np.zeros(0), dict(disp_index=np.nan, s3=np.nan, fert=np.nan, s_newborn=np.nan)))
            if stop_at_extinction:
                break
            continue
        # --- movimiento ---
        if mode == 'rw':
            nx = x + rng.integers(-1, 2, N); ny = y + rng.integers(-1, 2, N)
            bad = (nx < 0) | (nx >= G) | (ny < 0) | (ny >= G)
            while bad.any():  # rechazo => uniforme sobre celdas válidas de Moore
                nb = bad.sum()
                nx[bad] = x[bad] + rng.integers(-1, 2, nb); ny[bad] = y[bad] + rng.integers(-1, 2, nb)
                bad = (nx < 0) | (nx >= G) | (ny < 0) | (ny >= G)
            x, y = nx, ny
        else:
            x = rng.integers(0, G, N); y = rng.integers(0, G, N)
        cell = x * G + y
        occ = np.bincount(cell, minlength=G * G)
        m = occ[cell]
        # --- estado social ---
        s = np.clip(s + P['r'] - P['d'] * np.maximum(m - K, 0), 0.0, 1.0)
        # --- reproducción ---
        rep = (age >= P['a_min']) & (age <= P['a_max'])
        im = np.flatnonzero(rep & ~sex); iF = np.flatnonzero(rep & sex)
        order = np.argsort(cell[im], kind='stable'); im = im[order]
        cm = cell[im]
        cntM = np.bincount(cm, minlength=G * G); start = np.r_[0, np.cumsum(cntM)[:-1]]
        cf = cell[iF]
        ok = cntM[cf] > 0
        iF = iF[ok]; cf = cf[ok]
        j = start[cf] + (rng.random(len(iF)) * cntM[cf]).astype(int)
        mates = im[j]
        pconc = P['p0'] * s[iF] * s[mates]
        fert = (s[iF] ** 2 * s[mates]).mean() if len(iF) else np.nan
        conc = rng.random(len(iF)) < pconc
        moms = iF[conc]
        lit = rng.choice(P['L'], size=len(moms), p=P['pL'])
        mom_rep = np.repeat(moms, lit)
        surv = rng.random(len(mom_rep)) < P['q0'] * s[mom_rep]
        mom_rep = mom_rep[surv]
        nb_ = len(mom_rep)
        s_new = s[mom_rep].mean() if nb_ else np.nan
        x = np.r_[x, x[mom_rep]]; y = np.r_[y, y[mom_rep]]
        sex = np.r_[sex, rng.random(nb_) < 0.5]
        age = np.r_[age, np.zeros(nb_, int)]
        s = np.r_[s, s[mom_rep]]
        births = nb_
        # --- envejecimiento y muerte ---
        age = age + 1
        die = rng.random(len(age)) < hazard(age, P)
        deaths = int(die.sum())
        keep = ~die
        x, y, sex, age, s = x[keep], y[keep], sex[keep], age[keep], s[keep]
        extra = dict(disp_index=occ.var() / occ.mean() if occ.mean() > 0 else np.nan,
                     s3=(s.mean() ** 3) if len(s) else np.nan, fert=fert, s_newborn=s_new)
        m_now = occ[x * G + y] if len(s) else np.zeros(0)
        out.append(stats(t, m_now, extra))
    return out


if __name__ == '__main__':
    import pandas as pd, time
    t0 = time.time()
    h = run(T=500, seed=1, mode='rw')
    print('tiempo', time.time() - t0)
    df = pd.DataFrame(h)
    print(df[['time', 'population', 'births', 'deaths', 'mean_social', 'low_social_fraction', 'disp_index', 'mean_excess']].iloc[::10].to_string())
