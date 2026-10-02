"""Ensamble del ABM rápido (alpha=0): extinción, picos sucesivos, comparación rw vs mix."""
import numpy as np, pandas as pd, sys, time
from abm_rapido import run
from scipy.signal import find_peaks

def analiza(h, T):
    df = pd.DataFrame(h)
    N = df.population.values
    ext = df.time[N == 0].min() if (N == 0).any() else np.nan
    pk, _ = find_peaks(N, prominence=500, distance=40)
    tr, _ = find_peaks(-N, prominence=500, distance=40)
    return dict(ext=ext, Nmax=N.max(), t_pico=int(df.time[np.argmax(N)]), Smin=df.mean_social[N>0].min(),
                npicos=len(pk), picos=list(N[pk]), t_picos=list(df.time.values[pk]), valles=list(N[tr]),
                periodo=np.diff(df.time.values[pk]).mean() if len(pk) > 1 else np.nan)

if __name__ == '__main__':
    mode = sys.argv[1]; R = int(sys.argv[2]); T = int(sys.argv[3])
    t0 = time.time(); rows = []
    for seed in range(R):
        rows.append(analiza(run(T=T, seed=seed, mode=mode), T))
    df = pd.DataFrame(rows); df.to_pickle(f'ens_{mode}_{R}x{T}.pkl')
    print(mode, 'tiempo total', round(time.time()-t0,1), 's')
    print(df[['ext','Nmax','t_pico','Smin','npicos','periodo']].describe().to_string())
    print('fraccion extinta en T:', df.ext.notna().mean())
    print('picos (primeras 6 corridas):'); [print(p, v) for p, v in zip(df.picos[:6], df.valles[:6])]
