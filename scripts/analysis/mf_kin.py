"""
Campo medio CINÉTICO con desorden de celdas (sin geometría): Monte Carlo de individuos con edad, sexo y s.
Las celdas tienen pesos fijos w_c ~ Gamma(kappa_c, 1/kappa_c) (media 1): la ocupación es Poisson-Gamma
(binomial negativa) con índice de dispersión 1 + lambda/kappa_c, como la del ABM con afinidades.
Cada individuo permanece en su celda y con probabilidad 1/tau_F por semana se reubica en una celda
elegida con probabilidad ∝ w_c (persistencia de 'refugio' o 'cluster').  Los neonatos nacen en la celda
de la madre.  Sin estructura espacial no hay difusión ni fronteras: es la aproximación de campo medio con
heterogeneidad local persistente ('quenched').

Reglas (flags): R1 ventana crítica a_w (recuperación sólo si edad < a_w); R2 aprendizaje social lam2
(recuperación * [(1-lam2) + lam2 * s medio de los OTROS de la celda], aislado => 0); escenario longevo
(a_life, a_fert_max); umbral de no retorno s_h (R2').

Uso: python mf_kin.py calib | sweep | transplante
"""
import numpy as np, sys, time

LIT = np.array([3, 4, 5, 6, 7, 8, 9, 10]); PLIT = np.array([.03, .07, .15, .25, .23, .15, .08, .04])


def run(T=1500, seed=0, N0=400, age0=4, s0=1.0, G2=900, kappa_c=0.15, tau_F=20.0, K=8, rec=0.01, gamma=0.10,
        p0=0.30, q0=0.95, a_min=6, a_fert_max=50, a_life=60.0, k_mort=0.15, a_w=np.inf, lam2=0.0, s_h=0.0,
        init_same_cell=False, init_ages=None, stop_at_ext=True, record=True):
    rng = np.random.default_rng(seed)
    w = rng.gamma(kappa_c, 1.0 / kappa_c, G2) if np.isfinite(kappa_c) else np.ones(G2)
    pw = w / w.sum()
    N = N0
    if init_same_cell:
        cell = np.full(N, int(np.argmax(pw)))  # todos juntos en una celda
    else:
        cell = rng.choice(G2, N, p=pw)
    sex = rng.random(N) < 0.5  # True = hembra
    age = np.full(N, age0, int) if init_ages is None else np.asarray(init_ages, int).copy()
    s = np.full(N, float(s0))
    out = []
    for t in range(T + 1):
        N = len(s)
        if record:
            if N:
                rep = (age >= a_min) & (age <= a_fert_max)
                nf = (rep & sex).sum()
                out.append((t, N, s.mean(), (s < 0.2).mean(), ((rep & sex & (s > 0.3)).sum() / nf) if nf else 0.0, age.mean()))
            else:
                out.append((t, 0, np.nan, np.nan, 0.0, np.nan))
        if N == 0:
            if stop_at_ext: break
            continue
        # reubicación con persistencia
        mv = rng.random(N) < 1.0 / tau_F
        if mv.any():
            cell[mv] = rng.choice(G2, mv.sum(), p=pw)
        occ = np.bincount(cell, minlength=G2)
        m = occ[cell]
        # estado social
        R = np.ones(N)
        if lam2 > 0:
            ssum = np.bincount(cell, weights=s, minlength=G2)
            others = m - 1
            sbar = np.where(others > 0, (ssum[cell] - s) / np.maximum(others, 1), 0.0)
            R = (1 - lam2) + lam2 * sbar
        if np.isfinite(a_w):
            R = R * (age < a_w)
        if s_h > 0:
            R = R * (s >= s_h)
        s = np.clip(s + rec * R - gamma * np.maximum(m - K, 0), 0.0, 1.0)
        # reproducción
        rep = (age >= a_min) & (age <= a_fert_max)
        im = np.flatnonzero(rep & ~sex); iF = np.flatnonzero(rep & sex)
        if len(im) and len(iF):
            order = np.argsort(cell[im], kind='stable'); im = im[order]; cm = cell[im]
            cnt = np.bincount(cm, minlength=G2); start = np.r_[0, np.cumsum(cnt)[:-1]]
            cf = cell[iF]; ok = cnt[cf] > 0; iF = iF[ok]; cf = cf[ok]
            mates = im[start[cf] + (rng.random(len(iF)) * cnt[cf]).astype(int)]
            conc = rng.random(len(iF)) < p0 * s[iF] * s[mates]
            moms = iF[conc]
            lit = rng.choice(LIT, size=len(moms), p=PLIT)
            surv_n = rng.binomial(lit, np.clip(q0 * s[moms], 0, 1))
            mom_rep = np.repeat(moms, surv_n)
            nb = len(mom_rep)
            if nb:
                cell = np.r_[cell, cell[mom_rep]]; sex = np.r_[sex, rng.random(nb) < 0.5]
                age = np.r_[age, np.zeros(nb, int)]; s = np.r_[s, s[mom_rep]]
        # envejecer y morir
        age = age + 1
        die = rng.random(len(age)) < 1 / (1 + np.exp(-k_mort * (age - a_life)))
        keep = ~die
        cell, sex, age, s = cell[keep], sex[keep], age[keep], s[keep]
    return np.array(out) if record else None


def resumen(o, T):
    N = o[:, 1]; ext = N[-1] == 0
    i = int(np.argmax(N)); post = N[i:]
    reb = post[np.argmin(post):].max() / N[i] if len(post) > 1 else 0
    return dict(N_max=N[i], t_pk=o[i, 0], ext=bool(ext), t_ext=o[-1, 0] if ext else np.nan, N_min=post.min(), rebote=reb,
                S_min=np.nanmin(o[:, 2]), ID=np.nan)


def ensamble(n=10, T=1500, **kw):
    rs = [resumen(run(T=T, seed=k, **kw), T) for k in range(n)]
    P = np.mean([r['ext'] for r in rs]); P200 = np.mean([(r['t_ext'] <= 250) if r['ext'] else False for r in rs])
    return dict(P_ext=P, P_ext250=P200, N_max=np.mean([r['N_max'] for r in rs]), rebote=np.mean([r['rebote'] for r in rs]),
                t_ext=np.nanmean([r['t_ext'] for r in rs]) if P > 0 else np.nan, N_min=np.median([r['N_min'] for r in rs]))


def transplante(n=50, s_tr=0.02, edades=(6, 30), k_pairs=4, T=300, G2_local=25, tau_F=5.0, **kw):
    rng = np.random.default_rng(123)
    ok = 0
    for k in range(n):
        ages = rng.integers(edades[0], edades[1] + 1, 2 * k_pairs)
        # grilla local de G2_local celdas uniformes: los fundadores se mueven en un vecindario pequeño (sin Allee global)
        o = run(T=T, seed=1000 + k, N0=2 * k_pairs, s0=s_tr, init_same_cell=True, init_ages=ages, kappa_c=np.inf, G2=G2_local, tau_F=tau_F, **kw)
        ok += o[:, 1].max() > 10 * 2 * k_pairs
    return ok / n


if __name__ == '__main__':
    modo = sys.argv[1] if len(sys.argv) > 1 else 'calib'
    t0 = time.time()
    if modo == 'calib':
        print("Calibración (base, alpha=1.4 equivalente): objetivo P_ext(250)≈0.5, P_ext(2000)≈0.55, N_max≈5500, QS N≈1700")
        for kappa_c in (0.1, 0.15, 0.25, 0.5):
            for tau_F in (5, 20, 60):
                r = ensamble(n=8, T=1500, kappa_c=kappa_c, tau_F=tau_F)
                print(f" kappa_c={kappa_c} tau_F={tau_F}: {r}")
    elif modo == 'sweep':
        kc, tf = float(sys.argv[2]), float(sys.argv[3])
        base = dict(kappa_c=kc, tau_F=tf)
        for lon in (False, True):
            ex = dict(a_life=120.0, a_fert_max=80) if lon else {}
            print(f"\n== R1 (longevo={lon}) P_ext | rebote medio")
            for rec in (0.01, 0.03):
                row = []
                for aw in (6, 12, 24, 48, np.inf):
                    r = ensamble(n=10, T=1500, rec=rec, a_w=aw, **base, **ex)
                    row.append(f"a_w={aw}: {r['P_ext']:.1f}|{r['rebote']:.2f}")
                print(f" rec={rec}: " + "  ".join(row))
            print(f"== R2 (longevo={lon}) P_ext | rebote medio")
            for rec in (0.01, 0.03):
                row = []
                for l2 in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
                    r = ensamble(n=10, T=1500, rec=rec, lam2=l2, **base, **ex)
                    row.append(f"λ={l2}: {r['P_ext']:.1f}|{r['rebote']:.2f}")
                print(f" rec={rec}: " + "  ".join(row))
    elif modo == 'transplante':
        print("P_T (4 pares fase D, s=0.02, edades 6-30, misma celda, n=50)")
        for nombre, kw in [("base rec=.01", dict(rec=0.01)), ("base rec=.03", dict(rec=0.03)),
                           ("R1 a_w=12 rec=.03", dict(rec=0.03, a_w=12)), ("R1 a_w=12 rec=.01", dict(rec=0.01, a_w=12)),
                           ("R2 λ=1 rec=.03", dict(rec=0.03, lam2=1.0)), ("R2 λ=0.7 rec=.03", dict(rec=0.03, lam2=0.7)), ("R2 λ=0.5 rec=.03", dict(rec=0.03, lam2=0.5)),
                           ("base rec=.01 longevo", dict(rec=0.01, a_life=120., a_fert_max=80)), ("base rec=.03 longevo", dict(rec=0.03, a_life=120., a_fert_max=80)),
                           ("R1 a_w=12 rec=.03 longevo", dict(rec=0.03, a_w=12, a_life=120., a_fert_max=80)),
                           ("R2 λ=1 rec=.03 longevo", dict(rec=0.03, lam2=1.0, a_life=120., a_fert_max=80)),
                           ("R2 λ=0.7 rec=.03 longevo", dict(rec=0.03, lam2=0.7, a_life=120., a_fert_max=80))]:
            print(f" {nombre}: P_T={transplante(**kw):.2f}")
        print("Control: adultos de fase B (s=1, edades 6-30): base", transplante(s_tr=1.0), " R1:", transplante(s_tr=1.0, rec=0.03, a_w=12), " R2 λ=1:", transplante(s_tr=1.0, lam2=1.0))
    print("tiempo", round(time.time() - t0, 1), "s")
