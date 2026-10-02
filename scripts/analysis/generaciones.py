"""Tabla por generaciones y cronología de fases.

Registro por id que sobrevive a la muerte del animal (así la fecundidad realizada es la de toda la vida).
Subclase de Universe, sin tocar u25/.

Uso (desde la raíz; n_jobs=2):
  nice -n 10 python3 scripts/analysis/generaciones.py OUT.pkl preset:seeds[:k=v,...] ...
  ej.: calhoun_C1:1-8  calhoun_C1p:1-6  longevo:1-4  calhoun_C1:1-6:ages=4-52
La opción ages=a-b da una condición inicial NO sincrónica exploratoria: edades uniformes enteras en
[a, b], s = 1, posiciones al azar (la versión de producción es Params.initial_age_dist de u25/).
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from multiprocessing import Pool
from u25.params import preset
from u25.model import Universe

AGES_REC = (6, 12, 26, 52)


class GenU(Universe):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cap = 0
        self._grow(4096)
        n = self.N
        self.r_gen[:n] = 0
        self.r_male[:n] = self.male
        self.r_birth[:n] = -self.age
        self.r_sbirth[:n] = np.nan
        self.log = []
        self._bg = np.zeros(12, np.int64)

    def _grow(self, need):
        if need <= self._cap:
            return
        new = max(need, 2 * self._cap)
        def ext(a, fill, dt):
            b = np.full(new, fill, dt)
            if a is not None:
                b[: a.shape[0]] = a
            return b
        g = lambda n: getattr(self, n, None)
        self.r_gen = ext(g("r_gen"), -1, np.int64)
        self.r_male = ext(g("r_male"), False, bool)
        self.r_birth = ext(g("r_birth"), 0, np.int64)
        self.r_death = ext(g("r_death"), -1, np.int64)
        self.r_sbirth = ext(g("r_sbirth"), np.nan, float)
        for A in AGES_REC:
            setattr(self, f"r_s{A}", ext(g(f"r_s{A}"), np.nan, float))
        self.r_pups = ext(g("r_pups"), 0, np.int64)
        self.r_daught = ext(g("r_daught"), 0, np.int64)
        self.r_dmg = ext(g("r_dmg"), 0.0, float)      # daño por hacinamiento acumulado
        self._cap = new

    def social_delta(self, m):
        p = self.p
        d = super().social_delta(m)
        det = p.gamma * np.maximum(0, m - p.m_c)
        self.r_dmg[self.id] += det
        return d

    def update_social_state(self, m):
        super().update_social_state(m)
        for A in AGES_REC:
            k = self.age == A
            if k.any():
                getattr(self, f"r_s{A}")[self.id[k]] = self.s[k]

    def newborn_s(self, mothers):
        self._mrep = mothers
        return super().newborn_s(mothers)

    def reproduce(self, cell):
        n0 = self.N
        self._mrep = None
        super().reproduce(cell)
        nb = self.N - n0
        self._bg[:] = 0
        if nb > 0:
            self._grow(self.next_id + 1)
            ids = self.id[-nb:]
            mom = self.id[self._mrep]
            self.r_gen[ids] = self.r_gen[mom] + 1
            self.r_male[ids] = self.male[-nb:]
            self.r_birth[ids] = self.t
            self.r_sbirth[ids] = self.s[-nb:]
            np.add.at(self.r_pups, mom, 1)
            np.add.at(self.r_daught, mom, (~self.male[-nb:]).astype(np.int64))
            self._bg = np.bincount(np.minimum(self.r_gen[ids], 11), minlength=12)

    def adult_deaths(self):
        before = self.id.copy()
        super().adult_deaths()
        dead = np.setdiff1d(before, self.id, assume_unique=True)
        self.r_death[dead] = self.t

    def step(self):
        super().step()
        n = self.N
        ad = self.age >= self.p.repro_age_min
        gen_alive = np.bincount(np.minimum(self.r_gen[self.id], 11), minlength=12) if n else np.zeros(12, int)
        self.log.append(dict(t=self.t, N=n, B=self.births, Nad=int(ad.sum()),
                             Sad=float(self.s[ad].mean()) if ad.any() else np.nan,
                             S=float(self.s.mean()) if n else np.nan,
                             f_over=float((self._m > self.p.m_c).mean()) if n else np.nan,
                             f0=float(1 - (np.bincount(self.x * self.p.grid_size + self.y,
                                       minlength=self.p.grid_size ** 2) > 0).mean()) if n else np.nan,
                             bg=self._bg.copy(), gen_alive=gen_alive))


def parse(spec):
    parts = spec.split(":")
    name = parts[0]
    a, b = parts[1].split("-")
    seeds = list(range(int(a), int(b) + 1))
    kw, ages = {}, None
    for extra in parts[2:]:
        for kv in extra.split(","):
            k, v = kv.split("=")
            if k == "ages":
                ages = tuple(int(x) for x in v.split("-"))
            else:
                try:
                    kw[k] = float(v) if "." in v else int(v)
                except ValueError:
                    kw[k] = v
    return name, seeds, kw, ages


def one(job):
    name, seed, kw, ages, tag = job
    p = preset(name, **kw)
    rng = np.random.default_rng(10_000 + seed)
    U = GenU(p, rng=rng)
    if ages is not None:
        U.age = rng.integers(ages[0], ages[1] + 1, size=U.N).astype(np.int64)
        U.birth_t = -U.age.copy()
        U.r_birth[: U.N] = -U.age
    S, summ = U.run(600)
    n = U.next_id
    reg = {k[2:]: getattr(U, k)[:n].copy() for k in dir(U) if k.startswith("r_") and isinstance(getattr(U, k), np.ndarray)}
    return dict(tag=tag, preset=name, seed=seed, kw=kw, ages=ages, log=U.log, reg=reg,
                series=S[["t", "N", "births", "S_adult", "S_mean", "f_over", "occ_frac_cells", "N_adult",
                          "phi_BO", "frac_crowded_low", "frac_withdrawn", "corr_s_m", "h", "mean_age"]].copy(),
                summ={k: v for k, v in summ.items() if np.isscalar(v)})


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = []
    for spec in sys.argv[2:]:
        name, seeds, kw, ages = parse(spec)
        for s in seeds:
            jobs.append((name, s, kw, ages, spec))
    with Pool(2) as P:
        res = P.map(one, jobs, chunksize=1)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print("ok", len(res))
