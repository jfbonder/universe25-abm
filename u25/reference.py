"""Puerto fiel (lento) del código del notebook original (implementación previa del modelo), sólo para validación.

Se reemplazaron las constantes globales por un `Params` y el RNG global por instancias
`random.Random` / `np.random.RandomState` con semilla. La lógica es la del notebook.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass

import numpy as np

from .params import Params


@dataclass
class Mouse:
    id: int
    sex: str
    x: int
    y: int
    age: int
    social: float


def notebook_update_affinities(affinity: dict, occupation: dict, p: Params) -> set:
    """`Universe.update_affinities` del notebook (modifica `affinity` in situ)."""
    R = 1
    active_pairs = set()
    offsets = []
    for dx in range(R + 1):
        for dy in range(-R, R + 1):
            if dx == 0 and dy < 0:
                continue
            offsets.append((dx, dy))
    for (x, y), ids1 in occupation.items():
        for i in range(len(ids1)):
            for j in range(i):
                active_pairs.add(tuple(sorted((ids1[i], ids1[j]))))
        for dx, dy in offsets[1:]:
            cell2 = (x + dx, y + dy)
            if cell2 not in occupation:
                continue
            for id1 in ids1:
                for id2 in occupation[cell2]:
                    active_pairs.add(tuple(sorted((id1, id2))))
    for key in active_pairs:
        F = affinity.get(key, 0.0)
        affinity[key] = (1 - p.eta) * F + p.eta
    to_remove = []
    for key in list(affinity.keys()):
        if key in active_pairs:
            continue
        F = (1 - p.delta) * affinity[key]
        if F < p.eps:
            to_remove.append(key)
        else:
            affinity[key] = F
    for key in to_remove:
        del affinity[key]
    return active_pairs


class ReferenceUniverse:
    def __init__(self, p: Params | None = None, seed: int = 1):
        self.p = p or Params()
        self.rnd = _random.Random(seed)
        self.nrnd = np.random.RandomState(seed)
        self.grid_size = self.p.grid_size
        self.time = 0
        self.mice = {}
        self.affinity = {}
        self.next_id = 0
        self.births = 0
        self.deaths = 0
        for _ in range(self.p.n_males):
            self.add_mouse("M", self.nrnd.randint(self.grid_size), self.nrnd.randint(self.grid_size))
        for _ in range(self.p.n_females):
            self.add_mouse("F", self.nrnd.randint(self.grid_size), self.nrnd.randint(self.grid_size))

    def add_mouse(self, sex, x, y, age=None, social=None):
        age = self.p.initial_age if age is None else age
        social = self.p.initial_s if social is None else social
        self.mice[self.next_id] = Mouse(self.next_id, sex, x, y, age, social)
        self.next_id += 1

    def neighbors(self, x, y):
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < self.grid_size and 0 <= yy < self.grid_size:
                    out.append((xx, yy))
        return out

    def build_occupation(self):
        occ = {}
        for m in self.mice.values():
            occ.setdefault((m.x, m.y), []).append(m.id)
        self.occupation = occ
        return occ

    def choose_destination(self, mouse):
        cells = self.neighbors(mouse.x, mouse.y)
        weights = []
        for cell in cells:
            w = 1.0
            for oid in self.occupation.get(cell, []):
                if oid == mouse.id:
                    continue
                w += self.p.alpha * self.affinity.get(tuple(sorted((mouse.id, oid))), 0.0)
            weights.append(w)
        return self.rnd.choices(cells, weights=weights, k=1)[0]

    def move(self):
        self.occupation = self.build_occupation()
        newpos = {m.id: self.choose_destination(m) for m in self.mice.values()}
        for mid, (x, y) in newpos.items():
            self.mice[mid].x = x
            self.mice[mid].y = y
        self.time += 1

    def update_social_state(self, occ):
        p = self.p
        for m in self.mice.values():
            ex = max(0, len(occ.get((m.x, m.y), [])) - p.m_c)
            m.social = max(0.0, min(1.0, m.social + p.rec - p.gamma * ex))

    def reproduce(self, occ):
        p = self.p
        newborns = []
        for ids in occ.values():
            males = [self.mice[i] for i in ids if self.mice[i].sex == "M"
                     and p.repro_age_min <= self.mice[i].age <= p.repro_age_max]
            females = [self.mice[i] for i in ids if self.mice[i].sex == "F"
                       and p.repro_age_min <= self.mice[i].age <= p.repro_age_max]
            if not males or not females:
                continue
            for mother in females:
                father = self.rnd.choice(males)
                if self.rnd.random() > p.p0 * mother.social * father.social:
                    continue
                L = self.nrnd.choice(list(p.litter_sizes), p=list(p.litter_probs))
                for _ in range(L):
                    if self.rnd.random() > p.neonatal_survival * mother.social:
                        continue
                    newborns.append(dict(sex=self.rnd.choice(["M", "F"]), x=mother.x, y=mother.y,
                                         age=0, social=mother.social))
        for b in newborns:
            self.add_mouse(b["sex"], b["x"], b["y"], b["age"], b["social"])
        self.births = len(newborns)

    def adult_deaths(self):
        p = self.p
        dead = set()
        for m in self.mice.values():
            k = p.k_mort * (1.0 + p.social_mortality_factor * (1.0 - m.social))
            if self.rnd.random() < 1.0 / (1.0 + np.exp(-k * (m.age - p.a_max))):
                dead.add(m.id)
        for i in dead:
            del self.mice[i]
        for key in list(self.affinity.keys()):
            if key[0] in dead or key[1] in dead:
                del self.affinity[key]
        self.deaths = len(dead)

    def step(self):
        self.move()
        occ = self.build_occupation()
        notebook_update_affinities(self.affinity, occ, self.p)
        self.update_social_state(occ)
        self.reproduce(occ)
        for m in self.mice.values():
            m.age += 1
        self.adult_deaths()

    def run(self, T):
        hist = []
        for _ in range(T):
            self.step()
            n = len(self.mice)
            hist.append(dict(t=self.time, N=n, births=self.births, deaths=self.deaths,
                             S_mean=np.mean([m.social for m in self.mice.values()]) if n else np.nan,
                             n_aff=len(self.affinity)))
            if n == 0:
                break
        return hist
