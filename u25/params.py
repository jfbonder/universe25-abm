"""Parámetros del modelo Universe 25.

Los valores por defecto reproducen el notebook original del modelo (preset `notebook`; u25/reference.py)
con alpha = 1.4 (valor elegido para la línea base).  Los campos agrupados bajo
"EXTENSIONES" tienen defaults neutros: con esos valores el modelo es exactamente el
del notebook.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, fields, asdict
from typing import Any


@dataclass
class Params:
    # ------------------------------------------------------------ geometría
    grid_size: int = 30
    periodic: bool = False          # notebook: bordes duros (sin toro)

    # ------------------------------------------------------------ población inicial
    n_males: int = 200
    n_females: int = 200
    initial_age: int = 4
    initial_s: float = 1.0
    #   condición inicial no sincrónica (v2): "fixed" (todos con initial_age, notebook) |
    #   "uniform" (edades enteras uniformes en [initial_age, initial_age_hi], sorteadas con el RNG principal
    #   después de las posiciones; birth_t = -edad). Con "fixed" no se consume ningún número aleatorio.
    initial_age_dist: str = "fixed"
    initial_age_hi: int = 30

    # ------------------------------------------------------------ afinidad
    eta: float = 0.20               # refuerzo  F <- (1-eta) F + eta
    delta: float = 0.01             # decaimiento F <- (1-delta) F
    eps: float = 1e-6               # poda F < eps
    alpha: float = 1.4              # peso de movimiento w = 1 + alpha * sum_j F_ij

    # ------------------------------------------------------------ estado social
    m_c: int = 8                    # umbral de comunidad
    rec: float = 0.01               # alpha_rec (recuperación)
    gamma: float = 0.10             # deterioro por hacinamiento

    # ------------------------------------------------------------ reproducción
    p0: float = 0.30
    neonatal_survival: float = 0.95  # q = neonatal_survival * s_madre (notebook usa 0.95)
    repro_age_min: int = 6
    repro_age_max: int = 50
    litter_sizes: tuple = (3, 4, 5, 6, 7, 8, 9, 10)
    litter_probs: tuple = (0.03, 0.07, 0.15, 0.25, 0.23, 0.15, 0.08, 0.04)
    p_male: float = 0.5

    # ------------------------------------------------------------ mortalidad
    a_max: float = 60.0             # LIFE_EXPECTANCY
    k_mort: float = 0.15            # MORTALITY_STEEPNESS
    # Mortalidad social "multiplicativa". El notebook usaba k -> k (1 + mu_s (1 - s)), que para
    # a < a_max REDUCE la mortalidad de los deteriorados (signo invertido). Corregido: con
    # social_mortality_form="shift" (default) el deterioro adelanta la edad de muerte,
    #     p = 1/(1+exp(-k (a - a_max (1 - mu_s (1 - s))))),
    # que es creciente en mu_s para toda edad. "notebook" reproduce la forma original (sólo para
    # comparación). Con mu_s = 0 ambas coinciden. Preferir social_hazard (aditiva).
    social_mortality_factor: float = 0.0
    social_mortality_form: str = "shift"


    # ============================================================ EXTENSIONES
    # (1) alpha dependiente de s_i:
    #     alpha_i = alpha_low + (alpha - alpha_low) * f(s_i)
    #     alpha_s_mode: "const" (f=1, notebook) | "power" (f = s^alpha_s_power)
    #                   | "threshold" (f = 1[s >= alpha_s_threshold])
    #                   | "sigmoid" (f = 1/(1+exp(-(s-alpha_s_threshold)/alpha_s_width)))
    alpha_s_mode: str = "const"
    alpha_low: float = 0.0
    alpha_s_power: float = 1.0
    alpha_s_threshold: float = 0.5
    alpha_s_width: float = 0.05
    #     Forma de los pesos: "linear" (w = 1 + alpha_i * A, notebook) | "exp" (w = exp(alpha_i * A)).
    #     Con alpha_i < 0 y forma lineal los pesos se recortan a weight_floor.
    weight_form: str = "linear"
    weight_floor: float = 1e-3
    #     Aversión a multitudes de los deteriorados ("beautiful ones"):
    #     w(y) *= exp(-crowd_aversion * (1 - s_i)^crowd_power * n_y),  n_y = ocupación de y (sin i)
    crowd_aversion: float = 0.0
    crowd_power: float = 1.0

    # (2) ventana crítica de plasticidad de s por edad:
    #     dentro de [plast_age_min, plast_age_max] ds se aplica completo; fuera, la parte de
    #     recuperación se multiplica por plast_outside_rec y la de deterioro por plast_outside_det.
    plast_age_min: float = 0.0
    plast_age_max: float = math.inf
    plast_outside_rec: float = 1.0
    plast_outside_det: float = 1.0
    #     herencia: s_neonato = inherit_weight * s_madre + (1 - inherit_weight) * s_newborn_base
    inherit_weight: float = 1.0
    s_newborn_base: float = 1.0

    # (3) territorialidad de machos:
    #     machos reproductivos: w(y) *= exp(-territory_strength * M_y), M_y = nº de otros machos
    #     reproductivos en y (snapshot pre-movimiento).
    territory_strength: float = 0.0
    #     regla de apareamiento: "random" (notebook: cada hembra elige un macho uniforme de su celda)
    #     | "dominant" (el macho de mayor s de la celda)
    mating_rule: str = "random"
    #     período refractario de la hembra tras parir (semanas); 0 = notebook (puede parir cada semana)
    refractory_weeks: int = 0

    # (5) R2: recuperación por aprendizaje social:
    #     rec_i = rec * [(1-lambda) + lambda * sbar_V(i)]
    #     social_rec_source: "moore" (media de s de los otros en el vecindario de Moore post-movimiento)
    #                        | "affinity" (sum_j F_ij s_j / sum_j F_ij sobre todo el almacén de afinidades)
    #     social_rec_isolated: valor de sbar sin vecinos/conocidos: "zero" (no aprende) | "self" (s_i)
    social_rec_lambda: float = 0.0
    social_rec_source: str = "moore"
    social_rec_isolated: str = "zero"
    #     R2': umbral de no retorno s_h: sin recuperación si s_i < s_h
    rec_threshold: float = 0.0
    # (6) R1 con techo de competencia: al cumplir competence_age, c_i := s_i y luego s_i <= c_i
    #     (None = desactivado). R1 pura: plast_age_max=a_w, plast_outside_rec=0.
    competence_age: float | None = None
    # (7) R3: mezcla de largo alcance. Con prob. jump_prob por semana, tras el paso local, i salta:
    #     "social": a la celda pre-movimiento de otro individuo uniforme;
    #     "affinity": a la de un conocido j con prob. ∝ F_ij (sin conocidos -> "social");
    #     "uniform": a una celda uniforme de la grilla.
    jump_prob: float = 0.0
    jump_mode: str = "social"
    # (8) R5: territorialidad por derrota: machos adultos (edad >= repro_age_min) no dominantes de su
    #     celda (dominante = mayor s, empates al azar) pierden defeat_gamma de s por semana.
    defeat_gamma: float = 0.0
    # (9) fundadores: todos los iniciales en la celda central
    founders_same_cell: bool = False
    # (10) subpasos de movimiento por semana (cambio de escala temporal de la movilidad). En cada
    #      subpaso se recalculan ocupación, pares vecinos y pesos con las afinidades vigentes (sin
    #      reforzarlas); interacciones sociales, afinidades y demografía se evalúan una vez por semana con
    #      las posiciones finales. Los saltos R3 se aplican una vez por semana, en el último subpaso.
    move_substeps: int = 1
    # (11) refugios: fracción de celdas (fijas, sorteadas con refuge_seed en un RNG aparte) de capacidad
    #      refuge_capacity: nadie entra (ni se queda) en un refugio que ya tiene >= capacidad *otros*
    #      ocupantes (snapshot pre-subpaso; capacidad blanda por el movimiento simultáneo).
    #      refuge_seed fija la disposición de los refugios: es la MISMA para todas las réplicas de un
    #      ensamble (todas las corridas de v1 usan refuge_seed = 0; 8 disposiciones de C1 no mostraron
    #      un efecto apreciable).
    refuge_frac: float = 0.0
    refuge_capacity: int = 1
    refuge_seed: int = 0
    #      R7: preferencia de los deteriorados por los refugios. El peso de
    #      un refugio no lleno se multiplica por 1 + refuge_pref (1 - s_i)^refuge_pref_power.
    refuge_pref: float = 0.0
    refuge_pref_power: float = 4.0
    #      capacidad dura (refuge_hard; ver Params.refuge_hard_mode):
    #        False | "off"     capacidad blanda (sólo los pesos);
    #        True  | "strict"  (v2; presets C1, C1p, C2) capacidad dura de verdad: tras cada subpaso, los que
    #                 no se movieron tienen prioridad, los que entran se admiten en un orden sorteado
    #                 (permutación) hasta la capacidad y los rechazados vuelven a su celda anterior; la
    #                 admisión se itera hasta un punto fijo (un rechazado que vuelve a un refugio cuenta como
    #                 residente y desplaza a quien hubiera tomado su plaza). Las crías que nacerían en un
    #                 refugio lleno se ubican en una celda vecina no refugio (una por madre, uniforme), y lo
    #                 mismo los fundadores que excedan la capacidad. Invariante: ocupación <= capacidad en
    #                 todo refugio después de cada subpaso y al final de cada paso;
    #        "legacy"  (v1, commit 253b3fc) una sola pasada: los rechazados que vuelven y las crías no se
    #                 cuentan; un refugio puede quedar por encima de la capacidad.
    refuge_hard: bool | str = False
    #      base de la preferencia R7: "current" (s_i actual; C1) | "maternal" (R7'': h_i = s de la madre al
    #      nacer i; fundadores h = 1). Factor 1 + refuge_pref (1 - h_i)^refuge_pref_power.
    refuge_pref_basis: str = "current"
    # (12) radio de apareamiento: 0 = misma celda (notebook); 1 = vecindario de Moore de la hembra
    mate_radius: int = 0
    # (13) período de nido (v2): las crías con edad <= nest_period semanas (en el momento del
    #      movimiento; nacen con edad 0 y tienen edad 1 en su primer movimiento) quedan inmóviles en la
    #      celda donde nacieron (el nido), no saltan (R3) y no cuentan en la ocupación que produce daño
    #      (m de los demás, ni en n_y de la aversión); su propia m es la de los no-crías de su celda + 1.
    #      Sí cuentan para la capacidad de los refugios y forman afinidades. 0 = sin nido (notebook).
    nest_period: int = 0
    # (14) actualización asincrónica del movimiento (v2): en cada subpaso los agentes se
    #      mueven de a uno en un orden aleatorio; cada uno ve las posiciones actuales de los demás
    #      (ocupación, afinidades exactas del almacén, refugios llenos). False = simultáneo (notebook).
    async_move: bool = False

    # (4) mortalidad dependiente de s (aditiva, además del factor del notebook):
    #     p = 1 - (1 - p_edad) * (1 - social_hazard * (1 - s)^social_hazard_power)
    social_hazard: float = 0.0
    social_hazard_power: float = 1.0
    #     mortalidad de fondo (v2): riesgo constante por semana, independiente de edad y de s,
    #     p = 1 - (1 - p_edad,social) * (1 - mu_bg). No se atribuye a la mortalidad social (A4).
    mu_bg: float = 0.0

    # ------------------------------------------------------------ simulación
    T: int = 2000
    seed: int | None = 1
    max_pop: int = 60000            # salvaguarda: corta la corrida si N > max_pop (flag 'exploded')

    # ------------------------------------------------------------ observables
    obs_s_v: float = 0.3            # umbral de viabilidad para h (hembras reproductivas "sanas")
    obs_s_low: float = 0.2          # umbral de deterioro (refugiados: s > obs_s_low)
    obs_iso_m: int = 2              # "beautiful ones": s < obs_s_low y m_i <= obs_iso_m ...
    obs_iso_weeks: int = 4          # ... sostenido obs_iso_weeks semanas
    obs_blocks: int = 4             # nacimientos por bloque (obs_blocks x obs_blocks) para Gini

    # ------------------------------------------------------------ utilidades
    @property
    def refuge_hard_mode(self) -> str:
        """'off' | 'strict' | 'legacy' (normaliza bool y strings de CSV/CLI)."""
        v = self.refuge_hard
        if isinstance(v, (bool, int)) and not isinstance(v, str):
            return "strict" if v else "off"
        v = str(v).strip().lower()
        if v in ("true", "1", "yes", "strict", "hard"):
            return "strict"
        if v in ("false", "0", "no", "off", "", "none", "nan"):
            return "off"
        if v == "legacy":
            return "legacy"
        raise ValueError(f"refuge_hard inválido: {self.refuge_hard!r} (False, True/'strict', 'legacy')")

    def replace(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["litter_sizes"] = list(self.litter_sizes)
        d["litter_probs"] = list(self.litter_probs)
        return d

    def to_json(self) -> str:
        d = self.to_dict()
        d = {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in d.items()}
        return json.dumps(d, indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        for k in ("litter_sizes", "litter_probs"):
            if k in kw:
                kw[k] = tuple(kw[k])
        if "plast_age_max" in kw and kw["plast_age_max"] is None:
            kw["plast_age_max"] = math.inf
        return cls(**kw)

    def validate(self) -> None:
        assert self.grid_size >= 1
        assert not self.periodic or self.grid_size >= 3, "periodic requiere grid_size >= 3"
        assert len(self.litter_sizes) == len(self.litter_probs)
        assert abs(sum(self.litter_probs) - 1.0) < 1e-9, "litter_probs debe sumar 1"
        assert self.alpha_s_mode in ("const", "power", "threshold", "sigmoid")
        assert self.weight_form in ("linear", "exp")
        assert self.mating_rule in ("random", "dominant")
        assert 0 <= self.eta <= 1 and 0 <= self.delta <= 1
        assert self.social_rec_source in ("moore", "affinity")
        assert self.social_rec_isolated in ("zero", "self")
        assert self.jump_mode in ("social", "affinity", "uniform")
        assert self.social_mortality_form in ("shift", "notebook")
        assert 0 <= self.social_rec_lambda <= 1 and 0 <= self.jump_prob <= 1
        assert self.move_substeps >= 1 and 0 <= self.refuge_frac <= 1 and self.refuge_capacity >= 1
        assert self.mate_radius in (0, 1) and self.refuge_pref >= 0
        assert self.refuge_pref_basis in ("current", "maternal")
        self.refuge_hard_mode  # noqa: B018 - valida el valor
        assert self.initial_age_dist in ("fixed", "uniform")
        assert self.initial_age_dist == "fixed" or self.initial_age_hi >= self.initial_age
        assert 0 <= self.mu_bg < 1 and self.nest_period >= 0
        assert not (self.async_move and self.periodic and self.grid_size < 5), "async_move periódico: L >= 5"


def parse_value(p: Params, key: str, raw: str) -> Any:
    """Convierte un string de CLI (`key=value`) al tipo del campo correspondiente."""
    fmap = {f.name: f for f in fields(Params)}
    if key not in fmap:
        raise KeyError(f"parámetro desconocido: {key}")
    cur = getattr(p, key)
    if key == "refuge_hard":
        v = raw.strip().lower()
        return True if v in ("1", "true", "yes", "strict") else False if v in ("0", "false", "no", "off") else raw
    if isinstance(cur, bool):
        return raw.lower() in ("1", "true", "yes", "si", "sí")
    if isinstance(cur, tuple):
        return tuple(float(x) if "." in x else int(x) for x in raw.split(","))
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(float(raw))
    if isinstance(cur, float):
        return float(raw)
    if cur is None:
        if raw.lower() == "none":
            return None
        return float(raw) if key == "competence_age" else int(raw)
    return raw


# ------------------------------------------------------------------ presets
PRESETS = {
    "notebook": {},
    # longevidad compatible con O8 (fase D larga gobernada por envejecimiento)
    "longevo": dict(a_max=120.0, repro_age_max=80),
    # escenario de fundadores de Calhoun: 4 + 4 de 48 días (≈ 7 semanas) [ORIG p.82], juntos
    "founders": dict(n_males=4, n_females=4, initial_age=7, founders_same_cell=True),
    "founders_longevo": dict(n_males=4, n_females=4, initial_age=7, founders_same_cell=True,
                             a_max=120.0, repro_age_max=80),
    # candidato C0 de la calibración: R1 + R2 + aversión, longevo
    "calhoun_C0": dict(a_max=120.0, repro_age_max=80, alpha=4.0, crowd_aversion=0.15, crowd_power=4.0,
                       plast_age_max=12.0, plast_outside_rec=0.0, social_rec_lambda=1.0, rec=0.1,
                       inherit_weight=0.2, s_newborn_base=0.0, max_pop=15000, n_males=200, n_females=200),
    # candidato principal C1 = C0 + refugios + R7 con capacidad dura. v2: capacidad dura "strict"; la de v1 es refuge_hard="legacy".
    "calhoun_C1": dict(a_max=120.0, repro_age_max=80, alpha=4.0, crowd_aversion=0.15, crowd_power=4.0,
                       plast_age_max=12.0, plast_outside_rec=0.0, social_rec_lambda=1.0, rec=0.1,
                       inherit_weight=0.2, s_newborn_base=0.0, max_pop=15000, n_males=200, n_females=200,
                       refuge_frac=0.3, refuge_capacity=2, refuge_seed=0, refuge_pref=100.0,
                       refuge_hard="strict"),
    # C1': conjunto mínimo R1 + R2 + R7, sin aversión (AV)
    "calhoun_C1p": dict(a_max=120.0, repro_age_max=80, alpha=4.0, crowd_aversion=0.0, crowd_power=4.0,
                        plast_age_max=12.0, plast_outside_rec=0.0, social_rec_lambda=1.0, rec=0.1,
                        inherit_weight=0.2, s_newborn_base=0.0, max_pop=15000, n_males=200, n_females=200,
                        refuge_frac=0.3, refuge_capacity=2, refuge_seed=0, refuge_pref=100.0,
                        refuge_hard="strict"),
    # variante C2: C1 con preferencia por refugios según la socialización materna.
    # Variante de estudio de la Fase 2b, no la base.
    "calhoun_C2": dict(a_max=120.0, repro_age_max=80, alpha=4.0, crowd_aversion=0.15, crowd_power=4.0,
                       plast_age_max=12.0, plast_outside_rec=0.0, social_rec_lambda=1.0, rec=0.1,
                       inherit_weight=0.2, s_newborn_base=0.0, max_pop=15000, n_males=200, n_females=200,
                       refuge_frac=0.3, refuge_capacity=2, refuge_seed=0, refuge_pref=1000.0,
                       refuge_pref_power=1.0, refuge_pref_basis="maternal", refuge_hard="strict"),
    # candidato "tipo Calhoun" inicial, punto de partida para calibrar
    "calhoun_candidato": dict(a_max=120.0, repro_age_max=80, plast_age_max=12.0, plast_outside_rec=0.0,
                              jump_prob=0.05, jump_mode="social", alpha_s_mode="power",
                              alpha_s_power=1.0, crowd_aversion=0.1),
}


def preset(name: str, **overrides) -> Params:
    """Params de un preset (ver PRESETS) con overrides opcionales."""
    if name not in PRESETS:
        raise KeyError(f"preset desconocido: {name}; opciones: {sorted(PRESETS)}")
    return Params(**{**PRESETS[name], **overrides})
