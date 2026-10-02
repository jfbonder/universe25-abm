# Preregistered statistical plan (Phase 2b)

> **Status of this document.** This is the statistical plan that was preregistered before the production runs of
> the paper *An Agent-Based Model for the Social Collapse of Calhoun's Universe 25 Experiment* (Julián Fernández
> Bonder). It is the file `equipo/matematico/plan_2b.md` of the development repository, as committed in
> **commit `a8f868592e9bc2d8185ba151d854e0f5308df3c3` (`a8f8685`), 1 October 2026, 22:02:11 UTC**, before the
> launch of the production runs. Everything below the horizontal rule is reproduced **verbatim** (in Spanish, as
> written); nothing was edited. The development repository is not public; this repository has no history before
> its initial commit.
>
> **How to read it.**
> - The plan was written for the first production pipeline (`run_2b.sh`, outputs in `results/f2b_*`). The
>   production runs of the paper re-ran it, with the hard refuge capacity corrected and the additional stages and
>   primary evaluator described in the manuscripts, with `scripts/production/run_2b_v2.sh` into `results_v2/f2b_*`.
>   The decision rules of the stages that are not in this plan, and the primary evaluator, are not preregistered;
>   they were fixed on 2 October 2026, 05:34:30 UTC, before the production runs started (05:34:37 UTC; see
>   `results_v2/f2b_base/progress.log`), in the production plan reproduced in
>   [`production_plan.md`](production_plan.md).
> - Every deviation from this plan, and every post hoc analysis, is listed in the table of deviations of the
>   extended version (`manuscript/extended`, appendix of supplementary results) and in Table S12 of the
>   Supplemental Material of the PRE version (`manuscript/pre/supplement.pdf`).
> - Paths refer to the layout of the development repository. Their equivalents here: `equipo/matematico/run_2b.sh`
>   → `scripts/production/run_2b_v2.sh` (production orchestrator); `gen_designs.py`, `u25x.py`, `f2b_model.py` →
>   `scripts/production/`; `pipeline/` → `scripts/analysis/pipeline/`; `designs/calhoun_C1/` →
>   `scripts/production/designs_v2/calhoun_C1/`; `results/f2b_*` → `results_v2/f2b_*`.
> - The other development notes the plan cites (`calibracion.md`, `campo_medio.md`, `campo_medio_R1R2.md`,
>   `extincion.md`, `fase2b_grillas.md`, `objetivos_criterios.md`) are not distributed; their content is summarized
>   in the calibration, mean-field and observations sections of the extended version.
> - The plan names the roles of the AI-assisted workflow in which it was written ("agente matemático", "físico",
>   "coordinador", "Literatura"); see the declaration of generative AI use in the manuscripts.

---

# Plan de la Fase 2b (v2) — diseño estadístico sobre el preset base

Agente matemático. **Reemplaza** la versión anterior (candidato con R3 y α(s), ya descartado).

- **Base:** preset `calhoun_C1` (decisión del coordinador).
- **Controles:** `calhoun_C0`, la línea base del notebook y `longevo`.
- **Contraste confirmatorio:** `calhoun_C2`.
- **Todo está parametrizado por `PRESET`:** con otro preset, los diseños se regeneran solos y la salida va a `results/f2b_<etapa>_<tag>/`.

Archivos:
- `run_2b.sh`: orquestador;
- `gen_designs.py`: diseños y metadatos de etapa;
- `u25x.py`: lanzador sobre `u25.sweep.run_sweep` y transplantes;
- `f2b_model.py`: observables extra por corrida;
- `pipeline/f2b.py`: análisis por etapa e informe final;
- `designs/calhoun_C1/` (principal) y `designs/calhoun_C0/`.

## 0. Resumen

- **Cronograma:** un único cronograma secuencial de 20 etapas, en cuatro prioridades:

  | Prioridad | Contenido | Pared |
  |---|---|---|
  | P0 | núcleo confirmatorio y Morris | 1,3 h |
  | P1 | planos confirmatorios, P5, LHS y Sobol | 3,4 h |
  | P2 | LHS 2, P4, O12 en α×κ, refinamiento | 0,6–1,0 h |
  | P3 | sólo si sobra | — |

  **Total nominal: 5,4 h más ≈ 0,4 h de refinamiento, unas 5,8 h ≤ 6 h de pared** con 4 núcleos, suponiendo 1,5 s de CPU por corrida de C1. El costo real se mide en cada etapa, y el orquestador reescala las que siguen y aplica el orden de recorte (§4).
- **Lanzamiento:** `bash equipo/matematico/run_2b.sh`. Se relanza solo con `nohup`, es reanudable, usa `flock` y tiene un vigía que corta a las 6 h. Al terminar cada etapa corre el análisis de esa etapa y el informe final.
- **Confirmatorios:**
  - Q1–Q7: familia de 7, con Holm al 5 %;
  - familia QC: C1 frente a C2 en todos los objetivos, n = 200 cada uno, con Holm.
- **Exploratorios:** BH al 10 % por familia, más un BH global informativo. Morris y Sobol se reportan con IC bootstrap y un **piso de ruido** empírico (§8).
- **Criterio primario por corrida:** `O17p` = O17 (tipo Calhoun, evaluador oficial) **y** O13 con clase aislada *persistente*. Así lo pide C1, que tiene R7. Con un preset sin R7, el criterio primario pasa a ser `O17_run`.

## 1. Modelo base, reglas y parametrización

**C1** (`u25.params.PRESETS["calhoun_C1"]`; físico, `calibracion.md` §10.4):
- longevo: a_max = 120, ventana fértil 6–80;
- α = 4 constante;
- R1: a_w = 12, `plast_outside_rec` = 0;
- R2: λ = 1, Moore, aislado → 0, rec = 0,1, herencia w = 0,2, s_base = 0;
- **AV:** aversión de los deteriorados, κ = 0,15, p = 4;
- **R7:** refugios f_r = 0,3, capacidad 2 dura, preferencia de los deteriorados 100·(1 − s)⁴;
- N0 = 200 + 200, `max_pop` = 15 000.

**Reglas activas y su ablación** (`gen_designs.ABLATE`). "Sin R2" sigue la convención del físico en s3.

| Regla | Qué hace | Ablación |
|---|---|---|
| R1 | adulto sin recuperación (trinquete) | `plast_age_max=inf`, `plast_outside_rec=1` |
| R2 | socialización juvenil por aprendizaje social y herencia parcial | `social_rec_lambda=0`, `inherit_weight=1`, `rec=0.01` |
| AV | aversión a multitudes de los deteriorados | `crowd_aversion=0` |
| R7 | refugios con preferencia de los deteriorados | `refuge_frac=0`, `refuge_pref=0`, `refuge_hard=False` (= C0) |

**Otros presets:**
- **C0** = C1 sin R7, el modelo mínimo de tres reglas.
- **C2** = C1 con la preferencia por refugios según la socialización *materna*: `refuge_pref_basis="maternal"`, `refuge_pref=1000`, potencia 1 (`calibracion.md` §11.3).

**Parametrización:**
- `gen_designs.py all --preset P` escribe `designs/P/<etapa>/`, con `design.csv`, `labels.json`, `stage.json` y, para Morris, LHS y Sobol, `problem.json` y `X.npy`.
- Los diseños están **resueltos contra el preset**: toda columna que alguna fila cambia lleva en las demás el valor del preset.
- Las reglas activas, el factorial de ablación, los brazos de transplante, los factores de Morris, LHS y Sobol y el contraste (C1 ↔ C2) se deducen del preset.
- Para un C3 basta `PRESET=calhoun_C3 bash run_2b.sh`: genera `designs/calhoun_C3/` y escribe en `results/f2b_*_C3/`.
- Si un preset cambia después de empezar una etapa, el runner de u25 aborta esa etapa por incompatibilidad con `sweep.json`, en lugar de mezclar corridas.

## 2. Costos

**Medidos en este contenedor** (4 núcleos):

| Corrida | CPU por corrida |
|---|---|
| C1 (extinción en ≈ 165 semanas) | 1,4 s en un proceso (7,9 ms/semana); 1,8 s con 4 en paralelo (prueba de humo) |
| C0 | 0,95 s |
| Colonia persistente, N ≈ 3–6 mil | 6–14 ms/semana |
| γ = 0 (explota a `max_pop` en ≈ 30–45 semanas) | ≈ 1 s |
| Transplante, por réplica (madre ×2 + 4 colonias) | ≈ 5–6 s |

- **Modelo:** h = Σ corridas · c / (4·3600). Cada c de etapa (`stage.json`, `c_cpu`) es relativo a `C_RUN` = 1,5 s por corrida del candidato.
- **Corrección en línea:** tras cada etapa, `gen_designs.py measure` calcula el cociente entre el costo medido (`wall_time_s`) y el estimado. La mediana móvil de esos cocientes (`results/f2b_estado/escalas.txt`) multiplica la estimación de las etapas siguientes.
- **T:**
  - 600 en etapas comparables con la calibración del físico;
  - 700 en el contraste con C2, que tiene colas largas;
  - 1000 en la línea base;
  - 400 en el análisis global. Con a_max ≤ 150, toda extinción "tipo Calhoun" ocurre antes de t ≈ 250; lo que sigue vivo a T = 400 es "persistente".

## 3. Cronograma único (C_RUN = 1,5 s; pared con 4 núcleos)

Notas a la tabla:
- † El candidato y el brazo cruzado X comparten madre y trasladados (transplante pareado, §7 Q1).
- ‡ Los 8 factores de Sobol se eligen automáticamente con Morris. La predicción, que se usa si falta Morris, es rec, crowd_power, κ, α, γ, w, a_w y f_r.

| # | Etapa | P | Diseño | Corridas | T | Pared | Acumulado | Alimenta |
|---|---|---|---|---|---|---|---|---|
| 1 | `base` | 0 | C1, C0 y C1 con α = 0 (control de O14), × 200; serie completa | 600 | 600 | 0,09 h | 0,09 | Q3, objetivos oficiales, KM |
| 2 | `c1c2` | 0 | C1 y C2 × 200 | 400 | 700 | 0,06 | 0,16 | **QC** |
| 3 | `ref` | 0 | notebook (α = 1,4) y longevo puro, × 100; preset `notebook` | 200 | 1000 | 0,07 | 0,23 | Q3, línea base, drift |
| 4 | `abl1` | 0 | factorial: C1 y sus 4 ablaciones simples, más −R1 con α = 6, w = 0 con rec = 0,15, −pref y −hard; × 120 | 1 080 | 600 | 0,15 | 0,38 | **Q2**, ablaciones |
| 5 | `trans` | 0 | 7 brazos × 60 réplicas (B, D, MIX_M, MIX_F): C1, −R1, −R2, −R1−R2, **X: madre C1 → colonia −R1−R2 con rec = 0,1**†, C0 y C2 | 420 | 300 + 150 | 0,18 | 0,56 | **Q1**, O12, O12b |
| 6 | `amax` | 0 | a_max ∈ {90, 105, 120, 135, 150} (repro = 2/3), × 60 | 300 | 600 | 0,03 | 0,59 | **Q4** |
| 7 | `abl2` | 0 | resto del factorial 2⁴ (distancia ≥ 2) y γ = 0, × 40 | 480 | 600 | 0,17 | 0,76 | Q2 (interacciones) |
| 8 | `morris` | 0 | 15 factores, r = 20 (320 puntos), × 10, CRN | 3 200 | 400 | 0,56 | 1,31 | screening, selección para Sobol |
| 9 | `p2` | 1 | P2′: α ∈ {1…8} (7) × κ (8) + capa crowd_power ∈ {2, 8}, × 30 | 2 160 | 600 | 0,28 | 1,60 | **Q5**, f₀ |
| 10 | `p1` | 1 | P1′: rec (8) × w (7) × γ ∈ {0,1; 0}, × 30 | 3 360 | 600 | 0,53 | 2,12 | **Q7** |
| 11 | `p1aw` | 1 | P1′-a_w: a_w ∈ {4…24, ∞} × rec (8), × 30 | 1 680 | 600 | 0,29 | 2,41 | **Q6** |
| 12 | `p5` | 1 | P5: f_r (5) × cap (2) × PREF (5), capacidad dura, × 30 | 1 500 | 600 | 0,17 | 2,58 | O13 persistente, frontera f_r·cap |
| 13 | `lhs` | 1 | LHS random-cd de 200 puntos en los 15 factores, × 10 | 2 000 | 400 | 0,35 | 2,93 | región factible, emulador |
| 14 | `sobol` | 1 | Saltelli N = 256 sobre k = 8‡ (2560 puntos), **× 4** (mínimo 3), CRN | 10 240 | 400 | 1,78 | 4,71 | S1, ST |
| 15 | `lhs2` | 2 | segundo LHS independiente de 200 puntos, × 10 | 2 000 | 400 | 0,35 | 5,05 | duplica la resolución del LHS |
| 16 | `p4` | 2 | P4: N0 (7) × L ∈ {15, 20, 30} + capa `mate_radius` = 1 (3 puntos), × 30 | 720 | 600 | 0,09 | 5,15 | ρ_pk(N0/L²) |
| 17 | `trgrid` | 2 | O12 en α ∈ {2…8} × κ ∈ {0,10; 0,15; 0,20}, × 40 transplantes | 480 | 300 + 150 | 0,17 | 5,31 | O12 en el plano |
| 18 | `refine` | 2 | ≤ 40 puntos de frontera (0,1 < f < 0,9) de p1, p2, p1aw y p5, × 100; se genera solo | ≤ 4 000 | 600 | ≈ 0,4 | ≈ 5,7 | fronteras |
| 19 | `p6` | 3 | fundadores (`mate_radius` = 1): initial_s ∈ {0,2; 0,3; 0,5} × k ∈ {1, 2} × a_max ∈ {120, 200}, × 30 | 360 | 900 | 0,08 | ≈ 5,8 | exploratorio (O1–O3) |
| 20 | `sobolx` | 3 | Sobol de 4 a 8 réplicas (extiende `f2b_sobol`) | +10 240 | 400 | 1,78 | — | sólo si sobra (C1 más barato de lo supuesto) |

Con C0 como preset, las mismas etapas suman 3,2 h:
- sin P5 ni el criterio O13p;
- con 12 factores;
- con C_RUN = 0,95.

## 4. Prioridades y orden de recorte

**Reglas automáticas** (`gen_designs.py budget`, evaluadas antes de cada etapa con la escala de costo medida):
1. **P0 y P1 nunca se saltean.** Si el plazo vence durante una de ellas, el vigía manda SIGTERM al runner: lo hecho queda en disco y relanzar el script la completa.
2. **Sobol** baja sus réplicas para entrar en el tiempo restante, con un mínimo de 3. No se recorta N, porque el análisis exige el diseño completo.
3. **P2 y P3** se saltean si la estimación supera el tiempo restante menos 5 minutos. Para analizar, se reservan 5 minutos.
4. `sobolx` (P3) corre sólo si al final sobra tiempo.

**Orden de recorte manual**, si la escala medida supera ≈ 1,1 o el presupuesto baja de 6 h. El ahorro está entre paréntesis.

1. `sobolx` y `p6`.
2. `refine` (−0,4 h).
3. `trgrid` (−0,17).
4. `p4` (−0,09).
5. `lhs2` (−0,35).
6. Sobol de 4 a 3 réplicas (−0,45).
7. Sobol con N de 256 a 128: regenerar `sobol` (−0,9).
8. Morris con r de 20 a 12 (−0,22).

**Nunca se recortan** `base`, `c1c2`, `ref`, `abl1`, `trans`, `amax`, `abl2`, `p2`, `p1` y `p1aw`, porque llevan los tests confirmatorios.

Cómo aplicarlo:
- para saltear etapas, `STAGES="..."`;
- para un presupuesto menor, `DEADLINE_H=4`: las reglas 1–4 hacen el resto.

## 5. Etapas en detalle

**`base`**
- n = 200 por punto, con serie completa (todas las columnas).
- El análisis corre el **evaluador oficial** `u25.objectives.evaluate_ensemble`. Para O14, G_B se toma del control con α = 0; O12 viene de la etapa `trans`, asignado por etiqueta.
- También: tabla por punto con IC de Wilson, KM y ajustes Weibull y exponencial con censura.

**`c1c2`**
- C1 y C2 a T = 700, n = 200 cada uno, semillas independientes.
- Fisher bilateral por objetivo: O1–O15b, O13p, O13b y O13b como "PASS o MARG", O17_run, O17p y A1–A4.
- Mann–Whitney para las continuas: ρ_pk, f₀, Φ_BO,4sem, ρ8, T_ext − t_última y T_ext.
- Holm sobre la familia.

**`ref`**
- Línea base del notebook y longevo puro, con preset `notebook`.
- Supervivencia con mezcla temprana y tardía, Weibull de la cola y drift de log(pico) por ciclo (pipeline de la Fase 1).
- Es el "contraste con la línea base" de Q3.

**`abl1` y `abl2`: factorial completo 2⁴ en las reglas activas (R1, R2, AV, R7)**
- Las celdas a distancia ≤ 1 de C1 van con n = 120 (abl1) y las demás con n = 40 (abl2). n = 40 alcanza, porque en ellas el criterio primario es ≈ 0: IC [0; 0,09].
- Variantes del físico en abl1: −R1 con α = 6 (R1 prescindible con más agregación), w = 0 con rec = 0,15 (sin herencia), −pref (refugios sin preferencia: no hay persistencia) y −hard (capacidad blanda: estampidas).
- En abl2, el control γ = 0: el hacinamiento como disparador.
- Análisis:
  - Q2 (IUT);
  - mapa regla × objetivo (Fisher, exploratorio);
  - logística de Firth del criterio primario con efectos principales y las 6 interacciones dobles (exploratorio);
  - KM y log-rank entre celdas.

**`trans`**
- 60 réplicas por brazo, con T_main = 300 y T_after = 150 (protocolo O12 de `u25.experiments`).
- El brazo **X** es un **transplante cruzado** (`u25x.cross_experiment`):
  - la madre es C1, con la misma semilla que el brazo C1, así que los snapshots y los 8 trasladados son *los mismos*;
  - la colonia nueva no tiene R1 ni R2 y recupera individualmente a rec = 0,1;
  - el brazo C1 corre por el mismo camino (cruce identidad), con un flujo aleatorio por fase, de modo que el pareado es exacto (verificado en la prueba de humo).
- Separa la irreversibilidad *estructural* (las reglas) de la irreversibilidad *por edad*. En C1 la natalidad muere en ≈ 0,9·t_pk, así que en t_D = 1,75·t_pk los trasladados tienen 47–80 semanas, con una ventana fértil restante de 0–33 semanas.

**`amax`:** ley de la fase D (Q4).

**`morris`**
- 15 factores para C1, 12 para C0:

  | Factor | Rango | Escala |
  |---|---|---|
  | α | 2–8 | lineal |
  | κ | 0,05–0,3 | lineal |
  | crowd_power | 1–8 | log |
  | rec | 0,03–0,3 | log |
  | w | 0–0,6 | lineal |
  | a_w | 4–24 | log |
  | γ | 0,03–0,3 | log |
  | a_max | 90–150, con `repro_age_max` = 2/3·a_max | lineal |
  | p0 | 0,15–0,45 | lineal |
  | δ | 0,003–0,05 | log |
  | m_c | 4–16 | entero |
  | η | 0,05–0,5 | lineal |
  | f_r | 0,05–0,4 | lineal |
  | PREF | 3–1000 | log |
  | cap | 1–3 | entero |

- r = 20 trayectorias, 4 niveles, 10 réplicas con CRN.
- Métricas: f_PRIMARIO, f_O13, f_O13p, f_O4, f_O11, P_ext, ρ_pk, f₀, Φ_BO,4sem y la mediana KM.
- **Selección para Sobol:** los 8 factores con mayor μ* relativo medio sobre las métricas primarias (`analisis/seleccion_sobol.json`). Sobol se regenera solo, en `designs/<P>/sobol_auto/`.

**Planos del físico** (`fase2b_grillas.md` §2–5, sin cambios de ejes):
- P2′ (α × κ) y P1′ (rec × w, con la capa γ = 0);
- P1′-a_w;
- P5 (refugios, requisito de C1).
- Análisis: tabla por punto, mapas (`fig_<etapa>_<métrica>.png`) de f_PRIMARIO, O13, O13p, O4, O11, P_ext, ρ_pk, f₀, Φ_BO,4sem y la fracción explotada, y KM. Los puntos de frontera alimentan `refine`.

**`lhs` y `lhs2`**
- Dos LHS independientes (random-cd), de 200 puntos cada uno.
- Región factible = fracción de puntos con f_PRIMARIO ≥ 0,8, con IC bootstrap.
- Spearman por factor y métrica, y logística de Firth del criterio primario sobre factores estandarizados, con BH por familia.

**`sobol`:** S1 y ST con 1000 remuestreos (SALib), ST_ruido (§8), ST corregido y la fracción de varianza intra-punto.

**`p4` y `p6`:** siembra y escala, y fundadores (O1–O3), exploratorios. `p6` usa la variante de `calibracion.md` §11.4, `mate_radius` = 1 con `initial_s` < 1, que da O17 = 0,17 con C1.

## 6. Observables por corrida

**Del evaluador** (`obj_*`, en el worker y a resolución completa):
- O1–O16, O8b, O11b, O13b y O15b;
- A1–A4;
- `obj_O17_run` y G_B.

**De `f2b_model.U2b`**, una subclase de `Universe` que no cambia la dinámica ni el RNG:

| Columna | Qué mide |
|---|---|
| `obj_O13p` | O13 con Φ_BO sostenido 4 semanas en lugar del instantáneo: PASS si en t_pk, t_pk + 10 o t_pk + 20 (con ≥ 20 adultos) Φ_BO,4sem ≥ 0,10 y Φ_CR ≥ 0,10 |
| `x_rho_pk` | N_max/K |
| `x_f0_pk` | celdas vacías en t_pk |
| `x_BO`, `x_BO4`, `x_CR` | clases en t_pk, t_pk + 10 y t_pk + 20 |
| `x_in_ref_pk` | fracción en refugios en t_pk |
| `x_rho8` | (T_ext − t_pk)/t_pk |
| `x_t_lastbirth`, `x_D_last` | última cría y T_ext − última cría |
| `x_n_last` | crías de las últimas 10 semanas |
| `x_smat_B` | s al destete de las cohortes de la fase B |
| `x_smat_sen` | pendiente de Sen de s_mat: la cascada |

**Por punto:**
- f = PASS/(PASS + MARG + FAIL), con IC de Wilson; O17 y O17p como PASS sobre el total;
- P_ext(T) con Wilson;
- mediana KM con IC;
- medias de las x_*;
- CV de T_ext;
- fracción con ρ8 ≥ 1,84.

## 7. Confirmatorios preregistrados

**Familia Q1–Q7:** Holm a α = 0,05. Un p faltante, por etapa no corrida, se excluye de la familia y se informa. Cada Q tiene una predicción cuantitativa que sale del campo medio (`campo_medio.md`, `campo_medio_R1R2.md`, `extincion.md`) o de la calibración. "cumple_prediccion" se reporta aparte del rechazo.

### Q1. Irreversibilidad estructural (transplante cruzado pareado)

- **Datos:** `trans`, brazos C1 y X.
- **Teoría:**
  - con R1 el adulto es un trinquete, y con R2 (λ = 1, aislado → 0) S = 0 es absorbente;
  - la fecundidad de un linaje deteriorado es ∝ s³ (`campo_medio_R1R2.md` §2.1), así que la colonia nueva no despega;
  - con recuperación individual a rec = 0,1, en cambio, s pasa de 0,01 a 0,3 en unas 3 semanas, dentro de la ventana fértil restante.
- **Test:** McNemar exacto, unilateral, sobre los pares discordantes de la fase D. H₁: el brazo X funda más.
- **Predicción:** P_T(D | C1) ≤ 0,05 y P_T(D | X) ≥ 0,25. El techo es la edad: ventana restante de 0–33 semanas.
- **Potencia:** con P_T(D | X) = 0,25 se esperan unos 15 pares discordantes, todos 0 → 1. P(≥ 6) ≈ 0,99 y p ≈ 0,5¹⁵.
- **Secundarios** (exploratorios): Fisher de −R1, −R2, −R1−R2, C0 y C2 frente a C1 en la fase D, y O12/O12b por brazo.

### Q2. Cada regla es necesaria (test de intersección-unión)

- **Datos:** `abl1`.
- **Test:** para cada regla r ∈ {R1, R2, AV, R7}, Fisher unilateral de f_PRIMARIO(C1) > f_PRIMARIO(C1 − r), con n = 120 cada uno.
- p_Q2 = máx_r p_r. Por ser una IUT, no requiere ajuste interno.
- **Predicción:** f(C1) ≥ 0,9 y f(C1 − r) ≤ 0,5 para toda r. Referencias de s3 y §10:

  | Ablación | Esperado |
  |---|---|
  | −R2 | 0 |
  | −AV | 0 |
  | −R1 | 0,4 |
  | −R7 | O13p ≈ 0, porque Φ_BO,4sem = 0,006 en C0 |

- **Potencia:** ≈ 1 para 0,9 frente a 0,5 con n = 120.

### Q3. Colapso casi determinista frente a la línea base

- **Datos:** `base` y `ref`.
- **Test:** Fisher unilateral de P_ext(C1, T = 600, n = 200) > P_ext(notebook, T = 1000, n = 100).
- **Predicción:**
  - C1: P_ext = 1;
  - notebook: P_ext ≈ 0,53 (en la Fase 1, S_KM(1000) = 0,47);
  - CV(T_ext | C1) ≤ 0,10 (s5: 0,07).
- **Secundarios:**
  - Weibull: k ≫ 1 en C1 (riesgo creciente: vejez) frente a k < 1 en la cola del notebook (mezcla, `extincion.md` §2);
  - log-rank;
  - P_ext de C1 frente a longevo puro.

### Q4. Ley de la fase D (tabla de vida)

- **Datos:** `amax`, 300 corridas.
- **Teoría:**
  - cuando muere la natalidad, la colonia se extingue por vejez;
  - T_ext − t_última ≈ E[máx de n_last vidas], con el riesgo logístico discreto del modelo (`pipeline/f2b._amax_expected`);
  - por la invariancia del logístico ante traslaciones, esa diferencia crece 1:1 con a_max, con un corrimiento que depende sólo de n_last. Para n = 6–12 da a_max − 8 a a_max − 6.
  - Ya se verifica: en la prueba de humo C1 da 113 frente a 112–114 predicho, y la OAT del físico da D_B0 = 88/119/156 para a_max = 90/120/150.
- **Test:** pendiente de MCO de T_ext − t_última sobre a_max, con EE HC3. TOST de equivalencia con margen [0,8; 1,25].
- **Potencia:** EE ≈ 0,03, así que la potencia es ≈ 1 si la pendiente verdadera está en [0,9; 1,15].
- **Corolario** (exploratorio):
  - ρ8 = (T_ext − t_pk)/t_pk ≈ (t_última − t_pk + a*)/t_pk;
  - O8 se cumple si y sólo si t_pk ≲ a*/1,84;
  - eso explica que O8 pase en C1 y se pierda en C2, que tiene colas de edad largas.
  - Residuo medio frente a la fórmula: t de una muestra.

### Q5. La banda de O13 está centrada en n* ≈ m_c

- **Datos:** `p2`, capa crowd_power = 4, puntos con κ > 0 y n* > 0,5 (unos 49 × 30).
- **Teoría:**
  - el peso de una celda para un deteriorado vinculado es (1 + αn)·e^(−κn), con máximo en n* = 1/κ − 1/α;
  - con n* ≫ m_c todos terminan en pools (no hay BO); con n* ≪ m_c se dispersan (no hay CR, y O4 se rompe);
  - las dos clases coexisten sólo si n* ≈ m_c = 8.
- **Test:** logística de Firth de O13 (instantáneo) sobre u = log n* y u². H₁: b₂ < 0 (Wald unilateral).
- **Criterio:** IC bootstrap paramétrico de n*_opt = exp(−b₁/2b₂) ⊂ [4; 12].
- **Exploratorio:** f₀ crece con α y decrece con κ (Spearman).

### Q6. La ventana a_w pesa menos que la tasa rec

- **Datos:** `p1aw`, a_w finita (48 puntos × 30).
- **Teoría:**
  - la competencia adquirida es G = rec·∫₀^{a_w} s̄_V(a) da;
  - por eso ∂ln G/∂ln a_w = s̄_V(a_w)/⟨s̄_V⟩ < 1 = ∂ln G/∂ln rec;
  - las crías en la celda natal ven más s que los juveniles en los pools.
- **Test:** logística de Firth del criterio primario sobre z = log rec + θ·log a_w (cuadrática en z), con θ por verosimilitud perfilada. LR unilateral de θ < 1.
- **Predicción:** θ ∈ (0, 1). La OAT del físico sugiere θ ≈ 0,2–0,5: a_w de 8 a 16 cambia O17 de 0,93 a 0,87; rec de 0,07 a 0,15, de 1 a 0,17.

### Q7. Colapso endógeno sin hacinamiento

- **Datos:** `p1`, capa γ = 0.
- **Teoría:**
  - mapa generacional de la competencia: S_{g+1} = w·S̃_g + φΛ·S_g, con Λ = rec·a_w, S̃ ≥ S la s materna ponderada por fecundidad y φ ≤ 1 la dilución por juveniles;
  - con R₀(s) ≈ 69 s³ (longevo), si w + φΛ < 1, S → 0 y R₀(S) < 1: la colonia se extingue **sin hacinamiento**;
  - con C1 (w = 0,2, Λ = 1,2) el régimen es supercrítico: con γ = 0 explota hasta `max_pop`. **El hacinamiento es el disparador y la cascada R2 el amplificador.**
- **Test:** Fisher unilateral. P_ext(γ = 0) en la esquina subcrítica (w ≤ 0,1, rec ≤ 0,03; 4 puntos × 30) frente al punto C1 con γ = 0 (30 corridas).
- **Predicción:** ≥ 0,8 frente a ≤ 0,2. La prueba de humo dio 4/4 frente a 0/2, con 2/2 que explotan.
- **Exploratorio:** frontera logística en (w, Λ) y φ̂.

### Familia QC: C1 frente a C2, todos los objetivos

- **Datos:** `c1c2`, n = 200 cada uno.
- **Test:** Fisher bilateral por objetivo (≈ 25 binarios), Mann–Whitney en 6 continuas, con Holm sobre la familia.
- **Predicciones** (`calibracion.md` §11.3):

  | Objetivo | C1 | C2 |
  |---|---|---|
  | O8 | 0,67 | 0 |
  | O9 | 0,57 | 0,17 |
  | O13b (PASS o MARG) | 0 | ≈ 1 |
  | O17 | sin diferencia | sin diferencia |

  Además, C2 pierde algo en O5, O6 y O7: 2/60 colonias persistentes.
- **Potencia:** con n = 200 y Holm (α efectivo ≈ 0,0017), Δ = 0,15 se detecta con potencia ≈ 0,7 y Δ ≥ 0,2 con ≈ 0,95.

**Relación con el plan anterior:**
- la Q1 vieja (transplante) pasa a ser la Q1 nueva, cruzada y pareada;
- la Q2 vieja (trinquete) queda contenida en Q1 y Q2;
- la Q3 vieja (R1 × mezcla) es obsoleta, porque no hay R3;
- las Q4 y Q5 viejas (umbrales a_w y λ) pasan a ser Q6 y Q7 sobre los planos P1′;
- la Q6 vieja (ralentización) no aplica: C1 no oscila, y queda en `ref`;
- la Q7 vieja (colapso por edad) pasa a ser la Q4 nueva, cuantitativa.

## 8. Exploratorios, sensibilidad global y corrección múltiple

**Familias con BH al 10 %** (`f2b_final/exploratorio.csv`):

| Familia | Contenido |
|---|---|
| `ablacion_x_objetivo` | celda × objetivo |
| `factorial_logit` | efectos principales e interacciones |
| `trans_D_vs_candidato` | brazos de transplante frente al candidato en la fase D |
| `supervivencia` | Weibull, log-rank |
| `fase_D` | residuo de Q4 |
| `planos_tendencias` | tendencias en los planos |
| `frontera_gamma0` | frontera de la capa γ = 0 |
| `LHS_spearman`, `LHS_logit` | efectos por factor en el LHS (en el informe de la etapa) |

Se reporta además un BH global. Los p-valores que no son tests (φ̂) quedan como NaN.

**Morris y Sobol no son tests.** Se reportan μ* y ST con IC bootstrap, y además un **piso de ruido empírico**:
- **Split-half:** por punto, e = (media de réplicas pares − media de impares)/2. Tiene esperanza 0 y la misma varianza y estructura de CRN que el ruido de la media por punto.
- **Morris:** μ* de e por factor, frente a μ* de la métrica. Distinguible si el cociente es > 2.
- **Sobol:** ST_ruido,i = E[(e_A − e_AB_i)²] / (2·Var(Ȳ)), el estimador de Jansen aplicado a e.
- **Relevante:** ST − ST_ruido > 0,05 y ST − IC > ST_ruido.
- **Por qué hace falta:** con datos sintéticos de 3 réplicas sin CRN, los factores nulos dan ST ≈ 0,3. Sin el piso se declararían relevantes.

## 9. Supervivencia

- Kaplan–Meier con Greenwood (log–log), mediana KM con IC por inversión de bandas y P_ext(T) con Wilson, en **todas** las etapas, por punto.
- Log-rank entre puntos cuando hay ≤ 40.
- Ajustes exponencial y Weibull con censura (`base`).
- Mezcla temprana/tardía y riesgo de la cola (`ref`).
- KM consolidado: C1, C0, notebook, longevo y las ablaciones simples (`f2b_final/fig_km_consolidado.png`).
- **Nunca** se promedia t_ext sólo entre las extintas: lo censurado entra como censura a la derecha en T.

## 10. Tamaños muestrales

**Semiancho de Wilson al 95 % con p = 0,5:**

| n | Semiancho | Si f = 0 |
|---|---|---|
| 30 | ±0,17 | IC [0; 0,11] |
| 40 | ±0,15 | [0; 0,09] |
| 60 | ±0,12 | [0; 0,06] |
| 120 | ±0,09 | [0; 0,03] |
| 200 | ±0,07 | [0; 0,02] |

- Los planos llevan 30 réplicas: alcanza para ubicar un punto lejos de la frontera, y la frontera va a 100 réplicas en `refine`.
- Morris y LHS llevan 10 réplicas. El DE de f por punto es ≤ 0,16, y CRN reduce el ruido de los efectos elementales.
- Sobol lleva 4 réplicas con CRN: se informa la fracción de varianza intra-punto y el piso de ruido.
- La potencia de cada Q está en §7.

## 11. Uso, reanudación y salidas

```bash
bash equipo/matematico/run_2b.sh                         # C1, 6 h, segundo plano (imprime PID y log)
PRESET=calhoun_C0 bash equipo/matematico/run_2b.sh      # controles completos de C0 -> results/f2b_*_C0/
STAGES="base c1c2 trans" FG=1 bash equipo/matematico/run_2b.sh   # subconjunto, en primer plano
SMOKE=1 STAGES=p1 SMOKE_POINTS=30,86,56,63 FG=1 bash equipo/matematico/run_2b.sh   # prueba de humo
tail -f results/f2b_estado/orquestador.log               # avance; por etapa: results/f2b_<etapa>/progress.log
```

**Variables:**

| Variable | Default |
|---|---|
| `PRESET` | `calhoun_C1` |
| `SUFFIX` | `""` para C1, `_<tag>` para otros |
| `STAGES` | orden del manifiesto |
| `N_JOBS` | 4 |
| `DEADLINE_H` | 6 |
| `C_RUN` | 1,5; sólo al regenerar diseños |
| `RES` | `results/` |
| `SMOKE`, `SMOKE_REPS`, `SMOKE_T`, `SMOKE_POINTS` | prueba de humo |

**Reanudación:**
- relanzar el mismo comando: u25 saltea los pares (punto, réplica) hechos y los transplantes se saltean por brazo;
- `sobol` recuerda su diseño en `.design_src`;
- `sobolx` extiende las réplicas del mismo directorio.

**Salidas:**
- por etapa, `results/f2b_<etapa>/`:
  - `summary.csv`, `params.json`, `meta.json`, `stage.json`, `labels.json`, `progress.log`;
  - `series.parquet`, sólo en `base` y `ref`;
  - `analisis/`: `informe.md`, `tabla_puntos.csv`, figuras, y según la etapa `morris.csv` (con `seleccion_sobol.json`), `sobol.csv`, `contraste.csv`, `objetivos_oficial.csv` o `tabla_brazos.csv`;
- `results/f2b_final/`: `informe_Q.md`, `Q.csv` (Holm), `exploratorio.csv` (BH), `contraste_c1c2.csv` (Holm) y `fig_km_consolidado.png`;
- `results/f2b_estado/`: `orquestador.log`, `analisis.log`, `escalas.txt` y `run_2b_*.log`.

**Disco y git:** las series de `base` y `ref` ocupan ≈ 50–100 MB y quedan ignoradas por git (`results/**/series*`). Las semillas van de `base_seed` 25101 a 25119, una por etapa, distintas de las del físico.

## 12. Prueba de humo (hecha, ≤ 2 min de CPU cada una)

1. **Etapa `p1` recortada** (`SMOKE=1 STAGES=p1 SMOKE_POINTS=30,86,56,63`): 4 puntos × 2 réplicas, T = 250.
   - 21 s de pared;
   - el runner, las columnas extra, el análisis por etapa (tabla, Wilson, KM, mapas) y el informe final funcionan;
   - señal de Q7: con γ = 0 la esquina subcrítica se extingue en 4/4 (N_max ≈ 6–7 mil, S → 0,02) y C1 explota en 2/2;
   - C1 con γ = 0,1 da O17p 2/2 y T_ext − t_última = 113.
2. **`base`, `c1c2` y `trans` recortadas:** 50 s.
   - Funcionan la serie completa, `u25 collect`, el evaluador oficial con G_B, el contraste C1–C2 y el lanzador de transplantes.
   - El **transplante cruzado** se probó aparte: 2 brazos × 4 réplicas, pareado exacto, con los mismos trasladados y la misma s.
3. **Modo en segundo plano** (`nohup` automático): `amax` recortada, 1 corrida.
   - Se corrigió un sufijo duplicado en el proceso hijo.
   - El log, el plazo y el lock funcionan.
4. **Datos sintéticos, sin simular,** para las etapas que no se pueden recortar (Morris, LHS, Sobol, factorial, `amax`, `p2`, `p1aw`, final). Todas corren y recuperan los efectos plantados:
   - Q5: n*_opt = 6,97, plantado 7;
   - Q6: θ̂ = 0,40, plantado 0,4;
   - Q2 e IUT, Q4 y Q7 en la dirección correcta;
   - la selección de Morris ordena primero los dos factores con señal.

   Se corrigieron dos problemas:
   - un log-rank de k grupos con 2560 puntos (O(k²)); ahora sólo se calcula con ≤ 40 puntos;
   - la inflación de ST por ruido, que llevó a agregar el piso de §8.

## 13. Riesgos y pendientes

- **O13b** no puede dar PASS con N0 = 400: f_late está limitado (`calibracion.md` §11.1). En QC se reporta "PASS o MARG". Literatura decide si se reformula.
- **Costo:** C1 midió 1,4–1,8 s, frente a los 1,2 supuestos por el coordinador. El plan usa 1,5 s y se autocorrige. Si la escala medida pasa de 1,15, se pierden primero `refine` y `trgrid`.
- **Fundadores** (O1–O3): sólo exploratorio (`p6`, P3) y `p4` con `mate_radius` = 1.
- `move_substeps` = 1 en todo el plan (físico §10.1: k ≥ 2 destruye O13).
- Con `PRESET=calhoun_C2`, el contraste se invierte solo (C2 frente a C1) y R7 se detecta por `refuge_pref` > 0.
