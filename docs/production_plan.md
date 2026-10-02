# Production plan (stages and decision rules fixed before the production runs)

> **Status of this document.** This is the plan of the production runs of the paper *An Agent-Based Model for the
> Social Collapse of Calhoun's Universe 25 Experiment* (Julián Fernández Bonder): the stages that are not in the
> preregistered plan ([`preregistered_plan.md`](preregistered_plan.md)), their designs and their decision rules, and
> the primary evaluator. It was fixed on **2 October 2026, 05:34:30 UTC**, before the production runs started
> (05:34:37 UTC; first line of `results_v2/f2b_base/progress.log`). It is the file
> `equipo/matematico/plan_v2_corridas.md` of the development repository, as committed in commit
> `0dad573ba5ac2e4c14ef8ef806b28dae1c120c02` (`0dad573`). Everything below the horizontal rule is reproduced
> **verbatim** (in Spanish, as written); nothing was edited. The development repository is not public; this
> repository has no history before its initial commit.
>
> **How to read it.**
> - It complements the preregistered plan (`plan_2b.md` there, `docs/preregistered_plan.md` here), which remains in
>   force for everything this document does not change. Neither the stages added here nor the primary evaluator
>   are preregistered; they are labelled as such in the manuscripts, whose tables of deviations list every change.
> - Paths refer to the layout of the development repository. Their equivalents here:
>   `equipo/matematico/run_2b_v2.sh` → `scripts/production/run_2b_v2.sh`; `gen_designs_v2.py`, `gen_designs.py`,
>   `u25x.py`, `f2b_model_v2.py`, `trans_v2.py`, `api_check_v2.py`, `v2_api.json` → `scripts/production/`;
>   `designs_v2/calhoun_C1/` → `scripts/production/designs_v2/calhoun_C1/`; `pipeline/v2.py` →
>   `scripts/analysis/pipeline/v2.py`; `equipo/programador/fig_inputs.py` → `scripts/figures/fig_inputs.py`;
>   `plan_2b.md` → `docs/preregistered_plan.md`. The outputs are in `results_v2/` as in the plan.
> - The other development notes it cites (`especificacion_v2.md`, the programmer's notes) are not distributed; their
>   content is summarized in the manuscripts. References to review items (codes such as F2, M3 or P1) and to the
>   roles of the AI-assisted workflow in which it was written ("agente matemático", "coordinador", "físico",
>   "programador") are part of the original text; see the declaration of generative AI use in the manuscripts.

---

# Plan de corridas de la V2 (respuesta a la revisión externa, ronda 1)

Agente matemático. Complementa `plan_2b.md` (prerregistro de la Fase 2b, commit `a8f8685`), que sigue vigente para todo lo que acá no se cambia. Este documento **es el prerregistro de la V2**: las reglas de decisión de §4 quedan fijadas en el commit que lo incluye, antes de lanzar las corridas.

**Archivos:**

| Archivo | Para qué |
|---|---|
| `run_2b_v2.sh` | orquestador reanudable |
| `gen_designs_v2.py` | diseños (reutiliza `gen_designs.py` con semillas nuevas) y `designs_v2/calhoun_C1/` |
| `v2_api.json` | evaluador primario, variantes y nombres de los flags de u25 |
| `api_check_v2.py` | verificación previa de la API del evaluador |
| `f2b_model_v2.py` | clase `U2v2`: observables, parámetros de orden y re-puntuaciones por corrida, sin tocar la dinámica ni el RNG |
| `trans_v2.py` | protocolo de transplantes |
| `pipeline/v2.py` | análisis de las etapas nuevas e informe `f2b_final/v2/informe_v2.md` |

## 0. Resumen

**Qué se corre.** Toda la Fase 2b otra vez:
- con R7 corregido (capacidad dura de verdad, `refuge_hard="strict"`);
- con el **evaluador v2 como primario**: t′_pk (`plateau_start`), O5 con t_B99, O4 v2, O10/O11 v2, clase hacinada m > 8 fijo y O17p estricto, según la decisión del coordinador;
- puntuando además cada corrida con el **evaluador v1 completo**, para la tabla de desvíos.

Se agregan siete etapas que pidió la revisión:

| Etapa | Qué mide | Revisión |
|---|---|---|
| `nulos` | poder discriminante de cada criterio frente a modelos nulos | F2, M3; consolidado 3 |
| columnas por corrida | re-puntuación con t′_pk, t_B99, umbral fijo y O4 v2, cada opción por separado | M7, F6, F8 |
| `rob`, `robmu` | mortalidad de fondo, nido, actualización asincrónica, R7 legacy, s0 ∈ {0,25; 0,5} | F4, F5, F9.3, P1 |
| `ci` | condición inicial no sincrónica U{4..52} y U{4..80} | F1.4 |
| `trans` | tiempos de muestreo igualados, distribuciones de s y edad, banda de s intermedia | M1, F3, f9, M7 |
| `lhsx` | réplicas extra en la frontera del LHS y estimador de la región factible por punto | M5 |
| `q7aw` | Q7 con a_w variable: ¿Λ = r·a_w? | M3 |
| `size` | tamaño a densidad fija | F7, m15 |

**Presupuesto.** P0–P2 ≈ 3,8 h de cómputo, más ≈ 0,3 h de análisis: ≈ 4,1 h de pared en 4 CPUs. Con la extensión de Sobol (P3), ≈ 5,0 h; si no entra, el presupuesto la saltea. El vigía corta a las 5 h (`DEADLINE_H=5`).

**Lanzamiento** (reanudable: relanzar el mismo comando retoma donde quedó; avance en `results_v2/f2b_estado/orquestador.log`):
```bash
RES=results_v2 FG=1 bash equipo/matematico/run_2b_v2.sh
```

**Humo: pasado** (§6.3):
- todas las etapas recortadas, 27/27, en 16 minutos;
- un humo dirigido con el evaluador primario final: base, nulos, ci, q7aw, abl2, rob async y trans con X.

## 1. Qué cambia respecto de la Fase 2b

### Código (programador; `u25/`, commits 8b8fcaa y cc026a4)

- **R7 con capacidad dura estricta** (`True`/`"strict"`). La regla de la v1 queda como `"legacy"`, que `rob` corre como variante.
- **Criterios redefinidos:**
  - O10 sobre la s adulta;
  - O10b (deseable): fase B funcional;
  - O11 da NA si las cohortes tardías tienen menos de 30 animales maduros;
  - O4 v2 exige además f₀ ≥ 0,10.
- **O13p y O17p en el evaluador.** O17p es **estricto**: un NA en un esencial (salvo O1–O3 con N0 > 20) cuenta como FAIL, y ERR nunca aprueba.
- **Opciones del evaluador:** `peak_time`, `o5_birth_ref`, `o4_version` y `crowd_class_threshold`, validadas por `validate_eval_kw`.
- **Flags nuevos:** `mu_bg`, `nest_period`, `async_move` y `initial_age_dist`/`initial_age_hi`.

### Evaluador primario de la V2

La decisión del coordinador está en `v2_api.json`, `evaluador.primario`:
```
peak_time = "plateau_start"  (t′_pk = primera semana con N ≥ 0,98 N_max)
o5_birth_ref = "B99"         (fin efectivo de la natalidad: 99 % de los nacimientos acumulados)
o4_version = "v2"            (ρ_pk ≤ 0,8 y f₀ ≥ 0,10)
crowd_class_threshold = 8    (clase hacinada m > 8, independiente de m_c)
O10, O11 v2; O17p estricto
```
- El barrido escribe las columnas `obj_*` con estas opciones (`eval_kw` de `u25.sweep`), y `u25x.py` las valida antes de empezar.
- El orquestador aborta si la versión de u25 no las soporta (`api_check_v2.py`).

**Motivo de O5 con B99** (físico, `especificacion_v2.md` §6):
- con t′_pk y el t_last de la v1, O5 falla en el 75 % de las corridas de C1 por puro efecto de valores extremos de la cola de nacimientos;
- en el humo de la V2, O17p de C1 dio 0/4 con t′_pk + t_last (variante `l5`) y 4/4 con t′_pk + B99.

### Variantes por corrida (`evaluador.variantes`; columnas `obj_<prefijo>_*`)

| Prefijo | Variante | Uso |
|---|---|---|
| `v1` | evaluador v1 completo: argmax, t_last, O4 v1, m > m_c, O10/O11 v1 (`eval_O10_v1`, `eval_O11_v1`) | tabla de desvíos; réplica de Q1–Q7 "como se preregistró" (`f2b_final/v1/`) |
| `ta` | primario con argmax | sensibilidad de t_pk |
| `te` | primario con el fin de la meseta (`plateau_end`) | sensibilidad de t_pk |
| `mc` | primario con m > m_c | efecto del umbral fijo de clase (F8) |
| `l5` | primario con O5 sobre t_last | efecto de B99 |
| `o4a` | primario con O4 v1 | efecto de O4 v2 (F2.4) |

### Otras columnas por corrida (`f2b_model_v2.U2v2`)

- **Meseta:** `x_tpk2` (t′_pk), `x_tleave`, `x_tmid`, `x_plateau`.
- **Fracciones de clase en t_pk primario + {0, 10, 20}:** `x_BO4_k`, `x_BOi_k`, `x_CR_k`, `x_CRf_k` (m > 8) y `x_nad_k`. Sirven para re-puntuar O13 y O13p con cualquier umbral θ sin simular.
- **Parámetros de orden** (físico, `especificacion_v2.md` §3):
  - `x_Psi`: Ψ, el valor de O13p primario;
  - `x_rhoR_pk`: ρ con la capacidad de los refugios;
  - `x_f0_tp` y `x_h_tp`: f₀ y h en t′_pk;
  - `x_t_h`, `x_t_on`, `x_tB99`;
  - `x_tauB`: τ_B, la cola de nacimientos;
  - `x_GB`: duplicaciones con adultos competentes;
  - `x_t_s05ad`, `x_tB50`, `x_tB90`;
  - los instantes del evaluador primario (`x_t_pk_prim`, `x_t_B99_prim`, …).
- **m9:** `x_n_last1` y `x_n_last4`.
- **Serie:** `ref_young` (fracción de ocupantes de refugios con edad < a_w; F4, f10) y `S_juv`.

### Diseño

- **Semillas nuevas** (25301–25336): la V2 es una réplica independiente.
- **`base`** incluye C1′: es **el ensamble de referencia único** (n = 200, T = 600) de C1, C1′, C0 y α = 0, y de ahí sale cada cifra del texto (m1, f2, p4). `c1c2` es la réplica independiente de C1 y el ensamble de C2.
- **`trans`**: protocolo nuevo (§4.5), con el brazo C1′ agregado.
- **`sobol`**: k = 11, los 8 de Morris ∪ {w, r, a_w} (M5.3), con N = 256 y 3 réplicas (mínimo 2). `sobolx` lo lleva a 6. Si Morris no corre, usa la selección de la v1 ∪ {w, r, a_w}.
- **`lhs2`**: sube a P1.

## 2. Etapas, en el orden de ejecución

La pared se estima con 4 núcleos y con el costo medido en la Fase 2b (`gen_designs_v2.py all` lo imprime). El orquestador mide el costo real en cada etapa y re-escala las siguientes. En el humo, el cociente entre el costo medido y el estimado dio 1,0–1,1.

| # | Etapa | P | Diseño | Corridas | T | Pared | Alimenta |
|---|---|---|---|---|---|---|---|
| 1 | `base` | 0 | C1, C1′, C0 y C1 con α = 0, × 200; serie completa | 800 | 600 | 3,5 min | referencia; Q3; nulos α = 0 y C0 |
| 2 | `c1c2` | 0 | C1 y C2 × 200 | 400 | 700 | 1,8 | QC (Holm) |
| 3 | `ref` | 0 | notebook y longevo × 100 (preset notebook) | 200 | 1000 | 1,9 | Q3, línea base, m7 |
| 4 | `nulos` | 0 | γ = 0; −R2 (w = 1, sin aprendizaje social); longevo con la CI de C1; C1 sin reglas; × 200 | 800 | 600 | 3,9 | **(i)** |
| 5 | `abl1` | 0 | factorial a distancia ≤ 1 y variantes, × 120 | 1 080 | 600 | 4,2 | Q2, necesidad (θ, τ) |
| 6 | `trans` | 0 | 8 brazos (C1, −R1, −R2, −R1−R2, X, C0, C2, C1′) × 60 réplicas × 11 transplantes | 480 madres | 300 + 150 | 7,0 | Q1, O12, **(v)** |
| 7 | `amax` | 0 | a_max ∈ {90…150} × 60 | 300 | 600 | 0,9 | Q4, m9 |
| 8 | `rob` | 0 | C1 y C1′ × {R7 legacy, μ_bg = 0,003, μ_bg = 0,008, nido 3, async, async con capacidad blanda, s0 = 0,25, s0 = 0,5}, × 200 | 3 200 | 700 | 19 | **(iii)**, P1, F9.3 |
| 9 | `ci` | 0 | C1 y C1′ × edades uniformes en {4..52} y {4..80}, × 200; serie compacta | 800 | 700 | 3,5 | **(iv)** |
| 10 | `abl2` | 0 | factorial a distancia ≥ 2 y γ = 0, × 40 | 480 | 600 | 4,7 | Q2 |
| 11 | `size` | 1 | L ∈ {15, 20, 30, 45, 60} con N0/L² = 0,44, × 60 | 300 | 600 | 3,6 | F7, m15 |
| 12 | `q7aw` | 1 | γ = 0; a_w ∈ {6, 12, 24} × Λ ∈ {0,24…1,2} (6) × w ∈ {0; 0,05; 0,1}, × 20 | 1 080 | 600 | 16 | **(vii)** |
| 13 | `p1` | 1 | rec × w × γ ∈ {0,1; 0}, × 30 | 3 360 | 600 | 15 | Q7 |
| 14 | `p1aw` | 1 | a_w × rec, × 30 | 1 680 | 600 | 8 | Q6 |
| 15 | `p2` | 1 | α × κ, más la capa crowd_power, × 30 | 2 160 | 600 | 8 | Q5 |
| 16 | `p5` | 1 | refugios f_R × c_R × π, × 30 | 1 500 | 600 | 5 | persistencia |
| 17 | `lhs` | 1 | LHS de 200 puntos y 15 factores, × 10 | 2 000 | 400 | 10 | región factible |
| 18 | `lhs2` | 1 | segundo LHS independiente, × 10 | 2 000 | 400 | 10 | región factible |
| 19 | `lhsx` | 1 | ≤ 60 puntos frontera de lhs y lhs2, × 40 (se genera sola) | ≤ 2 400 | 400 | ≤ 12 | **(vi)** |
| 20 | `morris` | 1 | 15 factores, r = 20, × 10, CRN | 3 200 | 400 | 16 | selección para Sobol |
| 21 | `sobol` | 1 | k = 11, N = 256 (3 328 puntos), × 3, CRN | 9 984 | 400 | 49 | S1 y ST (condicionales a 4 factores fijos) |
| 22 | `robmu` | 2 | C1 y C1′ × μ_bg ∈ {0,001…0,01} (6), × 60 | 720 | 700 | 3 | curva de μ_bg (F4.1) |
| 23 | `p4` | 2 | N0 × L, × 30 | 720 | 600 | 3 | ρ_pk(N0/L²) |
| 24 | `trgrid` | 2 | O12 en α × κ, × 40 | 480 | 300 + 150 | 5 | O12 en el plano |
| 25 | `refine` | 2 | ≤ 40 puntos frontera de los planos, × 100 (se genera sola) | ≤ 4 000 | 600 | ≤ 14 | fronteras |
| 26 | `p6` | 3 | fundadores, × 30 | 360 | 900 | 2 | O1–O3 |
| 27 | `sobolx` | 3 | Sobol de 3 a 6 réplicas | +9 984 | 400 | 49 | sólo si sobra tiempo |

**Notas:**
- **`q7aw`**: con γ = 0, las corridas cercanas a la frontera llegan a 10⁴ animales antes de extinguirse o de explotar, y cuestan 2–10 s cada una (v1, p1 con γ = 0). Por eso la grilla se concentró en la frontera de la v1 (w ≤ 0,1) y se usan 20 réplicas.
- **Totales:** P0 ≈ 50 min, P1 ≈ 152 min, P2 ≈ 24 min, P3 ≈ 51 min.
- **Informes consolidados** (Q1–Q7, QC, el informe de la V2 y el evaluador v1): corren después de `abl2`, `q7aw` y `sobol`, y al final.

## 3. Prioridades y orden de recorte

**Reglas automáticas** (`gen_designs.budget`, antes de cada etapa, con la escala de costo medida):
1. P0 y P1 nunca se saltean. Si el plazo vence durante una de ellas, el vigía corta y relanzar la completa.
2. Sobol baja de 3 réplicas a 2 si no entra.
3. P2 y P3 se saltean si su estimación supera el tiempo restante menos 5 minutos.

**Orden de recorte manual**, si la escala medida supera ≈ 1,2 (por ejemplo, si async resulta más cara que ×3) o si hay menos de 5 h. Entre paréntesis, el ahorro:

1. `sobolx` (−49 min) y `p6` (−2).
2. `refine` (−14).
3. `trgrid` (−5) y `p4` (−3).
4. `robmu` (−3): la curva de μ_bg queda con los dos puntos de `rob`.
5. Sobol de 3 a 2 réplicas (−16).
6. Morris de r = 20 a 12 (−6).
7. `lhsx` con 20 réplicas en lugar de 40 (−6).

**Nunca se recortan:** `base`, `c1c2`, `ref`, `nulos`, `abl1`, `trans`, `amax`, `rob`, `ci`, `abl2`, `q7aw`, `p1`, `p1aw` y `p2`.

**Cómo aplicarlo:** `STAGES="..."`, o `DEADLINE_H=4` para que las reglas 1–3 recorten solas.

## 4. Estimandos y reglas de decisión (prerregistro de la V2)

### 4.0 Confirmatorios de la Fase 2b, replicados

Q1–Q7 y la familia QC se repiten con los mismos tests, predicciones y corrección de Holm de `plan_2b.md` §7, con semillas nuevas y el código corregido, **dos veces**:
- **con el evaluador primario v2** (`f2b_final/`): es el resultado que va al texto;
- **con el evaluador v1** (`f2b_final/v1/`, `F2B_VARIANT=v1`): es la réplica de lo preregistrado. Su comparación con la v1 publicada separa el efecto del código (R7) del efecto del evaluador.

En Q1, la fase D del evaluador primario está a 1,75 t′_pk; con el v1, en `Darg`, a 1,75 t_pk.

Excepciones de cálculo (van en la tabla de desvíos):
- en Q6, el p del Holm es el cuasi-binomial (M3);
- los p de Holm a menos de 0,01 del nivel se dan con cuatro cifras (M4.4).

### 4.1 (i) Poder discriminante frente a modelos nulos (`nulos` + `base` + `ref`)

**Modelos:**
- C1 y C1′ de `base`, como referencia;
- **nulos verdaderos** (F2.1):
  - γ = 0;
  - −R2: herencia total w = 1, sin aprendizaje social, recuperación individual a r = 0,01;
  - longevo: notebook con a_max = 120 y la CI de C1;
  - C1 sin las cuatro reglas;
- ablaciones parciales: α = 0 y C0, de `base`;
- notebook (`ref`), como columna informativa.

**Estimandos**, para cada criterio E (O4–O7, O10, O10b, O11, O13, O13p, O17_run, O17p, O8, O8b, O9, O11b y O14) y cada modelo M:
- **f_E(M) sobre todas las corridas**: NA y ERR cuentan como "no pasa", y el número de NA se informa aparte;
- su IC de Wilson;
- Δ_E(M) = f_E(C1) − f_E(M), con IC de Newcombe y Fisher unilateral; BH descriptivo.

Se calcula con el evaluador primario y con el v1, que es la columna "antes y después" de O10, O11 y O4.

**Reglas:**
- E es **genérico** si f_E(M) ≥ 0,8 en al menos 2 de los 4 nulos verdaderos;
- E **discrimina frente a M** si la cota superior de Wilson de f_E(M) es < 0,8 y el IC de Δ excluye 0;
- un criterio genérico no cuenta como evidencia en el texto.

**Tamaño:** n = 200 por modelo. El semiancho de Wilson es ≤ 0,07, y una Δ ≥ 0,15 se detecta con potencia ≈ 0,9.

**Vista previa** con la v1 (respaldo con γ = 0 de `abl2`, n = 40; −R2 de `abl1`; longevo de `ref`; criterios v1):
- O10 pasa en todos los nulos;
- O5, O6 y O7 pasan en −R2, longevo y "sin reglas";
- O11 pasa en vacío con γ = 0;
- discriminan O13p y, frente a −R2 y longevo, O11 y O13.

Con los criterios v2 (programador, n = 6), O10 da FAIL con γ = 0 y O11 da NA.

### 4.2 (ii) Re-puntuaciones (todas las etapas, sin simular)

**Estimandos:**
1. O4, O5, O8, O10, O11, O13p y O17p con el evaluador primario, con el v1 y con cada opción cambiada por separado (`ta`, `te`, `mc`, `l5`, `o4a`), en todos los brazos de `base`, `c1c2`, `nulos`, `abl1`, `rob` y `ci` (M7, F6.3, F8, F2.4).
2. El transplante a 1,75 t′_pk (fase D) y a 1,75 t_pk (`Darg`).
3. La fracción tipo Calhoun de los LHS con m > 8 fijo (primario) y con m > m_c (`mc`; F8.2).
4. Las curvas de O17p contra el umbral de clase θ ∈ {0,05; 0,075; …; 0,20} para C1, C1′ y C2, re-puntuadas desde `x_*_k` (M6.1).
5. La **necesidad de cada regla** en la grilla (θ, τ), con τ ∈ {0,5; …; 0,9} el umbral de ensamble (M6, F9.2).

**Reglas:**
- la minimalidad se enuncia como la región (θ, τ) en la que C1′ cumple f ≥ τ y cada ablación de C1′ queda por debajo de τ;
- no se elige un punto después de ver la tabla;
- el texto informa siempre primario y v1, uno al lado del otro.

### 4.3 (iii) Robustez (`rob`, `robmu`)

**Brazos:**
- C1 y C1′ con R7 legacy, μ_bg = 0,003 y 0,008 por semana, nido de 3 semanas, actualización asincrónica (capacidad dura y blanda), y s0 = 0,25 y 0,5 (F9.3);
- en `robmu`, μ_bg ∈ {0,001; 0,002; 0,003; 0,005; 0,008; 0,01} con n = 60.

**Estimandos:**
- O17p (con IC), O13p, O11, O5, O8, ρ_pk, f₀, Ψ, Φ_CR, la meseta, τ_B y la mediana KM de T_ext;
- diferencias contra el brazo de `base`: IC de Newcombe, Fisher bilateral y Mann–Whitney, con BH sobre la tabla.

**Regla:** la variante "preserva el resultado" si O17p ≥ 0,8. Si no, se informa qué criterio falla.

**Expectativas declaradas** (corridas exploratorias de los revisores y del físico, n = 8–16):

| Brazo | Esperado |
|---|---|
| μ_bg = 0,003 | ≈ 0,5 (v1, por O5) |
| μ_bg = 0,008 | ≈ 0 |
| nido | ≈ 0,9, con ρ_pk ≈ 0,43 |
| async con capacidad blanda | ≈ 1, con ρ_pk ≈ 0,45 |
| R7 legacy | O8 de 0,71 → 0,625 con strict (programador, n = 200) |
| s0 = 0,25 | ≈ 8/16 (físico, criterios v1, por O11) |
| s0 = 0,5 | ≈ 5/16 |

**Uso en el texto:**
- si μ_bg = 0,003 baja O17p de 0,8, la dependencia de la mortalidad nula se declara como limitación principal (F4.4);
- si s0 = 0,25 lo baja, O11 se declara en parte definicional (F9.3).

### 4.4 (iv) Condición inicial no sincrónica (`ci`)

- **Variantes:** edades uniformes enteras en {4..52} y en {4..80} semanas, con s = 1 y posiciones al azar; C1 y C1′; n = 200.
- **Estimandos:** los de 4.3, más la cronología: t_ad½, t′_pk, t_B50, t_B90, t_B99, la meseta y G_B.
- **Serie compacta:** se guarda para la cronología.
- **Regla:** la de 4.3. Si alguna CI baja O17p de 0,8, se declara en el abstract (F1.4).
- **Expectativa** (físico): con U{4..80}, 12/16 con B99 y 8/16 con t_last.

### 4.5 (v) Transplantes (`trans`, `trans_v2.py`)

**Fases por réplica.** Cada réplica tiene su madre; X comparte madre y animales con C1.

| Fase | Qué es |
|---|---|
| B | control sano |
| **D** | 1,75 t_pk, con el t_pk primario (t′_pk) y el fallback early-D; es la de Q1 y O12 |
| **Darg** | 1,75 t_pk con argmax: el protocolo de la v1 |
| **Dfix** | t = 95 semanas absolutas para todos los brazos (M1) |
| MIX_M/F | en D, como en la v1 |
| MIXfix_M/F | en t_fix |
| **BAND** | 4 + 4 adultos pasada la ventana (edad ≥ a_w) con s ∈ [0,2; 0,5], en la primera semana en que existen (F3.2) |
| BAND_MIX_M/F | los de la banda de un sexo, con sanos del otro |

**Registros por transplante:**
- de los trasladados (en los mixtos, los de la madre): s_mean, s_med, s_max, frac_s0 (s < 10⁻³) y age_med/min/max;
- de la colonia: éxito, t_found y N_max;
- en BAND, hasta 40 semanas después de fundar (o N > 400), la competencia de los nacidos en la colonia: s_desc_ad (edad ≥ a_w), s_desc_mat y n_desc_ad.

**Estimandos:**
- P_T con IC de Wilson por brazo y fase;
- las distribuciones de s y edad (M1.2, f9);
- McNemar de C1 contra X en D, Darg, Dfix y BAND;
- Fisher entre brazos **sólo en Dfix y BAND**, a tiempo igual, con BH. Reemplaza la familia `trans_D_vs_candidato` de la v1, que mezclaba regla y tiempo.

**Regla para el texto:**
- Q1 y O12 son una **verificación de consistencia de R1** (M1.1, F3.1). Se espera mediana de s = 0 en D bajo R1;
- el contenido no trivial se busca en BAND:
  - si bajo R1 + R2 la descendencia de animales intermedios no recupera (s_desc_ad < 0,2) y sin R1 ni R2 sí, es evidencia del mecanismo de linaje;
  - si recupera, la irreversibilidad sólo vale para s ≈ 0.

**Humo** (2 réplicas): BAND aparece en las semanas 8–10, con fundadores de 12–14 semanas y s ≈ 0,35. Con pareja sana hay fundaciones con s_desc_ad ≈ 0,2–0,4.

### 4.6 (vi) Región factible con incertidumbre por punto (`lhs`, `lhs2`, `lhsx`)

**Tres estimadores**, siempre relativos a la caja y a las escalas de muestreo (M5.2):
1. **ingenuo**: k/n ≥ 0,8, con bootstrap sobre puntos (el de la v1);
2. **volumen ponderado**: media de O17p sobre la caja, insesgado para E_x[p(x)] (M5.1);
3. **empírico-Bayes**: p_i ~ Beta(μ_i φ, (1 − μ_i) φ), con μ_i del emulador logístico (Firth, efectos principales) y φ por máxima verosimilitud beta-binomial. F̂ = media de P(p_i ≥ 0,8 | datos), con IC bootstrap sobre puntos que re-ajusta el emulador y φ.

**`lhsx`:**
- puntos de lhs y lhs2 con 0,4 ≤ k/n ≤ 0,95 en el criterio primario, como máximo 60, ordenados por |k/n − 0,8|;
- 40 réplicas nuevas por punto;
- se recalcula el estimador 3 y se cuentan los puntos inciertos.

**Regla:**
- el texto cita el estimador 3 con su IC y el 2 como "volumen ponderado";
- se informan también el primario y el v1 (`mc` para el umbral de clase);
- los índices de Sobol se declaran condicionales a los 4 factores fijos.

**Vista previa con la v1:** ingenuo 6,8 % [4,5; 9,3]; volumen 10,1 % [7,9; 13,0]; empírico-Bayes 5,5 % [3,6; 7,9], con 25 puntos inciertos y φ̂ = 0,73.

### 4.7 (vii) Q7b: ¿r y a_w entran sólo como Λ = r·a_w? (`q7aw`)

- **Modelo:** logit P_ext(γ = 0) = b0 + b_w w + b_1 (log r + θ log a_w), más una variante cuadrática; Firth y cuasi-binomial.
- **Predicción:** el mapa generacional con φ constante da θ = 1.
- **Test confirmatorio nuevo, único de la V2:**
  - TOST de equivalencia con márgenes [2/3; 3/2], con el IC de perfil al 90 % dentro de los márgenes;
  - se informa además el LR de θ = 1;
  - θ < 1 significa aprendizaje concentrado al principio de la ventana (como en Q6), es decir Λ_eff < r a_w.
- **Exploratorio:**
  - la frontera σw + φΛ = 1, con IC bootstrap paramétrico de σ̂ y φ̂. Si el IC de φ̂ excluye φ ≤ 1, se declara la contradicción con el prerregistro;
  - el **ancho de la transición** de P_ext en log Λ con w = 0, para cada a_w: f₋ + (f₊ − f₋)/(1 + e^{−(x − x_c)/Δ}) por máxima verosimilitud binomial, con IC bootstrap (físico §8.2).

### 4.8 Tamaño a densidad fija (`size`; F7, m15)

**Estimandos:**
- la dispersión de T_ext con cuantiles de Kaplan–Meier (m5) e IC bootstrap;
- la mediana y el IQR de t_last y de t_B99;
- D_last;
- ρ_pk, f₀ y Ψ contra L, con Kruskal–Wallis a densidad fija (m15);
- O17p contra L.

**Lectura prevista** (valores extremos, F7): D_last crece como log n, y la dispersión la fija la abruptez del cese de natalidad (τ_B).

### 4.9 Análisis sin corridas nuevas (`pipeline/v2.py`)

- **Q6 (M3):** índice único contra aditivo cuadrático en log r y log a_w, por QAIC, y re-ajuste sin los puntos que fallan por persistencia.
  - Vista previa v1: con todos los puntos gana el índice (ΔQAIC = 3,2). Sin los de persistencia gana el aditivo (ΔQAIC = −11), con θ̂ = 0.
- **m5:** dispersión con cuantiles de KM. La diferencia C1′ − longevo con IC bootstrap tiene margen ±0,05 (v1: −0,000 [−0,046; 0,048]).
- **m7:** mezcla del notebook con t_split ∈ {1,25; 1,5; 2; 2,5} × la mediana. Hecho con la v1, donde `ref` no depende de R7: p_temprana 0,48 / 0,54 / 0,55 / 0,55.
- **m9:** residuo de Q4 con n_last1 y n_last4.
- **m1:** tabla de ensambles en `f2b_final/v2/ensambles.csv`.

## 5. Observables por corrida

Ver §1 y el docstring de `f2b_model_v2.py`. Los criterios, sus variantes y O17p salen **del evaluador de u25**: fuente única. Todas las variantes son evaluaciones del mismo `evaluate_run` con otras opciones; cuestan ≈ 4 ms cada una, contra ≈ 1 s de la corrida.

## 6. Uso, API y prueba de humo

### 6.1 Uso

```bash
RES=results_v2 FG=1 bash equipo/matematico/run_2b_v2.sh                  # todo, en primer plano, 5 h
RES=results_v2 STAGES="base nulos rob" FG=1 bash equipo/matematico/run_2b_v2.sh
REGEN=1 RES=results_v2 FG=1 bash equipo/matematico/run_2b_v2.sh          # tras cambiar v2_api.json o los presets
tail -f results_v2/f2b_estado/orquestador.log
```

**Salidas:**
- por etapa, `results_v2/f2b_<etapa>/`: `summary.csv`, `stage.json`, `labels.json`, `eval_kw.json` y `analisis/informe.md`, más:
  - en `trans`: `tabla_brazos_fases.csv` y `contraste_tiempo_igual.csv`;
  - en `rob` y `ci`: `contraste_variantes.csv`;
  - en `q7aw`: `q7aw_theta.csv`, `q7aw_ancho.csv` y `q7aw.json`;
  - en `size`: `tabla_tamano.csv`;
- `results_v2/f2b_final/informe_Q.md` (primario);
- `f2b_final/v1/informe_Q.md` (evaluador v1);
- `f2b_final/v2/informe_v2.md`, con los CSV de discriminante, re-puntuación, θ, región, Q6, dispersión, t_split y ensambles.

### 6.2 API (`v2_api.json`)

- **Campos:** `refuge_hard` ∈ {True/"strict", "legacy", False}, `mu_bg`, `nest_period`, `async_move`, `initial_age_dist`/`initial_age_hi` y `s_newborn_base`.
- **Evaluador:** `primario` y `variantes`, con la sintaxis del docstring.
- **Verificaciones:**
  - `gen_designs_v2.py check` comprueba que cada campo exista y que el valor sobreviva el viaje design.csv → `design_to_params`;
  - `api_check_v2.py` valida el evaluador primario (si falla, aborta) y las variantes (si falla una, sólo avisa).
- **Cambiar algo:** si cambia un nombre o una opción, editar `v2_api.json` y relanzar con `REGEN=1`.
- **Resguardo:** cambiar `primario` después de empezar una etapa aborta esa etapa, porque `eval_kw` forma parte de la configuración del barrido. Así no se mezclan evaluadores en un `summary.csv`.

### 6.3 Prueba de humo (hecha, 2026-10-02)

1. **Humo completo** (`SMOKE=1 RES=results_v2 FG=1 bash …`): las 27 etapas recortadas a ≤ 4 puntos × 2 réplicas y T ≤ 250, en 16 minutos (≈ 25–35 s por etapa con análisis).
   - Runner, análisis por etapa, informes Q, informe V2 y vigía: OK.
   - Fallaron sólo los análisis de Morris y Sobol, como se esperaba: el diseño recortado no es un diseño completo. Esos análisis ya se probaron con datos sintéticos y con la v1.
   - Tiempo medido contra estimado: 1,0–1,1.
2. **Humo dirigido con el evaluador primario final:** `base nulos ci q7aw abl2`, `rob` con los cuatro brazos async (puntos 4, 5, 12 y 13) y `trans` con C1, X, C2 y C1′ (puntos 0, 4, 6 y 7); 4 minutos. El resultado está en §6.4.
3. **El humo no toca los diseños adaptativos reales:** usa una copia en `f2b_estado_smoke/des_smoke`.
4. **Para repetirlo:** `SMOKE=1 RES=results_v2 FG=1 bash equipo/matematico/run_2b_v2.sh`. Antes de la corrida real hay que borrar `results_v2/*_smoke`.

### 6.4 Resultado del humo dirigido (2 réplicas por punto, T = 250)

- **Evaluador primario:** `api_check_v2.py` lo validó con todas las variantes, y cada etapa escribe `eval_kw.json` con sus opciones.
- **Costo por corrida** (s de CPU), contra 1,2 s de una corrida de C1 en `base`:

  | Brazo | s por corrida | Cociente contra C1 |
  |---|---|---|
  | C1 async | 3,3 | ×2,8 |
  | C1′ async | 1,7–2,0 | — |
  | `q7aw` | 3,7 en promedio, 4,9 como máximo | — |

  Los dos están dentro de lo estimado: `rob` ≈ 20 min y `q7aw` ≈ 16–25 min.
- **Ningún error** de análisis en base, nulos, ci, q7aw, abl2, rob ni trans, ni en los informes Q (primario y v1) ni en el de la V2.
- **Señales** (no son resultados: n = 2):
  - C1 pasa O17p con el evaluador primario y con el v1, también en las variantes async y en las dos CI no sincrónicas;
  - en C1′ con CI no sincrónica, O5 da MARG en 2/4 con el primario, contra 1/4 con el v1.
- **Limpieza:** las salidas del humo (`results_v2/*_smoke`) se borraron. `results_v2/` queda vacío para la corrida real.

## 7. Riesgos y pendientes

- **Costo de async:** medido en el humo, ×2,8 en C1. Si en la corrida real la escala medida supera ≈ 1,2, se recorta según §3.
- **O17_run** (`calhoun_like`) sigue contando NA como aprobado. El texto usa siempre O17p, que es estricto.
- **El cambio de primario es un desvío del prerregistro** (t′_pk, B99, O4 v2, m > 8 fijo, O10/O11 v2). Va en `tab:deviations`, con la réplica v1 en `f2b_final/v1/`.
- **Sobol y Morris** usan el criterio primario v2. La selección de Morris, `lhsx` y `refine` se adaptan a él.
- **T_FIX = 95** quedó fijado de antemano. Si con R7 estricto el t_D de C1 se corre más de 15 semanas, Dfix deja de ser "mitad de la fase D" para C1, pero sigue siendo igual para todos los brazos, que es lo que pide M1.
- **Disco:** las series de `base` y `ci` ocupan ≈ 150 MB. Las cubre `.gitignore` (`results_v2/**/series*`). Para `make figs` en la V3, correr `python3 equipo/programador/fig_inputs.py --make results_v2/f2b_base` (notas del programador §5).
