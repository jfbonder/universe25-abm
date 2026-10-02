"""O11 medido al cierre de la ventana R1 (edad a_w) frente a la edad a_min = 6.

    python3 scripts/analysis/o11_aw.py run        # re-corre (≈ 10 min, 4 núcleos)
    python3 scripts/analysis/o11_aw.py report     # results_v2/o11_aw/resumen.md

`run` re-corre, con las mismas semillas y la misma clase de producción (f2b_model_v2.U2v2), las etapas
`results_v2/f2b_base` (C1, C1′, C0, α = 0), `results_v2/f2b_nulos` (los cuatro nulos) y los brazos s0 = 0,25 y
0,5 de `results_v2/f2b_rob` (puntos 6, 7, 14 y 15), evaluándolas con el evaluador primario v2 más
`o11_age="aw"`, y guarda series compactas a resolución completa (con s_mat/n_mat y s_aw/n_aw) en
`results_v2/o11_aw/<etapa>/`. El modelo es determinista: las columnas no-objetivo deben coincidir bit a bit con las
de la etapa de origen (se verifica en `report`, columna "idénticas").
`report` evalúa O11 con las dos edades sobre las mismas series (O11 "mature6" debe coincidir con el obj_O11
versionado), toma O17p con "mature6" del summary.csv versionado y O17p con "aw" del summary nuevo, y escribe la
tabla con IC de Wilson al 95 %.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "production"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
SRC = ROOT / "results_v2"
OUT = SRC / "o11_aw"
STAGES = [("f2b_base", None), ("f2b_nulos", None), ("f2b_rob", [6, 7, 14, 15])]


def _primary():
    from f2b_model_v2 import primary_kw
    return primary_kw()


def run(n_jobs=4):
    from u25.reeval import source_config, source_params, _import_cls
    from u25.sweep import run_sweep
    ekw = {**_primary(), "o11_age": "aw"}
    for st, pts in STAGES:
        src = SRC / st
        cfg = source_config(src)
        pmap = source_params(src, "auto", cfg["u25_version"])
        if pts is not None:
            pmap = {k: v for k, v in pmap.items() if k in pts}
        ddf = pd.read_csv(src / "design.csv")
        ddf = ddf[ddf.point.isin(list(pmap))]
        out = OUT / st
        r = run_sweep(ddf, cfg["reps"], out, T=cfg["T"], base_seed=cfg["base_seed"], n_jobs=n_jobs, crn=cfg["crn"],
                      model_cls=_import_cls(cfg["model_cls"]), quiet=True, params_map=pmap, eval_kw=ekw,
                      series_every=1)
        (out / "labels.json").write_text((src / "labels.json").read_text())
        print(st, r, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    f = k / n
    d = 1 + z * z / n
    c = (f + z * z / (2 * n)) / d
    h = z * np.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / d
    return f, c - h, c + h


def fmt(k, n):
    f, lo, hi = wilson(k, n)
    return f"{k}/{n} = {f:.3f} [{lo:.3f}; {hi:.3f}]".replace(".", ",")


def report():
    from u25 import objectives as OB
    from u25.params import Params
    ekw = _primary()
    rows, checks = [], []
    for st, pts in STAGES:
        src, out = SRC / st, OUT / st
        labels = json.load(open(src / "labels.json"))
        old = pd.read_csv(src / "summary.csv", low_memory=False, keep_default_na=False, na_values=[""])
        new = pd.read_csv(out / "summary.csv", low_memory=False, keep_default_na=False, na_values=[""])
        old = old[old.point.isin(new.point.unique())]
        plist = json.load(open(src / "params.json"))
        from u25.reeval import compare
        chk = compare(old, new)
        same = not chk["differing_non_objective_columns"]
        changed = sorted(chk["objective_status_changes"])
        checks.append((st, chk["n_rows"], same, changed))
        o11 = {}
        for f in sorted((out / "series").glob("p*_r*.parquet")):
            S = pd.read_parquet(f)
            pt, rp = int(S.point.iloc[0]), int(S.rep.iloc[0])
            p = Params.from_dict(plist[pt])
            S = S.sort_values("t").reset_index(drop=True)
            ct = OB.characteristic_times(S, p, peak_time=ekw.get("peak_time", "argmax"))
            o11[(pt, rp)] = (OB.eval_O11(S, p, ct, "mature6"), OB.eval_O11(S, p, ct, "aw"))
        old = old.set_index(["point", "rep"])
        new = new.set_index(["point", "rep"])
        for pt in sorted(new.index.get_level_values(0).unique()):
            keys = [k for k in o11 if k[0] == pt]
            n = len(keys)
            m6 = [o11[k][0] for k in keys]
            aw = [o11[k][1] for k in keys]
            agree = sum(o11[k][0].status == old.loc[k, "obj_O11"] for k in keys)
            v6 = np.nanmedian([r.value for r in m6]) if any(np.isfinite(r.value) for r in m6) else np.nan
            vw = np.nanmedian([r.value for r in aw]) if any(np.isfinite(r.value) for r in aw) else np.nan
            rows.append(dict(
                etapa=st, punto=pt, brazo=labels[pt], n=n, agree=agree,
                O11_m6=fmt(sum(r.status == "PASS" for r in m6), n),
                O11_aw=fmt(sum(r.status == "PASS" for r in aw), n),
                na_m6=sum(r.status == "NA" for r in m6), na_aw=sum(r.status == "NA" for r in aw),
                s_m6=v6, s_aw=vw,
                O17p_m6=fmt(int((old.loc[pt].obj_O17p == "PASS").sum()), len(old.loc[pt])),
                O17p_aw=fmt(int((new.loc[pt].obj_O17p == "PASS").sum()), len(new.loc[pt])),
                k6=sum(r.status == "PASS" for r in m6), kw=sum(r.status == "PASS" for r in aw)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "resumen.csv", index=False)
    L = ["# O11 a la edad a_w frente a la edad 6", "",
         "Generado por `scripts/analysis/o11_aw.py report` (datos: `results_v2/o11_aw/<etapa>/`, tabla en `resumen.csv`).",
         "",
         "- **Definición.** `o11_age=\"mature6\"` (default v2) mide s a la edad a_min = 6 (columnas s_mat/n_mat); "
         "`o11_age=\"aw\"` la mide al cierre de la ventana R1, después de las a_w actualizaciones de las edades 1..a_w "
         "(columnas s_aw/n_aw; a_w = plast_age_max = 12 en C1, C1′ y C0; en los brazos sin ventana finita se usa la "
         "misma edad calendario, 12). Las mismas cohortes de nacimiento se observan a_w semanas después de nacer: "
         "tendencia en [t_lag, t_B0] + (a_w − 6) y cohortes tardías en [t_B50 + a_w, t_B0 + a_w]. Umbrales sin "
         "cambios (s̄ < 0,2, Mann–Kendall p < 0,05, ≥ 30 animales; NA si hay menos, y NA cuenta como FAIL en O17p).",
         "- **Corridas.** Re-corridas con las mismas semillas y la clase de producción `f2b_model_v2.U2v2`; resto del "
         "evaluador = primario v2 (t′_pk, O5 con t_B99, O4 v2, clase hacinada m > 8).",
         "- **IC.** Wilson al 95 %. s̄ tarde = mediana entre corridas del valor de O11 (s̄ de las cohortes tardías).",
         "", "| Etapa | Brazo | O11 (edad 6) | O11 (edad a_w) | NA 6 / a_w | s̄ tarde 6 → a_w | O17p (edad 6) | O17p (edad a_w) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        s6 = "—" if not np.isfinite(r.s_m6) else f"{r.s_m6:.3f}".replace(".", ",")
        sw = "—" if not np.isfinite(r.s_aw) else f"{r.s_aw:.3f}".replace(".", ",")
        L.append(f"| `{r.etapa}` | {r.brazo} | {r.O11_m6} | {r.O11_aw} | {r.na_m6} / {r.na_aw} | {s6} → {sw} | "
                 f"{r.O17p_m6} | {r.O17p_aw} |")
    def g(stage, lab, col):
        r = R[(R.etapa == stage) & (R.brazo == lab)]
        return r[col].iloc[0] if len(r) else "—"
    nul = R[R.etapa == "f2b_nulos"]
    L += ["", "## Lectura", "",
          f"- **C1 y C1′ casi no cambian** (O11: {g('f2b_base', 'C1', 'O11_m6')} → {g('f2b_base', 'C1', 'O11_aw')}; "
          f"C1′ {g('f2b_base', 'C1p', 'O11_m6')} → {g('f2b_base', 'C1p', 'O11_aw')}): a la edad a_w las cohortes "
          "tardías siguen por debajo de 0,2, y más bajo que a la edad 6.",
          "- **O11 deja de discriminar frente a los nulos con daño por hacinamiento y sin ventana.** Medido a la edad "
          "a_w (12 semanas también en los nulos sin R1), O11 pasa en "
          + "; ".join(f"{r.brazo}: {r.O11_aw}" for r in nul.itertuples())
          + ". γ = 0 sigue sin pasar (NA: la colonia explota antes de que maduren las cohortes tardías) y −R2* "
          "separa sólo en parte. A la edad 6 la separación venía en buena medida del s de nacimiento (s_base = 0 "
          "en C1); a la edad a_w el daño por hacinamiento acumulado lleva también al longevo y a 'sin reglas' por "
          "debajo de 0,2. Medido a a_w, el fracaso de las cohortes es sobre "
          "todo el del daño por hacinamiento.",
          f"- **O17p.** En los nulos sigue en 0 salvo −R2* ({g('f2b_nulos', 'C1 -R2 (w=1, sin aprendizaje social)', 'O17p_m6')} → "
          f"{g('f2b_nulos', 'C1 -R2 (w=1, sin aprendizaje social)', 'O17p_aw')}): frente a −R2* la discriminación de "
          "O17p descansaba sólo en O11 (O13p pasa en 200/200). Los demás nulos siguen fallando por otros criterios.",
          f"- **s0.** Con s0 = 0,25 y 0,5 O11 a la edad 6 cae (C1: {g('f2b_rob', 'C1 s0=0.25', 'O11_m6')}, "
          f"{g('f2b_rob', 'C1 s0=0.5', 'O11_m6')}), y a la edad a_w se recupera ({g('f2b_rob', 'C1 s0=0.25', 'O11_aw')}, "
          f"{g('f2b_rob', 'C1 s0=0.5', 'O11_aw')}): la dependencia de s_base (la parte "
          "definicional de O11) casi desaparece."]
    L += ["", "## Controles", ""]
    for st, n, same, changed in checks:
        L.append(f"- `{st}`: {n} corridas re-corridas; columnas no-objetivo idénticas al summary versionado: "
                 f"**{'sí' if same else 'NO'}**; columnas obj_* con cambios de estado: {', '.join(changed) or 'ninguna'}.")
    ag = int(R.agree.sum()); nn = int(R.n.sum())
    L.append(f"- O11 \"mature6\" recalculado sobre las series nuevas coincide con el `obj_O11` versionado en {ag}/{nn} corridas.")
    (OUT / "resumen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        run()
    report()
