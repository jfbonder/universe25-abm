"""Gamma = t_det/T_gen sin dilución por crías, y O10/O10b medidos sobre los
adultos fuera de la ventana (edad >= a_w = 12 cumplida).

Re-simula (subclase de generaciones.GenU, sin tocar u25/) y registra por semana la s media de
  * los fundadores (generación 0, cohorte cerrada: no hay mortalidad antes de la senescencia),
  * los adultos con edad >= a_min = 6 (S_adult, la definición de la v2, diluida por las crías F1),
  * los animales fuera de la ventana R1, que ya recibieron sus a_w actualizaciones (edades 1..a_w): al registrar,
    después de `age += 1`, tienen edad >= a_w + 1 = 13. a_w = u25.stats.aw_obs_age(p): plast_age_max si la ventana
    es finita y 12 en los nulos sin R1 (12 en todos los brazos de este script). Es la misma frontera que usa
    u25.stats.Recorder para s_aw de O11 (`age == a_w + 1`: la cohorte que acaba de cerrar la ventana); aquí se
    toman todas las edades a partir de ella (con edad >= 12 al registrar, que incluye animales con 11 de sus 12
    actualizaciones, Gamma_aw de C1 da el mismo valor, 1,111).
t_det = t_1/2 - t_on, con t_on la primera semana con f_over >= 0.05 y t_1/2 la primera semana con s media < 0.5.
T_gen = intervalo entre las semanas medianas de nacimiento de F1 y F2 (definición de la v2); también la edad media
de las madres al parir.

Bloques (n_jobs = 2):
  base   réplicas exactas de results_v2/f2b_base (mismos Params y semillas; se verifica N_max contra summary.csv)
  nulos  réplicas exactas de results_v2/f2b_nulos (4 nulos, primeras NREP_NUL) y de f2b_ref punto 0 (notebook,
         primeras NREP_NB, T = 1000 como en el meta)
  rob    réplicas exactas de brazos de robustez (f2b_rob: mu_bg 0.003/0.008, nido, async, s0 0.25/0.5; f2b_ci:
         edades U{4..52}, U{4..80}), primeras NREP_ROB: estructura generacional y nu5/Delta5
  barrido gamma x s_base, semillas nuevas
Salida: OUT.parquet con una fila por corrida (escalares) y
OUT_series.parquet con las series por semana (t <= T_SERIES) de los bloques base y barrido, en formato largo.
Uso (desde la raíz): nice -n 10 python3 scripts/analysis/gamma_v3.py results/fisico_v3/gamma_v3.parquet [bloques] [n_jobs]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from multiprocessing import Pool
from u25.params import Params, preset
from u25.ensemble import rep_seed
from u25.stats import aw_obs_age
import u25.objectives as OBJ
from generaciones import GenU

KW = dict(peak_time="plateau_start", o5_birth_ref="B99", o4_version="v2", crowd_class_threshold=8)
A_W = 12       # a_w de C1 (informativo: la frontera de la ventana se toma de aw_obs_age(p) en GU3.step)
NREP_BASE = {0: 200, 1: 30, 2: 30, 3: 30}
NREP_ROB = 20
ROB_PTS = {"f2b_rob": [1, 2, 3, 4, 6, 7], "f2b_ci": [0, 1]}
NREP_NUL = 30
NREP_NB = 20
SWEEP = [(g, sb) for sb in (0.0, 0.25) for g in (0.02, 0.05, 0.1)]
NREP_SWEEP = 20
T_SERIES = 130


class GU3(GenU):
    def _grow(self, need):
        old = self._cap
        super()._grow(need)
        if self._cap != old or not hasattr(self, "r_mom"):
            b = np.full(self._cap, -1, np.int64)
            a = getattr(self, "r_mom", None)
            if a is not None:
                b[: a.shape[0]] = a
            self.r_mom = b

    def reproduce(self, cell):
        n0 = self.N
        super().reproduce(cell)
        nb = self.N - n0
        if nb > 0:
            self._grow(self.next_id + 1)
            self.r_mom[self.id[-nb:]] = self.id[self._mrep]

    def step(self):
        super().step()
        d = self.log[-1]
        n = self.N
        if n:
            g0 = self.r_gen[self.id] == 0
            ad = self.age >= self.p.repro_age_min
            old = self.age >= aw_obs_age(self.p) + 1     # ya recibieron las a_w actualizaciones de R1
            d.update(s_f0=float(self.s[g0].mean()) if g0.any() else np.nan, n_f0=int(g0.sum()),
                     s_aw=float(self.s[old].mean()) if old.any() else np.nan, n_aw=int(old.sum()),
                     frac_f0_ad=float((g0 & ad).sum() / max(ad.sum(), 1)),
                     frac_f0_aw=float((g0 & old).sum() / max(old.sum(), 1)))
        else:
            d.update(s_f0=np.nan, n_f0=0, s_aw=np.nan, n_aw=0, frac_f0_ad=np.nan, frac_f0_aw=np.nan)


def first_below(t, x, thr=0.5):
    w = np.flatnonzero(np.nan_to_num(x, nan=1.0) < thr)
    return float(t[w[0]]) if w.size else np.nan


def at(t, x, tt):
    if not np.isfinite(tt):
        return np.nan
    i = np.searchsorted(t, tt)
    return float(x[i]) if i < len(x) and t[i] == tt else np.nan


def o10_like(t_half, s_at_pk, t_pk):
    """Misma regla que u25.objectives.eval_O10 / eval_O10b (versión vigente), con otra serie de s:
    NA si t_1/2 < t_pk pero s(t_pk) es NaN (no aprueba en vacío); PASS si t_1/2 < t_pk y s(t_pk) < 0.5."""
    if not np.isfinite(t_half):
        o10 = "FAIL"
    elif t_half < t_pk and not np.isfinite(s_at_pk):
        o10 = "NA"
    elif t_half < t_pk and s_at_pk < 0.5:
        o10 = "PASS"
    elif abs(t_half - t_pk) <= 2:
        o10 = "MARG"
    else:
        o10 = "FAIL"
    r = np.inf if not np.isfinite(t_half) else t_half / t_pk
    o10b = "PASS" if r >= 0.5 else "MARG" if r >= 0.3 else "FAIL"
    return o10, o10b


def one(job):
    block, label, point, rep, pdict, seed, T = job
    p = Params.from_dict(pdict)
    U = GU3(p, rng=np.random.default_rng(seed))
    S, summ = U.run(T)
    S = S.sort_values("t").reset_index(drop=True)
    r = OBJ.evaluate_run(S, p, summ, **KW)
    ct = OBJ.characteristic_times(S, p, peak_time="plateau_start")
    L = pd.DataFrame([{k: v for k, v in x.items() if k not in ("bg", "gen_alive")} for x in U.log])
    t = L.t.to_numpy(float)
    tS = S.t.to_numpy(float)
    fo = S.f_over.to_numpy(float)
    w = np.flatnonzero(np.nan_to_num(fo) >= 0.05)
    t_on = float(tS[w[0]]) if w.size else np.nan
    t_ad = ct["t_sad05"]
    t_f0 = first_below(t, L.s_f0.to_numpy())
    t_aw = first_below(t, L.s_aw.to_numpy())
    t_pk = ct["t_pk"]
    s_aw_pk = at(t, L.s_aw.to_numpy(), t_pk)
    o10aw, o10baw = o10_like(t_aw, s_aw_pk, t_pk)
    s_f0_pk = at(t, L.s_f0.to_numpy(), t_pk)
    o10f0, o10bf0 = o10_like(t_f0, s_f0_pk, t_pk)
    # registro generacional
    n = U.next_id
    gen, birth, male = U.r_gen[:n], U.r_birth[:n], U.r_male[:n]
    daught, mom = U.r_daught[:n], U.r_mom[:n]
    b = gen >= 1
    med = {g: float(np.median(birth[gen == g])) if (gen == g).any() else np.nan for g in (1, 2, 3)}
    mat_age = (birth[b] - birth[mom[b]]).astype(float) if b.any() else np.array([np.nan])
    f0f, f1f = (gen == 0) & ~male, (gen == 1) & ~male
    tB99 = ct["t_B99"]
    nu5 = OBJ._at(ct, ct["N"], tB99) / ct["N_pk"] if np.isfinite(tB99) else np.nan
    out = dict(block=block, label=label, point=point, rep=rep, N_max=int(S.N.max()),
               O17p=r["O17p"].status, O10=r["O10"].status, O10b=r["O10b"].status, O11=r["O11"].status,
               O13p=r["O13p"].status, O10_aw=o10aw, O10b_aw=o10baw, O10_f0=o10f0, O10b_f0=o10bf0,
               t_on=t_on, t_ad=t_ad, t_f0=t_f0, t_aw=t_aw, t_pk=t_pk,
               s_f0_at_tad=at(t, L.s_f0.to_numpy(), t_ad), frac_f0_ad_at_tad=at(t, L.frac_f0_ad.to_numpy(), t_ad),
               s_aw_at_tad=at(t, L.s_aw.to_numpy(), t_ad), s_ad_pk=at(tS, S.S_adult.to_numpy(float), t_pk),
               s_aw_pk=s_aw_pk, s_f0_pk=s_f0_pk,
               med_F1=med[1], med_F2=med[2], med_F3=med[3], T_gen=med[2] - med[1],
               T_gen_mat=float(np.mean(mat_age)), births=int(b.sum()),
               gB=float(gen[b].mean()) if b.any() else np.nan,
               F1=float((gen[b] == 1).mean()) if b.any() else np.nan,
               F2=float((gen[b] == 2).mean()) if b.any() else np.nan,
               F3p=float((gen[b] >= 3).mean()) if b.any() else np.nan,
               plateau=ct["t_plateau_end"] - ct["t_plateau_start"], t_pk_argmax=ct["t_pk_argmax"],
               O5=r["O5"].status, t_B99=tB99, nu5=nu5, d5=(tB99 - t_pk) / t_pk if t_pk > 0 else np.nan,
               Rnet0=float(daught[f0f].mean()) if f0f.any() else np.nan,
               Rnet1=float(daught[f1f].mean()) if f1f.any() else np.nan,
               rho_pk=ct["N_pk"] / ct["K"], extinct=ct["extinct"], exploded=bool(summ.get("exploded", False)))
    # control de alineación: S_adult del registro y de la serie oficial
    out["sad_align_maxdiff"] = float(np.nanmax(np.abs(
        L.Sad.to_numpy(float)[: min(len(L), 60)] - np.interp(t[: min(len(L), 60)], tS, S.S_adult.to_numpy(float)))))
    if block == "base" or block == "barrido":
        k = t <= T_SERIES
        out["series"] = dict(t=t[k], s_f0=L.s_f0.to_numpy()[k], s_ad=L.Sad.to_numpy()[k], s_aw=L.s_aw.to_numpy()[k],
                             frac_f0_ad=L.frac_f0_ad.to_numpy()[k], N=L.N.to_numpy()[k])
    return out


def jobs_from(stage, points, nrep, block, maxrep=None):
    meta = json.load(open(f"results_v2/{stage}/meta.json"))
    plist = json.load(open(f"results_v2/{stage}/params.json"))
    labels = json.load(open(f"results_v2/{stage}/labels.json"))
    J = []
    for pt in points:
        nr = nrep[pt] if isinstance(nrep, dict) else nrep
        for rep in range(nr):
            J.append((block, f"{stage}:{labels[pt]}", pt, rep, plist[pt],
                      rep_seed(meta["base_seed"], pt, rep, meta.get("crn", False)), int(meta["T"])))
    return J


KEYS = ["block", "label", "point", "rep"]


def series_path(out):
    out = Path(out)
    return out.with_name(out.stem + "_series" + out.suffix)


def write_results(res, out):
    """Escalares (una fila por corrida) y series (formato largo) en Parquet."""
    D = pd.DataFrame([{k: v for k, v in r.items() if k != "series"} for r in res])
    D.to_parquet(out, index=False, compression="zstd")
    rows = []
    for r in res:
        if "series" in r:
            S = pd.DataFrame(r["series"])
            for k in KEYS:
                S.insert(KEYS.index(k), k, r[k])
            rows.append(S)
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(series_path(out), index=False, compression="zstd")


def read_results(out):
    """Inverso de write_results: (D escalar, dict (block, point, rep) -> DataFrame de la serie)."""
    D = pd.read_parquet(out)
    sp = series_path(out)
    ser = {}
    if sp.exists():
        S = pd.read_parquet(sp)
        ser = {(b, int(pt), int(rp)): g.drop(columns=KEYS).reset_index(drop=True)
               for (b, pt, rp), g in S.groupby(["block", "point", "rep"], sort=False)}
    return D, ser


if __name__ == "__main__":
    out = sys.argv[1]
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "nulos", "rob", "barrido"]
    J = []
    if "base" in which:
        J += jobs_from("f2b_base", [0, 1, 2, 3], NREP_BASE, "base")
    if "nulos" in which:
        J += jobs_from("f2b_nulos", [0, 1, 2, 3], NREP_NUL, "nulos")
        J += jobs_from("f2b_ref", [0], NREP_NB, "nulos")
    if "rob" in which:
        for stage, pts in ROB_PTS.items():
            J += jobs_from(stage, pts, NREP_ROB, "rob")
    if "barrido" in which:
        for i, (g, sb) in enumerate(SWEEP):
            p = preset("calhoun_C1", gamma=g, s_newborn_base=sb)
            for s in range(NREP_SWEEP):
                J.append(("barrido", f"gamma={g},s_base={sb}", i, s, p.to_dict(), 91_000 + 100 * i + s, 600))
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print("corridas:", len(J), flush=True)
    with Pool(n_jobs) as P:
        res = P.map(one, J, chunksize=2)
    write_results(res, out)
    D = pd.DataFrame([{k: v for k, v in r.items() if k != "series"} for r in res])
    sb = D[D.block == "base"]
    if len(sb):
        summ = pd.read_csv("results_v2/f2b_base/summary.csv").set_index(["point", "rep"])
        bad = [(a, b) for a, b, nm in zip(sb.point, sb.rep, sb.N_max) if int(summ.loc[(a, b), "N_max"]) != nm]
        print("base: réplicas con N_max distinto del summary:", len(bad), bad[:5])
    sn = D[D.block.isin(["nulos", "rob"])]
    for stage in ("f2b_nulos", "f2b_ref", "f2b_rob", "f2b_ci"):
        s2 = sn[sn.label.str.startswith(stage)]
        if len(s2):
            summ = pd.read_csv(f"results_v2/{stage}/summary.csv").set_index(["point", "rep"])
            bad = [(a, b) for a, b, nm in zip(s2.point, s2.rep, s2.N_max) if int(summ.loc[(a, b), "N_max"]) != nm]
            print(stage, ": réplicas con N_max distinto del summary:", len(bad), bad[:5])
    print("máx |Sad registro - S_adult serie| (60 semanas):", D.sad_align_maxdiff.max())
