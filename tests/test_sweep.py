"""Tests del runner de barridos reanudable (u25/sweep.py)."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from u25 import Params
from u25.sweep import run_sweep, collect_series, load_summary, read_done

ROOT = Path(__file__).resolve().parents[1]
BASE = Params(grid_size=10, n_males=20, n_females=20)
KEY = ["point", "rep"]
CMP = ["N_max", "t_peak", "total_births", "max_aff", "t_end"]


def _design(tmp_path, n=3):
    d = tmp_path / "design.csv"
    pd.DataFrame(dict(point=np.arange(n), alpha=np.linspace(0.5, 2.5, n), rec=[0.01, 0.02, 0.01][:n])).to_csv(d, index=False)
    return d


def _run(design, out, reps=2, **kw):
    kw.setdefault("T", 40)
    kw.setdefault("n_jobs", 1)
    return run_sweep(design, reps, out, base=BASE, quiet=True, **kw)


def test_resume_equals_uninterrupted(tmp_path):
    d = _design(tmp_path)
    full = tmp_path / "full"
    _run(d, full)
    part = tmp_path / "part"
    r1 = _run(d, part, max_runs=2)
    assert r1["new"] == 2
    r2 = _run(d, part)
    assert r2["new"] == 4 and r2["done"] == 6
    a, b = load_summary(full), load_summary(part)
    assert len(b) == 6 and not b.duplicated(KEY).any()
    pd.testing.assert_frame_equal(a[KEY + CMP], b[KEY + CMP])
    r3 = _run(d, part)                       # nada por hacer
    assert r3["new"] == 0


def test_parallel_equals_serial(tmp_path):
    d = _design(tmp_path)
    _run(d, tmp_path / "s", n_jobs=1)
    _run(d, tmp_path / "p", n_jobs=2)
    a, b = load_summary(tmp_path / "s"), load_summary(tmp_path / "p")
    pd.testing.assert_frame_equal(a[KEY + CMP], b[KEY + CMP])


def test_truncated_last_line_is_dropped(tmp_path):
    d = _design(tmp_path)
    out = tmp_path / "o"
    _run(d, out, max_runs=3)
    with open(out / "summary.csv", "a") as f:
        f.write("2,2.5,0.01,1,0,123")          # escritura cortada a mitad de línea
    header, done = read_done(out / "summary.csv")
    assert len(done) == 3
    _run(d, out)
    s = load_summary(out)
    assert len(s) == 6 and s["N_max"].notna().all()


def test_extend_reps_and_incompatible_config(tmp_path):
    d = _design(tmp_path)
    out = tmp_path / "o"
    _run(d, out, reps=1)
    r = _run(d, out, reps=2)
    assert r["new"] == 3
    with pytest.raises(RuntimeError):
        _run(d, out, reps=2, T=41)          # T distinto: incompatible
    r = _run(d, out, reps=2, T=41, force=True)
    assert r["new"] == 0


def test_lock_prevents_concurrent_runners(tmp_path):
    d = _design(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    f = open(out / ".lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError):
            _run(d, out)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def test_outputs_format_objectives_series_ties(tmp_path):
    d = _design(tmp_path, 2)
    out = tmp_path / "o"
    run_sweep(d, 2, out, T=40, n_jobs=1, base=BASE.replace(n_females=5), ties={"n_females": "n_males"},
              series_every=5, quiet=True)
    s = load_summary(out)
    cols = list(s.columns)
    assert cols[:4] == ["point", "alpha", "rec", "rep"]
    assert cols.index("N_max") > cols.index("rec")
    for c in ["obj_O1", "obj_O6", "obj_O13", "obj_A1", "obj_A4", "obj_O17_run", "n_peaks", "period",
              "h_at_peak", "N0"]:
        assert c in cols, c
    assert set(s["obj_O7"]) <= {"PASS", "FAIL", "MARG", "NA"}
    meta = json.load(open(out / "meta.json"))
    assert meta["design_columns"] == ["alpha", "rec"]
    pl = json.load(open(out / "params.json"))
    assert all(p["n_females"] == p["n_males"] == 20 for p in pl)
    assert (s["N0"] == 40).all()
    p = collect_series(out)
    S = pd.read_parquet(p)
    assert set(map(tuple, S[KEY].drop_duplicates().to_numpy())) == set(map(tuple, s[KEY].to_numpy()))
    assert (S.groupby(KEY)["t"].apply(lambda t: (t.iloc[:-1] % 5 == 0).all())).all()


def test_pipeline_reads_sweep_output(tmp_path):
    pip = ROOT / "scripts" / "analysis"
    if not (pip / "pipeline" / "io.py").exists():
        pytest.skip("pipeline estadístico no disponible")
    sys.path.insert(0, str(pip))
    try:
        from pipeline import io as pio
    finally:
        sys.path.remove(str(pip))
    d = _design(tmp_path)
    out = tmp_path / "o"
    _run(d, out)
    run = pio.load_run(out)
    assert pio.design_columns(run["summary"]) == ["alpha", "rec"]
    t, ev = pio.event_table(run["summary"])
    assert len(t) == 6


def test_sigterm_then_resume(tmp_path):
    d = _design(tmp_path)
    out = tmp_path / "o"
    cmd = [sys.executable, "-m", "u25", "sweep", "--design", str(d), "--reps", "4", "--T", "120",
           "--out", str(out), "--n-jobs", "2", "--quiet", "--set", "grid_size=10",
           "--set", "n_males=20", "--set", "n_females=20"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 120:
        if (out / "summary.csv").exists() and len((out / "summary.csv").read_text().splitlines()) >= 3:
            break
        time.sleep(0.2)
    pr.send_signal(signal.SIGTERM)
    pr.wait(timeout=60)
    _, done = read_done(out / "summary.csv")
    assert 2 <= len(done)
    pr2 = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=300)
    assert pr2.returncode == 0, pr2.stderr.decode()[-2000:]
    s = load_summary(out)
    assert len(s) == 12 and not s.duplicated(KEY).any()
    assert "interrumpido" in (out / "progress.log").read_text()


def test_sweep_has_new_objective_columns(tmp_path):
    d = _design(tmp_path, 1)
    out = tmp_path / "o"
    _run(d, out, reps=1)
    cols = set(load_summary(out).columns)
    assert {"obj_O8b", "obj_O11b", "obj_O13b", "obj_O15b", "obj_R_B", "obj_R_B_val", "o13b_delta_b"} <= cols
