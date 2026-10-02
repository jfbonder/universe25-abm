"""Corrección por comparaciones múltiples: Holm (FWER) y Benjamini-Hochberg (FDR)."""
import numpy as np


def holm(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)[::-1]  # de mayor a menor
    adj = np.empty(n); running = 1.0
    for k, i in enumerate(order):
        rank = n - k
        running = min(running, n * p[i] / rank)
        adj[i] = min(1.0, running)
    return adj


def adjust_table(df, pcol="p", methods=("holm", "bh")):
    """Agrega columnas p_holm / p_bh a un DataFrame con p-valores (ignora NaN)."""
    out = df.copy(); m = out[pcol].notna().to_numpy()
    for meth in methods:
        f = holm if meth == "holm" else bh
        col = np.full(len(out), np.nan)
        if m.any():
            col[m] = f(out.loc[m, pcol].to_numpy())
        out[f"p_{meth}"] = col
    return out
