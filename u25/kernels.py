"""Kernels numéricos (numba si está disponible; si no, Python puro -- lento pero correcto)."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depende del entorno
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


# Desplazamientos de Moore indexados d = (dx+1)*3 + (dy+1);  d=4 es "quedarse".
DX = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int64)
DY = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int64)


@njit(cache=True)
def cell_sort(cell, ncells):
    """Counting sort de agentes por celda. Devuelve (order, start, count)."""
    n = cell.shape[0]
    count = np.zeros(ncells, np.int64)
    for i in range(n):
        count[cell[i]] += 1
    start = np.zeros(ncells + 1, np.int64)
    for c in range(ncells):
        start[c + 1] = start[c] + count[c]
    pos = start[:ncells].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cell[i]
        order[pos[c]] = i
        pos[c] += 1
    return order, start, count


@njit(cache=True)
def build_pairs(x, y, G, periodic):
    """Todos los pares no ordenados (a, b) de agentes a distancia de Chebyshev <= 1.

    Recorre cada celda con los semi-desplazamientos (0,0) [mismos], (0,1), (1,-1), (1,0), (1,1)
    -- igual que `update_affinities` del notebook -- de modo que cada par aparece una sola vez.
    Devuelve índices de agente (no ids).
    """
    n = x.shape[0]
    ncells = G * G
    cell = x * G + y
    order, start, count = cell_sort(cell, ncells)
    offx = np.array([0, 1, 1, 1], np.int64)
    offy = np.array([1, -1, 0, 1], np.int64)
    # --- contar
    total = 0
    for cx in range(G):
        for cy in range(G):
            c = cx * G + cy
            nc = count[c]
            if nc == 0:
                continue
            total += nc * (nc - 1) // 2
            for k in range(4):
                xx = cx + offx[k]
                yy = cy + offy[k]
                if periodic:
                    xx %= G
                    yy %= G
                elif xx < 0 or xx >= G or yy < 0 or yy >= G:
                    continue
                c2 = xx * G + yy
                if c2 == c:
                    continue
                total += nc * count[c2]
    a = np.empty(total, np.int64)
    b = np.empty(total, np.int64)
    p = 0
    for cx in range(G):
        for cy in range(G):
            c = cx * G + cy
            nc = count[c]
            if nc == 0:
                continue
            s0 = start[c]
            for u in range(nc):
                iu = order[s0 + u]
                for v in range(u):
                    a[p] = iu
                    b[p] = order[s0 + v]
                    p += 1
            for k in range(4):
                xx = cx + offx[k]
                yy = cy + offy[k]
                if periodic:
                    xx %= G
                    yy %= G
                elif xx < 0 or xx >= G or yy < 0 or yy >= G:
                    continue
                c2 = xx * G + yy
                if c2 == c:
                    continue
                s2 = start[c2]
                n2 = count[c2]
                for u in range(nc):
                    iu = order[s0 + u]
                    for v in range(n2):
                        a[p] = iu
                        b[p] = order[s2 + v]
                        p += 1
    return a, b, cell, count


@njit(cache=True)
def accumulate_affinity_field(a, b, F, x, y, n, G, periodic):
    """A[i, d] = sum_{j en celda vecina d de i} F_ij  (matriz N x 9)."""
    A = np.zeros((n, 9), np.float64)
    for p in range(a.shape[0]):
        i = a[p]
        j = b[p]
        if i < 0 or j < 0:
            continue
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        if periodic:
            if dx > 1:
                dx -= G
            elif dx < -1:
                dx += G
            if dy > 1:
                dy -= G
            elif dy < -1:
                dy += G
        if dx < -1 or dx > 1 or dy < -1 or dy > 1:
            continue  # salvaguarda: un par que no es vecino no debe escribir fuera de A
        d = (dx + 1) * 3 + (dy + 1)
        A[i, d] += F[p]
        A[j, 8 - d] += F[p]
    return A


@njit(cache=True)
def sample_rows(W, u):
    """Elige una columna por fila con probabilidad proporcional a W (W >= 0, suma > 0)."""
    n = W.shape[0]
    out = np.empty(n, np.int64)
    for i in range(n):
        tot = 0.0
        for d in range(9):
            tot += W[i, d]
        r = u[i] * tot
        acc = 0.0
        k = -1
        last = 4
        for d in range(9):
            w = W[i, d]
            if w > 0.0:
                last = d
                acc += w
                if k < 0 and r < acc:
                    k = d
        if k < 0:
            k = last
        out[i] = k
    return out


@njit(cache=True)
def merge_reinforce_into(keys, val, t0, q, t, eta, eps, powtab, alive_id, K2, V2, T2, Fnew):
    """Fusión lineal del almacén ordenado (keys, val, t0) con los pares activos q (ordenados).

    - entradas del almacén con valor efectivo al final de t-1 < eps o con algún muerto se descartan
      (poda diferida: equivale a la poda paso a paso del notebook);
    - las claves de q se refuerzan: F = (1-eta) F_old + eta, t0 = t;
    - devuelve (keys2, val2, t02, Fnew) con Fnew alineado con q.
    """
    n = keys.shape[0]
    m = q.shape[0]
    mask = np.int64(0xFFFFFFFF)
    i = 0
    j = 0
    p = 0
    while i < n or j < m:
        if j >= m or (i < n and keys[i] < q[j]):
            # entrada vieja no activa: conservar si sigue viva y >= eps (al final de t-1)
            k = keys[i]
            v = val[i] * powtab[t - 1 - t0[i]]
            if v >= eps and alive_id[k >> 32] and alive_id[k & mask]:
                K2[p] = k
                V2[p] = val[i]
                T2[p] = t0[i]
                p += 1
            i += 1
        elif i < n and keys[i] == q[j]:
            v = val[i] * powtab[t - 1 - t0[i]]
            if v < eps:
                v = 0.0
            f = (1.0 - eta) * v + eta
            Fnew[j] = f
            K2[p] = q[j]
            V2[p] = f
            T2[p] = t
            p += 1
            i += 1
            j += 1
        else:
            f = eta
            Fnew[j] = f
            K2[p] = q[j]
            V2[p] = f
            T2[p] = t
            p += 1
            j += 1
    return p


@njit(cache=True)
def merge_reinforce(keys, val, t0, q, t, eta, eps, powtab, alive_id):
    """Versión que asigna memoria nueva (ver `merge_reinforce_into`)."""
    n = keys.shape[0]
    m = q.shape[0]
    K2 = np.empty(n + m, np.int64)
    V2 = np.empty(n + m, np.float64)
    T2 = np.empty(n + m, np.int64)
    Fnew = np.empty(m, np.float64)
    p = merge_reinforce_into(keys, val, t0, q, t, eta, eps, powtab, alive_id, K2, V2, T2, Fnew)
    return K2[:p], V2[:p], T2[:p], Fnew


@njit(cache=True)
def count_effective(keys, val, t0, t, eps, powtab, alive_id):
    """(n, suma) de afinidades vivas con valor efectivo al final de t >= eps."""
    mask = np.int64(0xFFFFFFFF)
    c = 0
    s = 0.0
    for i in range(keys.shape[0]):
        k = keys[i]
        v = val[i] * powtab[t - t0[i]]
        if v >= eps and alive_id[k >> 32] and alive_id[k & mask]:
            c += 1
            s += v
    return c, s


@njit(cache=True)
def radix_sort_pairs(a, b, n, ids):
    """Ordena pares por (índice menor, índice mayor) con dos counting sorts estables (LSD, base n).

    O(m + n). Como `ids` crece con el índice, el resultado queda ordenado por clave
    (id_lo << 32 | id_hi). Devuelve (lo, hi, keys).
    """
    m = a.shape[0]
    lo0 = np.empty(m, np.int64)
    hi0 = np.empty(m, np.int64)
    cnt = np.zeros(n + 1, np.int64)
    for p in range(m):
        if a[p] < b[p]:
            lo0[p] = a[p]
            hi0[p] = b[p]
        else:
            lo0[p] = b[p]
            hi0[p] = a[p]
        cnt[hi0[p] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    lo1 = np.empty(m, np.int64)
    hi1 = np.empty(m, np.int64)
    for p in range(m):
        q = cnt[hi0[p]]
        lo1[q] = lo0[p]
        hi1[q] = hi0[p]
        cnt[hi0[p]] += 1
    cnt[:] = 0
    for p in range(m):
        cnt[lo1[p] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    keys = np.empty(m, np.int64)
    for p in range(m):
        q = cnt[lo1[p]]
        lo0[q] = lo1[p]
        hi0[q] = hi1[p]
        cnt[lo1[p]] += 1
    for p in range(m):
        keys[p] = (ids[lo0[p]] << 32) | ids[hi0[p]]
    return lo0, hi0, keys


@njit(cache=True)
def remap_pairs(pa, pb, pF, newidx):
    """Reindexa pares tras la compactación por muertes; descarta los que involucran muertos."""
    m = pa.shape[0]
    a2 = np.empty(m, np.int64)
    b2 = np.empty(m, np.int64)
    F2 = np.empty(m, np.float64)
    p = 0
    for k in range(m):
        i = newidx[pa[k]]
        j = newidx[pb[k]]
        if i >= 0 and j >= 0:
            a2[p] = i
            b2[p] = j
            F2[p] = pF[k]
            p += 1
    return a2[:p], b2[:p], F2[:p]


# ------------------------------------------------------------------ v2: actualización asincrónica
@njit(cache=True)
def build_pairs_r2(x, y, G, periodic):
    """Todos los pares no ordenados (a, b) de agentes a distancia de Chebyshev <= 2 (índices de agente).
    Con bordes periódicos requiere G >= 5 (si no, una celda se alcanzaría por dos desplazamientos)."""
    n = x.shape[0]
    ncells = G * G
    cell = x * G + y
    order, start, count = cell_sort(cell, ncells)
    offx = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], np.int64)
    offy = np.array([1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2], np.int64)
    total = 0
    for c in range(ncells):
        nc = count[c]
        if nc == 0:
            continue
        cx = c // G
        cy = c % G
        total += nc * (nc - 1) // 2
        for k in range(12):
            xx = cx + offx[k]
            yy = cy + offy[k]
            if periodic:
                xx %= G
                yy %= G
            elif xx < 0 or xx >= G or yy < 0 or yy >= G:
                continue
            total += nc * count[xx * G + yy]
    a = np.empty(total, np.int64)
    b = np.empty(total, np.int64)
    p = 0
    for c in range(ncells):
        nc = count[c]
        if nc == 0:
            continue
        cx = c // G
        cy = c % G
        s0 = start[c]
        for u in range(nc):
            for v in range(u):
                a[p] = order[s0 + u]
                b[p] = order[s0 + v]
                p += 1
        for k in range(12):
            xx = cx + offx[k]
            yy = cy + offy[k]
            if periodic:
                xx %= G
                yy %= G
            elif xx < 0 or xx >= G or yy < 0 or yy >= G:
                continue
            c2 = xx * G + yy
            s2 = start[c2]
            for u in range(nc):
                for v in range(count[c2]):
                    a[p] = order[s0 + u]
                    b[p] = order[s2 + v]
                    p += 1
    return a, b


@njit(cache=True)
def async_move_kernel(order, u, x, y, G, periodic, ptr, nbr, Fv, alpha, linear, use_floor, floor,
                      crowd, occ_c, terr, is_rm, occm, refuge, cap, has_refuge, rfac, mobile, occ):
    """Barrido secuencial (orden `order`): cada agente móvil elige su destino con los pesos de
    Universe.movement_weights evaluados sobre las posiciones *actuales* y se mueve antes que el siguiente.
    Modifica x, y, occ, occ_c y occm en el lugar. Devuelve el nº de filas con todos los pesos nulos."""
    nzero = 0
    A = np.zeros(9)
    w = np.zeros(9)
    cells = np.zeros(9, np.int64)
    for k in range(order.shape[0]):
        i = order[k]
        if not mobile[i]:
            continue
        xi = x[i]
        yi = y[i]
        for d in range(9):
            A[d] = 0.0
        for q in range(ptr[i], ptr[i + 1]):
            j = nbr[q]
            dx = x[j] - xi
            dy = y[j] - yi
            if periodic:
                if dx > G // 2:
                    dx -= G
                elif dx < -(G // 2):
                    dx += G
                if dy > G // 2:
                    dy -= G
                elif dy < -(G // 2):
                    dy += G
            if dx < -1 or dx > 1 or dy < -1 or dy > 1:
                continue
            A[(dx + 1) * 3 + (dy + 1)] += Fv[q]
        tot = 0.0
        for d in range(9):
            nx = xi + DX[d]
            ny = yi + DY[d]
            if periodic:
                nx %= G
                ny %= G
            elif nx < 0 or nx >= G or ny < 0 or ny >= G:
                w[d] = 0.0
                cells[d] = -1
                continue
            c = nx * G + ny
            cells[d] = c
            self_ = 1 if d == 4 else 0
            if linear:
                wd = 1.0 + alpha[i] * A[d]
                if use_floor and wd < floor:
                    wd = floor
            else:
                wd = np.exp(alpha[i] * A[d])
            if crowd[i] > 0.0:
                wd *= np.exp(-crowd[i] * (occ_c[c] - self_))
            if terr > 0.0 and is_rm[i]:
                wd *= np.exp(-terr * (occm[c] - self_))
            if has_refuge and refuge[c]:
                if occ[c] - self_ >= cap:
                    wd = 0.0
                else:
                    wd *= rfac[i]
            w[d] = wd
            tot += wd
        if tot <= 0.0:
            nzero += 1
            continue
        r = u[i] * tot
        acc = 0.0
        sel = -1
        last = 4
        for d in range(9):
            if w[d] > 0.0:
                last = d
                acc += w[d]
                if sel < 0 and r < acc:
                    sel = d
        if sel < 0:
            sel = last
        if sel != 4:
            c0 = cells[4]
            c1 = cells[sel]
            occ[c0] -= 1
            occ[c1] += 1
            occ_c[c0] -= 1
            occ_c[c1] += 1
            if is_rm[i]:
                occm[c0] -= 1
                occm[c1] += 1
            x[i] = c1 // G
            y[i] = c1 % G
    return nzero
