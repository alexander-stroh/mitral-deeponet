"""Evaluate one or more networks on a CFD case of the published database, with the metrics of the paper.

  in-plane / off-plane : NRMSE of |u|* (u, v, w) on the branch window extruded to 61 planes z* in [-0.2, 0.2], 150 instants
                         (every 2nd snapshot); pooled RMSE over all points and instants / pooled std of the CFD; in-plane =
                         z* = 0, off-plane = |z*| > 0.03 (3 mm). Per-plane values are written for the error-vs-|z| profile.
  aggregate            : NRMSE of u*, v*, w* on a fixed random 5 % of the nodes of the clipped mesh (seed 42), 101 instants
                         (every 3rd snapshot), pooled over space and time.
  RVol                 : regurgitant volume through the integration window at x* = 0.005 (0.5 mm) of the paper (Figs. 10, 13),
                         for the cases with defined integration lines; CFD integrated over the same window. Both the paper's
                         quadrature and a corrected one (full trapezoid weight of the centre line) are reported.
CFD reference: linear interpolation in a Delaunay tetrahedralisation of the mesh nodes around the window. For Slot-L-Bent the
network is queried in the frame of its adaptation (x*_CFD - 0.064). Uses the standalone PyTorch model (no Modulus needed).

Usage: python scripts/evaluate.py --cfd <radar root> --case Drop-XL --model baseline=weights/pretrained.safetensors \
           --model tta=runs/tta_dropxl/model.safetensors [--out eval/Drop-XL.json] [--workers 16] [--device cuda]"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mitral_deeponet.grid import branch_grid, U_C, L_C, T_C       # noqa: E402

# integration lines of the RVol window (z*, y*_min, y*_max per line, model frame), as used for the paper's figures
_sl = [(0, .12), (1, .12), (2, .115), (3, .115), (4, .11), (5, .11), (6, .10), (7, .10), (8, .09), (9, .09), (10, .08), (11, .08),
       (12, .07), (13, .07), (14, .06), (15, .04), (16, .02)]
_dx = [(0, -.109, .177), (1, -.108, .175), (2, -.107, .167), (3, -.1065, .155), (4, -.106, .14), (5, -.105, .127), (6, -.1035, .114),
       (7, -.102, .1), (8, -.1, .087), (9, -.098, .075), (10, -.095, .062), (11, -.093, .048), (12, -.088, .032), (13, -.084, .02),
       (14, -.078, .008), (15, -.07, -.005), (16, -.05, -.015)]
_pm = [.047, .0465, .0458, .0452, .0443, .043, .0415, .039, .0366, .0333, .03, .0245, .015]
_sm = [.075, .0723, .0684, .0631, .0578, .051, .0445, .0361, .023]
_w = {'Slot-L-Bent': 0.0363, 'Drop-XL': 0.065, 'Pinhole-M': 0.0435, 'Slot-M': 0.0222, 'Pinhole-L': 0.061}
LINES = {
    'Slot-L-Bent': (100, [(_w['Slot-L-Bent'] * k / 16, -(h - .005), h - .005) for k, h in _sl]),
    'Drop-XL': (100, [(_w['Drop-XL'] * k / 16, a, b) for k, a, b in _dx]),
    'Pinhole-M': (50, [(_w['Pinhole-M'] * k / 12, -h, h) for k, h in enumerate(_pm)]),
    'Slot-M': (60, [(_w['Slot-M'] * k / 8, -h, h) for k, h in enumerate(_sm)]),
    'Pinhole-L': (70, [(_w['Pinhole-L'] * k / 12, -_w['Pinhole-L'] / _w['Pinhole-M'] * h, _w['Pinhole-L'] / _w['Pinhole-M'] * h)
                       for k, h in enumerate(_pm)]),
}
ZS = np.linspace(-0.2, 0.2, 61); ZS[30] = 0.0


def window_points(case):
    """Points (y*, z*) of the RVol window (lines mirrored in z), line ids, dy per line, z weights (paper and trapezoid)."""
    n_c, lines = LINES[case]
    zv = np.array([l[0] for l in lines]); dy_t = (lines[0][2] - lines[0][1]) / (n_c - 1)
    ys, zs, ids, dys, zw_paper, zw_trap = [], [], [], [], [], []
    for i, (z, y0, y1) in enumerate(lines):
        dz = (zv[1] - zv[0]) / 2 if i == 0 else (zv[-1] - zv[-2]) / 2 if i == len(lines) - 1 else (zv[i + 1] - zv[i - 1]) / 2
        zw_paper.append(dz * (1.0 if z == 0 else 2.0))                       # as in the paper (centre line: half weight)
        zw_trap.append((zv[1] - zv[0]) if z == 0 else 2 * dz)               # trapezoid over -z_max ... z_max
        n = max(2, round((y1 - y0) / dy_t) + 1)
        ys.append(np.linspace(y0, y1, n)); zs.append(np.full(n, z)); ids.append(np.full(n, i)); dys.append((y1 - y0) / (n - 1))
    return np.concatenate(ys), np.concatenate(zs), np.concatenate(ids), np.array(dys), np.array(zw_paper), np.array(zw_trap)


def flux(u, ids, dys, zw):
    """Volume flow (m^3/s) from u* on the window points."""
    return sum(np.trapz(u[ids == i] * U_C, dx=dys[i] * L_C) * zw[i] * L_C for i in range(len(dys)))


def rvol(T, q, win=(-0.05, 0.35)):
    """Regurgitant volume (ml): positive flow integrated over the window, periodic continuation (Fig. 10 definition)."""
    t = T - T.min(); tt = np.r_[t - T_C, t, t + T_C]; qq = np.r_[q, q, q]; o = np.argsort(tt); tt, qq = tt[o], qq[o]
    m = (tt >= win[0]) & (tt <= win[1])
    return float(np.trapz(np.maximum(qq[m], 0), tt[m]) * 1e6)


G = {}


def _reference(args):
    """CFD reference at one snapshot: z-t points, aggregate sample, RVol points; plus the branch input."""
    root, case, i = args
    from mitral_deeponet.data.cfd import CFDCase
    if 'c' not in G:
        G['c'] = CFDCase(root, case)
    c = G['c']; g = c.clipped(i)
    _, F = c.snapshot(i, grid=g); bu, bv = c.branch(i, grid=g)
    out = dict(i=i, bu=bu, bv=bv)
    for k in ('Q', 'R'):
        if k in G['W']:
            W, V, ok = G['W'][k]
            U = np.full((len(ok), 3), np.nan, 'float32'); U[ok] = np.einsum('ij,ijk->ik', W, F[:, :3][V]); out[k] = U
    out['S'] = F[G['S'], :3]
    return out


def _init(W, S):
    G['W'], G['S'] = W, S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfd', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--model', action='append', required=True, help='label=path (.safetensors or .pt state dict); repeatable')
    ap.add_argument('--out'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--device', default='cpu')
    A = ap.parse_args()
    import torch
    from scipy.spatial import Delaunay
    from mitral_deeponet import load
    from mitral_deeponet.data.cfd import CFDCase
    t0 = time.time(); log = lambda *a: print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)
    c = CFDCase(A.cfd, A.case); sh = c.shift
    xyz, _ = c.snapshot(int(c.p_valid[0])); xyz = xyz.astype('float64')
    xy = branch_grid(); nw = len(xy)
    Q = np.c_[np.tile(xy[:, 0] + sh, len(ZS)), np.tile(xy[:, 1], len(ZS)), np.repeat(ZS, nw)]           # CFD frame
    sub = np.where((xyz[:, 0] >= Q[:, 0].min() - 0.004) & (xyz[:, 0] <= Q[:, 0].max() + 0.03) & (np.abs(xyz[:, 1]) <= 0.26) & (np.abs(xyz[:, 2]) <= 0.23))[0]
    tri = Delaunay(xyz[sub]); log('Delaunay', len(sub), 'nodes')

    def weights(P):
        s = tri.find_simplex(P); ok = s >= 0
        Tm = tri.transform[s[ok]]; bc = np.einsum('ijk,ik->ij', Tm[:, :3], P[ok] - Tm[:, 3])
        return np.c_[bc, 1 - bc.sum(1)], sub[tri.simplices[s[ok]]], ok
    W = {'Q': weights(Q)}
    if A.case in LINES:
        ry, rz, rid, rdy, zw_p, zw_t = window_points(A.case)
        R = np.c_[np.full(len(ry), 0.005 + sh), ry, rz]; W['R'] = weights(R)
    S = np.sort(np.random.default_rng(42).choice(len(xyz), len(xyz) // 20, replace=False))
    JQ = list(range(0, 300, 2))[:150]; JS = list(range(0, len(c), 3)); J = sorted(set(JQ) | set(JS) | (set(range(len(c))) if A.case in LINES else set()))
    with Pool(A.workers, initializer=_init, initargs=(W, S)) as pool:
        REF = {r['i']: r for r in pool.imap_unordered(_reference, [(A.cfd, A.case, i) for i in J])}
    log('reference read:', len(REF), 'snapshots')
    pid = np.repeat(np.arange(len(ZS)), nw); offz = np.abs(ZS) > 0.03 + 1e-9; iz0 = int(np.where(ZS == 0)[0][0])
    res = dict(case=A.case, frame_shift=sh, z=ZS.tolist())
    if A.case in LINES:
        T = c.T
        qc = np.array([flux(REF[i]['R'][:, 0], rid, rdy, zw_p) for i in range(len(c))])
        qc_t = np.array([flux(REF[i]['R'][:, 0], rid, rdy, zw_t) for i in range(len(c))])
        res['rvol_cfd_window'] = rvol(T, qc); res['rvol_cfd_window_trapezoid'] = rvol(T, qc_t)
    for spec in A.model:
        label, path = spec.split('=', 1)
        m = load(path, device=A.device)
        def pred(P, i):
            X = P.copy(); X[:, 0] -= sh
            return m.predict(REF[i]['bu'], REF[i]['bv'], X, c.tstar[i]).cpu().numpy().astype('float64')
        acc = np.zeros((len(ZS), 4))                                                    # n, sse, s, ss  (|u|)
        for i in JQ:
            ref = REF[i]['Q'].astype('float64'); ok = np.isfinite(ref).all(1); p = pred(Q, i)
            mr, mp = np.linalg.norm(ref[ok], axis=1), np.linalg.norm(p[ok, :3], axis=1); ip = pid[ok]
            acc += np.stack([np.bincount(ip, np.ones(len(mr)), len(ZS)), np.bincount(ip, (mp - mr) ** 2, len(ZS)),
                             np.bincount(ip, mr, len(ZS)), np.bincount(ip, mr ** 2, len(ZS))], 1)
        nrmse = lambda a: float(100 * np.sqrt(a[1] / a[0]) / np.sqrt(a[3] / a[0] - (a[2] / a[0]) ** 2))
        r = dict(inplane_mag=nrmse(acc[iz0]), offplane_mag=nrmse(acc[offz].sum(0)), zprofile_mag=[nrmse(a) for a in acc])
        agg = np.zeros((3, 4))
        for i in JS:
            ref = REF[i]['S'].astype('float64'); p = pred(xyz[S], i)[:, :3]
            agg += np.stack([np.full(3, len(ref)), ((p - ref) ** 2).sum(0), ref.sum(0), (ref ** 2).sum(0)], 1)
        for k, n in enumerate('uvw'):
            r[f'agg_{n}'] = nrmse(agg[k])
        if A.case in LINES:
            RP = np.c_[np.full(len(ry), 0.005 + sh), ry, rz]
            q = np.array([flux(pred(RP, i)[:, 0], rid, rdy, zw_p) for i in range(len(c))])
            q_t = np.array([flux(pred(RP, i)[:, 0], rid, rdy, zw_t) for i in range(len(c))])
            r['rvol'], r['rvol_trapezoid'] = rvol(c.T, q), rvol(c.T, q_t)
        res[label] = r
        log(label, {k: round(v, 2) for k, v in r.items() if not isinstance(v, list)}, 'CFD RVol', round(res.get('rvol_cfd_window', np.nan), 2))
    if A.out:
        os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True); json.dump(res, open(A.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
