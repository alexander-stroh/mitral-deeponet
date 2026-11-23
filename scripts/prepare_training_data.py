"""Build the training set of the DeepONet from the published URANS database.

For every snapshot of every training case: the branch input (u*, v* on the 10738-point grid at z* = 0) and all nodes of the
clipped mesh with (x*, y*, z*) and the targets (u*, v*, w*, p*, nu_t*). With --fraction < 1 a fixed uniform random subset of
the nodes is kept per snapshot (the published networks were pretrained on all 4.3e9 points; the reproduction runs in the
paper's response used 25 % for pretraining and 5 % for rehearsal during adaptation, see docs/REPRODUCE.md).

Output (<out>/): meta.json; branch_u.f32, branch_v.f32 (n_snapshots x 10738), branch_t.f32 (n_snapshots); points as flat
arrays x, y, z, u, v, w, p, nu_t (.f32) and snap.i32 (snapshot index of each point). Raw little-endian binaries, read with
np.memmap using the shapes in meta.json (mitral_deeponet.data.TrainingSet).

--legacy-pressure reproduces the arrays of the original training pipeline exactly: it replaces the pressure of the first
snapshot of every case by that of the second, also where the published first snapshot is valid (Drop-S, Pinhole-S,
Pinhole-L; see mitral_deeponet/data/cfd.py). Without it these three snapshots keep their valid pressure.

Usage: python scripts/prepare_training_data.py --data <radar root> --out <dir> [--fraction 1.0] [--cases Drop-S,...]
                                               [--exclude Slot-M] [--workers 16] [--seed 2026] [--legacy-pressure]"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mitral_deeponet.data.cfd import CFDCase, TRAINING_CASES          # noqa: E402

KEYS = ('x', 'y', 'z', 'u', 'v', 'w', 'p', 'nu_t')
_case = {}


def work(args):
    root, case, ci, i, fraction, seed, legacy = args
    if case not in _case:
        _case.clear(); _case[case] = CFDCase(root, case, legacy_pressure=legacy)
    c = _case[case]
    g = c.clipped(i)
    xyz, F = c.snapshot(i, grid=g)
    bu, bv = c.branch(i, grid=g)
    if fraction < 1:
        keep = np.random.default_rng([seed, ci, i]).random(len(xyz)) < fraction
        xyz, F = xyz[keep], F[keep]
    return i, float(c.tstar[i]), float(c.T[i]), bu, bv, np.c_[xyz, F].astype('float32')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='directory containing the unpacked cases (<Case>/<Case>.case)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', default=','.join(TRAINING_CASES))
    ap.add_argument('--exclude', default='', help='comma-separated cases to leave out (held-out test)')
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--legacy-pressure', action='store_true',
                    help='first/last snapshot pressure as in the original pipeline (copied from the neighbouring snapshot)')
    A = ap.parse_args()
    cases = [c for c in A.cases.split(',') if c and c not in A.exclude.split(',')]
    os.makedirs(A.out, exist_ok=True)
    fh = {k: open(os.path.join(A.out, f'{k}.f32'), 'wb') for k in KEYS + ('branch_u', 'branch_v', 'branch_t')}
    fs = open(os.path.join(A.out, 'snap.i32'), 'wb')
    meta = dict(cases=[], fraction=A.fraction, seed=A.seed, n_branch=10738, legacy_pressure=A.legacy_pressure,
                dataset='doi:10.35097/gycyvxthuw98sfpq')
    t0 = time.time(); s = 0; n = 0
    for ci, case in enumerate(cases):
        nsnap = len(CFDCase(A.data, case))
        info = dict(case=case, first_snapshot=s, n_snapshots=nsnap, points=[], tstar=[], T=[])
        with Pool(A.workers) as pool:
            for i, ts, T, bu, bv, P in pool.imap(work, [(A.data, case, ci, i, A.fraction, A.seed, A.legacy_pressure) for i in range(nsnap)], chunksize=2):
                if not (np.isfinite(bu).all() and np.isfinite(bv).all()):
                    raise ValueError(f'{case} snapshot {i}: NaN in the branch input')
                for k, col in zip(KEYS, P.T):
                    fh[k].write(np.ascontiguousarray(col).tobytes())
                fs.write(np.full(len(P), s + i, 'int32').tobytes())
                fh['branch_u'].write(bu.tobytes()); fh['branch_v'].write(bv.tobytes())
                fh['branch_t'].write(np.float32(ts).tobytes())
                info['points'].append(len(P)); info['tstar'].append(ts); info['T'].append(T); n += len(P)
                if i % 50 == 0:
                    print(f'[{time.time() - t0:7.0f}s] {case} {i}/{nsnap}: {len(P)} points, t* = {ts:.4f}', flush=True)
        meta['cases'].append(info); s += nsnap
    for f in list(fh.values()) + [fs]:
        f.close()
    meta.update(n_snapshots=s, n_points=n)
    json.dump(meta, open(os.path.join(A.out, 'meta.json'), 'w'), indent=1)
    print(f'done: {s} snapshots, {n} points, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
