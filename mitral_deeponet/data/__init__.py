"""Data access: published URANS database (cfd), published PIV (piv), and the prepared training set (TrainingSet)."""
import json
import os

import numpy as np


class TrainingSet:
    """Prepared training set (scripts/prepare_training_data.py), memory-mapped.

    Attributes: meta (dict), branch_u / branch_v (n_snapshots, 10738), branch_t (n_snapshots,), points: dict of flat arrays
    x, y, z, u, v, w, p, nu_t (n_points,), snap (n_points,) snapshot index of each point."""

    KEYS = ('x', 'y', 'z', 'u', 'v', 'w', 'p', 'nu_t')

    def __init__(self, path, load=False):
        self.path = path
        self.meta = json.load(open(os.path.join(path, 'meta.json')))
        S, N, nb = self.meta['n_snapshots'], self.meta['n_points'], self.meta['n_branch']
        mm = lambda name, dtype, shape: np.memmap(os.path.join(path, name), dtype=dtype, mode='r', shape=shape)
        self.branch_u = mm('branch_u.f32', 'float32', (S, nb))
        self.branch_v = mm('branch_v.f32', 'float32', (S, nb))
        self.branch_t = mm('branch_t.f32', 'float32', (S,))
        self.snap = mm('snap.i32', 'int32', (N,))
        self.points = {k: mm(f'{k}.f32', 'float32', (N,)) for k in self.KEYS}
        if load:                                  # into RAM (random access on a disk-backed memmap is slow)
            self.snap = np.array(self.snap)
            self.points = {k: np.array(v) for k, v in self.points.items()}

    def __len__(self):
        return self.meta['n_points']

    def case_rows(self, case):
        """Range [a, b) of point rows that belong to one case (points are stored case by case)."""
        c = next(c for c in self.meta['cases'] if c['case'] == case)
        a = sum(sum(d['points']) for d in self.meta['cases'][:self.meta['cases'].index(c)])
        return a, a + sum(c['points'])


def load_target(path):
    """Target file of scripts/prepare_target.py (npz): branch inputs, plane observations, pressure probes."""
    return dict(np.load(path, allow_pickle=False))
