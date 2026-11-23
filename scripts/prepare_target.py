"""Prepare the data of one target case for test-time adaptation (TTA) and evaluation.

CFD target (synthetic validation, e.g. Drop-XL, Slot-L-Bent, or any case of the database):
  branch inputs and the planar observations (u*, v* on the branch grid at z* = 0) for all snapshots, and the CFD pressure at
  the nodes nearest to the LV and LA probe positions. With --xz also the observations u*, w* on an x-z plane at y* = 0
  (the "second, orthogonal plane"; window x* in [0.005, 0.69], z* in [-0.22, 0.22], linear interpolation in a Delaunay
  tetrahedralisation of the left-atrial nodes).
PIV target (in vitro: Pinhole-L, Slot-L, Drop-XL, EccJet):
  branch inputs and planar observations from the phase-resolved PIV. The paper's adaptations to Pinhole-L, Slot-L and
  Drop-XL also used the measured LV/LA pressures (EccJet did not). These entered through a manual synchronisation that is
  not part of the dataset, so PIV targets are prepared without pressure. In an ablation the pressure signals changed the
  RVol by at most 1.5 ml and the errors by at most 3 points.

Frames: the branch window of Slot-L-Bent is shifted by +0.064 in x* (deformed plate). The planar observations are stored with
the coordinates of the branch grid, i.e. in the frame in which the network sees the window (x*_CFD - 0.064 for Slot-L-Bent),
as in the paper's runs; the pressure probes keep their nominal positions.

Output npz keys: case, source, tstar, T, bu, bv (n, 10738), wx, wy, wz (10738,) observation points, [pxyz (2, 3), p (n, 2)],
[xz (m, 3), Uxz (n, m, 3)].
--legacy-pressure (CFD) reproduces the paper's targets exactly: the original preparation replaced the pressure of the first
snapshot by that of the second, although the published first snapshot of Drop-XL is valid (2 of 602 probe values).

Usage: python scripts/prepare_target.py --cfd <radar root> --case Drop-XL --out targets/Drop-XL.npz [--xz] [--workers 16]
                                        [--legacy-pressure]
       python scripts/prepare_target.py --piv <radar root>/PIV --case EccJet --out targets/EccJet_piv.npz"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mitral_deeponet.grid import branch_grid            # noqa: E402

_c = {}


def _cfd_snapshot(args):
    root, case, i, pidx, xz, legacy = args
    from mitral_deeponet.data.cfd import CFDCase
    if case not in _c:
        _c[case] = CFDCase(root, case, legacy_pressure=legacy)
    c = _c[case]
    g = c.clipped(i)
    bu, bv = c.branch(i, grid=g)
    _, F = c.snapshot(i, grid=g)
    out = dict(i=i, bu=bu, bv=bv, p=F[pidx, 3])
    if xz is not None:
        W, V, ok = xz['W'], xz['V'], xz['ok']
        U = np.full((len(ok), 3), np.nan, 'float32'); U[ok] = np.einsum('ij,ijk->ik', W, F[:, :3][V])
        out['Uxz'] = U
    return out


def xz_plane(c, shift):
    """x-z plane at y* = 0 on the window grid (x*: 119 points, z*: the 102 window y values), fluid points only."""
    from scipy.spatial import Delaunay, cKDTree
    from mitral_deeponet.grid import NX, NY, X_RANGE, Y_RANGE
    xyz, _ = c.snapshot(int(c.p_valid[0]))
    gx = np.linspace(*X_RANGE, NX) + shift; gz = np.linspace(*Y_RANGE, NY)
    XZ = np.c_[np.repeat(gx, len(gz)), np.zeros(len(gx) * len(gz)), np.tile(gz, len(gx))]
    sub = np.where((xyz[:, 0] >= gx.min() - 0.004) & (xyz[:, 0] <= gx.max() + 0.03) & (np.abs(xyz[:, 1]) <= 0.26) & (np.abs(xyz[:, 2]) <= 0.23))[0]
    tri = Delaunay(xyz[sub].astype('float64'))
    dist, _ = cKDTree(xyz[sub]).query(XZ)
    s = tri.find_simplex(XZ); keep = (s >= 0) & (dist < 1.5 * np.diff(gz).mean())
    XZ = XZ[keep]; s = s[keep]
    Tm = tri.transform[s]; bc = np.einsum('ijk,ik->ij', Tm[:, :3], XZ - Tm[:, 3])
    return dict(points=XZ, W=np.c_[bc, 1 - bc.sum(1)], V=sub[tri.simplices[s]], ok=np.ones(len(XZ), bool))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfd', help='root of the unpacked CFD cases')
    ap.add_argument('--piv', help='root of the unpacked PIV part (…/PIV)')
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--xz', action='store_true', help='CFD only: also extract the x-z plane at y* = 0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--legacy-pressure', action='store_true',
                    help='CFD only: first/last snapshot pressure as in the original pipeline (copied from the neighbour)')
    A = ap.parse_args()
    t0 = time.time()
    xy = branch_grid()
    res = dict(case=A.case, wx=xy[:, 0], wy=xy[:, 1], wz=np.zeros(len(xy)))
    if A.piv:
        from mitral_deeponet.data.piv import PIVCampaign
        bu, bv, ts, T, ph = PIVCampaign(A.piv, A.case).branch_inputs()
        res.update(source='PIV', bu=bu, bv=bv, tstar=ts, T=T, phase=ph)
    else:
        from mitral_deeponet.data.cfd import CFDCase
        c = CFDCase(A.cfd, A.case, legacy_pressure=A.legacy_pressure)
        g = c.clipped(int(c.p_valid[0])); pidx = c.probe_nodes(grid=g)
        xyz, _ = c.snapshot(int(c.p_valid[0]), grid=g)
        xz = xz_plane(c, c.shift) if A.xz else None
        if xz is not None:
            xz['points'][:, 0] -= c.shift                                 # store in the frame of the branch window
        with Pool(A.workers) as pool:
            R = sorted(pool.map(_cfd_snapshot, [(A.cfd, A.case, i, pidx, xz, A.legacy_pressure) for i in range(len(c))]), key=lambda r: r['i'])
        res.update(source='CFD', bu=np.stack([r['bu'] for r in R]), bv=np.stack([r['bv'] for r in R]), tstar=c.tstar, T=c.T,
                   pxyz=xyz[pidx].astype('float64'), p=np.stack([r['p'] for r in R]).astype('float32'))
        if xz is not None:
            res.update(xz=xz['points'], Uxz=np.stack([r['Uxz'] for r in R]))
    for k in ('bu', 'bv'):
        if not np.isfinite(res[k]).all():
            raise ValueError(f'NaN in {k}')
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    np.savez(A.out, **res)
    print(f'{A.case} ({res["source"]}): {len(res["tstar"])} snapshots, t* {res["tstar"].min():.4f}..{res["tstar"].max():.4f} -> {A.out}'
          f' ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
