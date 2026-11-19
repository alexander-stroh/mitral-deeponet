"""Reading and preprocessing of the published URANS database (RADAR4KIT, doi:10.35097/gycyvxthuw98sfpq).

Each case is an EnSight Gold data set <root>/<Case>/<Case>.case with per-node fields u, v, w (m/s), p (Pa), mu_t (Pa s).
The preprocessing reproduces the pipeline used to train the published networks:

1. rotate the mesh by -30 deg about z: the .geo coordinates are in the frame of the CFD set-up. The velocity components
   are already given in the jet-aligned frame (STAR-CCM+ "rotated_cos") and are therefore not rotated;
2. clip the domain to z >= -0.05 m and y <= 0.06 m (left atrium and adjacent ventricle; vtkClipDataSet, which adds
   interpolated points on the clip planes);
3. scale coordinates by L_C and fields by U_C, P_C and MU_T_C (see grid.py), time to t* in [0, 1];
4. branch input: cut the clipped domain with the plane z* = 0 (vtkCutter) and interpolate u*, v* linearly (scipy griddata)
   onto the 10738 points of grid.branch_grid(), optionally shifted in x (Slot-L-Bent: +0.064).

Snapshots without a pressure file (first/last step of a simulation run, see the dataset README) receive the pressure of the
nearest snapshot with valid pressure, as in the original pipeline.
Requires vtk (tested with 9.3.0) and scipy."""
import math
import os
import re

import numpy as np

from ..grid import L_C, T_C, U_C, P_C, MU_T_C, branch_grid, NX, NY, X_RANGE, Y_RANGE

TRAINING_CASES = ['Drop-S', 'Drop-M', 'Drop-L', 'Slot-S', 'Slot-M', 'Slot-L', 'Pinhole-S', 'Pinhole-M', 'Pinhole-L']
VALIDATION_CASES = ['Drop-XL', 'Slot-L-Bent']
BRANCH_SHIFT = {'Slot-L-Bent': 0.064}           # x* shift of the branch window / TTA plane (deformed plate)
# pressure probe positions of the in vitro rig, model frame, m (dataset README)
PROBE_LV = (-0.11825, -0.03680, 0.0)
PROBE_LA = (0.00986, 0.00008, -0.040)
FIELDS = ('u', 'v', 'w', 'p', 'mu_t')


def rescale_time(T):
    """Absolute simulation time (s) -> t* in [0, 1] (cycle phase; the first snapshot defines the cycle)."""
    T = np.asarray(T, dtype='float64')
    return (T - math.floor(T.min() / T_C) * T_C) / T_C


def _case_file(root, case):
    return os.path.join(root, case, f'{case}.case')


def _pressure_steps(case_file):
    """Indices (into time set 1) of the snapshots that have a pressure file, from time set 2 of the .case file."""
    txt = open(case_file).read()
    sets = re.split(r'time set:\s*', txt)[1:]
    def numbers(s):
        m = re.search(r'filename numbers:\s*(.*?)(?:\n\s*time values:|$)', s, re.S)
        return [int(x) for x in m.group(1).split()] if m else None
    all_steps = numbers(sets[0])
    p_steps = numbers(sets[1]) if len(sets) > 1 else all_steps
    pos = {n: i for i, n in enumerate(all_steps)}
    return np.array([pos[n] for n in p_steps])


class CFDCase:
    """One case of the published database, read with vtkEnSightGoldBinaryReader.

    >>> c = CFDCase('/data/radar', 'Drop-XL')
    >>> xyz, F = c.snapshot(10)          # model frame, non-dimensional; F columns u*, v*, w*, p*, nu_t*
    >>> bu, bv = c.branch(10)            # branch input (10738,), (10738,)
    """

    def __init__(self, root, case):
        import vtk
        self.vtk = vtk
        vtk.vtkObject.GlobalWarningDisplayOff()                 # polyhedral cells: harmless "non-manifold" contour warnings
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
        self.case, self.file = case, _case_file(root, case)
        r = vtk.vtkEnSightGoldBinaryReader()
        r.SetCaseFileName(self.file)
        r.UpdateInformation()
        r.Update()
        ts = r.GetTimeSets().GetItem(0)
        self.T = np.array([ts.GetTuple1(i) for i in range(ts.GetNumberOfTuples())])
        self.tstar = rescale_time(self.T)
        self.reader = r
        self.p_valid = _pressure_steps(self.file)          # snapshot indices with pressure
        self.shift = BRANCH_SHIFT.get(case, 0.0)
        self._clipped = {}

    def __len__(self):
        return len(self.T)

    def _read(self, i):
        self.reader.SetTimeValue(self.T[i])
        self.reader.Update()
        out = self.vtk.vtkUnstructuredGrid()
        out.DeepCopy(self.reader.GetOutput().GetBlock(0))
        return out

    def _transform(self, grid):
        """Rotate by -30 deg about z, clip (z >= -0.05 m, y <= 0.06 m), scale by 1 / L_C. Same filter chain as the original."""
        vtk = self.vtk
        rot = vtk.vtkTransform(); rot.PostMultiply(); rot.RotateZ(-30)
        f = vtk.vtkTransformFilter(); f.SetInputData(grid); f.SetTransform(rot); f.Update()
        g = f.GetOutput()
        for origin, normal, inside_out in (((0, 0, -0.05), (0, 0, 1), False), ((0, 0.06, 0), (0, 1, 0), True)):
            pl = vtk.vtkPlane(); pl.SetOrigin(*origin); pl.SetNormal(*normal)
            c = vtk.vtkClipDataSet(); c.SetInputData(g); c.SetClipFunction(pl); c.SetInsideOut(inside_out); c.Update()
            g = c.GetOutput()
        s = vtk.vtkTransform(); s.Scale(1 / L_C, 1 / L_C, 1 / L_C)
        f = vtk.vtkTransformFilter(); f.SetInputData(g); f.SetTransform(s); f.Update()
        return f.GetOutput()

    def clipped(self, i):
        """vtkUnstructuredGrid of snapshot i in the model frame (non-dimensional coordinates, dimensional fields)."""
        g = self._transform(self._read(i))
        if i not in set(self.p_valid.tolist()):          # replace the missing pressure by that of the nearest valid snapshot
            j = int(self.p_valid[np.argmin(np.abs(self.p_valid - i))])
            src = self._transform(self._read(j)).GetPointData().GetArray('p')
            g.GetPointData().RemoveArray('p')
            g.GetPointData().AddArray(src)
        return g

    def snapshot(self, i, grid=None):
        """(xyz (N, 3), fields (N, 5)) of snapshot i: non-dimensional, model frame, float32."""
        from vtk.util.numpy_support import vtk_to_numpy
        g = self.clipped(i) if grid is None else grid
        xyz = vtk_to_numpy(g.GetPoints().GetData()).astype('float32')
        pd_ = g.GetPointData()
        F = np.stack([vtk_to_numpy(pd_.GetArray(a)) for a in FIELDS], 1).astype('float64')
        F[:, :3] /= U_C; F[:, 3] /= P_C; F[:, 4] /= MU_T_C
        return xyz, F.astype('float32')

    def plane(self, i, xy, z=0.0, grid=None):
        """u*, v* interpolated linearly onto the points xy (n, 2) of the plane z* = z (vtkCutter + griddata)."""
        from scipy.interpolate import griddata
        from vtk.util.numpy_support import vtk_to_numpy
        vtk = self.vtk
        g = self.clipped(i) if grid is None else grid
        pl = vtk.vtkPlane(); pl.SetNormal(0, 0, 1); pl.SetOrigin(0, 0, z)
        cut = vtk.vtkCutter(); cut.SetCutFunction(pl); cut.SetInputData(g); cut.Update()
        s = cut.GetOutput()
        P = vtk_to_numpy(s.GetPoints().GetData())
        out = []
        for a in ('u', 'v'):
            out.append(griddata((P[:, 0], P[:, 1]), vtk_to_numpy(s.GetPointData().GetArray(a)), (xy[:, 0], xy[:, 1]), method='linear') / U_C)
        return out[0].astype('float32'), out[1].astype('float32')

    def branch(self, i, grid=None):
        """Branch input (u*, v*) of snapshot i on the 10738-point grid; the window is shifted by +0.064 in x for Slot-L-Bent.
        Interpolation on the full 119 x 102 grid first, then masking, exactly as in the original pipeline."""
        xy, mask = branch_grid(return_mask=True)
        x = np.linspace(*X_RANGE, NX) + self.shift
        y = np.linspace(*Y_RANGE, NY)
        X, Y = np.meshgrid(x, y)
        u, v = self.plane(i, np.c_[X.ravel(), Y.ravel()], grid=grid)
        idx = np.where(mask.ravel())[0]
        return u[idx], v[idx]

    def probe_nodes(self, grid=None):
        """Indices of the clipped-mesh nodes nearest to the LV and LA pressure probes (as in the original pipeline)."""
        from vtk.util.numpy_support import vtk_to_numpy
        g = self.clipped(int(self.p_valid[0])) if grid is None else grid
        X = vtk_to_numpy(g.GetPoints().GetData()).astype('float64')
        return [int(np.argmin(((X - np.array(p) / L_C) ** 2).sum(1))) for p in (PROBE_LV, PROBE_LA)]
