"""Phase-resolved PIV of the published database (part record doi:10.35097/mcxab6y7kkyucumh) -> branch inputs.

Reproduces the preprocessing of the in vitro inputs used in the paper:

1. read the ensemble-averaged NetCDF field <prefix>_phaseNNN_avg.nc (velocity [1, H, W, 2] in m/s, grid from the attributes
   coord_min / coord_max in mm, rows from y = -coord_max[1] to -coord_min[1]);
2. rotate coordinates and velocity components by the campaign angle about (0, 0), then translate by the campaign offset
   (camera frame -> model frame, mm);
3. discard the vectors beyond a straight line near the distal window edge where the PIV is unreliable (three campaigns);
4. interpolate u, v linearly (scipy griddata) onto the 119 x 102 branch grid, fill points outside the convex hull of the PIV
   data with the nearest value, keep the 10738 masked points and divide by U_C;
5. time: phase NNN is the time after the pump trigger in ms; T = NNN / 1000 s + sync (manual alignment of the PIV trigger with
   the CFD cycle), phases with T >= T_first + 0.75 s are dropped, T > 0.75 s is wrapped into the cycle, t* = T / 0.75 s.
Requires scipy and netCDF4 or xarray."""
import os
import re

import numpy as np

from ..grid import U_C, T_C, branch_grid, NX, NY, X_RANGE, Y_RANGE

# angle (deg), offset (mm, applied after the rotation), sync (s), cut line (two points in x* / y*, keep below the line)
CAMPAIGNS = {
    'Pinhole-L': dict(folder='Pinhole-L_phase', prefix='result_12_pinhole13', angle=-30.0, offset=(-27.25, -10.70), sync=0.045,
                      cut=((0.665, 0.1), (0.69, -0.05))),
    'Drop-XL': dict(folder='Drop-XL_phase', prefix='result_11_drop14', angle=-29.0, offset=(-27.75, -11.1327), sync=0.07,
                    cut=((0.63, 0.12 + 0.033673), (0.675, -0.05 + 0.033673))),
    'Slot-L': dict(folder='Slot-L_phase', prefix='result_01_Slot8', angle=-29.0, offset=(-20.65, -9.40), sync=0.03, cut=None),
    'EccJet': dict(folder='EccJet_phase', prefix='result_02_EccJet8', angle=-28.5, offset=(-24.75, -13.40), sync=0.04,
                   cut=((0.605, 0.1), (0.67, -0.05))),
}


def read_avg_nc(path):
    """(x mm, y mm, u m/s, v m/s) flattened, camera frame, from a PIVview NetCDF average."""
    try:
        import netCDF4
        with netCDF4.Dataset(path) as d:
            vel = np.array(d.variables['velocity'][:]); cmin, cmax = d.variables['velocity'].coord_min, d.variables['velocity'].coord_max
    except ImportError:
        import xarray as xr
        with xr.open_dataset(path) as d:
            vel = d['velocity'].values; cmin, cmax = d['velocity'].coord_min, d['velocity'].coord_max
    H, W = vel.shape[1:3]
    X, Y = np.meshgrid(np.linspace(cmin[0], cmax[0], W), np.linspace(-cmax[1], -cmin[1], H))
    return X.ravel(), Y.ravel(), vel[0, :, :, 0].ravel(), vel[0, :, :, 1].ravel()


def to_model(x, y, u, v, angle, offset):
    a = np.deg2rad(angle); c, s = np.cos(a), np.sin(a)
    return c * x - s * y + offset[0], s * x + c * y + offset[1], c * u - s * v, s * u + c * v


class PIVCampaign:
    """>>> p = PIVCampaign('/data/radar/PIV', 'EccJet'); bu, bv, tstar, T, phases = p.branch_inputs()"""

    def __init__(self, root, case, plane='0mm'):
        self.case, self.cfg = case, CAMPAIGNS[case]
        self.dir = os.path.join(root, self.cfg['folder'], plane)
        pat = re.compile(re.escape(self.cfg['prefix']) + r'_phase(\d+)_avg\.nc$')
        self.phases = sorted(int(m.group(1)) for f in os.listdir(self.dir) if (m := pat.match(f)))

    def times(self):
        """(phases kept, T in s within the cycle, t*), ordered by t*."""
        T = np.array(self.phases) / 1000.0 + self.cfg['sync']
        keep = T < T[0] + T_C
        ph, T = np.array(self.phases)[keep], T[keep]
        T = np.where(T > T_C, T - T_C, T)
        o = np.argsort(T)
        return ph[o], T[o], T[o] / T_C

    def field(self, phase):
        """PIV vectors of one phase in the model frame: x*, y*, u (m/s), v (m/s), with the cut line applied."""
        x, y, u, v = read_avg_nc(os.path.join(self.dir, f"{self.cfg['prefix']}_phase{phase:03d}_avg.nc"))
        x, y, u, v = to_model(x, y, u, v, self.cfg['angle'], self.cfg['offset'])
        x, y = x / 100.0, y / 100.0                          # mm -> x* = x / 0.1 m
        if self.cfg['cut'] is not None:
            (x1, y1), (x2, y2) = self.cfg['cut']
            k = (y2 - y1) / (x2 - x1); keep = y < k * x + (y1 - k * x1)
            x, y, u, v = x[keep], y[keep], u[keep], v[keep]
        return x, y, u, v

    def branch(self, phase):
        """Branch input (u*, v*) of one phase on the 10738-point grid."""
        from scipy.interpolate import griddata
        x, y, u, v = self.field(phase)
        X, Y = np.meshgrid(np.linspace(*X_RANGE, NX), np.linspace(*Y_RANGE, NY))
        idx = np.where(branch_grid(return_mask=True)[1].ravel())[0]
        out = []
        for f in (u, v):
            lin = griddata((x, y), f, (X, Y), method='linear', fill_value=np.nan)
            near = griddata((x, y), f, (X, Y), method='nearest')
            lin = np.where(np.isnan(lin), near, lin)
            out.append((lin.ravel()[idx] / U_C).astype('float32'))
        return out[0], out[1]

    def branch_inputs(self):
        """All phases: bu, bv (n, 10738), t* (n,), T (n,), phase numbers (n,)."""
        ph, T, ts = self.times()
        B = [self.branch(p) for p in ph]
        return np.stack([b[0] for b in B]), np.stack([b[1] for b in B]), ts, T, ph
