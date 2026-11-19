"""Example: volumetric reconstruction for one in vitro case from its phase-resolved planar PIV input.

Loads the model adapted to the chosen case and the published branch inputs (PIV interpolated onto the branch grid), evaluates
the velocity on the supervised plane z* = 0 and on the perpendicular x-z plane y* = 0, and writes an .npz with SI fields
(optionally a quick-look figure if matplotlib is installed).
Usage: python examples/predict_piv_case.py [pinholel|slotl|dropxl|eccjet] [snapshot index] [--device cuda]"""
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mitral_deeponet import load, branch_grid, to_si, L_C, T_C, U_C

case = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'pinholel'
j = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
dev = 'cuda' if '--device' in sys.argv and sys.argv[sys.argv.index('--device') + 1] == 'cuda' else 'cpu'

D = np.load(ROOT / 'weights' / 'piv_branch_inputs.npz')          # python -m mitral_deeponet.weights
u, v, ts = D[f'{case}_u'], D[f'{case}_v'], D[f'{case}_tstar']
j = len(ts) // 4 if j is None else j
model = load(ROOT / 'weights' / f'tta_{case}_piv.safetensors', device=dev)
baseline = load(ROOT / 'weights' / 'pretrained.safetensors', device=dev)

xy = branch_grid()                                               # supervised plane z* = 0 (same points as the input)
gx, gz = np.meshgrid(np.linspace(0.005, 0.69, 138), np.linspace(-0.2, 0.2, 81))
xz = np.c_[gx.ravel(), np.zeros(gx.size), gz.ravel()]            # x-z plane y* = 0 (not supervised)
res = {}
for name, m in (('tta', model), ('baseline', baseline)):
    for plane, P in (('xy', np.c_[xy, np.zeros(len(xy))]), ('xz', xz)):
        out = m.predict(u[j], v[j], P, ts[j]).cpu().numpy()
        for k, f in to_si(out).items(): res[f'{name}_{plane}_{k}'] = f
res.update(xy_m=xy * L_C, xz_m=xz * L_C, piv_u=u[j] * U_C, piv_v=v[j] * U_C, tstar=ts[j], t_s=ts[j] * T_C)
out = ROOT / f'prediction_{case}_{j:03d}.npz'; np.savez_compressed(out, **res)
err = np.hypot(res['tta_xy_u'] - res['piv_u'], res['tta_xy_v'] - res['piv_v'])
print(f'{case}: snapshot {j} of {len(ts)}, t* = {ts[j]:.3f}; in-plane RMS |u_pred - u_PIV| = {np.sqrt(np.mean(err ** 2)):.3f} m/s; wrote {out.name}')

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    mag = lambda p, pl: np.hypot(np.hypot(res[f'{p}_{pl}_u'], res[f'{p}_{pl}_v']), res[f'{p}_{pl}_w'])
    vmax = float(np.hypot(res['piv_u'], res['piv_v']).max())
    for a, (val, P, title, yl) in zip(ax, [(np.hypot(res['piv_u'], res['piv_v']), xy, 'PIV input, z* = 0', 'y (cm)'),
                                           (mag('tta', 'xy'), xy, 'TTA prediction |u|, z* = 0', 'y (cm)'),
                                           (mag('tta', 'xz'), xz[:, [0, 2]], 'TTA prediction |u|, y* = 0', 'z (cm)')]):
        sc = a.tricontourf(P[:, 0] * 100 * L_C, P[:, 1] * 100 * L_C, val, levels=40, vmin=0, vmax=vmax, cmap='viridis')
        a.set_aspect('equal'); a.set_title(title); a.set_xlabel('x (cm)'); a.set_ylabel(yl)
    fig.colorbar(sc, ax=ax, label='m/s'); fig.savefig(ROOT / f'prediction_{case}_{j:03d}.png', dpi=130)
except ImportError:
    pass
