"""Check that the standalone implementation reproduces the original Modulus predictions stored in reference/.

The reference outputs were computed with NVIDIA Modulus Sym 1.5.0 in float64 on CPU, at the 10738 branch-grid points on the
supervised plane z* = 0 plus 2000 random points in the volume, for two instants per model and branch input.
The weights are expected in weights/ (python -m mitral_deeponet.weights) or in $MITRAL_DEEPONET_WEIGHTS; missing ones are skipped.
Usage: python tests/test_reference.py [--float32]      (exit code 0 if all available models agree within tolerance)"""
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
from mitral_deeponet import load
WEIGHTS = pathlib.Path(os.environ.get('MITRAL_DEEPONET_WEIGHTS', ROOT / 'weights'))

F32 = '--float32' in sys.argv
TOL = 2e-4 if F32 else 1e-9            # max abs deviation in non-dimensional units (|u*| reaches ~1)
R = np.load(ROOT / 'tests' / 'data' / 'reference_outputs.npz')
pts = R['points']; tags = sorted({k.rsplit('__', 1)[0] for k in R.files if k.endswith('__out')})
worst = 0.0; ok = True; models = {}; n = 0
for tag in tags:
    mid = tag.split('__')[0]
    w = WEIGHTS / f'{mid}.safetensors'
    if not w.exists():
        print(f'{tag:45s} skipped ({w} missing)'); continue
    if mid not in models:
        models = {mid: load(w).to(torch.float32 if F32 else torch.float64)}
    m = models[mid]
    pr = m.predict(R[f'{tag}__u_plane'], R[f'{tag}__v_plane'], pts, float(R[f'{tag}__tstar'])).double().numpy()
    d = float(np.abs(pr - R[f'{tag}__out']).max()); worst = max(worst, d)
    print(f'{tag:45s} max |diff| = {d:.2e}', 'OK' if d < TOL else 'FAIL')
    ok &= d < TOL; n += 1
ok &= n > 0
print(f'{"float32" if F32 else "float64"}: {n} checks, worst deviation {worst:.2e} (tolerance {TOL:.0e}) ->', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
