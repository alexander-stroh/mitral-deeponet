# mitral-deeponet

Deep Operator Network (DeepONet) with test-time adaptation for reconstructing the 4D flow of mitral regurgitation from planar
velocity data. This is the code and model release of

> J. M. Hoffmann, Y. Hasegawa, A. Stroh, "Reconstruction of 4D Mitral Regurgitation Hemodynamics from Sparse Planar Data using
> Deep Operator Networks with Test-Time Adaptation", arXiv:2609.20857 (2026), https://doi.org/10.48550/arXiv.2609.20857.

The network maps a masked planar two-component velocity snapshot (u, v), as measured by 2D2C PIV in the mid-plane of an in vitro
left-heart simulator, to the phase-averaged 3D velocity, pressure and eddy-viscosity fields of the regurgitant flow at the same
instant.

- **Pretraining.** The network is pretrained on URANS simulations of nine mitral-regurgitation orifice phantoms (MROPs).
- **Test-time adaptation (TTA).** For each new case the network is fine-tuned for 400 iterations on the case's planar
  observations, while the training data stay in the loss (rehearsal).

The repository contains:

| Part | Needs | Content |
|---|---|---|
| `mitral_deeponet/` (model, grid) | PyTorch, NumPy, safetensors | standalone implementation for inference, scales, frames, branch-input grid |
| `mitral_deeponet/data/` | + SciPy, VTK, netCDF4 | preprocessing of the published URANS and PIV data |
| `scripts/train.py` | NVIDIA Modulus Sym 1.5.0 | pretraining and TTA, as in the paper |
| `scripts/evaluate.py` | as data | the paper's error metrics against the CFD |
| `tests/`, `examples/` | as model | reference test against the original model, PIV example |

The data are published separately on RADAR4KIT (CC BY 4.0): J. Hoffmann, R. Leister, R. Karl, L. Neff, "URANS simulations and
PIV measurements of mitral regurgitation orifice phantoms in a pulsatile in vitro heart simulator" (2026),
https://doi.org/10.35097/gycyvxthuw98sfpq.

Release v1.0.0 (source code and all weight files) is archived on Zenodo: https://doi.org/10.5281/zenodo.24464090.

## Installation

```bash
git clone https://github.com/alexander-stroh/mitral-deeponet && cd mitral-deeponet
pip install -e .                 # inference only: torch, numpy, safetensors
pip install -e .[data]           # + preprocessing and evaluation: scipy, vtk, netCDF4, matplotlib
python -m mitral_deeponet.weights            # download the published checkpoints into weights/
python tests/test_reference.py               # standalone model == original Modulus model (float64: < 1e-9)
```

Training needs NVIDIA Modulus Sym 1.5.0. The paper used the container `nvcr.io/nvidia/modulus/modulus:24.04` with `vtk==9.3.0`
installed; `docs/REPRODUCE.md` gives the set-up.

## Quick start

```python
import numpy as np
from mitral_deeponet import load, branch_grid, to_si

model = load('weights/tta_eccjet_piv.safetensors')            # or device='cuda'
D = np.load('weights/piv_branch_inputs.npz')
u, v, ts = D['eccjet_u'], D['eccjet_v'], D['eccjet_tstar']    # (n_snapshots, 10738) each, non-dimensional

xyz = np.array([[0.02, 0.0, 0.01]])                          # model frame, x* = x / 0.1 m
out = model.predict(u[7], v[7], xyz, ts[7]).numpy()          # (N, 5): u*, v*, w*, p*, nu_t*
print(to_si(out))                                            # m/s, Pa, m^2/s
```

`python examples/predict_piv_case.py eccjet` writes a reconstruction on the supervised plane and on the perpendicular x-z plane.

## Checkpoints

The checkpoints are the 7 networks of the paper (21.1 M parameters each, float32, safetensors, 81 MiB), attached to the GitHub
release. `docs/MODELS.md` lists their provenance and checksums.

| File | Target case | Adaptation data |
|---|---|---|
| `pretrained` | – (9 training MROPs) | – (the *baseline* of the paper) |
| `tta_dropxl_cfd` | Drop-XL (unseen, larger than all training orifices) | CFD plane z* = 0 + virtual pressure probes |
| `tta_slotlbent_cfd` | Slot-L-Bent (unseen, deformed plate) | CFD plane z* = 0 + virtual pressure probes |
| `tta_pinholel_piv` | Pinhole-L | phase-resolved PIV + measured LV/LA pressures |
| `tta_slotl_piv` | Slot-L | phase-resolved PIV + measured LV/LA pressures |
| `tta_dropxl_piv` | Drop-XL | phase-resolved PIV + measured LV/LA pressures |
| `tta_eccjet_piv` | EccJet (unseen eccentric jet) | phase-resolved PIV (no pressures) |

All adapted checkpoints start from `pretrained` (step 50,000) and end at step 50,400. `pretrained_optimizer.pt` holds the
step counter, the AdamW moments and the learning-rate schedule at the end of pretraining, so that new adaptations start as
in the paper. The NTK loss weights are not stored: they are recomputed in the first adaptation step (step 50,000), as in the
original runs.

The parameter names are those of NVIDIA Modulus Sym 1.5.0. The files therefore also load with `strict=True` into a
`DeepONetArch` built from two `FullyConnectedArch` sub-networks (`layer_size=640, nr_layers=9, activation_fn=silu,
adaptive_activations=True`):
- branch keys `u_piv`, `v_piv` (10,738 each);
- trunk keys `x, y, z, t`;
- outputs `u, v, w, p, nu_t`.

## Inputs, outputs and conventions

### Branch input

1. The in-plane velocities (u*, v*) on the plane z* = 0 are interpolated onto a 119 × 102 equidistant grid,
   x ∈ [0.05, 6.9] cm and y ∈ [−2.2, 2.2] cm.
2. Two cut-offs (−30° and 60°) crop the grid to 10,738 points. `branch_grid()` returns them in the expected order.
3. The input vector is [u*₁ … u*₁₀₇₃₈, v*₁ … v*₁₀₇₃₈].

The branch network sees no coordinates, so new data must be interpolated onto exactly these points. The input must not
contain NaN.

### Trunk input

The trunk takes (x*, y*, z*, t*) in the model frame, a jet-aligned Cartesian frame:
- origin at the centre of the flat orifice opening;
- x along the regurgitant jet into the left atrium;
- z normal to the PIV plane.

t* = (T − T₀) / 0.75 s ∈ [0, 1] is the phase in the cardiac cycle (80 bpm), with T₀ the start of the simulated cycle. For Slot-L-Bent, the branch window and the
adaptation were shifted by 6.4 mm, x*_model = x*_CFD − 0.064.

### Outputs

The outputs are non-dimensional, phase-averaged (URANS) fields. `to_si()` converts them.

| Quantity | Scale |
|---|---|
| u*, v*, w* | U_C = 4.988348 m/s |
| p* | P_C = 13,439.05 Pa |
| ν_t* | ν_t,C = 3.917 × 10⁻⁴ m²/s |
| x*, y*, z* | L_C = 0.1 m |

## Reproducing the paper

`docs/REPRODUCE.md` gives the full pipeline:

1. download the dataset;
2. prepare the training set and the targets;
3. pretrain (about 5 h on one A6000);
4. adapt (about 5 min per case);
5. evaluate.

It also lists what was verified: the preprocessing reproduces the original training data (bit for bit with
`--legacy-pressure`), the targets and the PIV inputs.

## Intended use and limitations

- **Research use only.** The model reconstructs flow in one specific in vitro left-heart simulator (rigid or foil MROPs,
  fixed chamber geometry, 80 bpm). It is not a medical device and must not be used for clinical decisions.
- **Accuracy decays away from the supervised plane.** TTA corrects the flow in the observed plane. On the synthetic cases,
  the gain has halved about 3 mm from the plane. Further out, the volumetric field is close to the pretrained prior, which
  off the plane performs about as well as selecting the closest training case.
- **Snapshot-based.** Each prediction uses only the planar snapshot of the same instant. Diastolic reconstructions are
  weaker than systolic ones.
- **Case-specific weights.** Each `tta_*` checkpoint is calibrated to one case; for a new case, adapt `pretrained`.

## License

- Code: MIT (`LICENSE`).
- Network weights and other files of the release: CC BY 4.0 (`LICENSE-weights`).
- Data: CC BY 4.0, see the dataset record.

## Citation

Please cite the paper and the dataset above (`CITATION.cff`). The code and the weights of release v1.0.0 can be cited as
https://doi.org/10.5281/zenodo.24464090.
