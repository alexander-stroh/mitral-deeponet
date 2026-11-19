# Published checkpoints

All checkpoints are state dicts of the same network:
- branch and trunk MLPs with 9 × 640 hidden layers, adaptive SiLU and weight normalization;
- a 640 × 5 output projection;
- 21.1 M parameters, float32.

They were exported unchanged from the original NVIDIA Modulus Sym 1.5.0 runs (`don.0.pth`) to safetensors. The standalone
implementation reproduces the original Modulus forward pass to 1.1 × 10⁻¹⁵ in float64 and 8.4 × 10⁻⁷ in float32 for every
checkpoint (`tests/test_reference.py`).

| File | Original run | Final step | Target | Adaptation data | SHA-256 (file) |
|---|---|---|---|---|---|
| `pretrained.safetensors` | `don_larger2` | 50,000 | – | – | `4f77647c…` |
| `tta_dropxl_cfd.safetensors` | `don_larger_hybrid_dd2` | 50,400 | Drop-XL | CFD plane z* = 0 (u, v), CFD pressure at the LV and LA probes | `3ded484c…` |
| `tta_slotlbent_cfd.safetensors` | `don_larger_hybrid_dd2_slotlphasebent` | 50,400 | Slot-L-Bent | as above, window shifted by +6.4 mm in x | `1df65bba…` |
| `tta_pinholel_piv.safetensors` | `don_larger_hybrid_dd2_pinholelphase_active_withp` | 50,400 | Pinhole-L | PIV z = 0 (u, v), measured LV and LA pressures | `a09ce111…` |
| `tta_slotl_piv.safetensors` | `don_larger_hybrid_dd2_slotlphase` | 50,400 | Slot-L | as above | `5645eda0…` |
| `tta_dropxl_piv.safetensors` | `don_larger_hybrid_dd2_dropxlphase` | 50,400 | Drop-XL | as above | `83ed68f1…` |
| `tta_eccjet_piv.safetensors` | `don_larger_hybrid_dd2_eccjetphase` | 50,400 | EccJet | as above | `6be2c524…` |
| `pretrained_optimizer.pt` | `don_larger2/optim_checkpoint.0.pth` | 50,000 | – | optimizer, scheduler and NTK aggregator state | `ca24b0a6…` |
| `piv_branch_inputs.npz` | – | – | – | branch inputs of the four in vitro cases, as used in the paper | `e5e94406…` |

The full checksums are in `mitral_deeponet/weights.py` and are checked on download.

## Training settings

These are the settings of the original runs; `scripts/train.py` uses the same ones.

**Pretraining**
- 50,000 steps on all nodes of all 2,706 snapshots of the nine training cases (4.28 × 10⁹ points).
- Batches of 2,500 points with the branch input of their snapshot, gradients aggregated over 4 batches.
- AdamW (learning rate 1 × 10⁻³, weight decay 0.01), learning rate × 0.95 every 4,000 steps.
- Sum of squared errors on u*, v*, w*, p*, ν_t*, with NTK-based loss weights updated every 100 steps.
- Hardware: one RTX 6000 Ada, 0.375 s per step without validation.

**Adaptation (TTA)**
- 400 steps, continuing the optimizer state of the pretraining (learning rate ≈ 5.3 × 10⁻⁴).
- Constraints:
  - training set (2,500 points per batch, rehearsal);
  - target plane (2,500 points per batch);
  - two pressure probes (100 samples per batch).
- NTK weighting of all terms.

**Configuration files**
- `conf/config.yaml` is the original `config_larger_hybrid.yaml`.
- The pretraining configuration differed only in its checkpoint and validation frequencies.
- The Slot-L-Bent adaptation used the same settings with less frequent validation.

## Known properties

The checkpoints were evaluated in the paper and in its review; see `docs/REPRODUCE.md` for the numbers.

- **Drop-XL, z-t metric (CFD adaptation).**
  - In-plane NRMSE of |u| falls from 46 % (baseline) to 6 %.
  - Off-plane (|z| > 3 mm) it falls from 62 % to 52 %.
  - Regurgitant volume in the integration window: 41 → 49 ml (CFD 84 ml).
- **Five TTA seeds** with otherwise identical settings: in-plane 6.3 ± 1.6 %, off-plane 49.8 ± 1.1 %, RVol 56 ± 2.4 ml.
  `tta_dropxl_cfd` is one run of this distribution.
