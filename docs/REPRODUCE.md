# Reproducing the paper

The full pipeline goes from the published data to adapted networks and their errors. All commands run from the repository
root.

## 1. Environment

Preprocessing and evaluation need Python with `pip install -e .[data]` (SciPy, VTK, netCDF4). Training needs NVIDIA Modulus
Sym 1.5.0; the original runs used the NVIDIA container `nvcr.io/nvidia/modulus/modulus:24.04` (PyTorch 2.3, Modulus Sym 1.5.0)
with VTK 9.3.0 added:

```bash
apptainer build modulus.sif docker://nvcr.io/nvidia/modulus/modulus:24.04
apptainer exec --nv modulus.sif pip install --user vtk==9.3.0 safetensors==0.4.5
alias run='apptainer exec --nv modulus.sif'
```

With Docker, use the same image and `pip install vtk==9.3.0 safetensors==0.4.5` inside it. Every command below works in this
container. With VTK 9.3.0 and `--legacy-pressure` (Section 3) the preprocessing reproduces the original training data bit
for bit; other VTK versions may differ in the points that clipping adds.

## 2. Data

```bash
scripts/download_data.sh data                     # all 11 CFD cases + PIV, about 200 GB; or e.g. ... data Drop-XL PIV
```

RADAR4KIT delivers each record as a BagIt container. The script extracts the archive listed in the dataset's `SHA256SUMS`,
verifies it and unpacks it to `data/<Case>/<Case>.case` and `data/PIV/`. Refused requests (rate limit) are retried.

## 3. Training set

```bash
run python scripts/prepare_training_data.py --data data --out cache/train_full --workers 32                 # 4.28e9 points, ~155 GB
run python scripts/prepare_training_data.py --data data --out cache/train_25pct --fraction 0.25 --workers 32  # ~39 GB
run python scripts/prepare_training_data.py --data data --out cache/train_5pct --fraction 0.05 --workers 32   # ~8 GB, rehearsal
```

For every snapshot of the nine training cases, the script does the following:
- rotates the mesh into the jet-aligned model frame and clips it to the left atrium and the adjacent ventricle;
- non-dimensionalizes the coordinates and all fields;
- cuts the plane z* = 0 and interpolates the branch input.

Each snapshot takes about 45 s on one core. With 32 workers the full set takes about 1 h.

**Verified.** The 5 % subset built from the published data was compared with the original training arrays for all nine
cases (every fifth snapshot, plus the first and last two of each case). Coordinates, u*, v*, w*, ν_t*, t* and the branch
inputs are identical, with 0.0 difference. The pressure is identical except in the first snapshot of Drop-S, Pinhole-S and
Pinhole-L (3 of 2,706 snapshots).

**Pressure of the first and last snapshot.** The original pipeline replaced the pressure of the first snapshot of every case
by that of the second, and the pressure of the last by that of the second-to-last. In most cases these snapshots have no
pressure file, and the script does the same. In Drop-S, Pinhole-S and Pinhole-L the published first snapshot has a valid
pressure, which the script keeps (mean difference 0.02 in p*). `--legacy-pressure` applies the original rule and then
reproduces the original arrays exactly, also for these three snapshots.

## 4. Pretraining

```bash
run python scripts/train.py --mode pretrain --data cache/train_full --out runs/pretrain              # 50,000 steps
run python scripts/train.py --mode pretrain --data cache/train_full --out runs/pretrain_noslotm --exclude Slot-M
```

`--load-ram 1` (default) holds the training set in RAM: about 155 GB for the full set, which the original run did too. Use
`--fraction 0.25` in step 3 for a 39 GB set.

- **Speed.** 0.30–0.41 s per step on one RTX A6000, i.e. about 5 h for 50,000 steps.
- **25 % subset.** Two pretrainings on the 25 % subset (seeds 1 and 2) and the published network gave, after adaptation
  to Drop-XL:
  - in-plane NRMSE 5.4–6.0 %;
  - off-plane 45–52 %;
  - RVol 50–56 ml.

  The baseline before adaptation varies much more between pretrainings.

## 5. Targets

```bash
run python scripts/prepare_target.py --cfd data --case Drop-XL --out targets/Drop-XL.npz --xz          # synthetic validation
run python scripts/prepare_target.py --cfd data --case Slot-L-Bent --out targets/Slot-L-Bent.npz --xz
run python scripts/prepare_target.py --piv data/PIV --case EccJet --out targets/EccJet_piv.npz         # in vitro
```

- **CFD targets** contain the branch inputs, the plane observations at z* = 0 and the CFD pressure at the LV and LA probe
  nodes. With `--xz` they also contain an x-z plane at y* = 0 (the second, orthogonal plane).
- **PIV targets** contain the branch inputs and the plane observations.

**Verified.**
- **PIV.** The branch inputs of all four campaigns are identical to those of the paper (0.0 difference in the paper's
  container, all phases), including the selection of phases and the time synchronisation (`mitral_deeponet/data/piv.py`).
- **Drop-XL and Slot-L-Bent (CFD).** Snapshot times, t*, branch inputs, probe nodes and probe pressures are identical to the
  data of the paper's runs. One exception: for Drop-XL the original preparation replaced the pressure of the first snapshot
  by that of the second, although the published first snapshot is valid. The code keeps the valid value, so 2 of 602
  pressure samples differ; `--legacy-pressure` reproduces the original values. For Slot-L-Bent the option changes nothing.
- **x-z plane.** The paper's runs used only the x-y plane. The x-z plane (`--xz`) belongs to the two-plane runs of the review
  and is identical to the one used there.

**Pressures of the in vitro cases.** The paper's adaptations to Pinhole-L, Slot-L and Drop-XL also used the measured LV and
LA pressures; the adaptation to EccJet used none. The pressures were synchronised to the cycle by hand and offset; the
published pressure logs do not contain that processing, so PIV targets are prepared without pressure. In an ablation the
pressure signals changed the RVol by at most 1.5 ml and the errors by at most 3 points.

## 6. Adaptation (TTA)

```bash
run python scripts/train.py --mode adapt --data cache/train_5pct --target targets/Drop-XL.npz \
    --init weights/pretrained.safetensors --init-optimizer weights/pretrained_optimizer.pt --out runs/tta_dropxl
```

The defaults are the paper's settings: 400 iterations, x-y plane, rehearsal and pressure on, all weights updated. Two
points differ from the paper's runs. PIV targets carry no pressures (Section 5), so their pressure constraint is skipped.
The training-set batches are drawn with replacement, whereas the original drew a random permutation of all points; for
400 × 4 × 2,500 draws the two are statistically equivalent. Variants used in the review:
- `--planes xyxz` (second plane);
- `--rehearsal 0` and `--pressure 0` (ablations);
- `--train trunk|branch|last`;
- `--steps 1600 --save-at 50,100,200,400,800,1600` (error against iteration);
- `--seed`.

An adaptation takes 4–6 min on one A6000 (0.55–0.75 s per iteration, depending on GPU load).

**Rehearsal subset.** The original adaptation drew the rehearsal batches from the full training set held in RAM. The 5 %
subset is statistically equivalent for 400 × 4 × 2,500 draws and reproduces the published Drop-XL adaptation within the seed
spread (Section 8).

## 7. Evaluation

```bash
python scripts/evaluate.py --cfd data --case Drop-XL --model baseline=weights/pretrained.safetensors \
    --model tta=runs/tta_dropxl/model.safetensors --out eval/Drop-XL.json --workers 16
```

The script reports the paper's metrics:
- in-plane and off-plane NRMSE on the z-extruded branch window, plus the per-plane profile against |z|;
- aggregate NRMSE of u, v, w on a 5 % node sample;
- the regurgitant volume in the integration window, with CFD in the same window.

Two details differ from the evaluation behind the paper's tables:
- **Regurgitant volume.** `rvol` uses the quadrature of the original evaluation, which gives the centre line of the window
  half its weight. `rvol_trapezoid` uses the full weight; these are the values of the paper (3–5 % higher). The paper's CFD
  reference was integrated on the cut plane itself and is 0.4 ml higher for Drop-XL than the value interpolated here.
- **Aggregate NRMSE.** The 5 % sample is drawn from the nodes of the clipped mesh. The paper's values come from a sample of
  the raw mesh nodes around the left atrium and are 0.4–0.9 points higher.

## 8. Verification of this repository

| Check | Result |
|---|---|
| Standalone model vs original Modulus model, 7 checkpoints | ≤ 1.1 × 10⁻¹⁵ (float64), ≤ 8.4 × 10⁻⁷ (float32) |
| Training data from the published data vs original training arrays, all nine cases (5 % subset) | identical (0.0), except the pressure of 3 first snapshots; identical throughout with `--legacy-pressure` |
| Drop-XL and Slot-L-Bent targets vs the data of the paper's runs | identical (0.0); Drop-XL first-snapshot pressure only with `--legacy-pressure` |
| PIV branch inputs from the published PIV vs those of the paper | identical (0.0), all four campaigns, all phases |
| Published checkpoints on the published Drop-XL (`scripts/evaluate.py`), baseline / TTA | in-plane 46.3 / 6.0 %, off-plane 62.4 / 51.6 %, RVol 42.7 / 51.6 ml (CFD 85.9 ml), aggregate u 54.3 / 47.0 %. Paper: 46.3 / 6.0, 62.4 / 51.6, 42.7 / 51.6 (CFD 86.3), 54.8 / 47.5 |
| Published checkpoints on the published Slot-L-Bent, baseline / TTA | in-plane 41.3 / 6.4 %, off-plane 74.5 / 76.2 %, RVol 53.2 / 61.1 ml (CFD 61.0 ml), aggregate u 47.9 / 43.3 %. Paper: 41.3 / 6.4, 74.5 / 76.2, 53.2 / 61.1 (CFD 60.9), 48.5 / 43.8 |
| Three new adaptations to Drop-XL (seeds 0–2) from the published data, weights and optimizer state, 5 % rehearsal set | in-plane 5.4–6.2 %, off-plane 49.6–53.8 %, aggregate u 44.6–46.6 %, RVol 57.5–60.1 ml. Published run with the same script: 6.0 %, 51.6 %, 47.0 %, 51.6 ml |
| Five adaptations of the review (seeds 0–4, 5 % rehearsal subset of the original arrays, evaluation of the review) | in-plane 4.9–8.9 %, off-plane 48.6–51.2 %, aggregate u 44.4–47.1 %, RVol 55–61 ml |
| Pretraining from scratch (200-step smoke test, Slot-M held out) | runs, 0.39 s per step |

## 9. Notes on the dataset

Three points of the dataset README differ from the procedure that produced the paper's data. This code implements the
procedure that was actually used:

1. **Coordinate frame of the mesh.** The velocity components are given in the jet-aligned frame. The mesh coordinates of the
   `.geo` files must be rotated by −30° about z to reach it (`CFDCase._transform`).
2. **Drop-XL PIV.** The paper's inputs were transformed with θ = −29°, not −30°.
3. **Download format.** RADAR delivers each record as a BagIt container, which `scripts/download_data.sh` handles.
