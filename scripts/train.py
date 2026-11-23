"""Pretraining and test-time adaptation (TTA) of the DeepONet with NVIDIA Modulus Sym 1.5.0.

Pretraining (--mode pretrain): data-driven training on the prepared training set (u*, v*, w*, p*, nu_t* at all mesh nodes of
all snapshots of the training cases; branch input = the planar slice of the same snapshot). The paper's network was trained
for 50,000 steps.

TTA (--mode adapt): starts from pretrained weights and continues training for --steps iterations (paper: 400) on
  supervision       the training set (rehearsal), u v w p nu_t            [--rehearsal 1]
  supervision_pinn  the target's x-y plane at z* = 0, u v                  [--planes xy | xyxz]
  supervision_xz    the target's x-z plane at y* = 0, u w (if prepared)    [--planes xz | xyxz]
  pressure_nodes    the target's LV and LA pressures, p (CFD targets)      [--pressure 1]
with the optimizer, learning-rate schedule and NTK loss weighting of the pretraining. Constraint names are those of the
original runs. --train restricts the update to the branch net, the trunk net or the last layers (ablation).

Settings (conf/config.yaml): 9 x 640 MLPs with adaptive SiLU, AdamW (lr 1e-3, weight decay 0.01), lr x 0.95 every 4,000 steps,
NTK weights every 100 steps, sum loss, gradient aggregation over 4 batches of 2,500 points per constraint.
The training points are drawn uniformly with replacement from the prepared set (see docs/REPRODUCE.md for the subset used).

Usage (inside the Modulus container, see README):
  python scripts/train.py --mode pretrain --data cache/train --out runs/pretrain [--steps 50000] [--exclude Slot-M]
  python scripts/train.py --mode adapt --data cache/train_5pct --target targets/Drop-XL.npz \
         --init weights/pretrained.safetensors --init-optimizer weights/pretrained_optimizer.pt --out runs/tta_dropxl"""
import argparse
import json
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--mode', required=True, choices=['pretrain', 'adapt'])
ap.add_argument('--data', required=True, help='prepared training set (scripts/prepare_training_data.py)')
ap.add_argument('--target', help='target file (scripts/prepare_target.py), adapt mode')
ap.add_argument('--out', required=True)
ap.add_argument('--init', help='initial weights: .safetensors or Modulus don.0.pth (adapt mode: required)')
ap.add_argument('--init-optimizer', help='optimizer checkpoint of the initial weights (optim_checkpoint.0.pth); restores the step, '
                'learning-rate schedule and Adam moments of the pretraining (paper setting). The NTK weights are not stored; with '
                'the restored step 50000 they are computed in the first iteration. Without it, AdamW starts fresh at --lr from '
                'step 0, where Modulus computes the first NTK weights only at step 100 and gives steps 1-99 zero loss weight.')
ap.add_argument('--steps', type=int, default=None, help='iterations (default: 50000 pretrain, 400 adapt)')
ap.add_argument('--lr', type=float, default=None, help='initial learning rate if no optimizer checkpoint is given '
                '(default 1e-3 for pretraining; 1e-3 * 0.95**(50000/4000) for adaptation, i.e. the rate at the end of pretraining)')
ap.add_argument('--planes', default='xy', choices=['xy', 'xyxz', 'xz', 'none'])
ap.add_argument('--rehearsal', type=int, default=1)
ap.add_argument('--pressure', type=int, default=1)
ap.add_argument('--train', default='all', choices=['all', 'branch', 'trunk', 'last'])
ap.add_argument('--exclude', default='', help='comma-separated training cases left out of the training set (held-out test)')
ap.add_argument('--load-ram', type=int, default=1, help='load the training set into RAM (recommended unless on a fast SSD)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save-every', type=int, default=0, help='checkpoint frequency (default: pretrain 5000, adapt end only)')
ap.add_argument('--save-at', default='', help='comma-separated iterations at which to keep a copy of the weights')
A = ap.parse_args()
steps = A.steps or (50000 if A.mode == 'pretrain' else 400)
if A.mode == 'adapt' and not (A.init and A.target):
    ap.error('adapt mode needs --init and --target')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s', stream=sys.stdout, force=True)
log = logging.getLogger('train')
T0 = time.time()
random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
OUT = os.path.abspath(A.out)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
json.dump(vars(A), open(f'{OUT}/args.json', 'w'), indent=1)

# ---------------- initial weights / optimizer state into the Modulus network directory
START = 0
if A.init:
    if A.init.endswith('.safetensors'):
        from safetensors.torch import load_file
        torch.save(load_file(A.init), f'{OUT}/don.0.pth')
    else:
        shutil.copy(A.init, f'{OUT}/don.0.pth')
if A.init_optimizer:
    shutil.copy(A.init_optimizer, f'{OUT}/optim_checkpoint.0.pth')
    START = int(torch.load(f'{OUT}/optim_checkpoint.0.pth', map_location='cpu')['step'])

from modulus.sym.hydra import instantiate_arch, to_yaml                     # noqa: E402
from modulus.sym.hydra.utils import compose                                 # noqa: E402
from modulus.sym.key import Key                                             # noqa: E402
from modulus.sym.models.deeponet import DeepONetArch                        # noqa: E402
from modulus.sym.domain import Domain                                       # noqa: E402
from modulus.sym.domain.constraint.continuous import DeepONetConstraint     # noqa: E402
from modulus.sym.dataset import Dataset                                     # noqa: E402
from modulus.sym.loss import PointwiseLossNorm                              # noqa: E402
from modulus.sym.solver import Solver                                       # noqa: E402
import modulus.sym.domain.constraint.constraint as mcon                     # noqa: E402
import modulus.sym.trainer as mtrain                                        # noqa: E402
from mitral_deeponet.data import TrainingSet, load_target                   # noqa: E402

cfg = compose(config_path='../conf', config_name='config')
cfg.training.max_steps = START + steps
cfg.training.ntk.save_name = f'{OUT}/ntk'
cfg.network_dir = OUT
cfg.training.save_network_freq = A.save_every or (5000 if A.mode == 'pretrain' else (50 if A.save_at else steps))
for k in ('rec_validation_freq', 'rec_inference_freq', 'rec_monitor_freq', 'rec_constraint_freq'):
    cfg.training[k] = 10 ** 9                                     # no validators / inferencers in this script
if not A.init_optimizer:
    cfg.optimizer.lr = A.lr or (1e-3 if A.mode == 'pretrain' else 1e-3 * 0.95 ** (50000 / 4000))
log.info('\n' + to_yaml(cfg))
SAVE_AT = [int(s) for s in A.save_at.split(',') if s]


# ---------------- uniform sampling with replacement for the (huge) training set
class ReplacementSampler(torch.utils.data.Sampler):
    def __init__(self, ds):
        self.n = len(ds)

    def __len__(self):
        return self.n

    def __iter__(self):
        while True:
            yield from torch.randint(high=self.n, size=(65536,), dtype=torch.int64).tolist()


_RS = mcon.RandomSampler
mcon.RandomSampler = lambda ds: ReplacementSampler(ds) if len(ds) > 10 ** 7 else _RS(ds)

# ---------------- network
trunk_net = instantiate_arch(cfg=cfg.arch.trunk, input_keys=[Key('x'), Key('y'), Key('z'), Key('t')],
                             output_keys=[Key('trunk', cfg.arch.trunk.layer_size)])
branch_net = instantiate_arch(cfg=cfg.arch.branch, input_keys=[Key('u_piv', 10738), Key('v_piv', 10738)],
                              output_keys=[Key('branch', cfg.arch.branch.layer_size)])
deeponet = DeepONetArch(output_keys=[Key('u'), Key('v'), Key('w'), Key('p'), Key('nu_t')], branch_net=branch_net, trunk_net=trunk_net)
nodes = [deeponet.make_node(name='don')]
update = {'all': lambda n: True, 'branch': lambda n: n.startswith('branch_net'), 'trunk': lambda n: n.startswith('trunk_net'),
          'last': lambda n: n.startswith('output_linear') or '.final_layer.' in n}[A.train]
FROZEN = [p for n, p in deeponet.named_parameters() if not update(n)]
log.info(f'updated parameters: {sum(p.numel() for n, p in deeponet.named_parameters() if update(n))} ({A.train})')
# frozen parameters keep requires_grad (identical NTK weights); their gradients are dropped before the optimizer step
_apply = mtrain.AdamMixin.adam_apply_gradients


def apply_gradients(self):
    for p in FROZEN:
        p.grad = None
    _apply(self)


mtrain.AdamMixin.adam_apply_gradients = apply_gradients
_save = Solver.save_checkpoint


def save_checkpoint(self, step):
    _save(self, step)
    if step - START in SAVE_AT:
        shutil.copy(f'{OUT}/don.0.pth', f'{OUT}/don.it{step - START}.pth')


Solver.save_checkpoint = save_checkpoint


# ---------------- datasets
class TrainingDataset(Dataset):
    """Points of the prepared training set with the branch input of their snapshot."""
    auto_collation = True

    def __init__(self, path, exclude):
        ts = TrainingSet(path, load=bool(A.load_ram))
        self.bu, self.bv = torch.from_numpy(np.array(ts.branch_u)), torch.from_numpy(np.array(ts.branch_v))
        self.t = torch.from_numpy(np.array(ts.branch_t))[:, None]
        self.snap, self.f = ts.snap, ts.points
        self.cuts = sorted(ts.case_rows(c) for c in exclude if c)      # excluded row ranges
        self.n = len(ts) - sum(b - a for a, b in self.cuts)
        for c in exclude:
            if c:
                log.info(f'excluding {c}: rows {ts.case_rows(c)}')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = np.sort(np.asarray(idx, dtype='int64'))
        for a, b in self.cuts:                                         # map indices past the excluded ranges
            idx = idx + (idx >= a) * (b - a)
        s = torch.from_numpy(np.asarray(self.snap[idx]).astype('int64'))
        F = {k: torch.from_numpy(np.asarray(v[idx], dtype='float32'))[:, None] for k, v in self.f.items()}
        invar = {'u_piv': self.bu[s], 'v_piv': self.bv[s], 't': self.t[s], 'x': F['x'], 'y': F['y'], 'z': F['z']}
        outvar = {k: F[k] for k in ('u', 'v', 'w', 'p', 'nu_t')}
        return invar, outvar, {k: torch.ones_like(v) for k, v in outvar.items()}

    @property
    def invar_keys(self):
        return ['u_piv', 'v_piv', 'x', 'y', 'z', 't']

    @property
    def outvar_keys(self):
        return ['u', 'v', 'w', 'p', 'nu_t']


class PlaneDataset(Dataset):
    """Planar observations of the target: every snapshot at the same points."""
    auto_collation = True

    def __init__(self, bu, bv, t, X, out):
        self.bu, self.bv = torch.from_numpy(bu).float(), torch.from_numpy(bv).float()
        self.t = torch.from_numpy(t.reshape(-1, 1)).float(); self.X = torch.from_numpy(X).float()
        self.out = {k: torch.from_numpy(v).float() for k, v in out.items()}          # (n_snapshots, n_points)
        self.ns, self.np_ = self.out[next(iter(out))].shape

    def __len__(self):
        return self.ns * self.np_

    def __getitem__(self, idx):
        idx = torch.as_tensor(idx); s, k = idx // self.np_, idx % self.np_
        invar = {'u_piv': self.bu[s], 'v_piv': self.bv[s], 'x': self.X[k, :1], 'y': self.X[k, 1:2], 'z': self.X[k, 2:3], 't': self.t[s]}
        outvar = {c: v[s, k][:, None] for c, v in self.out.items()}
        return invar, outvar, {c: torch.ones_like(v) for c, v in outvar.items()}

    @property
    def invar_keys(self):
        return ['u_piv', 'v_piv', 'x', 'y', 'z', 't']

    @property
    def outvar_keys(self):
        return list(self.out)


dom = Domain()
if A.mode == 'pretrain' or A.rehearsal:
    ds = TrainingDataset(A.data, A.exclude.split(','))
    dom.add_constraint(DeepONetConstraint(nodes=nodes, dataset=ds, loss=PointwiseLossNorm(), batch_size=2500, shuffle=True,
                                          drop_last=True, num_workers=4), 'supervision')
    log.info(f'training set {A.data}: {len(ds)} points')
if A.mode == 'adapt':
    P = load_target(A.target)
    bu, bv, ts = P['bu'], P['bv'], P['tstar'].astype('float32')

    def plane(X, out, name):
        ds = PlaneDataset(bu, bv, ts, X.astype('float32'), out)
        dom.add_constraint(DeepONetConstraint(nodes=nodes, dataset=ds, loss=PointwiseLossNorm(), batch_size=2500, shuffle=True,
                                              drop_last=True, num_workers=4), name)
        log.info(f'{name}: {len(ds)} points ({ds.np_} per snapshot), outputs {list(out)}')
    if A.planes in ('xy', 'xyxz'):
        plane(np.c_[P['wx'], P['wy'], P['wz']], {'u': bu, 'v': bv}, 'supervision_pinn')
    if A.planes in ('xz', 'xyxz'):
        if 'Uxz' not in P:
            ap.error('the target has no x-z plane (prepare it with --xz)')
        plane(P['xz'], {'u': P['Uxz'][:, :, 0], 'w': P['Uxz'][:, :, 2]}, 'supervision_xz')
    if A.pressure and 'p' in P:
        n, px = len(ts), P['pxyz']
        inv = {'x': np.tile(px[:, :1], (n, 1)), 'y': np.tile(px[:, 1:2], (n, 1)), 'z': np.tile(px[:, 2:3], (n, 1)),
               't': np.repeat(ts, 2)[:, None], 'u_piv': np.repeat(bu, 2, axis=0), 'v_piv': np.repeat(bv, 2, axis=0)}
        inv = {k: v.astype('float32') for k, v in inv.items()}
        dom.add_constraint(DeepONetConstraint.from_numpy(nodes=nodes, invar=inv, outvar={'p': P['p'].reshape(-1, 1).astype('float32')},
                                                         batch_size=100), 'pressure_nodes')
        log.info(f'pressure_nodes: {len(inv["x"])} samples at {px.tolist()}')
    elif A.pressure:
        log.info('target has no pressure data; pressure constraint skipped')

t1 = time.time()
slv = Solver(cfg, dom)
slv.solve()
t2 = time.time()
json.dump({'setup_s': t1 - T0, 'solve_s': t2 - t1, 'steps': steps, 'start_step': START, 'host': os.uname().nodename,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}, open(f'{OUT}/timing.json', 'w'), indent=1)
try:                                                    # weights for the standalone package
    from safetensors.torch import save_file
    sd = torch.load(f'{OUT}/don.0.pth', map_location='cpu')
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, f'{OUT}/model.safetensors')
except ImportError:
    log.info('safetensors not installed: weights only as don.0.pth (convert with scripts/export_weights.py)')
log.info(f'TRAIN_DONE {OUT}: solve {t2 - t1:.0f} s')
os._exit(0)
