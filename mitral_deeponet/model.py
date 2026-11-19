"""Standalone PyTorch implementation of the DeepONet used in Hoffmann, Hasegawa & Stroh (2026).

The network was trained with NVIDIA Modulus Sym 1.5.0 (DeepONetArch with two FullyConnectedArch sub-networks). This module
re-implements the same forward pass with plain PyTorch, so that the published weights can be used without Modulus. Parameter
names are identical to the Modulus state dict, so a checkpoint loads with ``strict=True`` into either implementation.

Architecture (per sub-network): 9 hidden layers of width 640 with weight normalization and SiLU activation, where the
pre-activation is scaled by a single trainable factor shared by all hidden layers of that sub-network ("adaptive
activation"), followed by a linear output layer of width 640 without activation. The branch output b and trunk output
tau are combined as y = W (b * tau) with W of shape 5 x 640 and no bias. Outputs: u, v, w, p, nu_t (non-dimensional).
"""
import torch
from torch import nn
import torch.nn.functional as F

N_BRANCH = 10738            # masked points of the 119 x 102 branch-input grid (see grid.py)
WIDTH, DEPTH = 640, 9
OUTPUTS = ('u', 'v', 'w', 'p', 'nu_t')


class WeightNormLinear(nn.Module):
    """Linear layer with weight normalization, weight = g * v / ||v|| (row-wise), as in Modulus."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.weight_g = nn.Parameter(torch.ones(n_out, 1))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(x, self.weight_g * self.weight / self.weight.norm(dim=1, p=2, keepdim=True), self.bias)


class FCLayer(nn.Module):
    def __init__(self, n_in, n_out, activation_par=None, weight_norm=True):
        super().__init__()
        self.linear = WeightNormLinear(n_in, n_out) if weight_norm else nn.Linear(n_in, n_out)
        self.activation_par = activation_par          # shared nn.Parameter (adaptive activation) or None (no activation)

    def forward(self, x):
        x = self.linear(x)
        return x if self.activation_par is None else F.silu(self.activation_par * x)


class _Core(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        a = nn.Parameter(torch.ones(1))                # one adaptive-activation factor shared by all hidden layers
        self.layers = nn.ModuleList([FCLayer(n_in if i == 0 else WIDTH, WIDTH, a) for i in range(DEPTH)])
        self.final_layer = FCLayer(WIDTH, WIDTH, None, weight_norm=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class _Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self._impl = _Core(n_in)

    def forward(self, x):
        return self._impl(x)


class MitralDeepONet(nn.Module):
    """Branch input: masked planar velocity snapshot (u*, v*) on the 10738-point grid of grid.branch_grid(), concatenated as
    [u*_1..u*_10738, v*_1..v*_10738]. Trunk input: (x*, y*, z*, t*) in the model frame. Output: (u*, v*, w*, p*, nu_t*)."""

    def __init__(self):
        super().__init__()
        self.branch_net = _Net(2 * N_BRANCH)
        self.trunk_net = _Net(4)
        self.output_linear = nn.Linear(WIDTH, len(OUTPUTS), bias=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # In the saved state dict every hidden layer carries a copy of the shared adaptive-activation factor; the copies are
        # identical. Check this so that a manipulated checkpoint cannot silently load with a different value per layer.
        for net in ('branch_net', 'trunk_net'):
            vals = [state_dict[f'{net}._impl.layers.{i}.activation_par'] for i in range(DEPTH)]
            if any(not torch.equal(v, vals[0]) for v in vals):
                raise ValueError(f'{net}: adaptive-activation factors differ between layers')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def encode_branch(self, u_plane, v_plane):
        """u_plane, v_plane: (B, 10738) non-dimensional in-plane velocities on the branch grid -> (B, 640)."""
        return self.branch_net(torch.cat([u_plane, v_plane], dim=-1))

    def decode(self, branch_code, xyzt):
        """branch_code: (640,) or (1, 640); xyzt: (N, 4) -> (N, 5) non-dimensional (u, v, w, p, nu_t)."""
        return self.output_linear(branch_code.reshape(1, -1) * self.trunk_net(xyzt))

    def forward(self, u_plane, v_plane, xyzt):
        """Pointwise form, one snapshot per query point: u_plane, v_plane (N, 10738), xyzt (N, 4) -> (N, 5)."""
        return self.output_linear(self.encode_branch(u_plane, v_plane) * self.trunk_net(xyzt))

    @torch.no_grad()
    def predict(self, u_plane, v_plane, xyz, t, batch_size=131072):
        """Field of one snapshot at many points. u_plane, v_plane: (10738,); xyz: (N, 3) model frame; t: scalar t*.
        Returns an (N, 5) tensor of non-dimensional (u, v, w, p, nu_t)."""
        p = next(self.parameters())
        as_t = lambda a: torch.as_tensor(a, dtype=p.dtype, device=p.device)
        b = self.encode_branch(as_t(u_plane)[None], as_t(v_plane)[None])
        xyz = as_t(xyz); out = []
        for s in range(0, len(xyz), batch_size):
            x = xyz[s:s + batch_size]
            out.append(self.decode(b, torch.cat([x, torch.full_like(x[:, :1], float(t))], dim=1)))
        return torch.cat(out)


def load(path, device='cpu'):
    """Load a published checkpoint (.safetensors or .pt state dict) into a MitralDeepONet in eval mode."""
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(str(path), device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    m = MitralDeepONet()
    m.load_state_dict(sd)
    return m.to(device).eval()
