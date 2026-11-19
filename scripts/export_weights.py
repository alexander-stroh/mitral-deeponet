"""Convert a Modulus checkpoint (don.0.pth, state dict) to .safetensors for the standalone package, and check that it loads.
Usage: python scripts/export_weights.py runs/tta_dropxl/don.0.pth runs/tta_dropxl/model.safetensors"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mitral_deeponet import MitralDeepONet      # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
sd = {k: v.detach().clone().contiguous() for k, v in torch.load(src, map_location='cpu', weights_only=True).items()}
MitralDeepONet().load_state_dict(sd, strict=True)
save_file(sd, dst)
print(f'{src} -> {dst}')
