"""Published checkpoints: names, checksums and download.

The weight files are attached to the GitHub release of this repository (too large for git). Set BASE_URL (or the environment
variable MITRAL_DEEPONET_WEIGHTS_URL) to another location if they are mirrored elsewhere."""
import hashlib
import os
import urllib.request

BASE_URL = os.environ.get('MITRAL_DEEPONET_WEIGHTS_URL', 'https://github.com/OWNER/mitral-deeponet/releases/download/v1.0.0')

# name: (sha256, description). Provenance in docs/MODELS.md.
CHECKPOINTS = {
    'pretrained.safetensors': ('4f77647cbe57a25ba94ad0cf5dc172be1bd848b24773e317ed590c3e229067fc', 'pretrained on the 9 training MROPs, 50,000 steps (baseline)'),
    'pretrained_optimizer.pt': ('ca24b0a6e7c5a73cc4c8f3f3518029fd6650add83029745aff52c8f84cec14d9', 'step, AdamW moments and learning-rate schedule at the end of pretraining (TTA restarts as in the paper)'),
    'tta_dropxl_cfd.safetensors': ('3ded484c1022a0ca9a66599be8dd15062d71024570ea4cfa720105cd92217eb8', 'adapted to Drop-XL, CFD plane z* = 0 + virtual pressure probes'),
    'tta_slotlbent_cfd.safetensors': ('1df65bbaaddd0d942fb9d38b69d1e59cd3118632313a92e3d88cdc358986fd67', 'adapted to Slot-L-Bent, CFD plane z* = 0 + virtual pressure probes'),
    'tta_pinholel_piv.safetensors': ('a09ce111eaf4ecaf73856874e5b15778766999e9aa7026600ff8ffc1c5a31339', 'adapted to Pinhole-L, phase-resolved PIV + measured pressures'),
    'tta_slotl_piv.safetensors': ('5645eda08f0cf11c531642dc9ff1f523a064dfb386ff6933ad740bed9d6131f8', 'adapted to Slot-L, phase-resolved PIV + measured pressures'),
    'tta_dropxl_piv.safetensors': ('83ed68f1db55dc4e5e3f68d0adb0940caffa5b994b44bfa9a5b304f196ff626e', 'adapted to Drop-XL, phase-resolved PIV + measured pressures'),
    'tta_eccjet_piv.safetensors': ('6be2c5248bc6d98c67a98ab308bd6b4e7bb81280e66494c6c13fb6d0e62ff914', 'adapted to EccJet, phase-resolved PIV + measured pressures'),
    'piv_branch_inputs.npz': ('e5e944065b5aa1d7f4ca2aea3b397db224435985e627e43b1ee07e12c6309ac7', 'branch inputs of the four in vitro cases (PIV on the branch grid)'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, dest='weights', base_url=None):
    """Download one file of the release into dest/ (skipped if present with the right checksum); returns its path."""
    if name not in CHECKPOINTS:
        raise KeyError(f'unknown file {name}; available: {", ".join(CHECKPOINTS)}')
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, name); ref = CHECKPOINTS[name][0]
    if not (os.path.exists(path) and (not ref or sha256(path) == ref)):
        url = f'{(base_url or BASE_URL).rstrip("/")}/{name}'
        print(f'downloading {url}')
        urllib.request.urlretrieve(url, path + '.part'); os.replace(path + '.part', path)
    if ref and sha256(path) != ref:
        raise IOError(f'checksum mismatch for {path}')
    return path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Download the published checkpoints.')
    ap.add_argument('names', nargs='*', default=list(CHECKPOINTS)); ap.add_argument('--dest', default='weights')
    ap.add_argument('--base-url')
    a = ap.parse_args()
    for n in a.names:
        print(fetch(n, a.dest, a.base_url))
