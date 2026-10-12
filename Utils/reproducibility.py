"""Content identity and RNG helpers for the fixed MotionVQ/controller pipeline."""
from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

IMPLEMENTATION_VERSION = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def artifact_fingerprints(db, vqvae_ckpt, norm_stats):
    db = Path(db)
    # Include frame contents: file names/shape alone cannot detect an overwritten DB.
    files = ('meta.json', 'segments.json', 'StyleNames.json', 'Labels.i32', 'X.f32')
    parts = {name: file_sha256(db / name) for name in files if (db / name).exists()}
    from dataset import norm_stats_fingerprint
    from Utils.runtime import load_norm_stats
    from Utils.model_loading import build_vqvae_from_checkpoint
    mean, std = load_norm_stats(norm_stats)
    model = build_vqvae_from_checkpoint(Path(vqvae_ckpt))
    return {'db_sha256': hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
            'vq_state_sha256': state_fingerprint(model.state_dict()),
            'norm_fingerprint': norm_stats_fingerprint(mean, std),
            'split_sha256': parts.get('segments.json')}


def validate_fingerprints(recorded, expected, *, required=True):
    if not recorded:
        if required:
            raise ValueError('Artifact lacks content fingerprints; regenerate a version 2 artifact')
        return
    for key, value in expected.items():
        if key not in recorded or recorded[key] != value:
            raise ValueError(f'Artifact fingerprint mismatch: {key}')


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


@contextmanager
def isolated_rng(seed):
    saved = rng_state()
    try:
        from Utils.runtime import set_seed
        set_seed(seed)
        yield
    finally:
        restore_rng_state(saved)


def case_noise_plan(model, starts, targets, frames, *, seed=42, noise_std=1.0):
    """Raw-space Gaussian sources; per-case prefixes are independent of batch size/NFE."""
    chunks = []
    for start, target in zip(starts, targets):
        key = f'{seed}:{int(start)}:{int(target)}'.encode()
        key_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        chunks.append(np.random.default_rng(key_seed).standard_normal(
            (frames - 1, model.latent_dim), dtype=np.float32))
    eps = torch.from_numpy(np.stack(chunks)).to(model.latent_offset.device)
    return model.latent_offset + model.latent_scale * model.latent_transport_std * float(noise_std) * eps


def cluster_interval(values, groups, *, samples=1000, seed=42, weights=None):
    """Percentile interval resampling original source clips, keeping all their windows together."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape != groups.shape or values.shape != weights.shape:
        raise ValueError('Bootstrap values, groups and weights must have identical shapes')
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, groups, weights = values[valid], groups[valid], weights[valid]
    if not values.size:
        return float('nan'), float('nan')
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values * weights)
    counts = np.bincount(inverse, weights=weights)
    draws = np.random.default_rng(seed).integers(0, len(unique), (samples, len(unique)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return tuple(np.quantile(means, [.025, .975]).tolist())
