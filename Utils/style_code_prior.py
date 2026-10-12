from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


# How the offline style-code prior supervises the selector (see style_prior_divergence).
STYLE_PRIOR_LOSS_TYPES = ("dense", "js", "sparse_eps", "legacy")
STYLE_PRIOR_LOSS_EPS = 1e-3


def sparse_code_histogram(indices, *, selected_slots, num_embeddings, topk=4):
    """Clip → slot histograms → top-k probabilities shared by training and recipe export."""
    indices = np.asarray(indices)
    slots = tuple(int(slot) for slot in selected_slots)
    if indices.ndim != 2 or not len(indices) or not slots or topk < 1 or num_embeddings < 1:
        raise ValueError('Recipe histogram requires nonempty [frames,slots] indices and positive topk/K')
    if not np.issubdtype(indices.dtype, np.integer) or min(slots) < 0 or max(slots) >= indices.shape[1]:
        raise ValueError('Recipe indices/selected slots are invalid')
    selected = indices[:, slots]
    if (selected < 0).any() or (selected >= num_embeddings).any():
        raise ValueError('Recipe code indices exceed the frozen codebook')
    result = np.zeros((len(slots), num_embeddings), dtype=np.float32)
    for local, slot in enumerate(slots):
        counts = np.bincount(indices[:, slot], minlength=num_embeddings).astype(np.float64)
        if topk >= num_embeddings:
            result[local] = counts / counts.sum()
            continue
        codes = np.argsort(-counts, kind='stable')[:min(topk, num_embeddings)]
        result[local, codes] = counts[codes] / counts[codes].sum()
    return result


def _full_code_profile(indices, num_embeddings):
    """Full histogram in sparse storage; avoid building and scanning a dense table twice."""
    if ((indices < 0).any() or (indices >= num_embeddings).any()):
        raise ValueError('Recipe code indices exceed the frozen codebook')
    slots, codes, weights = [], [], []
    for slot in range(indices.shape[1]):
        counts = np.bincount(indices[:, slot], minlength=num_embeddings)
        used = np.flatnonzero(counts)
        slots.append(np.full(len(used), slot, dtype=np.uint8 if indices.shape[1] <= 256 else np.int32))
        codes.append(used.astype(np.uint16 if num_embeddings <= 65536 else np.int32))
        weights.append((counts[used].astype(np.float64) / len(indices)).astype(np.float32))
    return np.concatenate(slots), np.concatenate(codes), np.concatenate(weights)


class TrainStyleRecipePool:
    """Independent train exemplars only; mirrors never count as additional source clips."""

    def __init__(self, *, db, model, cache, mean, std, frames=240, exemplars=5,
                 exemplar_choices=None, bank_size=0, seed=42, bank_max_entries=131072):
        if frames < 1 or exemplars < 1 or not model.use_style_code_selector:
            raise ValueError('Train recipes require a selector and positive frames/exemplars')
        db = Path(db)
        segments = json.loads((db / 'segments.json').read_text())
        self.by_style = {}
        seen = set()
        for index, segment in enumerate(segments):
            if segment.get('split') != 'train' or segment.get('mirror') or int(segment['num_frames']) < frames:
                continue
            source_id = segment.get('source_entry_id', index)
            if source_id in seen:
                continue
            seen.add(source_id)
            self.by_style.setdefault(int(segment['label']), []).append({**segment, 'source_entry_id': source_id})
        missing = sorted(set(range(model.num_styles)) - set(self.by_style))
        if missing:
            raise ValueError(f'Train recipe pool lacks {frames}-frame unmirrored exemplars for styles {missing}')
        self.model, self.cache = model, cache
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.frames, self.exemplars = frames, exemplars
        if bank_size < 0:
            raise ValueError('Recipe bank size must be non-negative')
        self.bank_size, self.seed = int(bank_size), int(seed)
        self.bank = OrderedDict()  # Raw statistics only; never cache trainable selector outputs.
        self._candidate_cache = OrderedDict()
        self.full_histogram = getattr(model, 'conditioning_mode', 'label') == 'clip_recipe'
        self.bank_max_entries = int(bank_max_entries)
        if self.bank_max_entries < 1:
            raise ValueError('Recipe bank max entries must be positive')
        self.source_ids = {label: {s['source_entry_id'] for s in clips} for label, clips in self.by_style.items()}
        self.exemplar_choices = tuple(exemplar_choices or (exemplars,))
        if (not self.exemplar_choices or any(int(n) != n or n < 1 for n in self.exemplar_choices)
                or len(set(self.exemplar_choices)) != len(self.exemplar_choices)):
            raise ValueError('Recipe exemplar choices must be distinct positive integers')
        if getattr(model, 'conditioning_mode', 'label') in ('recipe', 'clip_recipe'):
            need = max(self.exemplar_choices) + 1  # Leave out the supervised source clip.
            small = {label: len(clips) for label, clips in self.by_style.items() if len(clips) < need}
            if small:
                raise ValueError(f'Pure recipe training needs at least {need} independent train clips per style; found {small}')
        self.last_sample_stats = {'exemplars_mean': 0., 'shortfall_fraction': 0., 'bank_hit_fraction': 0.}
        self.source = None
        if cache is None:
            meta = json.loads((db / 'meta.json').read_text())
            self.source = np.memmap(db / 'X.f32', mode='r', dtype='float32', shape=(meta['num_frames'], 375))

    @torch.no_grad()
    def sample(self, labels, *, probability, excluded_source_ids=(),
               per_sample_exclusions=None, require_all=False, labels_cpu=None):
        if not 0 <= probability <= 1:
            raise ValueError('Recipe probability must be in [0,1]')
        if labels.ndim != 1 or (require_all and probability != 1.):
            raise ValueError('Pure recipe sampling requires 1D labels and probability=1')
        if per_sample_exclusions is not None and len(per_sample_exclusions) != labels.numel():
            raise ValueError('Recipe per-sample exclusions must match the batch')
        self.last_sample_stats = {'exemplars_mean': 0., 'shortfall_fraction': 0., 'bank_hit_fraction': 0.}
        selected = torch.rand(labels.numel(), device=labels.device) < probability
        if not labels.numel() or probability == 0.:
            return None, 0.
        if probability == 1.:
            mask = np.ones(labels.numel(), dtype=bool)
        else:
            mask = selected.cpu().numpy()
            if not mask.any():
                return None, 0.
        if labels_cpu is None:
            requested = labels.detach().cpu().numpy()
        else:
            # Only pass the original host labels while target labels remain unchanged.
            if (labels_cpu.device.type != 'cpu' or labels_cpu.shape != labels.shape
                    or labels_cpu.dtype != labels.dtype):
                raise ValueError('Host recipe labels must match the batch shape and dtype on CPU')
            requested = labels_cpu.detach().numpy()
        excluded = set(excluded_source_ids)
        slots = self.model.selected_style_slots
        size = self.model.vqvae.quantizer.num_embeddings
        # Upload each sparse profile once, then gather repeated rows on the GPU.
        # The bank holds raw histograms; selector outputs still change every update.
        sparse_upload = self.full_histogram and labels.is_cuda
        recipes = None if sparse_upload else np.zeros((len(requested), len(slots), size), dtype=np.float32)
        profiles = []
        profile_rows = np.zeros(len(requested), dtype=np.int64)
        groups = {}
        for row in np.flatnonzero(mask):
            blocked = excluded | (set(per_sample_exclusions[row]) if per_sample_exclusions is not None else set())
            label = int(requested[row])
            blocked &= self.source_ids[label]
            groups.setdefault((label, frozenset(blocked)), []).append(row)
        exemplar_total, shortfalls, hits = 0, 0, 0
        bank_choices = None
        if self.bank_size and require_all:
            # CPU randint uses one raw 32-bit draw for these small bounds. A common
            # multiple preserves every scalar result and the complete RNG state,
            # without hundreds of scalar Tensor allocations/item() calls per batch.
            bounds = ([len(self.exemplar_choices)] if len(self.exemplar_choices) > 1 else []) + [self.bank_size]
            common = math.lcm(*bounds)
            if common < 1 << 28:
                bank_choices = torch.randint(common, (len(groups), len(bounds))).numpy()
        for group_index, ((label, blocked), rows) in enumerate(groups.items()):
            candidate_key = (label, blocked)
            candidates = self._candidate_cache.get(candidate_key)
            if candidates is None:
                candidates = [s for s in self.by_style[label] if s['source_entry_id'] not in blocked]
                self._candidate_cache[candidate_key] = candidates
                if len(self._candidate_cache) > 4096:
                    self._candidate_cache.popitem(last=False)
            else:
                self._candidate_cache.move_to_end(candidate_key)
            if not candidates:
                if require_all:
                    raise ValueError(f'Pure recipe has no independent train exemplar for style {label}')
                mask[rows] = False
                continue
            choice = (int(bank_choices[group_index, 0] % len(self.exemplar_choices)) if bank_choices is not None
                      else int(torch.randint(len(self.exemplar_choices), ()).item())) if len(self.exemplar_choices) > 1 else 0
            desired = self.exemplar_choices[choice]
            if require_all and len(candidates) < desired:
                raise ValueError(f'Pure recipe style {label} needs {desired} independent exemplars; only {len(candidates)} remain')
            count = min(desired, len(candidates))
            exemplar_total += count * len(rows)
            shortfalls += (count < desired) * len(rows)
            variant = (int(bank_choices[group_index, -1] % self.bank_size) if bank_choices is not None
                       else int(torch.randint(self.bank_size, ()).item())) if self.bank_size else None
            key = (label, blocked, desired, variant)
            cached = self.bank.get(key) if self.bank_size else None
            if cached is not None:
                self.bank.move_to_end(key)
                if self.full_histogram:
                    slot_ids, codes, weights = cached
                    if sparse_upload:
                        profiles.append(cached)
                        profile_rows[rows] = len(profiles)
                    else:
                        recipes[np.asarray(rows)[:, None], slot_ids[None], codes[None]] = weights[None]
                else:
                    codes, weights = cached
                    recipes[np.asarray(rows)[:, None, None], np.arange(len(slots))[None, :, None], codes[None]] = weights[None]
                hits += len(rows)
                continue
            private_rng = None
            if self.bank_size:
                identity = json.dumps([self.seed, label, desired, variant, sorted(str(i) for i in blocked)])
                private_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')
                private_rng = np.random.default_rng(private_seed)
            chosen = (private_rng.permutation(len(candidates))[:count].tolist() if private_rng is not None
                      else torch.randperm(len(candidates))[:count].tolist())
            parts = []
            for index in chosen:
                segment = candidates[index]
                crop_count = int(segment['num_frames']) - self.frames + 1
                offset = int(private_rng.integers(crop_count) if private_rng is not None else torch.randint(crop_count, ()).item())
                start = int(segment['frame_start']) + offset
                if self.cache is not None:
                    indices = np.asarray(self.cache.indices[start:start + self.frames, slots], dtype=np.int64)
                else:
                    raw = np.array(self.source[start:start + self.frames], dtype=np.float32, copy=True)
                    x = torch.from_numpy((raw - self.mean) / self.std).to(labels.device)
                    _, encoded, _ = self.model.vqvae.encode(x)
                    indices = encoded.cpu().numpy()[:, slots]
                parts.append(indices)
            if self.full_histogram:
                profile = _full_code_profile(np.concatenate(parts), size)
                si, codes, weights = profile
                if sparse_upload:
                    profiles.append(profile)
                    profile_rows[rows] = len(profiles)
                else:
                    recipes[np.asarray(rows)[:, None], si[None], codes[None]] = weights[None]
            else:
                dense = sparse_code_histogram(np.concatenate(parts), selected_slots=range(len(slots)),
                    num_embeddings=size, topk=self.model.style_selector.topk)
                recipes[rows] = dense
            if self.bank_size:
                if self.full_histogram:
                    self.bank[key] = profile
                else:
                    codes = np.argsort(-dense, axis=-1, kind='stable')[:, :min(size, self.model.style_selector.topk)]
                    self.bank[key] = (codes.astype(np.uint16 if size <= 65536 else np.int32),
                                      np.take_along_axis(dense, codes, axis=-1))
                if self.full_histogram and len(self.bank) > self.bank_max_entries:
                    self.bank.popitem(last=False)
        actual = int(mask.sum())
        self.last_sample_stats = {'exemplars_mean': exemplar_total / max(actual, 1),
                                 'shortfall_fraction': shortfalls / max(actual, 1),
                                 'bank_hit_fraction': hits / max(actual, 1)}
        if sparse_upload and mask.any():
            flat_indices = np.concatenate([(i * len(slots) + si.astype(np.int64)) * size + codes
                for i, (si, codes, _) in enumerate(profiles, 1)])
            weights = np.concatenate([weights for _, _, weights in profiles])
            table = torch.zeros((len(profiles) + 1, len(slots), size), device=labels.device, dtype=torch.float32)
            table.flatten().scatter_(0, torch.from_numpy(flat_indices).to(labels.device, non_blocking=True),
                                     torch.from_numpy(weights).to(labels.device, non_blocking=True))
            result = table[torch.from_numpy(profile_rows).to(labels.device, non_blocking=True)]
        else:
            result = torch.from_numpy(recipes).to(labels.device, non_blocking=True) if mask.any() else None
        return result, float(mask.mean())


@torch.no_grad()
def recipe_probabilities_from_indices(indices, model, *, clip_lengths=None):
    """Shared clip/export/evaluation front end; output is always final manual probabilities."""
    if getattr(model,'conditioning_version',0)==2:
        return model.recipe_from_indices(indices,clip_lengths=clip_lengths)
    size = int(model.vqvae.quantizer.num_embeddings)
    learned = getattr(model, 'conditioning_mode', 'label') == 'clip_recipe'
    probs = sparse_code_histogram(indices, selected_slots=model.selected_style_slots,
        num_embeddings=size, topk=size if learned else model.style_selector.topk)
    if learned:
        device = model.vqvae.quantizer.embedding_weight.device
        probs = model.recipe_from_histogram(torch.from_numpy(probs).to(device))[0].cpu().numpy()
    return probs


def recipe_slots_from_probs(probs, selected_slots):
    result = []
    for row, slot in zip(probs, selected_slots):
        codes = np.flatnonzero(row > 0)
        codes = codes[np.argsort(-row[codes], kind='stable')]
        result.append(dict(slot=int(slot), codes=codes.tolist(), weights=row[codes].tolist()))
    return result


def recipe_conditioning_metadata(model):
    from Utils.reproducibility import state_fingerprint
    selector = getattr(model, 'histogram_selector', None)
    if getattr(model,'conditioning_version',0)==2:
        selector_hash=state_fingerprint({**{'encoder.'+k:v for k,v in model.clip_encoder.state_dict().items()},
            **{'selector.'+k:v for k,v in model.clip_selector.state_dict().items()}})
        return dict(conditioning_mode='clip_recipe',conditioning_version=2,probability_space='control',
            recipe_source='temporal_clip_selector',selector_sha256=selector_hash)
    return dict(conditioning_mode=getattr(model, 'conditioning_mode', 'label'),
                probability_space='control',
                recipe_source='learned_histogram_selector' if selector is not None else 'topk_histogram',
                selector_sha256=state_fingerprint(selector.state_dict()) if selector is not None else None)


def style_prior_divergence(
    target_prior: torch.Tensor,
    probs: torch.Tensor,
    *,
    logits: torch.Tensor | None = None,
    loss_type: str = "dense",
    temperature: float = 1.0,
    eps: float = STYLE_PRIOR_LOSS_EPS,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Divergence between the offline style-code prior and the selector distribution.

    `target_prior` is the smoothed prior pi_bar[s, j, k] (rank 2 or 3); `probs` is the selector's
    TopK-sparsified pi[s, j, k]; `logits` are the pre-softmax scores a[s, j, k].

    The paper's L_prior = (1/|Omega|) * sum_j KL(prior || pi) cannot be evaluated on the paper's
    pi: pi is TopK-sparsified (exact zeros) while the smoothed prior is strictly positive, so the
    KL is +inf. Clamping the zeros introduces the contribution
    `-log(1e-8) * (prior mass outside the top-k)` without useful gradients to excluded codes.
    Its share of total loss is not evidence of gradient dominance. The modes below differ
    in how they make the supervision well posed:

    * ``dense``      -- supervise ``softmax(logits / temperature)``, the distribution the selector
                        actually parameterizes. KL = 0 stays attainable, the gradient is the
                        standard distillation gradient ``p - prior``, and TopK remains the
                        selection operator used for the style token. Recommended.
    * ``js``         -- keep the literal sparse ``pi`` but use the bounded Jensen-Shannon
                        divergence (the same toolbox as the paper's slot saliency). Note the
                        floor: on this prior JS(prior || uniform) = 0.302 < JS(prior || best
                        top-k) = 0.535, so a sparse selector cannot approach the prior.
    * ``sparse_eps`` -- minimal patch: ``log(probs + eps)`` instead of ``log(clamp(probs, 1e-8))``.
    * ``legacy``     -- the original ``log(clamp(probs, 1e-8))`` cross-entropy, kept so existing
                        checkpoints can be reproduced bit-for-bit.
    """
    if loss_type not in STYLE_PRIOR_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {STYLE_PRIOR_LOSS_TYPES}, got {loss_type!r}")
    if weights is not None and loss_type != "dense":
        raise ValueError("weighted rows are only supported by loss_type='dense'")
    if target_prior.dim() == 2:
        target_prior = target_prior.unsqueeze(0)
    if target_prior.dim() != 3 or probs.dim() != 3:
        raise ValueError(
            f"target_prior and probs must be rank 3 [B, M, K], got {tuple(target_prior.shape)} and {tuple(probs.shape)}"
        )
    if target_prior.shape != probs.shape:
        raise ValueError(
            f"target_prior / probs shape mismatch: {tuple(target_prior.shape)} vs {tuple(probs.shape)}"
        )
    target = target_prior.to(device=probs.device, dtype=probs.dtype)

    if loss_type == "dense":
        if logits is None:
            raise ValueError(
                "loss_type='dense' needs the pre-softmax logits of the selector distribution "
                "(pass logits=...); see style_prior_divergence for the alternatives"
            )
        if tuple(logits.shape) != tuple(probs.shape):
            raise ValueError(f"logits shape {tuple(logits.shape)} != probs shape {tuple(probs.shape)}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        log_probs = F.log_softmax(logits.to(dtype=probs.dtype) / float(temperature), dim=-1)
        per_row = -(target * log_probs).sum(dim=-1)
        if weights is None:
            return per_row.mean()
        # Per-style rows weighted by batch frequency: equals the per-sample mean when the table
        # rows are styles and the weights are their occurrence counts.
        w = weights.to(device=per_row.device, dtype=per_row.dtype).reshape(-1, 1)
        return (per_row * w).sum() / (w.sum().clamp_min(1e-8) * per_row.size(-1))

    if loss_type == "js":
        p = target.clamp_min(0.0)
        q = probs.clamp_min(0.0)
        mixture = 0.5 * (p + q)
        log_mixture = torch.log(mixture.clamp_min(1e-12))

        def _kl(a: torch.Tensor) -> torch.Tensor:
            return (a * (torch.log(a.clamp_min(1e-12)) - log_mixture)).sum(dim=-1)

        return 0.5 * (_kl(p) + _kl(q)).mean()

    if loss_type == "sparse_eps":
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        return -(target * torch.log(probs.clamp_min(0.0) + float(eps))).sum(dim=-1).mean()

    # legacy
    return -(target * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


def style_code_entropy(probs: torch.Tensor, *, weights: torch.Tensor | None = None) -> torch.Tensor:
    """H(pi) = -sum_k pi_k log pi_k, the paper's sparsity-promoting regularizer (Eq. 7).

    Positive by construction: the trainer adds `+ weight * H`, so minimizing it sharpens the
    selection. Zeros contribute nothing (0 * log 0 := 0), which is why the clamp here is harmless
    -- unlike in the prior divergence, where the clamp is what breaks the KL.

    `weights` ([N] for `probs` [N, M, K]) replaces the uniform mean over rows with a weighted mean
    `sum_n w_n H_n / sum_n w_n`; it is used by the per-style selector fast path, where rows are
    styles weighted by how often they occur in the batch. `weights=None` is bit-identical to the
    original uniform mean.
    """
    p = probs.clamp_min(0.0)
    per_row = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
    if weights is None:
        return per_row.mean()
    w = weights.to(device=per_row.device, dtype=per_row.dtype).reshape(-1, 1)
    # Match the uniform mean's normalisation: average over the slot dimension as well.
    return (per_row * w).sum() / (w.sum().clamp_min(1e-8) * per_row.size(-1))


def validate_recipe_metadata(metadata, *, vq_fingerprint, selected_slots, allow_legacy=False, norm_fingerprint=None):
    if (metadata or {}).get('probability_space', 'control') != 'control':
        raise ValueError('Recipe must contain final control probabilities, not raw clip histograms')
    recorded = (metadata or {}).get("vq_state_sha256")
    if recorded is None and not allow_legacy:
        raise ValueError("Recipe lacks codebook provenance; regenerate it or explicitly allow legacy recipes")
    if recorded is not None and recorded != vq_fingerprint:
        raise ValueError("Recipe belongs to a different VQ codebook")
    slots = (metadata or {}).get("selected_slots")
    if slots is None and not allow_legacy:
        raise ValueError('Recipe lacks selected slot identities/order')
    if slots is not None and tuple(slots) != tuple(selected_slots):
        raise ValueError("Recipe selected slots/order differ from the controller")
    recorded_norm = (metadata or {}).get('norm_fingerprint')
    if norm_fingerprint is not None and recorded_norm is not None and recorded_norm != norm_fingerprint:
        raise ValueError('Recipe normalization differs from the controller/VQ manifold')




def save_style_code_prior(
    path: str | Path,
    *,
    prior: np.ndarray,
    global_prior: np.ndarray,
    slot_saliency: np.ndarray,
    selected_slots: np.ndarray,
    style_names: list[str] | np.ndarray,
    counts: np.ndarray | None = None,
    split: str = "train",
    smoothing: float = 1.0,
    vqvae_ckpt: str | Path | None = None,
    db: str | Path | None = None,
    fingerprints: dict | None = None,
    cache_sha256: str | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prior": np.asarray(prior, dtype=np.float32),
        "global_prior": np.asarray(global_prior, dtype=np.float32),
        "slot_saliency": np.asarray(slot_saliency, dtype=np.float32),
        "selected_slots": np.asarray(selected_slots, dtype=np.int64),
        "style_names": np.asarray(style_names, dtype=np.str_),
        "split": np.asarray(str(split)),
        "smoothing": np.asarray(float(smoothing), dtype=np.float32),
    }
    if counts is not None:
        payload["counts"] = np.asarray(counts, dtype=np.int64)
    # Provenance: two VQ runs share the same (num_codebooks, num_embeddings) shape, so the arrays
    # alone cannot tell which codebook the prior was estimated from.
    if vqvae_ckpt is not None:
        payload["vqvae_ckpt"] = np.asarray(str(vqvae_ckpt))
    if db is not None:
        payload["db"] = np.asarray(str(db))
    payload["num_codebooks"] = np.asarray(int(np.asarray(prior).shape[1]), dtype=np.int64)
    if fingerprints is not None:
        payload["fingerprints"] = np.asarray(json.dumps(fingerprints, sort_keys=True))
    if cache_sha256 is not None:
        payload["cache_sha256"] = np.asarray(cache_sha256)
    np.savez_compressed(path, **payload)
    return path


def load_style_code_prior(path: str | Path) -> dict[str, np.ndarray | list[str] | str | float]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"style code prior file not found: {path}")
    data = np.load(path, allow_pickle=True)
    required = {"prior", "global_prior", "slot_saliency", "selected_slots"}
    if not required.issubset(set(data.files)):
        raise KeyError(f"{path} must contain keys {sorted(required)}")

    prior = np.asarray(data["prior"], dtype=np.float32)
    global_prior = np.asarray(data["global_prior"], dtype=np.float32)
    slot_saliency = np.asarray(data["slot_saliency"], dtype=np.float32).reshape(-1)
    selected_slots = np.asarray(data["selected_slots"], dtype=np.int64).reshape(-1)
    style_names = [str(name) for name in np.asarray(data["style_names"], dtype=np.str_).tolist()] if "style_names" in data else []
    split = str(data["split"].item()) if "split" in data else "train"
    smoothing = float(np.asarray(data["smoothing"]).item()) if "smoothing" in data else 1.0
    counts = np.asarray(data["counts"], dtype=np.int64) if "counts" in data else None
    vqvae_ckpt = str(data["vqvae_ckpt"]) if "vqvae_ckpt" in data else None
    db = str(data["db"]) if "db" in data else None
    num_codebooks = int(np.asarray(data["num_codebooks"]).item()) if "num_codebooks" in data else None

    if prior.ndim != 3:
        raise ValueError(f"prior must have rank 3 [num_styles, num_slots, num_embeddings], got {prior.shape}")
    if global_prior.ndim != 2:
        raise ValueError(f"global_prior must have rank 2 [num_slots, num_embeddings], got {global_prior.shape}")
    if prior.shape[1:] != global_prior.shape:
        raise ValueError(f"prior/global_prior shape mismatch: {prior.shape} vs {global_prior.shape}")
    if slot_saliency.shape != (prior.shape[1],):
        raise ValueError(
            f"slot_saliency must have shape ({prior.shape[1]},), got {slot_saliency.shape}"
        )
    if selected_slots.size == 0:
        raise ValueError("selected_slots must be non-empty")
    if selected_slots.min() < 0 or selected_slots.max() >= prior.shape[1]:
        raise ValueError(
            f"selected_slots out of range for num_slots={prior.shape[1]}: {selected_slots.tolist()}"
        )
    return {
        "path": str(path),
        "prior": prior,
        "global_prior": global_prior,
        "slot_saliency": slot_saliency,
        "selected_slots": selected_slots,
        "style_names": style_names,
        "split": split,
        "smoothing": smoothing,
        "counts": counts,
        "vqvae_ckpt": vqvae_ckpt,
        "db": db,
        "num_codebooks": num_codebooks,
        "fingerprints": json.loads(str(data["fingerprints"].item())) if "fingerprints" in data else None,
        "cache_sha256": str(data["cache_sha256"].item()) if "cache_sha256" in data else None,
    }


__all__ = [
    "STYLE_PRIOR_LOSS_EPS",
    "STYLE_PRIOR_LOSS_TYPES",
    "load_style_code_prior",
    "save_style_code_prior",
    "style_code_entropy",
    "style_prior_divergence",
]
