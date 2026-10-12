from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from Utils.style_code_prior import (
    STYLE_PRIOR_LOSS_TYPES,
    style_code_entropy,
    style_prior_divergence,
)

from .basic_vqvae_375to291 import BasicVQVAE375to291, ControlSkipCatMLP


FlowPredictionMode = Literal["velocity", "target"]



def resolve_flowmatch_prediction_mode(prediction_mode: str) -> FlowPredictionMode:
    mode = str(prediction_mode).strip().lower()
    if mode not in {"velocity", "target"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode!r}")
    return mode


def sample_flow_times(reference, *, low_probability=0., low_max=.2):
    """Positive density on all flow times; zero probability preserves legacy RNG."""
    if not math.isfinite(low_probability) or not 0 <= low_probability < 1 or not 0 < low_max <= 1:
        raise ValueError('Low-time mixture requires probability in [0,1) and maximum in (0,1]')
    u = torch.rand((reference.size(0), 1), device=reference.device, dtype=reference.dtype)
    if low_probability == 0:
        return u
    return torch.where(u < low_probability, u * low_max / low_probability,
                       (u - low_probability) / (1 - low_probability))


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    t = t.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    emb = torch.cat((torch.sin(t[:, None] * freqs[None]), torch.cos(t[:, None] * freqs[None])), dim=-1)
    if emb.size(-1) < dim:
        emb = F.pad(emb, (0, dim - emb.size(-1)))
    return emb


class StyleEncoder(nn.Module):
    def __init__(self, num_styles: int, style_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_styles, style_dim)

    def forward(self, style_label: torch.Tensor) -> torch.Tensor:
        return self.embedding(style_label.long())


class HistogramCodeSelector(nn.Module):
    """Label-free residual scores on frozen code geometry; zero-init is histogram TopK."""

    def __init__(self, code_dim: int, slots: int):
        super().__init__()
        self.slot = nn.Parameter(torch.zeros(slots, 8))
        self.key = nn.Linear(2 * code_dim, 16)
        self.query = nn.Sequential(nn.Linear(2 * code_dim + 10, 32), nn.GELU(), nn.Linear(32, 16))
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)

    def forward(self, histogram, codebook, *, validate=True):
        if validate and (histogram.ndim != 3 or not torch.isfinite(histogram).all()
                or (histogram < 0).any() or (histogram.sum(-1) <= 0).any()):
            raise ValueError('Clip histograms require finite nonnegative, nonempty [B,M,K] rows')
        p = histogram / histogram.sum(-1, keepdim=True)
        book = codebook.detach()
        key = (book.data_ptr(), book._version, book.device, book.dtype)
        if getattr(self, '_geometry_cache_key', None) != key:
            e = (book - book.mean(0)) / book.std(0, unbiased=False).clamp_min(1e-6)
            self._geometry_cache = torch.cat((e, e.square()), -1)
            self._geometry_cache_key = key
        geometry = self._geometry_cache
        stats = p @ geometry
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1, keepdim=True) / math.log(max(2, p.size(-1)))
        slot = self.slot.unsqueeze(0).expand(p.size(0), -1, -1)
        query = self.query(torch.cat((stats, entropy, p.amax(-1, keepdim=True), slot), -1))
        scores = (query @ self.key(geometry).t()) / 4.
        return p.clamp_min(1e-8).log() + 4. * torch.tanh(scores / 4.)


class StyleCodeSelector(nn.Module):
    def __init__(
        self,
        *,
        num_styles: int,
        num_embeddings: int,
        code_dim: int,
        selected_slots: Sequence[int],
        style_dim: int,
        hidden_dim: int = 128,
        topk: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.0,
        prototype_normalization: str = 'layernorm',
    ):
        super().__init__()
        if num_styles < 1:
            raise ValueError(f"num_styles must be >= 1, got {num_styles}")
        if num_embeddings < 1:
            raise ValueError(f"num_embeddings must be >= 1, got {num_embeddings}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")
        slots = tuple(int(slot) for slot in selected_slots)
        if not slots:
            raise ValueError("selected_slots must be non-empty")
        if any(slot < 0 for slot in slots):
            raise ValueError(f"selected_slots must be non-negative, got {slots}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if style_dim < 1:
            raise ValueError(f"style_dim must be >= 1, got {style_dim}")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.num_styles = int(num_styles)
        self.num_embeddings = int(num_embeddings)
        self.code_dim = int(code_dim)
        self.selected_slots = slots
        self.num_style_slots = len(slots)
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.temperature = float(temperature)
        if prototype_normalization not in ('layernorm', 'none'):
            raise ValueError('prototype_normalization must be layernorm or none')
        self.prototype_normalization = prototype_normalization
        proto_dim = self.num_style_slots * self.code_dim
        logits_hidden_dim = max(self.hidden_dim, self.style_dim, proto_dim)
        proto_hidden_dim = max(self.style_dim, proto_dim)

        self.style_embedding = nn.Embedding(self.num_styles, self.hidden_dim)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, logits_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(logits_hidden_dim, self.num_style_slots * self.num_embeddings),
        )
        self.proto_norm = nn.LayerNorm(proto_dim)
        if prototype_normalization == 'none':
            self.proto_norm.requires_grad_(False)
        self.proto_to_style = nn.Sequential(
            nn.Linear(proto_dim, proto_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(proto_hidden_dim, self.style_dim),
        )

    @torch.no_grad()
    def initialize_from_prior(self, prior: torch.Tensor) -> dict:
        """Fit the existing output Linear to prior logits; no optimizer steps or VQ changes."""
        if tuple(prior.shape) != (self.num_styles, self.num_style_slots, self.num_embeddings):
            raise ValueError('Initialization prior must match selected style slots')
        if not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError('Prior-logit initialization requires a positive finite smoothed prior')
        output = self.to_logits[-1]
        if self.num_styles > output.in_features + 1:
            raise ValueError('Selector hidden dimension is too small for exact per-style prior initialization')
        was_training = self.training
        self.eval()
        try:
            labels = torch.arange(self.num_styles, device=output.weight.device)
            hidden = self.to_logits[:-1](self.style_embedding(labels)).double().cpu()
            design = torch.cat((hidden, torch.ones(self.num_styles, 1, dtype=torch.float64)), dim=-1)
            prior_cpu = prior.detach().double().cpu()
            prior_cpu = prior_cpu / prior_cpu.sum(-1, keepdim=True)
            logits = self.temperature * prior_cpu.log()
            logits = (logits - logits.mean(-1, keepdim=True)).reshape(self.num_styles, -1)
            fit = torch.linalg.lstsq(design, logits, driver='gelsd')
            if int(fit.rank) != self.num_styles:
                raise ValueError('Selector features cannot distinguish every style for prior initialization')
            output.weight.copy_(fit.solution[:-1].T.to(output.weight))
            output.bias.copy_(fit.solution[-1].to(output.bias))
            actual = F.softmax(self.to_logits(self.style_embedding(labels)).reshape_as(prior) / self.temperature, -1)
            expected = prior_cpu.to(actual)
            error = float((actual - expected).abs().max())
            if error > 1e-4:
                raise ValueError(f'Prior-logit initialization is numerically unstable: max probability error {error}')
            return {'method': 'prior_logits_lstsq', 'rank': int(fit.rank), 'max_probability_error': error}
        finally:
            self.train(was_training)

    def _expand_style_label(self, style_label: torch.Tensor, batch: int) -> torch.Tensor:
        style_label = style_label.reshape(-1).long()
        if style_label.numel() == batch:
            return style_label
        if style_label.numel() == 1:
            return style_label.expand(batch)
        raise ValueError(f"Cannot align {style_label.numel()} style labels with batch size {batch}")

    def _resolve_batch(self, style_label, manual_probs, batch_size=None) -> int:
        sizes = [int(batch_size)] if batch_size is not None else []
        if style_label is not None:
            sizes.append(style_label.numel())
        if manual_probs is not None:
            if manual_probs.dim() not in (2, 3):
                raise ValueError("manual_probs must have rank 2 [M,K] or 3 [B,M,K]")
            sizes.append(1 if manual_probs.dim() == 2 else manual_probs.size(0))
        if not sizes:
            raise ValueError("One of style_label or manual_probs must be provided.")
        batch = max(sizes)
        if batch < 1 or any(size not in (1, batch) for size in sizes):
            raise ValueError(f"Incompatible conditioning batch sizes: {sizes}")
        return batch

    def _sparsify_probs(self, probs: torch.Tensor) -> torch.Tensor:
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs / denom
        if self.topk >= self.num_embeddings:
            return probs
        top_values, top_indices = torch.topk(probs, self.topk, dim=-1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(-1, top_indices, top_values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _normalize_manual_probs(self, manual_probs: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        if manual_probs.dim() == 2:
            manual_probs = manual_probs.unsqueeze(0)
        if manual_probs.dim() != 3:
            raise ValueError(
                f"manual_probs must have rank 2 [M,K] or 3 [B,M,K], got {tuple(manual_probs.shape)}"
            )
        if manual_probs.size(1) != self.num_style_slots or manual_probs.size(2) != self.num_embeddings:
            raise ValueError(
                f"manual_probs must have shape [B,{self.num_style_slots},{self.num_embeddings}], "
                f"got {tuple(manual_probs.shape)}"
            )
        if manual_probs.size(0) == 1 and batch > 1:
            manual_probs = manual_probs.expand(batch, -1, -1)
        elif manual_probs.size(0) != batch:
            raise ValueError(
                f"manual_probs batch mismatch: expected 1 or {batch}, got {manual_probs.size(0)}"
            )
        if not torch.isfinite(manual_probs).all() or (manual_probs < 0).any():
            raise ValueError("manual_probs must be finite and non-negative")
        original_probs = manual_probs
        already_normalized = torch.isclose(manual_probs.sum(-1, keepdim=True),
                                          manual_probs.new_ones(()), rtol=0., atol=1e-6)
        row_scale = manual_probs.amax(dim=-1, keepdim=True)
        manual_probs = manual_probs / torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
        manual_sum = manual_probs.sum(dim=-1, keepdim=True)
        manual_mask = manual_sum > 0
        manual_probs = manual_probs / torch.where(manual_mask, manual_sum, torch.ones_like(manual_sum))
        # Preserve FP32-normalized selector probabilities exactly. A second normalization
        # otherwise perturbs the token by a few ulps and can flip nearest-code ties.
        manual_probs = torch.where(already_normalized, original_probs, manual_probs)
        return manual_probs, manual_mask

    def forward(
        self,
        *,
        codebook_weight: torch.Tensor,
        style_label: torch.Tensor | None = None,
        manual_probs: torch.Tensor | None = None,
        manual_strength: float = 1.0,
        temperature: float | None = None,
        batch_size: int | None = None,
        manual_probs_validated: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_batch(style_label, manual_probs, batch_size)
        if not math.isfinite(float(manual_strength)) or not 0 <= float(manual_strength) <= 1:
            raise ValueError("manual_strength must be finite and in [0,1]")
        style_logits = None
        style_probs = None
        # True only when style_logits holds the pre-softmax scores of the learned selector; with
        # manual (recipe) probabilities there is no logit parameterization to supervise.
        logits_are_dense = False
        learned_probs = None

        if style_label is not None:
            style_label = self._expand_style_label(style_label.to(device=codebook_weight.device), batch)
            style_hidden = self.style_embedding(style_label)
            style_logits = self.to_logits(style_hidden).reshape(batch, self.num_style_slots, self.num_embeddings)
            logits_are_dense = True
            tau = self.temperature if temperature is None else float(temperature)
            if not math.isfinite(tau) or tau <= 0.0:
                raise ValueError(f"temperature must be > 0, got {tau}")
            style_probs = F.softmax(style_logits / tau, dim=-1)
            style_probs = self._sparsify_probs(style_probs)
            learned_probs = style_probs

        if manual_probs is not None:
            manual_probs = manual_probs.to(device=codebook_weight.device, dtype=codebook_weight.dtype)
            if manual_probs_validated:
                # Internal histogram selector emits normalized, nonempty TopK rows.
                # External recipes keep all validation and normalization below.
                manual_mask = None
            else:
                manual_probs, manual_mask = self._normalize_manual_probs(manual_probs, batch)
            if style_probs is None:
                if float(manual_strength) != 1.0 or (manual_mask is not None and not bool(manual_mask.all())):
                    raise ValueError("Pure recipes require all style slots and manual_strength=1")
                style_probs = manual_probs
            else:
                mix = float(manual_strength)
                mix = max(0.0, min(1.0, mix))
                mixed_probs = ((1.0 - mix) * style_probs) + (mix * manual_probs)
                style_probs = mixed_probs if manual_mask is None else torch.where(manual_mask, mixed_probs, style_probs)
                style_probs = style_probs / style_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if style_probs is None:
            raise RuntimeError("style_probs was not resolved.")

        codebook_weight = codebook_weight.to(device=style_probs.device, dtype=style_probs.dtype)
        style_codes = torch.einsum("bmk,kd->bmd", style_probs, codebook_weight)
        style_proto = style_codes.reshape(batch, self.num_style_slots * self.code_dim)
        projected = self.proto_norm(style_proto) if self.prototype_normalization == 'layernorm' else style_proto
        style_feature = self.proto_to_style(projected)
        if style_logits is None:
            style_logits = torch.log(style_probs.clamp_min(1e-8))
        return {
            "style_logits": style_logits,
            "style_logits_are_dense": bool(logits_are_dense),
            "style_probs": style_probs,
            "learned_style_probs": learned_probs,
            "style_codes": style_codes,
            "style_proto": style_proto,
            "style_feature": style_feature,
        }


class ConditionalVelocityField(nn.Module):
    num_tokens = 6

    def __init__(
        self,
        *,
        latent_dim: int,
        motion_dim: int,
        traj_dim: int,
        style_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_heads: int = 4,
        ff_hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_motion_token: bool = True,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if time_embed_dim < 1:
            raise ValueError(f"time_embed_dim must be >= 1, got {time_embed_dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if ff_hidden_dim < 1:
            raise ValueError(f"ff_hidden_dim must be >= 1, got {ff_hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.fp32_math_attention = False
        # `use_motion_token=False` drops the direct 291-dim previous-motion path. At inference the
        # fed motion is exactly decoder(z_prev), so the quantized latent token already carries it;
        # the ablation removes the redundant raw path and leaves 5 tokens.
        self.use_motion_token = bool(use_motion_token)
        self.num_tokens = 6 if self.use_motion_token else 5
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim) if self.use_motion_token else None
        self.traj_proj = nn.Linear(traj_dim, hidden_dim)
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim)
        nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def prepare_conditions(self, z_prev, content, traj, style):
        # These tokens are constant throughout an Euler solve; keep their gradients.
        # Without the motion token `content` is ignored: the previous-motion information
        # stays reachable only through the quantized latent token z_prev.
        if not self.use_motion_token:
            return torch.stack((self.latent_proj(z_prev), self.traj_proj(traj),
                                self.style_proj(style)), dim=1)
        return torch.stack((self.latent_proj(z_prev), self.motion_proj(content),
                            self.traj_proj(traj), self.style_proj(style)), dim=1)

    def forward(
        self,
        *,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        traj: torch.Tensor,
        style: torch.Tensor,
        condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_emb = _timestep_embedding(t.to(dtype=z_t.dtype), self.time_embed_dim)
        if condition_tokens is None:
            condition_tokens = self.prepare_conditions(z_prev, content, traj, style)
        tokens = torch.cat((self.latent_proj(z_t)[:, None], condition_tokens,
                            self.time_proj(time_emb)[:, None]), dim=1)
        positions = self.position_embeddings
        if positions.device != tokens.device or positions.dtype != tokens.dtype:
            positions = positions.to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + positions
        if self.fp32_math_attention and self.training and tokens.is_cuda and tokens.dtype == torch.float32:
            # Six-token FP32 attention wastes work in a 64x64 memory-efficient tile.
            with sdpa_kernel(SDPBackend.MATH):
                encoded = self.encoder(tokens)
        else:
            encoded = self.encoder(tokens)
        # Use the updated z_t / z_bar token as the readout token.
        main_token = self.output_norm(encoded[:, 0])
        return self.out(main_token)


class ConditionalSkipCatMLPField(nn.Module):
    """Concatenative skip-MLP velocity field (ControlOperators ``SkipCatMLP``), no attention.

    Same interface as :class:`ConditionalVelocityField`, but the [z_t, z_prev, motion?,
    traj, style, time] inputs are concatenated into one flat vector and mapped through a
    skip-cat MLP instead of tokenizing + self-attention. `prepare_conditions` caches the
    constant part (z_prev, motion?, traj, style) so an Euler solve reuses it per step.
    """

    num_tokens = None

    def __init__(
        self,
        *,
        latent_dim: int,
        motion_dim: int,
        traj_dim: int,
        style_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_heads: int = 4,
        ff_hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_motion_token: bool = True,
    ):
        super().__init__()
        del num_heads, ff_hidden_dim  # attention-only knobs, accepted for a shared interface
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if time_embed_dim < 1:
            raise ValueError(f"time_embed_dim must be >= 1, got {time_embed_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.latent_dim = int(latent_dim)
        self.motion_dim = int(motion_dim)
        self.traj_dim = int(traj_dim)
        self.style_dim = int(style_dim)
        self.use_motion_token = bool(use_motion_token)
        self.fp32_math_attention = False
        self.dropout = nn.Dropout(dropout)
        condition_dim = self.latent_dim + self.traj_dim + self.style_dim + (
            self.motion_dim if self.use_motion_token else 0)
        self.condition_dim = condition_dim
        self.input_dim = condition_dim + self.latent_dim + self.time_embed_dim
        self.net = ControlSkipCatMLP(
            inp=self.input_dim,
            out=self.latent_dim,
            hidden=self.hidden_dim,
            depth=num_layers,
        )

    def prepare_conditions(self, z_prev, content, traj, style):
        # Flat concatenation in the same semantic order as the token field:
        # [z_prev, motion?, traj, style]. Cached across the Euler solve.
        if not self.use_motion_token:
            return torch.cat((z_prev, traj, style), dim=-1)
        return torch.cat((z_prev, content, traj, style), dim=-1)

    def forward(
        self,
        *,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        traj: torch.Tensor,
        style: torch.Tensor,
        condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_emb = _timestep_embedding(t.to(dtype=z_t.dtype), self.time_embed_dim)
        if condition_tokens is None:
            condition_tokens = self.prepare_conditions(z_prev, content, traj, style)
        x = torch.cat((z_t, condition_tokens, time_emb), dim=-1)
        return self.net(self.dropout(x))


class FlowMatchVQVAEController(nn.Module):
    def __init__(
        self,
        *,
        vqvae: BasicVQVAE375to291,
        num_styles: int,
        use_style_code_selector: bool = False,
        selected_style_slots: Sequence[int] | None = None,
        style_dim: int = 64,
        style_selector_hidden_dim: int = 128,
        style_selector_topk: int = 4,
        style_selector_temperature: float = 1.0,
        style_proto_normalization: str = 'layernorm',
        style_prior_loss: str = "dense",
        conditioning_mode: str = "label",
        time_embed_dim: int = 64,
        flow_hidden_dim: int = 256,
        flow_layers: int = 2,
        flow_heads: int = 4,
        flow_ffn_dim: int = 1024,
        dropout: float = 0.1,
        prediction_mode: FlowPredictionMode = "velocity",
        target_time_eps: float = 1e-3,
        flow_loss_space: str | None = None,
        inference_requantize: bool = False,
        inference_latent_clip: bool = True,
        label_selector_dedup: bool = True,
        use_motion_token: bool = True,
        field_arch: str = "transformer",
        history_conditioning: bool = False,
    ):
        super().__init__()
        self.vqvae = vqvae
        self.vqvae.eval()
        for parameter in self.vqvae.parameters():
            parameter.requires_grad_(False)

        self.input_dim = int(vqvae.input_dim)
        self.traj_dim = int(vqvae.traj_dim)
        self.motion_dim = int(vqvae.motion_dim)
        self.latent_dim = int(vqvae.latent_dim)
        self.hidden_dim = int(flow_hidden_dim)
        self.style_dim = int(style_dim)
        self.num_styles = int(num_styles)
        self.use_motion_token = bool(use_motion_token)
        field_arch = str(field_arch).strip().lower()
        if field_arch not in ("transformer", "skip_cat_mlp"):
            raise ValueError(f"field_arch must be 'transformer' or 'skip_cat_mlp', got {field_arch!r}")
        self.field_arch = field_arch
        self.history_conditioning = bool(history_conditioning)
        if self.history_conditioning and field_arch != "transformer":
            raise ValueError("History conditioning currently requires the transformer field")
        self.prediction_mode = resolve_flowmatch_prediction_mode(prediction_mode)
        self.target_time_eps = float(target_time_eps)
        if flow_loss_space is None:
            flow_loss_space = 'normalized' if prediction_mode == 'velocity' else 'transport'
        if flow_loss_space not in ("normalized", "transport"):
            raise ValueError("flow_loss_space must be normalized or transport")
        self.flow_loss_space = flow_loss_space
        self.inference_requantize = bool(inference_requantize)
        self.inference_latent_clip = bool(inference_latent_clip)
        # Label-mode speed path: the selector and its regularizers depend only on the style
        # label, so a whole-style table is computed once per step and rows are selected per
        # sample, instead of materializing [B, M, K] twice (forward + backward) for B=512.
        self.label_selector_dedup = bool(label_selector_dedup)
        self.use_style_code_selector = bool(use_style_code_selector)
        if conditioning_mode not in ('label', 'recipe', 'clip_recipe') or (conditioning_mode != 'label' and not self.use_style_code_selector):
            raise ValueError('Recipe conditioning requires the codebook style selector')
        self.conditioning_mode = conditioning_mode
        if str(style_prior_loss) not in STYLE_PRIOR_LOSS_TYPES:
            raise ValueError(
                f"style_prior_loss must be one of {STYLE_PRIOR_LOSS_TYPES}, got {style_prior_loss!r}"
            )
        self.style_prior_loss = str(style_prior_loss)
        if self.use_style_code_selector:
            if selected_style_slots is None or len(tuple(selected_style_slots)) == 0:
                selected_style_slots = tuple(range(min(8, int(vqvae.num_latents))))
            selected_style_slots = tuple(int(slot) for slot in selected_style_slots)
            if len(set(selected_style_slots)) != len(selected_style_slots):
                raise ValueError(f"selected_style_slots must be unique, got {selected_style_slots}")
            if any(slot < 0 or slot >= int(vqvae.num_latents) for slot in selected_style_slots):
                raise ValueError(
                    f"selected_style_slots must fall within [0, {int(vqvae.num_latents) - 1}], got {selected_style_slots}"
                )
            self.style_selector = StyleCodeSelector(
                num_styles=self.num_styles,
                num_embeddings=int(vqvae.quantizer.num_embeddings),
                code_dim=int(vqvae.code_dim),
                selected_slots=selected_style_slots,
                style_dim=self.style_dim,
                hidden_dim=int(style_selector_hidden_dim),
                topk=int(style_selector_topk),
                temperature=float(style_selector_temperature),
                dropout=dropout,
                prototype_normalization=style_proto_normalization,
            )
            self.style_encoder = None
            if self.conditioning_mode in ('recipe', 'clip_recipe'):
                self.style_selector.style_embedding.requires_grad_(False)
                self.style_selector.to_logits.requires_grad_(False)
            self.register_buffer("style_slot_indices", torch.tensor(selected_style_slots, dtype=torch.long))
            self.register_buffer(
                "style_prior_probs",
                torch.full(
                    (
                        self.num_styles,
                        len(selected_style_slots),
                        int(vqvae.quantizer.num_embeddings),
                    ),
                    1.0 / float(int(vqvae.quantizer.num_embeddings)),
                    dtype=torch.float32,
                ),
            )
            self.register_buffer("style_prior_ready", torch.tensor(False, dtype=torch.bool))
        else:
            self.style_selector = None
            self.style_encoder = StyleEncoder(num_styles=num_styles, style_dim=style_dim)
            self.register_buffer("style_slot_indices", torch.zeros((0,), dtype=torch.long))
            self.register_buffer("style_prior_probs", torch.zeros((0, 0, 0), dtype=torch.float32))
            self.register_buffer("style_prior_ready", torch.tensor(False, dtype=torch.bool))
        self.histogram_selector = (HistogramCodeSelector(int(vqvae.code_dim), len(selected_style_slots))
                                   if conditioning_mode == 'clip_recipe' else None)
        field_cls = ConditionalSkipCatMLPField if self.field_arch == 'skip_cat_mlp' else ConditionalVelocityField
        self.field = field_cls(
            latent_dim=self.latent_dim,
            motion_dim=self.motion_dim,
            traj_dim=self.traj_dim,
            style_dim=style_dim,
            hidden_dim=self.hidden_dim,
            time_embed_dim=time_embed_dim,
            num_heads=flow_heads,
            ff_hidden_dim=flow_ffn_dim,
            num_layers=flow_layers,
            dropout=dropout,
            use_motion_token=self.use_motion_token,
        )

        self.history_delta_proj = nn.Linear(self.latent_dim, self.hidden_dim) if self.history_conditioning else None
        if self.history_delta_proj is not None:
            nn.init.zeros_(self.history_delta_proj.weight)
            nn.init.zeros_(self.history_delta_proj.bias)

        self.register_buffer("latent_offset", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_scale", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_transport_std", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_data_min", torch.full((self.latent_dim,), -float("inf"), dtype=torch.float32))
        self.register_buffer("latent_data_max", torch.full((self.latent_dim,), float("inf"), dtype=torch.float32))
        self.register_buffer("latent_stats_ready", torch.tensor(False, dtype=torch.bool))
        self._latent_stats_initialized = False
        self._style_prior_initialized = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Read once on loading, not from GPU at every normalization/Euler substep.
        self._latent_stats_initialized = bool(self.latent_stats_ready.item())
        self._style_prior_initialized = bool(self.style_prior_ready.item())

    def train(self, mode: bool = True):
        super().train(mode)
        self.vqvae.eval()
        return self

    def set_latent_prior_stats(
        self,
        offset: torch.Tensor,
        scale: torch.Tensor,
        *,
        transport_std: torch.Tensor | None = None,
        data_min: torch.Tensor | None = None,
        data_max: torch.Tensor | None = None,
    ) -> None:
        offset = torch.as_tensor(offset, dtype=torch.float32, device=self.latent_offset.device).reshape(-1)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.latent_scale.device).reshape(-1)
        if offset.shape != (self.latent_dim,) or scale.shape != (self.latent_dim,):
            raise ValueError("Invalid latent stats shape")
        if transport_std is None:
            transport_std_t = torch.ones(self.latent_dim, dtype=torch.float32, device=self.latent_transport_std.device)
        else:
            transport_std_t = torch.as_tensor(
                transport_std,
                dtype=torch.float32,
                device=self.latent_transport_std.device,
            ).reshape(-1)
        if transport_std_t.shape != (self.latent_dim,):
            raise ValueError("Invalid latent transport_std shape")
        if data_min is None:
            data_min_t = torch.full((self.latent_dim,), -float("inf"), dtype=torch.float32, device=self.latent_data_min.device)
        else:
            data_min_t = torch.as_tensor(data_min, dtype=torch.float32, device=self.latent_data_min.device).reshape(-1)
        if data_max is None:
            data_max_t = torch.full((self.latent_dim,), float("inf"), dtype=torch.float32, device=self.latent_data_max.device)
        else:
            data_max_t = torch.as_tensor(data_max, dtype=torch.float32, device=self.latent_data_max.device).reshape(-1)
        if data_min_t.shape != (self.latent_dim,) or data_max_t.shape != (self.latent_dim,):
            raise ValueError("Invalid latent min/max stats shape")
        self.latent_offset.copy_(offset)
        self.latent_scale.copy_(scale.clamp_min(1e-6))
        self.latent_transport_std.copy_(transport_std_t.clamp_min(1e-6))
        self.latent_data_min.copy_(data_min_t)
        self.latent_data_max.copy_(data_max_t)
        self.latent_stats_ready.fill_(True)
        self._latent_stats_initialized = True

    def _require_latent_stats(self) -> None:
        if not self._latent_stats_initialized:
            raise RuntimeError("Call set_latent_prior_stats(...) before training or sampling.")

    @staticmethod
    def _stats_like(buffer: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Registered buffers already live on the model device/dtype; the conversion is only
        # needed for mismatched callers (e.g. fp64 diagnostics), so skip the dispatch otherwise.
        if buffer.device == reference.device and buffer.dtype == reference.dtype:
            return buffer
        return buffer.to(device=reference.device, dtype=reference.dtype)

    def normalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        offset = self._stats_like(self.latent_offset, latents)
        scale = self._stats_like(self.latent_scale, latents)
        return (latents - offset) / scale

    def denormalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        offset = self._stats_like(self.latent_offset, latents)
        scale = self._stats_like(self.latent_scale, latents)
        return offset + scale * latents

    def transport_normalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        transport_std = self._stats_like(self.latent_transport_std, latents)
        return latents / transport_std

    def transport_denormalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        transport_std = self._stats_like(self.latent_transport_std, latents)
        return transport_std * latents

    def sample_transport_noise_like(self, reference: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
        noise = torch.randn_like(reference)
        return float(noise_std) * self.transport_denormalize_latent(noise)

    def clip_latent_to_data_bounds(self, latents: torch.Tensor) -> torch.Tensor:
        latents_f32 = latents if latents.dtype == torch.float32 else latents.to(dtype=torch.float32)
        latents_norm = self.normalize_latent(latents_f32)
        data_min = self._stats_like(self.latent_data_min, latents_norm)
        data_max = self._stats_like(self.latent_data_max, latents_norm)
        latents_norm = latents_norm.clamp(min=data_min, max=data_max)
        clipped = self.denormalize_latent(latents_norm)
        return clipped if latents.dtype == torch.float32 else clipped.to(dtype=latents.dtype)

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    @property
    def selected_style_slots(self) -> tuple[int, ...]:
        # Slot identity is fixed while training. Do not synchronize CUDA every batch.
        key = (id(self.style_slot_indices), self.style_slot_indices._version)
        if getattr(self, '_style_slot_cache_key', None) != key:
            self._style_slot_cache = tuple(int(slot) for slot in self.style_slot_indices.detach().cpu().tolist())
            self._style_slot_cache_key = key
        return self._style_slot_cache

    def _flatten_frames(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(-1, dim)
        raise ValueError(f"Expected rank 2 or 3 input, got shape={tuple(x.shape)}")

    def _expand_style(self, style_label: torch.Tensor, n: int) -> torch.Tensor:
        style_label = style_label.reshape(-1)
        if style_label.numel() == n:
            return style_label.long()
        if style_label.numel() == 1:
            return style_label.long().expand(n)
        if n % style_label.numel() == 0:
            repeat = n // style_label.numel()
            return style_label.long().repeat_interleave(repeat)
        raise ValueError(f"Cannot align {style_label.numel()} style labels with batch size {n}")

    def set_style_code_prior(self, prior: torch.Tensor | None) -> None:
        if not self.use_style_code_selector:
            if prior is not None:
                raise RuntimeError("This controller uses StyleEncoder, not StyleCodeSelector.")
            return
        if prior is None:
            self.style_prior_ready.fill_(False)
            self._style_prior_initialized = False
            return
        prior_t = torch.as_tensor(prior, dtype=torch.float32, device=self.style_prior_probs.device)
        if prior_t.dim() != 3:
            raise ValueError(
                f"style code prior must have shape [num_styles, num_slots, num_embeddings], got {tuple(prior_t.shape)}"
            )
        if prior_t.size(0) != self.num_styles or prior_t.size(-1) != int(self.vqvae.quantizer.num_embeddings):
            raise ValueError(
                f"style code prior shape mismatch: expected num_styles={self.num_styles}, "
                f"num_embeddings={int(self.vqvae.quantizer.num_embeddings)}, got {tuple(prior_t.shape)}"
            )
        if prior_t.size(1) == len(self.selected_style_slots):
            prior_selected = prior_t
        elif prior_t.size(1) == int(self.vqvae.num_latents):
            prior_selected = prior_t.index_select(1, self.style_slot_indices.to(device=prior_t.device))
        else:
            raise ValueError(
                f"style code prior num_slots mismatch: expected {len(self.selected_style_slots)} or "
                f"{int(self.vqvae.num_latents)}, got {prior_t.size(1)}"
            )
        prior_selected = prior_selected / prior_selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if tuple(prior_selected.shape) != tuple(self.style_prior_probs.shape):
            raise ValueError(
                f"style code prior final shape mismatch: expected {tuple(self.style_prior_probs.shape)}, "
                f"got {tuple(prior_selected.shape)}"
            )
        self.style_prior_probs.copy_(prior_selected)
        self.style_prior_ready.fill_(True)
        self._style_prior_initialized = True

    def _prior_entropy_by_style(self, *, kl=False):
        prior = self.style_prior_probs
        key = (id(prior), prior._version)
        if getattr(self, '_prior_entropy_cache_key', None) != key:
            with torch.no_grad():
                self._prior_entropy_cache = -(prior * prior.clamp_min(1e-8).log()).sum(-1).mean(-1)
                self._prior_kl_entropy_cache = -(prior * prior.clamp_min(1e-12).log()).sum(-1).mean(-1)
            self._prior_entropy_cache_key = key
        return self._prior_kl_entropy_cache if kl else self._prior_entropy_cache

    def recipe_from_histogram(self, histogram, *, return_logits=False, validated=False):
        """Clip front end. Manual probabilities never pass through this function implicitly."""
        if self.histogram_selector is None:
            raise ValueError('Learned histogram selection requires a clip_recipe checkpoint')
        if histogram.ndim == 2:
            histogram = histogram.unsqueeze(0)
        book = self.vqvae.quantizer.embedding_weight.detach()
        histogram = histogram.to(book)
        if not validated:
            histogram, mask = self.style_selector._normalize_manual_probs(histogram, histogram.size(0))
            if not mask.all():
                raise ValueError('Clip histograms require all selected slots')
        if histogram.ndim != 3 or tuple(histogram.shape[1:]) != (self.style_selector.num_style_slots, book.size(0)):
            raise ValueError('Clip histogram shape must match selected slots and codebook')
        logits = self.histogram_selector(histogram, book, validate=False)
        dense = F.softmax(logits / self.style_selector.temperature, -1)
        # Stable ties agree with the histogram exporter at residual initialization.
        indices = torch.argsort(dense, dim=-1, descending=True, stable=True)[..., :self.style_selector.topk]
        probs = torch.zeros_like(dense).scatter(-1, indices, dense.gather(-1, indices))
        probs = probs / probs.sum(-1, keepdim=True)
        return (probs, logits) if return_logits else probs

    def encode_conditions(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        return_style_details: bool = False,
        clip_histograms: torch.Tensor | None = None,
        clip_histograms_validated: bool = False,
        selector_output: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        traj_curr, motion_curr = self.split_traj_motion(x_curr)
        if self.conditioning_mode in ('recipe', 'clip_recipe') and (style_label is not None or (manual_style_probs is None and clip_histograms is None)):
            raise ValueError('Recipe-only controller requires pure manual probabilities and style_label=None')
        clip_logits = None
        if clip_histograms is not None:
            if style_label is not None or manual_style_probs is not None:
                raise ValueError('Clip histograms and final manual recipes are mutually exclusive')
            if not clip_histograms_validated:
                clip_histograms, mask = self.style_selector._normalize_manual_probs(clip_histograms, x_curr.size(0))
                if not mask.all():
                    raise ValueError('Clip histograms require all selected slots')
            if clip_histograms.size(0) != x_curr.size(0):
                raise ValueError('Clip histogram batch must match conditioning frames')
            manual_style_probs, clip_logits = self.recipe_from_histogram(clip_histograms, return_logits=True, validated=True)
        # motion375 rows are packed as [traj_i, motion_{i-1}], so conditioning on
        # x_curr means "previous motion + current trajectory context" by design.
        if self.use_style_code_selector:
            if selector_output is None:
                selector_output = self.style_selector(
                    # `embedding_weight` is the quantizer-agnostic accessor exposed by both the
                    # library-backed and the legacy codebook implementations (shape [K, code_dim]).
                    codebook_weight=self.vqvae.quantizer.embedding_weight.detach(),
                    style_label=None
                    if style_label is None
                    else self._expand_style(style_label.to(device=x_curr.device), x_curr.size(0)),
                    manual_probs=manual_style_probs,
                    manual_strength=manual_style_strength,
                    batch_size=x_curr.size(0),
                    manual_probs_validated=clip_logits is not None,
                )
            if clip_logits is not None:
                selector_output.update(style_logits=clip_logits, style_logits_are_dense=True,
                                      learned_style_probs=selector_output['style_probs'])
            result = (motion_curr, traj_curr, selector_output["style_feature"])
            if return_style_details:
                return result + (selector_output,)
            return result
        if style_label is None:
            raise ValueError("style_label is required when using the embedding-based StyleEncoder.")
        style_label = self._expand_style(style_label.to(device=x_curr.device), x_curr.size(0))
        result = (motion_curr, traj_curr, self.style_encoder(style_label))
        if return_style_details:
            return result + ({},)
        return result

    def selector_output_all_styles(self) -> dict[str, torch.Tensor]:
        """Run the learned selector once on the full style table (label mode)."""
        labels = torch.arange(self.num_styles, device=self.vqvae.quantizer.embedding_weight.device)
        return self.style_selector(
            codebook_weight=self.vqvae.quantizer.embedding_weight.detach(),
            style_label=labels,
            manual_probs=None,
            manual_strength=1.0,
            batch_size=self.num_styles,
        )

    def _compute_style_regularizers_weighted(
        self,
        *,
        style_counts: torch.Tensor,
        style_probs: torch.Tensor,
        style_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regularizers over per-style rows, weighted by how often each style occurs."""
        hist_loss = style_probs.new_zeros(())
        entropy_loss = style_probs.new_zeros(())
        top1 = style_probs.amax(dim=-1)
        counts = style_counts.to(device=style_probs.device, dtype=style_probs.dtype).reshape(-1, 1)
        top1_prob = (top1 * counts).sum() / (counts.sum().clamp_min(1e-8) * top1.size(-1))
        if not self.use_style_code_selector:
            return hist_loss, entropy_loss, top1_prob
        if self._style_prior_initialized:
            target_prior = self.style_prior_probs.to(device=style_probs.device, dtype=style_probs.dtype)
            hist_loss = style_prior_divergence(
                target_prior,
                style_probs,
                logits=style_logits,
                loss_type=self.style_prior_loss,
                temperature=float(self.style_selector.temperature),
                weights=style_counts,
            )
        entropy_loss = style_code_entropy(style_probs, weights=style_counts)
        return hist_loss, entropy_loss, top1_prob

    def _compute_style_regularizers(
        self,
        *,
        style_label: torch.Tensor,
        style_probs: torch.Tensor,
        style_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """L_prior and L_ent of the paper (Eq. 6-7), with a well-posed L_prior.

        `style_probs` is the TopK-sparsified selector distribution (used for the style token and
        for the entropy term); `style_logits` are the pre-softmax scores, required by the default
        `dense` prior loss (see Utils.style_code_prior.style_prior_divergence for why the literal
        KL against the sparse distribution is not usable).
        """
        hist_loss = style_probs.new_zeros(())
        entropy_loss = style_probs.new_zeros(())
        top1_prob = style_probs.amax(dim=-1).mean()
        if not self.use_style_code_selector:
            return hist_loss, entropy_loss, top1_prob
        expanded_style = self._expand_style(style_label.to(device=style_probs.device), style_probs.size(0))
        if self._style_prior_initialized:
            target_prior = self.style_prior_probs.to(device=style_probs.device, dtype=style_probs.dtype).index_select(
                0,
                expanded_style.long(),
            )
            hist_loss = style_prior_divergence(
                target_prior,
                style_probs,
                logits=style_logits,
                loss_type=self.style_prior_loss,
                temperature=float(self.style_selector.temperature),
            )
        entropy_loss = style_code_entropy(style_probs)
        return hist_loss, entropy_loss, top1_prob

    @torch.no_grad()
    def frame_latent(self, x_frame: torch.Tensor) -> torch.Tensor:
        x_frame = self._flatten_frames(x_frame, self.input_dim)
        z_q, _, _ = self.vqvae.encode(x_frame.float())
        return z_q

    @torch.no_grad()
    def requantize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        # `quantize_only` is the bit-identical frozen nearest-code path without the library's
        # per-call one-hot materialization (5x faster on the rollout feedback step).
        return self.vqvae.quantizer.quantize_only(latents.float())

    def feedback_state(self, latents, *, requantize=None, latent_clip=None, legacy_order=False):
        """Return motion and its exact latent; clipping precedes codebook lookup."""
        requantize = self.inference_requantize if requantize is None else bool(requantize)
        latent_clip = self.inference_latent_clip if latent_clip is None else bool(latent_clip)
        if legacy_order and requantize:
            z = self.requantize_latent(latents)
            motion = self.vqvae.decode(z)
            return motion, self.clip_latent_to_data_bounds(z) if latent_clip else z
        z = self.clip_latent_to_data_bounds(latents) if latent_clip else latents
        if requantize:
            z = self.requantize_latent(z)
        return self.vqvae.decode(z), z

    def history_delta(self, z_prev, history_latent):
        if not self.history_conditioning:
            if history_latent is not None:
                raise ValueError("History supplied to a controller without history conditioning")
            return None
        if history_latent is None:
            return torch.zeros_like(z_prev)
        history_latent = history_latent.to(z_prev)
        if history_latent.shape != z_prev.shape:
            raise ValueError("History latent shape differs from previous latent")
        return z_prev - self.normalize_latent(history_latent)

    def prepare_history_conditions(self, z_prev, e_c, e_traj, e_s, history_delta=None):
        tokens = self.field.prepare_conditions(z_prev, e_c, e_traj, e_s)
        if self.history_delta_proj is not None:
            delta = torch.zeros_like(z_prev) if history_delta is None else history_delta
            previous = tokens[:, :1] + self.history_delta_proj(delta)[:, None]
            tokens = torch.cat((previous, tokens[:, 1:]), dim=1)
        return tokens

    def _predict_velocity(
        self,
        *,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        t: torch.Tensor,
        e_c: torch.Tensor,
        e_traj: torch.Tensor,
        e_s: torch.Tensor,
        condition_tokens: torch.Tensor | None = None,
        history_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition_tokens is None and self.history_conditioning:
            condition_tokens = self.prepare_history_conditions(z_prev, e_c, e_traj, e_s, history_delta)
        field_out = self.field(
            z_t=z_t,
            z_prev=z_prev,
            t=t,
            content=e_c,
            traj=e_traj,
            style=e_s,
            condition_tokens=condition_tokens,
        )
        if self.prediction_mode == "velocity":
            velocity = self.transport_denormalize_latent(field_out)
            remaining = (1.0 - t).clamp_min(self.target_time_eps)
            target_pred = z_t + remaining * velocity
            return velocity, field_out, target_pred
        target_pred = self.transport_denormalize_latent(field_out)
        remaining = (1.0 - t).clamp_min(self.target_time_eps)
        velocity = (target_pred - z_t) / remaining
        return velocity, field_out, target_pred

    def _integrate(
        self,
        *,
        z_prev: torch.Tensor,
        e_c: torch.Tensor,
        e_traj: torch.Tensor,
        e_s: torch.Tensor,
        num_steps: int,
        noise_std: float,
        z0: torch.Tensor | None = None,
        history_delta: torch.Tensor | None = None,
        condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        n = z_prev.size(0)
        if z0 is None:
            z = self.sample_transport_noise_like(
                torch.empty((n, self.latent_dim), device=z_prev.device, dtype=z_prev.dtype),
                noise_std=noise_std,
            )
        else:
            z = z0
        dt = 1.0 / float(num_steps)
        if condition_tokens is None:
            condition_tokens = self.prepare_history_conditions(z_prev, e_c, e_traj, e_s, history_delta)
        for step in range(num_steps):
            t = z.new_full((n, 1), step / float(num_steps))
            velocity, _, _ = self._predict_velocity(
                z_t=z,
                z_prev=z_prev,
                t=t,
                e_c=e_c,
                e_traj=e_traj,
                e_s=e_s,
                condition_tokens=condition_tokens,
            )
            z = z + dt * velocity
        return z

    def compute_loss(
        self,
        *,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        clip_histograms: torch.Tensor | None = None,
        clip_histograms_validated: bool = False,
        prior_labels: torch.Tensor | None = None,
        prev_latent: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        flow_weight: float = 1.0,
        recon_weight: float = 0.0,
        root_recon_weight: float = 0.0,
        recon_inference_path: bool = False,
        recon_anchor_weight: float = 0.0,
        style_hist_weight: float = 0.0,
        style_entropy_weight: float = 0.0,
        noise_std: float = 1.0,
        solver_steps: int = 8,
        return_prediction: bool = False,
        return_endpoint: bool = False,
        return_rollout_latent: bool = False,
        history_latent: torch.Tensor | None = None,
        prediction_no_grad: bool = False,
        compute_style_metrics: bool = True,
        return_per_sample: bool = False,
        style_table: dict[str, torch.Tensor] | None = None,
        flow_low_t_probability: float = 0.,
        flow_low_t_max: float = .2,
    ) -> dict[str, torch.Tensor]:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        x_next = self._flatten_frames(x_next, self.input_dim)
        if style_label is None and clip_histograms is None and (style_hist_weight > 0 or style_entropy_weight > 0):
            raise ValueError('Pure recipes have no learned selector to regularize; use zero prior/entropy weights')
        supervision_labels = prior_labels if clip_histograms is not None else style_label
        if clip_histograms is not None and (style_hist_weight > 0 or style_entropy_weight > 0) and prior_labels is None:
            raise ValueError('Clip selector prior supervision requires prior_labels (never a condition input)')
        traj_next, motion_next = self.split_traj_motion(x_next)
        # The observation (previous motion + its latent) is fed exactly as provided: GT rows
        # during pure teacher forcing, the model's own requantized feedback during rollout.
        x_curr_for_conditions = x_curr
        # Per-style selector fast path (see __init__). It also covers the fixed validation
        # windows (large batches, eval mode): the learned selector is deterministic and
        # label-driven, so running it once on the 90-row style table and gathering rows is
        # mathematically identical to one forward per sample (fp32 reduction order aside).
        # Recipes/manual probabilities parameterize style per sample and never qualify; small
        # batches (interactive/simulation, < 2*num_styles rows) keep the per-sample path.
        dedup_table = None
        dedup_counts = None
        dedup_selector_output = None
        if (self.label_selector_dedup and self.use_style_code_selector
                and self.conditioning_mode == 'label' and style_label is not None
                and manual_style_probs is None and clip_histograms is None
                and self.style_prior_loss == 'dense'
                and float(getattr(self.style_selector, 'dropout_prob', 0.0)) == 0.0):
            flat_labels = style_label.reshape(-1).long()
            if flat_labels.numel() >= 2 * self.num_styles:
                # Rollout training repeats the same style labels across unroll steps; the caller
                # can hoist this table once per window (unrolled_flow_loss does), which is
                # bit-identical because the selector is deterministic with dropout 0.
                table = style_table if style_table is not None else self.selector_output_all_styles()
                dedup_counts = torch.bincount(flat_labels, minlength=self.num_styles)
                dedup_table = table
                dedup_selector_output = dict(table)
                dedup_selector_output['style_feature'] = table['style_feature'].index_select(0, flat_labels)
        e_c, e_traj, e_s, style_details = self.encode_conditions(
            x_curr=x_curr_for_conditions,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            return_style_details=True,
            clip_histograms=clip_histograms,
            clip_histograms_validated=clip_histograms_validated,
            selector_output=dedup_selector_output,
        )

        with torch.no_grad():
            z_prev_raw = self.frame_latent(x_curr) if prev_latent is None else prev_latent.float()
            z1_raw = self.frame_latent(x_next) if target_latent is None else target_latent.float()

        z_prev = self.normalize_latent(z_prev_raw)
        z1 = self.normalize_latent(z1_raw)
        delta_history = self.history_delta(z_prev, history_latent)
        z0 = self.sample_transport_noise_like(z1, noise_std=noise_std)
        t = sample_flow_times(z1, low_probability=flow_low_t_probability if self.training else 0.,
                              low_max=flow_low_t_max)
        z_t = (1.0 - t) * z0 + t * z1
        velocity_target = z1 - z0
        conditions = self.prepare_history_conditions(z_prev, e_c, e_traj, e_s, delta_history)
        velocity_pred, field_out, target_pred = self._predict_velocity(
            z_t=z_t,
            z_prev=z_prev,
            t=t,
            e_c=e_c,
            e_traj=e_traj,
            e_s=e_s,
            history_delta=delta_history,
            condition_tokens=conditions,
        )

        if self.prediction_mode == "velocity":
            flow_loss = (F.mse_loss(velocity_pred, velocity_target) if self.flow_loss_space == 'normalized'
                         else F.mse_loss(field_out, self.transport_normalize_latent(velocity_target)))
        else:
            flow_loss = (F.mse_loss(target_pred, z1) if self.flow_loss_space == "normalized"
                         else F.mse_loss(field_out, self.transport_normalize_latent(z1)))
        style_hist_loss = flow_loss.new_zeros(())
        style_entropy_loss = flow_loss.new_zeros(())
        selector_entropy = flow_loss.new_zeros(())
        conditioning_entropy = flow_loss.new_zeros(())
        style_top1_prob = flow_loss.new_zeros(())
        dense_prior_ce = None
        if self.use_style_code_selector and (compute_style_metrics or style_hist_weight > 0 or style_entropy_weight > 0):
            learned_probs = style_details.get('learned_style_probs')
            selector_probs = learned_probs if learned_probs is not None else style_details['style_probs']
            shared_entropy = (learned_probs is not None and supervision_labels is not None
                              and selector_probs is style_details['style_probs'])
            if not shared_entropy:
                conditioning_entropy = style_code_entropy(style_details['style_probs'])
            style_top1_prob = selector_probs.amax(-1).mean()
            if style_hist_weight > 0.0 and not self._style_prior_initialized:
                raise RuntimeError("Style code prior is required when style_hist_weight > 0.")
            selector_logits = style_details.get("style_logits")
            if not bool(style_details.get("style_logits_are_dense", False)):
                # manual-probability path: no learned logits to supervise
                selector_logits = None
            if learned_probs is not None and supervision_labels is not None and dedup_table is not None:
                style_hist_loss, style_entropy_loss, style_top1_prob = self._compute_style_regularizers_weighted(
                    style_counts=dedup_counts,
                    style_probs=dedup_table['style_probs'],
                    style_logits=dedup_table.get('style_logits'),
                )
            elif learned_probs is not None and supervision_labels is not None:
                style_hist_loss, style_entropy_loss, style_top1_prob = self._compute_style_regularizers(
                    style_label=supervision_labels,
                    style_probs=selector_probs,
                    style_logits=selector_logits,
                )
            if learned_probs is not None and supervision_labels is not None:
                # Shared by both paths: the entropy metric mirrors the optimized entropy, and the
                # dense prior CE is remembered so the KL diagnostic can reuse it.
                selector_entropy = style_entropy_loss
                if shared_entropy:
                    conditioning_entropy = style_entropy_loss
                if self.style_prior_loss == 'dense' and selector_logits is not None:
                    dense_prior_ce = style_hist_loss
            if style_hist_weight == 0:
                style_hist_loss = flow_loss.new_zeros(())
            if style_entropy_weight == 0:
                style_entropy_loss = flow_loss.new_zeros(())

        recon_loss = z1.new_zeros(())
        anchor_loss = z1.new_zeros(())
        quant_error = z1.new_zeros(())
        z_hat = None
        motion_hat = None
        if recon_weight > 0.0 or root_recon_weight > 0.0 or return_prediction or return_rollout_latent:
            # Rollout training feeds the prediction back detached; building an autograd graph
            # that is immediately discarded only wastes time and memory. The flag is honored
            # only when no auxiliary loss consumes the prediction.
            skip_graph = bool(
                prediction_no_grad and (return_prediction or return_rollout_latent)
                and recon_weight == 0.0 and root_recon_weight == 0.0)
            context = torch.no_grad() if skip_graph else nullcontext()
            with context:
                z_hat = self.denormalize_latent(
                    self._integrate(
                        z_prev=z_prev,
                        e_c=e_c,
                        e_traj=e_traj,
                        e_s=e_s,
                        num_steps=solver_steps,
                        noise_std=noise_std,
                        z0=z0,
                        history_delta=delta_history,
                        condition_tokens=conditions.detach() if skip_graph else conditions,
                    )
                )
                if recon_inference_path:
                    # Supervise exactly what inference renders: clip -> requantize -> decode,
                    # the same order as `feedback_state`. The frozen nearest-code lookup is a
                    # value-only operation, so a straight-through estimator keeps the gradient
                    # on the predicted latent:
                    #   z_st = z_clipped + (z_quantized - z_clipped).detach()
                    # has forward value z_quantized (the displayed latent) and identity gradient
                    # d(z_st)/d(z_clipped) = I, so the L1 loss reaches the flow field.
                    z_clipped = self.clip_latent_to_data_bounds(z_hat)
                    z_quantized = self.requantize_latent(z_clipped)
                    # Commitment anchor (standard VQ-VAE recipe). The displayed value
                    # quantize(z_c) is a staircase in z_c: the straight-through gradient alone
                    # walks z_c off the codebook with nothing pulling it back, which showed up
                    # as divergence after a few epochs of fine-tuning. The anchor keeps the
                    # continuous latent near its code without touching the displayed value.
                    if recon_anchor_weight > 0.0:
                        anchor_loss = F.mse_loss(z_clipped, z_quantized.detach())
                    quant_error = (z_clipped - z_quantized).detach().norm(dim=-1).mean()
                    z_hat = z_clipped + (z_quantized - z_clipped).detach()
                    motion_hat = self.vqvae.decode(z_hat, from_indices=False)
                elif return_prediction or recon_weight > 0.0 or root_recon_weight > 0.0:
                    motion_hat = self.vqvae.decode(z_hat, from_indices=False)
        if recon_weight > 0.0:
            recon_loss = F.l1_loss(motion_hat, motion_next)
        root_loss = z1.new_zeros(())
        if root_recon_weight > 0:
            root_loss = F.l1_loss(motion_hat[:, :3], motion_next[:, :3])

        total = (
            flow_weight * flow_loss
            + recon_weight * recon_loss
            + root_recon_weight * root_loss
            + float(recon_anchor_weight) * anchor_loss
            + float(style_hist_weight) * style_hist_loss
            + float(style_entropy_weight) * style_entropy_loss
        )
        result = {
            "flow_time_mean": t.mean(),
            "loss": total,
            "flow_loss": flow_loss,
            "recon_loss": recon_loss,
            "recon_anchor_loss": anchor_loss,
            "recon_quant_error": quant_error,
            "root_recon_loss": root_loss,
            "style_consistency_loss": flow_loss.new_zeros(()),
            "style_hist_loss": style_hist_loss,
            "style_entropy": style_entropy_loss,
            "selector_entropy": selector_entropy,
            "conditioning_entropy": conditioning_entropy,
            "style_top1_prob": style_top1_prob,
            "latent_target_norm": z1.norm(dim=-1).mean(),
            "velocity_pred_norm": velocity_pred.norm(dim=-1).mean(),
            "velocity_target_norm": velocity_target.norm(dim=-1).mean(),
        }
        prior_entropy = flow_loss.new_zeros(())
        if return_per_sample:
            if self.flow_loss_space == 'normalized':
                error = target_pred - z1 if self.prediction_mode == 'target' else velocity_pred - velocity_target
            else:
                error = field_out - self.transport_normalize_latent(z1 if self.prediction_mode == 'target' else velocity_target)
            result['_flow_per_sample'] = error.square().mean(-1)
        prior_kl = flow_loss.new_zeros(())
        coverage = flow_loss.new_zeros(())
        if compute_style_metrics and self.use_style_code_selector and self._style_prior_initialized and supervision_labels is not None:
            if dedup_table is not None:
                # Fast path: metrics are computed over per-style rows, weighted by occurrence.
                counts = dedup_counts.to(device=flow_loss.device, dtype=flow_loss.dtype).reshape(-1, 1)
                prior = self.style_prior_probs
                prior_entropy = ((self._prior_entropy_by_style() * counts.reshape(-1)).sum()
                                 / counts.sum().clamp_min(1e-8))
                if bool(dedup_table.get('style_logits_are_dense', False)):
                    logp = F.log_softmax(dedup_table['style_logits'] / self.style_selector.temperature, dim=-1)
                    per_style_kl = (prior * (prior.clamp_min(1e-12).log() - logp)).sum(-1).mean(-1)
                    prior_kl = (per_style_kl * counts.reshape(-1)).sum() / counts.sum().clamp_min(1e-8)
                coverage = ((prior * (dedup_table['style_probs'] > 0)).sum(-1).mean(-1)
                            * counts.reshape(-1)).sum() / counts.sum().clamp_min(1e-8)
                prior_kl = prior_kl.detach()
                coverage = coverage.detach()
                prior_entropy = prior_entropy.detach()
            else:
                labels = self._expand_style(supervision_labels, x_curr.size(0))
                prior = self.style_prior_probs[labels]
                prior_entropy = self._prior_entropy_by_style()[labels].mean()
                logits = style_details.get("style_logits")
                if bool(style_details.get("style_logits_are_dense", False)):
                    if dense_prior_ce is not None:
                        # KL = CE - H(prior). Both optimized CE and its gradient stay unchanged.
                        prior_kl = dense_prior_ce - self._prior_entropy_by_style(kl=True)[labels].mean()
                    else:
                        logp = F.log_softmax(logits / self.style_selector.temperature, dim=-1)
                        prior_kl = (prior * (prior.clamp_min(1e-12).log() - logp)).sum(-1).mean()
                coverage = (prior * (selector_probs > 0)).sum(-1).mean()
        result.update(style_prior_kl=prior_kl, style_prior_target_entropy=prior_entropy,
                      style_code_coverage=coverage)
        if self.prediction_mode == "target":
            result.update(
                {
                    "target_pred_norm": target_pred.norm(dim=-1).mean(),
                    "target_error": (target_pred - z1).norm(dim=-1).mean(),
                    "target_weight_mean": self.latent_transport_std.to(device=z1.device, dtype=z1.dtype).mean(),
                }
            )
        if return_prediction:
            if z_hat is None or motion_hat is None:
                raise RuntimeError("Prediction tensors were not computed.")
            result.update(
                {
                    "pred_latent": z_hat,
                    "pred_motion": motion_hat,
                    "pred_frame": torch.cat((traj_next, motion_hat), dim=-1),
                }
            )
        if return_rollout_latent:
            result["pred_latent"] = z_hat
        if return_endpoint:
            # A single FM evaluation, for cheap auxiliary losses; no Euler/decode.
            result['endpoint_latent'] = self.denormalize_latent(target_pred)
        return result

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        prev_latent: torch.Tensor | None = None,
        num_steps: int = 8,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
        style_feature: torch.Tensor | None = None,
        history_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if solver != "euler":
            raise ValueError("Minimal flowmatch controller only keeps Euler integration.")
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        if style_feature is None:
            e_c, e_traj, e_s = self.encode_conditions(
                x_curr=x_curr, style_label=style_label, manual_style_probs=manual_style_probs,
                manual_style_strength=manual_style_strength)
        else:
            e_traj, e_c = self.split_traj_motion(x_curr)
            if style_feature.shape != (x_curr.size(0), self.style_dim):
                raise ValueError('Cached style feature shape differs from the current frame batch')
            e_s = style_feature
        if prev_latent is None:
            z_prev = self.normalize_latent(self.frame_latent(x_curr))
        else:
            prev_latent = prev_latent.to(device=x_curr.device, dtype=torch.float32)
            if prev_latent.shape != (x_curr.size(0), self.latent_dim):
                raise ValueError(
                    f"Expected prev_latent shape {(x_curr.size(0), self.latent_dim)}, got {tuple(prev_latent.shape)}"
                )
            z_prev = self.normalize_latent(prev_latent)
        z0_norm = None if z0 is None else self.normalize_latent(z0.float().to(device=x_curr.device))
        z_hat = self._integrate(
            z_prev=z_prev,
            e_c=e_c,
            e_traj=e_traj,
            e_s=e_s,
            num_steps=num_steps,
            noise_std=noise_std,
            z0=z0_norm,
            **({'history_delta': self.history_delta(z_prev, history_latent)} if self.history_conditioning else {}),
        )
        return self.denormalize_latent(z_hat)

    @torch.no_grad()
    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        prev_latent: torch.Tensor | None = None,
        history_latent: torch.Tensor | None = None,
        num_steps: int = 8,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_hat = self.sample_latent(
            x_curr=x_curr,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            prev_latent=prev_latent,
            history_latent=history_latent,
            num_steps=num_steps,
            noise_std=noise_std,
            z0=z0,
            solver=solver,
        )
        motion_hat = self.vqvae.decode(z_hat, from_indices=False)
        return motion_hat, z_hat


def build_flowmatch_controller(
    *,
    vqvae: BasicVQVAE375to291,
    num_styles: int,
    use_style_code_selector: bool = False,
    selected_style_slots: Sequence[int] | None = None,
    style_dim: int = 64,
    style_selector_hidden_dim: int = 128,
    style_selector_topk: int = 4,
    style_selector_temperature: float = 1.0,
    style_proto_normalization: str = 'layernorm',
    style_prior_loss: str = "dense",
    conditioning_mode: str = "label",
    time_embed_dim: int = 64,
    flow_hidden_dim: int = 256,
    flow_layers: int = 2,
    flow_heads: int = 4,
    flow_ffn_dim: int = 1024,
    dropout: float = 0.1,
    prediction_mode: FlowPredictionMode = "velocity",
    target_time_eps: float = 1e-3,
    flow_loss_space: str | None = None,
    inference_requantize: bool = False,
    inference_latent_clip: bool = True,
    label_selector_dedup: bool = True,
    use_motion_token: bool = True,
    field_arch: str = "transformer",
    history_conditioning: bool = False,
) -> FlowMatchVQVAEController:
    return FlowMatchVQVAEController(
        vqvae=vqvae,
        num_styles=num_styles,
        use_style_code_selector=use_style_code_selector,
        selected_style_slots=selected_style_slots,
        style_dim=style_dim,
        style_selector_hidden_dim=style_selector_hidden_dim,
        style_selector_topk=style_selector_topk,
        style_selector_temperature=style_selector_temperature,
        style_proto_normalization=style_proto_normalization,
        style_prior_loss=style_prior_loss,
        conditioning_mode=conditioning_mode,
        time_embed_dim=time_embed_dim,
        flow_hidden_dim=flow_hidden_dim,
        flow_layers=flow_layers,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        dropout=dropout,
        prediction_mode=prediction_mode,
        target_time_eps=target_time_eps,
        flow_loss_space=flow_loss_space,
        inference_requantize=inference_requantize,
        inference_latent_clip=inference_latent_clip,
        label_selector_dedup=label_selector_dedup,
        use_motion_token=use_motion_token,
        field_arch=field_arch,
        history_conditioning=history_conditioning,
    )


def build_flowmatch_controller_from_config(
    config: Mapping[str, object],
    *,
    vqvae: BasicVQVAE375to291,
    num_styles: int,
) -> FlowMatchVQVAEController:
    if int(config.get('conditioning_version',0)) == 2:
        from .clip_recipe_v2 import build_clip_recipe_v2
        return build_clip_recipe_v2(config,vqvae=vqvae,num_styles=num_styles)
    raw_style_slots = config.get("style_slot_indices", None)
    selected_style_slots = None
    if raw_style_slots not in (None, (), []):
        selected_style_slots = tuple(int(slot) for slot in raw_style_slots)
    model = build_flowmatch_controller(
        vqvae=vqvae,
        num_styles=num_styles,
        use_style_code_selector=bool(config.get("use_style_code_selector", False)),
        selected_style_slots=selected_style_slots,
        style_dim=int(config.get("style_dim", 64)),
        style_selector_hidden_dim=int(config.get("style_selector_hidden_dim", 128)),
        style_selector_topk=int(config.get("style_selector_topk", 4)),
        style_selector_temperature=float(config.get("style_selector_temperature", 1.0)),
        style_proto_normalization=str(config.get('style_proto_normalization', 'layernorm')),
        style_prior_loss=str(config.get("style_prior_loss", "dense")),
        conditioning_mode=str(config.get("conditioning_mode", "label")),
        time_embed_dim=int(config.get("time_embed_dim", 64)),
        flow_hidden_dim=int(config.get("flow_hidden_dim", config.get("hidden_dim", 256))),
        flow_layers=int(config.get("flow_layers", config.get("num_layers", 2))),
        flow_heads=int(config.get("flow_heads", 4)),
        flow_ffn_dim=int(config.get("flow_ffn_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        prediction_mode=resolve_flowmatch_prediction_mode(str(config.get("prediction_mode", "velocity"))),
        target_time_eps=float(config.get("target_time_eps", 1e-3)),
        flow_loss_space=config.get("flow_loss_space"),
        inference_requantize=bool(config.get("inference_requantize", False)),
        inference_latent_clip=bool(config.get("inference_latent_clip", True)),
        label_selector_dedup=bool(config.get("label_selector_dedup", True)),
        use_motion_token=bool(config.get("use_motion_token", True)),
        field_arch=str(config.get("field_arch", "transformer")),
        history_conditioning=bool(config.get("history_conditioning", False)),
    )
    model.field.fp32_math_attention = bool(config.get('fp32_math_attention', False))
    return model


__all__ = [
    "ConditionalVelocityField",
    "FlowPredictionMode",
    "FlowMatchVQVAEController",
    "StyleCodeSelector",
    "StyleEncoder",
    "build_flowmatch_controller",
    "build_flowmatch_controller_from_config",
    "resolve_flowmatch_prediction_mode",
]
