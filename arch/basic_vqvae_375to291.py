from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# First channels of the 291-dim motion block: root planar velocity, yaw rate, vertical
# velocity. Mirrors `Utils.motion375.decode_selected_state`.
ROOT_MOTION_DIMS = 3

try:  # pragma: no cover - import guard
    from vector_quantize_pytorch import VectorQuantize as _LibraryVectorQuantize
except ImportError:  # pragma: no cover
    _LibraryVectorQuantize = None

# `threshold_ema_dead_code` counts EMA assignments per batch, not "consecutive steps
# unused". With a 512-sample batch and 32 heads over 2048 codes the expected count per code
# is ~8, so the upstream examples use 2. Anything much above the per-code mean replaces the
# whole codebook every step and it never converges.
LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD = 2
LIBRARY_EXTREME_RESET_FRACTION = 0.5


@dataclass
class QuantizeResult:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    dead_codes_reset: torch.Tensor


class LibraryVectorQuantizer(nn.Module):
    """Adapter that runs `vector_quantize_pytorch.VectorQuantize` behind our VQ interface.

    The (dim, codebook_size, codebook_dim, heads) tuple is chosen so the library reproduces
    the layout this project already trains: a `num_codebooks * code_dim` latent split into
    `num_codebooks` chunks of `code_dim` dims, all quantized against one shared table.

    With `dim == codebook_dim * heads` the library inserts no projections, so `project_in`
    and `project_out` are `Identity` and the module holds zero trainable parameters - the
    codebook is an EMA buffer, exactly as before.

    Keeps the `QuantizeResult` contract (`quantized`, `indices`, `loss`, `codebook_loss`,
    `commitment_loss`, `perplexity`, `dead_codes_reset`) plus `lookup()`, `num_embeddings`,
    `num_codebooks`, `code_dim` and `latent_dim`, all of which the trainers, the VQ target
    cache and the checkpoints rely on.
    """

    def __init__(
        self,
        num_embeddings: int,
        code_dim: int,
        num_codebooks: int,
        commitment_cost: float = 0.25,
        use_rotation_trick: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        dead_code_threshold: int = LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD,
    ):
        super().__init__()
        if _LibraryVectorQuantize is None:  # pragma: no cover
            raise ImportError(
                "vector_quantize_pytorch is required for the quantizer; install it with "
                "`pip install vector-quantize-pytorch`"
            )
        self.num_embeddings = int(num_embeddings)
        self.code_dim = int(code_dim)
        self.num_codebooks = int(num_codebooks)
        self.num_latents = self.num_codebooks  # alias used by the model/checkpoint contract
        self.latent_dim = self.num_codebooks * self.code_dim
        self.commitment_cost = float(commitment_cost)
        self.dead_code_threshold = int(dead_code_threshold)
        self.use_rotation_trick = bool(use_rotation_trick)

        self.vq = _LibraryVectorQuantize(
            dim=self.latent_dim,
            codebook_size=self.num_embeddings,
            codebook_dim=self.code_dim,
            heads=self.num_codebooks,
            separate_codebook_per_head=False,
            decay=float(ema_decay),
            eps=float(ema_eps),
            kmeans_init=bool(kmeans_init),
            kmeans_iters=int(kmeans_iters),
            threshold_ema_dead_code=self.dead_code_threshold,
            commitment_weight=self.commitment_cost,
            rotation_trick=self.use_rotation_trick,
            use_cosine_sim=False,
            ema_update=True,
        )

        # The library replaces expired codes inside its own forward and does not report how
        # many, so count them by wrapping the instance's `replace`. Bound as an instance
        # attribute; if a future version renames it the counter simply stays at zero.
        self._last_dead_resets = 0
        self._extreme_reset_streak = 0
        self._warned_extreme_reset = False
        book = getattr(self.vq, "_codebook", None)
        if book is not None and hasattr(book, "replace"):
            original_replace = book.replace

            def counting_replace(batch_samples, batch_mask=None, seq_mask=None):
                if batch_mask is not None:
                    self._last_dead_resets += int(batch_mask.sum().item())
                return original_replace(batch_samples, batch_mask=batch_mask, seq_mask=seq_mask)

            book.replace = counting_replace

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.vq.codebook

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.size(-1) != self.num_codebooks:
            raise ValueError(
                f"indices last dim mismatch: got {indices.size(-1)}, expected {self.num_codebooks}"
            )
        return self.vq.get_codes_from_indices(indices.long()).reshape(
            *indices.shape[:-1], self.latent_dim
        )

    @staticmethod
    def _perplexity(indices: torch.Tensor, num_embeddings: int, eps: float) -> torch.Tensor:
        counts = torch.bincount(indices.reshape(-1), minlength=num_embeddings).to(torch.float32)
        avg_probs = counts / counts.sum().clamp_min(1.0)
        return torch.exp(-(avg_probs * (avg_probs + eps).log()).sum())

    @torch.no_grad()
    def quantize_only(self,z_bt,*,chunk_rows=64):
        """Frozen nearest-code path: chunked argmin + gather, no one-hot/diagnostics.

        The library forward materializes `F.one_hot(indices, K)` (int64 then cast to fp32)
        even in eval, which for B=512/32 heads/K=2048 is hundreds of MB of pure overhead.
        This path computes the identical nearest code per head (same ordering as the
        library's `argmax(-cdist(...))`, first-index tie break) and gathers the same fp32
        codebook rows, so outputs are bit-identical while cost drops to one small matmul
        per chunk.
        """
        if self.training:raise ValueError('quantize_only requires a frozen eval codebook')
        if z_bt.ndim!=2 or z_bt.shape[-1]!=self.latent_dim:raise ValueError('Expected [B,latent_dim]')
        if chunk_rows<1:raise ValueError('chunk_rows must be >= 1')
        z=z_bt.float()
        batch=z.size(0)
        heads=self.num_codebooks
        dim=self.code_dim
        embed=self.embedding_weight
        x=z.reshape(batch*heads,dim)
        if embed.dim()==2:
            # Single augmented matmul: argmin(-2 x e^T + ||e||^2) omits ||x||^2, which is a
            # per-row constant and cannot change the argmin. Measured ~1.8x faster than the
            # two-pass norm form at B=1024/K=2048; codes bit-identical on 6144 real latents.
            code=torch.empty(x.size(0),dtype=torch.long,device=x.device)
            weight=-2.0*embed.t()
            bias=embed.square().sum(-1)
            step=chunk_rows*heads
            for begin in range(0,x.size(0),step):
                score=torch.addmm(bias[None],x[begin:begin+step],weight)
                code[begin:begin+step]=score.argmin(dim=-1)
            return embed.index_select(0,code).reshape(batch,self.latent_dim)
        # separate codebook per head: same rule, per-head rows
        code=torch.empty(heads,x.size(0)//heads,dtype=torch.long,device=x.device)
        per_head=x.reshape(heads,x.size(0)//heads,dim)
        for head in range(heads):
            table=embed[head]
            table_norm=table.square().sum(-1)
            rows=per_head[head]
            for begin in range(0,rows.size(0),chunk_rows):
                piece=rows[begin:begin+chunk_rows]
                dist=piece.square().sum(-1,keepdim=True)+table_norm[None]-2.0*(piece@table.t())
                code[head,begin:begin+chunk_rows]=dist.argmin(dim=-1)
        gathered=embed.gather(1,code[...,None].expand(-1,-1,dim))
        return gathered.reshape(batch,self.latent_dim)

    def forward(self, z_bt: torch.Tensor) -> QuantizeResult:
        if z_bt.dim() != 2:
            raise ValueError(f"Expected z_bt shape [N, latent_dim], got {tuple(z_bt.shape)}")
        if z_bt.size(-1) != self.latent_dim:
            raise ValueError(
                f"z_bt latent_dim mismatch: got {z_bt.size(-1)}, expected {self.latent_dim}"
            )

        self._last_dead_resets = 0
        quantized, indices, library_loss = self.vq(z_bt)
        dead_codes_reset = z_bt.new_tensor(float(self._last_dead_resets))

        # The first training step legitimately expires the whole table while the codebook is
        # still being k-means initialised, so only complain once the rate stays extreme.
        if self.training and not self._warned_extreme_reset:
            if self._last_dead_resets > LIBRARY_EXTREME_RESET_FRACTION * self.num_embeddings:
                self._extreme_reset_streak += 1
                if self._extreme_reset_streak >= 3:
                    self._warned_extreme_reset = True
                    print(
                        f"[Warn] threshold_ema_dead_code={self.dead_code_threshold} keeps "
                        f"replacing more than {100 * LIBRARY_EXTREME_RESET_FRACTION:.0f}% of the "
                        f"codebook every step ({self._last_dead_resets}/{self.num_embeddings} "
                        f"for 3 steps in a row). This parameter counts EMA assignments per "
                        f"batch, so a value far above the per-code mean reinitialises the "
                        f"codebook continuously and it will never converge; the upstream "
                        f"default is 2."
                    )
            else:
                self._extreme_reset_streak = 0

        perplexity = self._perplexity(indices, self.num_embeddings, self.vq.eps)
        commitment_loss = library_loss / self.commitment_cost if self.commitment_cost > 0 else library_loss
        codebook_loss = z_bt.new_zeros(())
        return QuantizeResult(
            quantized=quantized,
            indices=indices,
            loss=library_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
            dead_codes_reset=dead_codes_reset,
        )


class ControlMLP(nn.Module):
    """
    ControlOperators-style plain MLP:
      Linear -> ELU -> ... -> Linear

    Supports both [B, D] and [B, T, D].
    """

    def __init__(
        self,
        inp: int,
        out: int,
        *,
        hidden: int = 512,
        depth: int = 2,
    ):
        super().__init__()
        if inp < 1:
            raise ValueError(f"inp must be >= 1, got {inp}")
        if out < 1:
            raise ValueError(f"out must be >= 1, got {out}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: list[nn.Module] = []
        in_dim = int(inp)
        hidden = int(hidden)
        for _ in range(int(depth) - 1):
            layers.extend((nn.Linear(in_dim, hidden), nn.ELU()))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, int(out)))

        self.net = nn.Sequential(*layers)
        self.inp = int(inp)
        self.out = int(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.inp:
            raise ValueError(f"Expected last dim {self.inp}, got {x.size(-1)}")
        if x.dim() == 2:
            return self.net(x)
        if x.dim() == 3:
            batch, steps, channels = x.shape
            y = self.net(x.reshape(batch * steps, channels))
            return y.reshape(batch, steps, self.out)
        raise ValueError(
            f"Expected x shape [B,{self.inp}] or [B,T,{self.inp}], got {tuple(x.shape)}"
        )


class ControlSkipMLP(nn.Module):
    """Residual MLP matching ControlOperators ``networks.SkipMLP``."""

    def __init__(
        self,
        inp: int,
        out: int,
        *,
        hidden: int = 512,
        depth: int = 6,
    ):
        super().__init__()
        if inp < 1:
            raise ValueError(f"inp must be >= 1, got {inp}")
        if out < 1:
            raise ValueError(f"out must be >= 1, got {out}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: list[nn.Module] = []
        in_dim = int(inp)
        for _ in range(int(depth) - 1):
            layers.extend((nn.Linear(in_dim, int(hidden)), nn.ELU()))
            in_dim = int(hidden)
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(in_dim, int(out))
        self.inp = int(inp)
        self.output_dim = int(out)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_idx in range(0, len(self.layers), 2):
            h_in = h
            h = self.layers[layer_idx](h)
            h = self.layers[layer_idx + 1](h)
            if h.shape == h_in.shape:
                h = h + h_in
        return self.out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.inp:
            raise ValueError(f"Expected last dim {self.inp}, got {x.size(-1)}")
        if x.dim() == 2:
            return self._forward_flat(x)
        if x.dim() == 3:
            batch, steps, channels = x.shape
            y = self._forward_flat(x.reshape(batch * steps, channels))
            return y.reshape(batch, steps, self.output_dim)
        raise ValueError(
            f"Expected x shape [B,{self.inp}] or [B,T,{self.inp}], got {tuple(x.shape)}"
        )


class ControlSkipCatMLP(nn.Module):
    """Concatenative skip MLP matching ControlOperators ``networks.SkipCatMLP``."""

    def __init__(
        self,
        inp: int,
        out: int,
        *,
        hidden: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        if inp < 1:
            raise ValueError(f"inp must be >= 1, got {inp}")
        if out < 1:
            raise ValueError(f"out must be >= 1, got {out}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers: list[nn.Module] = [nn.Linear(int(inp), int(hidden)), nn.GELU()]
        for _ in range(int(depth) - 1):
            layers.extend(
                (
                    nn.LayerNorm(int(hidden)),
                    nn.Linear(int(hidden) + int(inp), int(hidden)),
                    nn.GELU(),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(int(hidden) + int(inp), int(out))
        self.inp = int(inp)
        self.output_dim = int(out)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers[1](self.layers[0](x))
        for layer_idx in range(2, len(self.layers), 3):
            h = self.layers[layer_idx](h)
            h = torch.cat((h, x), dim=-1)
            h = self.layers[layer_idx + 1](h)
            h = self.layers[layer_idx + 2](h)
        return self.out(torch.cat((h, x), dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.inp:
            raise ValueError(f"Expected last dim {self.inp}, got {x.size(-1)}")
        if x.dim() == 2:
            return self._forward_flat(x)
        if x.dim() == 3:
            batch, steps, channels = x.shape
            y = self._forward_flat(x.reshape(batch * steps, channels))
            return y.reshape(batch, steps, self.output_dim)
        raise ValueError(
            f"Expected x shape [B,{self.inp}] or [B,T,{self.inp}], got {tuple(x.shape)}"
        )


def normalize_control_network_arch(architecture: str) -> str:
    arch = str(architecture).strip().lower().replace("-", "_")
    aliases = {
        "mlp": "mlp",
        "plain": "mlp",
        "control_mlp": "mlp",
        "skip": "skip_mlp",
        "skip_mlp": "skip_mlp",
        "residual": "skip_mlp",
        "residual_mlp": "skip_mlp",
        "skip_cat": "skip_cat_mlp",
        "skipcat": "skip_cat_mlp",
        "skip_cat_mlp": "skip_cat_mlp",
    }
    if arch not in aliases:
        raise ValueError(f"Unsupported control network architecture: {architecture!r}")
    return aliases[arch]


def build_control_network(
    architecture: str,
    *,
    inp: int,
    out: int,
    hidden: int,
    depth: int,
) -> nn.Module:
    arch = normalize_control_network_arch(architecture)
    network_types = {
        "mlp": ControlMLP,
        "skip_mlp": ControlSkipMLP,
        "skip_cat_mlp": ControlSkipCatMLP,
    }
    return network_types[arch](inp=inp, out=out, hidden=hidden, depth=depth)


class BasicVQVAE375to291(nn.Module):
    """
    Single-branch frame-wise VQ-VAE with a configurable encoder input mode.

    Ablation modes:
      - traj_motion: [traj, motion] -> encoder -> z_e -> codebook -> z_q -> decoder -> motion
      - motion_only: motion -> encoder -> z_e -> codebook -> z_q -> decoder -> motion

    Encoder and decoder can use the plain, residual-skip, or concatenative-skip
    ControlOperators MLP. The decoder always reconstructs motion291 directly
    from the quantized latent.
    """

    requires_context_window = False
    model_type = "input_mode_vqvae_375to291"
    decoder_arch = "mlp"

    def __init__(
        self,
        input_dim: int = 375,
        traj_dim: int = 84,
        motion_dim: int = 291,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 0,
        encoder_ffn_dim: int = 0,
        decoder_hidden_dim: int = 512,
        decoder_layers: int = 6,
        num_latents: int = 16,
        code_dim: int = 8,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        use_rotation_trick: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        dead_code_threshold: int = LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD,
        input_mode: str = "traj_motion",
        encoder_arch: str = "mlp",
        decoder_arch: str = "mlp",
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        # Kept for checkpoint/config compatibility with the older VQ-VAE signatures.
        del encoder_heads, encoder_ffn_dim, dropout, max_seq_len, rope_base

        if traj_dim + motion_dim != input_dim:
            raise ValueError(
                f"Expected traj_dim + motion_dim == input_dim, got "
                f"{traj_dim} + {motion_dim} != {input_dim}"
            )
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if decoder_hidden_dim < 1:
            raise ValueError(f"decoder_hidden_dim must be >= 1, got {decoder_hidden_dim}")
        if encoder_layers < 1:
            raise ValueError(f"encoder_layers must be >= 1, got {encoder_layers}")
        if decoder_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1, got {decoder_layers}")
        if num_latents < 1:
            raise ValueError(f"num_latents must be >= 1, got {num_latents}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")

        self.input_dim = int(input_dim)
        self.traj_dim = int(traj_dim)
        self.motion_dim = int(motion_dim)
        self.hidden_dim = int(hidden_dim)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.num_latents = int(num_latents)
        self.code_dim = int(code_dim)
        self.latent_dim = self.num_latents * self.code_dim
        self.use_rotation_trick = bool(use_rotation_trick)
        self.input_mode = self.normalize_input_mode(input_mode)
        self.encoder_arch = normalize_control_network_arch(encoder_arch)
        self.decoder_arch = normalize_control_network_arch(decoder_arch)
        self.encoder_input_dim = self.input_dim if self.input_mode == "traj_motion" else self.motion_dim

        self.encoder = build_control_network(
            self.encoder_arch,
            inp=self.encoder_input_dim,
            out=self.latent_dim,
            hidden=self.hidden_dim,
            depth=encoder_layers,
        )
        self.quantizer = LibraryVectorQuantizer(
            num_embeddings=num_embeddings,
            code_dim=self.code_dim,
            num_codebooks=self.num_latents,
            commitment_cost=commitment_cost,
            use_rotation_trick=self.use_rotation_trick,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = build_control_network(
            self.decoder_arch,
            inp=self.latent_dim,
            out=self.motion_dim,
            hidden=self.decoder_hidden_dim,
            depth=decoder_layers,
        )

    @staticmethod
    def normalize_input_mode(input_mode: str) -> str:
        mode = str(input_mode).strip().lower()
        if mode in {"traj_motion", "traj+motion", "full", "all"}:
            return "traj_motion"
        if mode in {"motion_only", "motion", "motion-only"}:
            return "motion_only"
        raise ValueError(f"Unsupported input_mode: {input_mode!r}")

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.size(-1)}")
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    def _to_btd(
        self,
        x: torch.Tensor,
        *,
        name: str,
        expected_last_dim: int,
    ) -> tuple[torch.Tensor, bool]:
        if x.size(-1) != expected_last_dim:
            raise ValueError(f"Expected {name} last dim {expected_last_dim}, got {x.size(-1)}")
        if x.dim() == 2:
            return x.unsqueeze(1), True
        if x.dim() == 3:
            return x, False
        raise ValueError(
            f"Expected {name} shape [B,{expected_last_dim}] or [B,T,{expected_last_dim}], got {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_from_btd(x: torch.Tensor, squeezed_time: bool) -> torch.Tensor:
        return x[:, 0] if squeezed_time else x

    @staticmethod
    def _ensure_3d(x: torch.Tensor, *, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected {name} dims to be 2 or 3, got {x.dim()}")

    def _select_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.encoder_input_dim:
            return x
        if self.input_mode == "motion_only" and x.size(-1) == self.input_dim:
            return self.split_traj_motion(x)[1]
        raise ValueError(
            f"Expected encoder input last dim {self.encoder_input_dim} for input_mode={self.input_mode}, "
            f"got {x.size(-1)}"
        )

    def _input_to_btd(self, x: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
        x = self._select_encoder_input(x)
        return self._to_btd(x, name=name, expected_last_dim=self.encoder_input_dim)

    def _encode_btd(self, x_btd: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_btd)

    def _quantize_btd(self, z_e_btd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, QuantizeResult]:
        if z_e_btd.dim() != 3 or z_e_btd.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected z_e_btd shape [B,T,{self.latent_dim}], got {tuple(z_e_btd.shape)}"
            )
        batch, steps, _ = z_e_btd.shape
        q = self.quantizer(z_e_btd.reshape(batch * steps, self.latent_dim))
        z_q_btd = q.quantized.reshape(batch, steps, self.latent_dim)
        indices_btd = q.indices.reshape(batch, steps, self.num_latents)
        return z_q_btd, indices_btd, q

    def _decode_btd(self, latents_btd: torch.Tensor) -> torch.Tensor:
        if latents_btd.dim() != 3 or latents_btd.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latents_btd shape [B,T,{self.latent_dim}], got {tuple(latents_btd.shape)}"
            )
        return self.decoder(latents_btd)

    def _resolve_latents_btd(
        self,
        latents: torch.Tensor,
        *,
        from_indices: bool,
        batch: int,
        steps: int,
    ) -> torch.Tensor:
        if from_indices:
            if latents.dim() == 2:
                expected = (batch, self.num_latents)
                if tuple(latents.shape) != expected:
                    raise ValueError(f"Expected indices shape {expected}, got {tuple(latents.shape)}")
                latents = latents.unsqueeze(1)
            elif latents.dim() == 3:
                expected = (batch, steps, self.num_latents)
                if tuple(latents.shape) != expected:
                    raise ValueError(f"Expected indices shape {expected}, got {tuple(latents.shape)}")
            else:
                raise ValueError(f"Expected indices rank 2 or 3, got {tuple(latents.shape)}")
            return self.quantizer.lookup(latents.long())

        if latents.dim() == 2:
            expected = (batch, self.latent_dim)
            if tuple(latents.shape) != expected:
                raise ValueError(f"Expected latents shape {expected}, got {tuple(latents.shape)}")
            return latents.unsqueeze(1)
        if latents.dim() == 3:
            expected = (batch, steps, self.latent_dim)
            if tuple(latents.shape) != expected:
                raise ValueError(f"Expected latents shape {expected}, got {tuple(latents.shape)}")
            return latents
        raise ValueError(f"Expected latents rank 2 or 3, got {tuple(latents.shape)}")

    def encode_pre_vq(self, x: torch.Tensor) -> torch.Tensor:
        x_btd, squeezed_time = self._input_to_btd(x, name="x")
        z_e_btd = self._encode_btd(x_btd)
        return self._restore_from_btd(z_e_btd, squeezed_time)

    def forward(self, x: torch.Tensor, return_details: bool = False, *, continuous_decode: bool = False):
        x_btd, squeezed_time = self._input_to_btd(x, name="x")
        z_e_btd = self._encode_btd(x_btd)
        z_q_btd, indices_btd, q = self._quantize_btd(z_e_btd)
        motion_hat = self._decode_btd(z_q_btd)
        # Decoding the unquantized latent costs one extra decoder pass, so it is opt-in and
        # only used by the code-alignment loss.
        continuous_hat = self._decode_btd(z_e_btd) if continuous_decode else None

        motion_hat = self._restore_from_btd(motion_hat, squeezed_time)
        if continuous_hat is not None:
            continuous_hat = self._restore_from_btd(continuous_hat, squeezed_time)
        z_e = self._restore_from_btd(z_e_btd, squeezed_time)
        z_q = self._restore_from_btd(z_q_btd, squeezed_time)
        indices = self._restore_from_btd(indices_btd, squeezed_time)

        if return_details:
            return {
                "input_mode": self.input_mode,
                "motion_hat": motion_hat,
                "motion_hat_continuous": continuous_hat,
                "latent": z_e,
                "quantized_latent": z_q,
                "indices": indices,
                "vq_loss": q.loss,
                "codebook_loss": q.codebook_loss,
                "commitment_loss": q.commitment_loss,
                "perplexity": q.perplexity,
                "dead_codes_reset": q.dead_codes_reset,
            }
        return motion_hat, q.loss, q.perplexity

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_btd, squeezed_time = self._input_to_btd(x, name="x")
        z_q, indices, result = self._quantize_btd(self._encode_btd(x_btd))
        return (self._restore_from_btd(z_q, squeezed_time),
                self._restore_from_btd(indices, squeezed_time), result.perplexity)

    def decode_step(
        self,
        latents: torch.Tensor,
        *,
        from_indices: bool = False,
    ) -> torch.Tensor:
        batch = latents.shape[0]
        latents_btd = self._resolve_latents_btd(
            latents,
            from_indices=from_indices,
            batch=batch,
            steps=1,
        )
        return self._decode_btd(latents_btd)[:, 0]

    def decode(
        self,
        latents: torch.Tensor,
        from_indices: bool = False,
    ) -> torch.Tensor:
        squeezed_time = False
        if latents.dim() == 2:
            batch = latents.size(0)
            steps = 1
            squeezed_time = True
        elif latents.dim() == 3:
            batch, steps, _ = latents.shape
        else:
            raise ValueError(f"Expected latents rank 2 or 3, got {tuple(latents.shape)}")
        latents_btd = self._resolve_latents_btd(
            latents,
            from_indices=from_indices,
            batch=batch,
            steps=steps,
        )
        motion_hat = self._decode_btd(latents_btd)
        return self._restore_from_btd(motion_hat, squeezed_time)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_details=True)["motion_hat"]

    @staticmethod
    def _loss_by_type(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["l1", "mse", "smooth_l1"],
    ) -> torch.Tensor:
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def compute_loss(
        self,
        x: torch.Tensor,
        target_motion: torch.Tensor | None = None,
        recon_type: Literal["l1", "mse", "smooth_l1"] = "l1",
        temporal_loss_type: Literal["l1", "mse", "smooth_l1"] | None = None,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        vel_weight: float = 0.0,
        acc_weight: float = 0.0,
        latent_l1_weight: float = 0.0,
        latent_vel_weight: float = 0.0,
        root_weight: float = 0.0,
        code_align_weight: float = 0.0,
        latent_margin_weight: float = 0.0,
        roundtrip_weight: float = 0.0,
        latent_margin_target: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        if code_align_weight < 0.0:
            raise ValueError(f"code_align_weight must be >= 0, got {code_align_weight}")
        if latent_margin_weight < 0.0:
            raise ValueError(f"latent_margin_weight must be >= 0, got {latent_margin_weight}")
        if roundtrip_weight < 0.0:
            raise ValueError(f"roundtrip_weight must be >= 0, got {roundtrip_weight}")
        details = self.forward(x, return_details=True,
                               continuous_decode=code_align_weight > 0.0 or roundtrip_weight > 0.0)
        pred = details["motion_hat"]
        latent = details["latent"]

        if target_motion is None:
            if x.size(-1) == self.input_dim:
                _, target_motion = self.split_traj_motion(x)
            elif x.size(-1) == self.motion_dim:
                target_motion = x
            else:
                raise ValueError(
                    f"Could not infer target_motion from input with last dim {x.size(-1)}"
                )
        if target_motion.shape != pred.shape:
            raise ValueError(
                f"target_motion shape mismatch: expected {tuple(pred.shape)}, got {tuple(target_motion.shape)}"
            )

        if temporal_loss_type is None:
            temporal_loss_type = recon_type

        recon_loss = self._loss_by_type(pred, target_motion, recon_type)

        pred_bt = self._ensure_3d(pred, name="pred")
        target_bt = self._ensure_3d(target_motion, name="target_motion")
        latent_bt = self._ensure_3d(latent, name="latent")

        vel_loss = pred.new_zeros(())
        acc_loss = pred.new_zeros(())
        latent_l1_loss = pred.new_zeros(())
        latent_vel_loss = pred.new_zeros(())
        root_loss = pred.new_zeros(())

        if vel_weight > 0.0 and pred_bt.size(1) >= 2:
            pred_vel = pred_bt[:, 1:] - pred_bt[:, :-1]
            target_vel = target_bt[:, 1:] - target_bt[:, :-1]
            vel_loss = self._loss_by_type(pred_vel, target_vel, temporal_loss_type)

        # Second-order (acceleration) term. The first-order term above matches per-step
        # motion, which still leaves per-frame output noise: measured on the quantized path
        # the second-difference error is 4x the autoencoder path's (0.055 vs 0.013 in
        # normalized units), and that is what the jitter/F.S. metrics see (the 32 head codes
        # change between adjacent frames on 100% of steps). Penalising |d2 pred - d2 target|
        # attacks exactly that band. Needs >= 3 frames per window to be defined.
        if acc_weight > 0.0 and pred_bt.size(1) >= 3:
            pred_acc = pred_bt[:, 2:] - 2.0 * pred_bt[:, 1:-1] + pred_bt[:, :-2]
            target_acc = target_bt[:, 2:] - 2.0 * target_bt[:, 1:-1] + target_bt[:, :-2]
            acc_loss = self._loss_by_type(pred_acc, target_acc, temporal_loss_type)

        # Dedicated root-trajectory term. The first ROOT_MOTION_DIMS channels of the motion
        # block are the root planar velocity, yaw rate and vertical velocity (see
        # Utils.motion375.decode_selected_state), so they are already a velocity signal and
        # this L1 is a per-step root velocity error. Averaged over all 291 motion dims the
        # root would only receive 3/291 of the reconstruction gradient, yet its error is the
        # one that integrates into trajectory drift.
        if root_weight > 0.0:
            root_loss = self._loss_by_type(
                pred_bt[..., :ROOT_MOTION_DIMS],
                target_bt[..., :ROOT_MOTION_DIMS],
                recon_type,
            )

        if latent_l1_weight > 0.0:
            latent_l1_loss = torch.mean(torch.abs(latent))

        if latent_vel_weight > 0.0 and latent_bt.size(1) >= 2:
            latent_delta = latent_bt[:, 1:] - latent_bt[:, :-1]
            latent_vel_loss = self._loss_by_type(
                latent_delta,
                torch.zeros_like(latent_delta),
                temporal_loss_type,
            )

        # Continuous/discrete code alignment. `latent_code_gap` is the plain squared gap
        # between the encoder output and its nearest code (the quantity the commitment term
        # penalizes); `code_align_loss` additionally asks the decoder to reproduce the
        # *discrete* path from the continuous latent, so the two code paths stay
        # interchangeable instead of only agreeing in latent space.
        latent_code_gap = torch.mean((latent - details["quantized_latent"]).square())
        code_align_loss = pred.new_zeros(())
        if code_align_weight > 0.0:
            continuous_hat = details["motion_hat_continuous"]
            if continuous_hat is None:
                raise RuntimeError("code_align_weight requires the continuous decode")
            code_align_loss = torch.mean((continuous_hat - pred.detach()).square())

        # Boundary margin: the measured failure mode of this codec is that the encoder output
        # lives almost ON its Voronoi boundary (median exact boundary distance 0.011 vs cell
        # spacing 0.154), so ANY equal-norm disturbance flips the code. Push each encoder
        # output away from the boundary with a hinge; the codebook is an EMA buffer (no grad)
        # so this trains the encoder only. Exact signed distance toward the second-nearest
        # code: (||z-c2||^2 - ||z-c1||^2) / (2 ||c1-c2||).
        latent_margin_loss = pred.new_zeros(())
        boundary_distance = pred.new_zeros(())
        if latent_margin_weight > 0.0:
            embed = self.quantizer.embedding_weight.detach()
            if embed.dim() == 3 and embed.size(0) == 1:
                embed = embed[0]
            z_flat = latent_bt.reshape(-1, self.code_dim)
            # Chunked nearest/second-nearest search: [B*T*32, 2048] distance matrices otherwise
            # dominate activation memory at batch 512.
            idx1 = torch.empty(z_flat.size(0), dtype=torch.long, device=z_flat.device)
            d1 = torch.empty(z_flat.size(0), device=z_flat.device)
            d2 = torch.empty_like(d1)
            idx2 = torch.empty_like(idx1)
            for begin in range(0, z_flat.size(0), 8192):
                end = min(begin + 8192, z_flat.size(0))
                dist = torch.cdist(z_flat[begin:end].float(), embed.float())
                d1[begin:end], idx1[begin:end] = dist.min(dim=-1)
                # Non-inplace masking: scatter_ would bump `dist`'s version counter and break
                # the autograd graph that cdist recorded.
                masked = dist.masked_fill(
                    torch.nn.functional.one_hot(idx1[begin:end], dist.size(-1)).bool(),
                    float('inf'))
                d2[begin:end], idx2[begin:end] = masked.min(dim=-1)
            c1 = embed[idx1]
            c2 = embed[idx2]
            gap12 = (c1 - c2).norm(dim=-1).clamp_min(1e-9)
            signed = (d2.square() - d1.square()) / (2.0 * gap12)
            boundary_distance = signed.mean()
            latent_margin_loss = torch.relu(float(latent_margin_target) - signed).mean()

        # Roundtrip consistency: feed the decoder's own output back through the encoder and
        # ask it to return to the code it came from (the controller's closed loop does exactly
        # this re-encode). Supervising towards the quantized code (not the continuous encoder
        # output) directly minimizes out-of-cell drift, i.e. code flips under self-feedback.
        roundtrip_loss = pred.new_zeros(())
        roundtrip_drift = pred.new_zeros(())
        if roundtrip_weight > 0.0:
            if self.input_mode == "traj_motion":
                feedback = torch.cat((x[..., : self.traj_dim], pred), dim=-1)
            else:
                feedback = pred
            z_rt = self.encoder(feedback.reshape(-1, self.encoder_input_dim))
            target_code = details["quantized_latent"].detach().reshape(-1, self.latent_dim)
            roundtrip_drift = (z_rt - target_code).norm(dim=-1).mean()
            roundtrip_loss = (z_rt - target_code).square().mean()

        total = (
            recon_weight * recon_loss
            + code_align_weight * code_align_loss
            + latent_margin_weight * latent_margin_loss
            + roundtrip_weight * roundtrip_loss
            + vq_weight * details["vq_loss"]
            + vel_weight * vel_loss
            + acc_weight * acc_loss
            + root_weight * root_loss
            + latent_l1_weight * latent_l1_loss
            + latent_vel_weight * latent_vel_loss
        )
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "vq_loss": details["vq_loss"],
            "vel_loss": vel_loss,
            "acc_loss": acc_loss,
            "root_loss": root_loss,
            "latent_l1_loss": latent_l1_loss,
            "latent_vel_loss": latent_vel_loss,
            "codebook_loss": details["codebook_loss"],
            "commitment_loss": details["commitment_loss"],
            "code_align_loss": code_align_loss,
            "latent_margin_loss": latent_margin_loss,
            "boundary_distance": boundary_distance,
            "roundtrip_loss": roundtrip_loss,
            "roundtrip_drift": roundtrip_drift,
            "latent_code_gap": latent_code_gap,
            "perplexity": details["perplexity"],
            "dead_codes_reset": details["dead_codes_reset"],
        }


__all__ = [
    "BasicVQVAE375to291",
    "LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD",
    "LibraryVectorQuantizer",
    "ControlMLP",
    "ControlSkipCatMLP",
    "ControlSkipMLP",
    "QuantizeResult",
    "build_control_network",
    "normalize_control_network_arch",
]
