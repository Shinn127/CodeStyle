from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from arch import BasicVQVAE375to291
from Utils.reproducibility import validate_fingerprints, file_sha256


VQ_TARGET_CACHE_FORMAT = "vq_target_cache_v2"
VQ_TARGET_CACHE_META_FILE = "meta.json"
VQ_TARGET_CACHE_INDICES_FILE_U16 = "indices.u16"
VQ_TARGET_CACHE_INDICES_FILE_I32 = "indices.i32"


def resolve_vq_indices_dtype(num_embeddings: int) -> tuple[np.dtype, str, str]:
    if num_embeddings < 1:
        raise ValueError(f"num_embeddings must be >= 1, got {num_embeddings}")
    if num_embeddings <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16), "uint16", VQ_TARGET_CACHE_INDICES_FILE_U16
    return np.dtype(np.int32), "int32", VQ_TARGET_CACHE_INDICES_FILE_I32


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


@dataclass(frozen=True)
class VQTargetCacheMeta:
    db: Path
    vqvae_ckpt: Path
    norm_stats: Path
    num_frames: int
    feature_dim: int
    num_codebooks: int
    code_dim: int
    latent_dim: int
    num_embeddings: int
    indices_dtype: str
    indices_file: str
    format: str = VQ_TARGET_CACHE_FORMAT
    fingerprints: dict | None = None
    runtime: dict | None = None
    indices_sha256: str | None = None
    complete: bool = True

    def to_json_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "db": str(self.db),
            "vqvae_ckpt": str(self.vqvae_ckpt),
            "norm_stats": str(self.norm_stats),
            "num_frames": int(self.num_frames),
            "feature_dim": int(self.feature_dim),
            "num_codebooks": int(self.num_codebooks),
            "code_dim": int(self.code_dim),
            "latent_dim": int(self.latent_dim),
            "num_embeddings": int(self.num_embeddings),
            "indices_dtype": self.indices_dtype,
            "indices_file": self.indices_file,
            "fingerprints": self.fingerprints,
            "runtime": self.runtime,
            "indices_sha256": self.indices_sha256,
            "complete": self.complete,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> "VQTargetCacheMeta":
        fmt = str(payload.get("format", ""))
        if fmt not in ("vq_target_cache_v1", VQ_TARGET_CACHE_FORMAT):
            raise ValueError(
                f"Unsupported VQ target cache format: {fmt!r}, expected {VQ_TARGET_CACHE_FORMAT!r}"
            )
        return cls(
            db=_resolve_path(str(payload["db"])),
            vqvae_ckpt=_resolve_path(str(payload["vqvae_ckpt"])),
            norm_stats=_resolve_path(str(payload["norm_stats"])),
            num_frames=int(payload["num_frames"]),
            feature_dim=int(payload.get("feature_dim", 375)),
            num_codebooks=int(payload["num_codebooks"]),
            code_dim=int(payload["code_dim"]),
            latent_dim=int(payload["latent_dim"]),
            num_embeddings=int(payload["num_embeddings"]),
            indices_dtype=str(payload["indices_dtype"]),
            indices_file=str(payload["indices_file"]),
            format=fmt,
            fingerprints=payload.get("fingerprints"), runtime=payload.get("runtime"),
            indices_sha256=payload.get("indices_sha256"),
            complete=bool(payload.get("complete", fmt == "vq_target_cache_v1")),
        )


def load_vq_target_cache_meta(cache_dir: str | Path) -> VQTargetCacheMeta:
    cache_dir = _resolve_path(cache_dir)
    meta_path = cache_dir / VQ_TARGET_CACHE_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"VQ target cache meta not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return VQTargetCacheMeta.from_json_dict(payload)


def save_vq_target_cache_meta(cache_dir: str | Path, meta: VQTargetCacheMeta) -> Path:
    cache_dir = _resolve_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / VQ_TARGET_CACHE_META_FILE
    temporary = meta_path.with_suffix(".json.partial")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(meta.to_json_dict(), f, ensure_ascii=False, indent=2)
    temporary.replace(meta_path)
    return meta_path


class VQTargetCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = _resolve_path(cache_dir)
        self.meta = load_vq_target_cache_meta(self.cache_dir)
        if not self.meta.complete:
            raise ValueError("VQ target cache is incomplete")

        if self.meta.feature_dim != 375:
            raise ValueError(
                f"Unsupported feature_dim in VQ target cache: {self.meta.feature_dim}, expected 375"
            )

        dtype_map = {
            "uint16": np.uint16,
            "int32": np.int32,
        }
        if self.meta.indices_dtype not in dtype_map:
            raise ValueError(f"Unsupported cache indices_dtype: {self.meta.indices_dtype}")
        self.indices_dtype = np.dtype(dtype_map[self.meta.indices_dtype])
        self.indices_path = self.cache_dir / self.meta.indices_file
        if not self.indices_path.exists():
            raise FileNotFoundError(f"VQ target cache indices file not found: {self.indices_path}")
        expected_bytes = self.meta.num_frames * self.meta.num_codebooks * self.indices_dtype.itemsize
        if self.indices_path.stat().st_size != expected_bytes:
            raise ValueError("VQ cache byte length does not match its metadata")

        self.indices = np.memmap(
            self.indices_path,
            dtype=self.indices_dtype,
            mode="r",
            shape=(self.meta.num_frames, self.meta.num_codebooks),
        )

    def validate(
        self,
        *,
        db: Path | None = None,
        vqvae_ckpt: Path | None = None,
        norm_stats: Path | None = None,
        vqvae: BasicVQVAE375to291 | None = None,
        fingerprints: dict | None = None,
    ) -> None:
        if fingerprints is not None:
            validate_fingerprints(self.meta.fingerprints, fingerprints)
            if not self.meta.indices_sha256 or file_sha256(self.indices_path) != self.meta.indices_sha256:
                raise ValueError("VQ target cache indices content checksum mismatch")
        if fingerprints is None and db is not None and self.meta.db != _resolve_path(db):
            raise ValueError(
                f"VQ target cache DB mismatch: cache={self.meta.db}, requested={_resolve_path(db)}"
            )
        if fingerprints is None and vqvae_ckpt is not None and self.meta.vqvae_ckpt != _resolve_path(vqvae_ckpt):
            raise ValueError(
                f"VQ target cache checkpoint mismatch: cache={self.meta.vqvae_ckpt}, "
                f"requested={_resolve_path(vqvae_ckpt)}"
            )
        if fingerprints is None and norm_stats is not None and self.meta.norm_stats != _resolve_path(norm_stats):
            raise ValueError(
                f"VQ target cache norm_stats mismatch: cache={self.meta.norm_stats}, "
                f"requested={_resolve_path(norm_stats)}"
            )
        if vqvae is not None:
            if int(vqvae.num_latents) != self.meta.num_codebooks:
                raise ValueError(
                    f"VQ target cache num_codebooks mismatch: cache={self.meta.num_codebooks}, "
                    f"model={int(vqvae.num_latents)}"
                )
            if int(vqvae.code_dim) != self.meta.code_dim:
                raise ValueError(
                    f"VQ target cache code_dim mismatch: cache={self.meta.code_dim}, "
                    f"model={int(vqvae.code_dim)}"
                )
            if int(vqvae.latent_dim) != self.meta.latent_dim:
                raise ValueError(
                    f"VQ target cache latent_dim mismatch: cache={self.meta.latent_dim}, "
                    f"model={int(vqvae.latent_dim)}"
                )
            if int(vqvae.quantizer.num_embeddings) != self.meta.num_embeddings:
                raise ValueError(
                    f"VQ target cache num_embeddings mismatch: cache={self.meta.num_embeddings}, "
                    f"model={int(vqvae.quantizer.num_embeddings)}"
                )

    def lookup_indices(self, frame_ids: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(frame_ids, torch.Tensor):
            frame_ids_np = frame_ids.detach().to(device="cpu", dtype=torch.long).numpy().reshape(-1)
        else:
            frame_ids_np = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
        if frame_ids_np.size == 0:
            return np.empty((0, self.meta.num_codebooks), dtype=self.indices_dtype)
        if frame_ids_np.min() < 0 or frame_ids_np.max() >= self.meta.num_frames:
            raise IndexError(
                f"frame_ids out of range for VQ target cache: "
                f"min={int(frame_ids_np.min())}, max={int(frame_ids_np.max())}, "
                f"num_frames={self.meta.num_frames}"
            )
        return np.asarray(self.indices[frame_ids_np])

    @torch.no_grad()
    def lookup_latents(
        self,
        frame_ids: torch.Tensor | np.ndarray,
        *,
        vqvae: BasicVQVAE375to291,
        device: torch.device,
    ) -> torch.Tensor:
        indices_np = self.lookup_indices(frame_ids)
        indices_t = torch.from_numpy(indices_np.astype(np.int64, copy=False)).to(device=device)
        return vqvae.quantizer.lookup(indices_t)


__all__ = [
    "VQTargetCache",
    "VQTargetCacheMeta",
    "VQ_TARGET_CACHE_FORMAT",
    "VQ_TARGET_CACHE_INDICES_FILE_I32",
    "VQ_TARGET_CACHE_INDICES_FILE_U16",
    "VQ_TARGET_CACHE_META_FILE",
    "load_vq_target_cache_meta",
    "resolve_vq_indices_dtype",
    "save_vq_target_cache_meta",
]
