"""
Precompute per-frame VQ target indices for FlowMatch training with the input-mode VQ model.

Output directory layout:
  <out_dir>/
    meta.json
    indices.u16 or indices.i32

The cache stores discrete VQ codebook indices for every frame after applying the
same normalization stats used by the frozen VQ encoder. During flow training,
these indices can be looked up and converted back to quantized latents cheaply,
avoiding repeated encoder passes for every next-frame target.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import load_motion375_metadata
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.reproducibility import artifact_fingerprints, file_sha256
from Utils.runtime import (
    MEMMAP_X_FILE,
    configure_torch_runtime,
    load_norm_stats,
    resolve_device,
    set_seed,
    require_unmirrored_db,
)
from Utils.vq_target_cache import (
    VQTargetCacheMeta,
    resolve_vq_indices_dtype,
    save_vq_target_cache_meta,
)


class _NormalizedFrameChunkDataset(Dataset):
    def __init__(
        self,
        *,
        x_path: Path,
        num_frames: int,
        chunk_rows: int,
        mean: np.ndarray,
        std: np.ndarray,
        feature_begin: int,
    ) -> None:
        self.x_path = x_path
        self.num_frames = int(num_frames)
        self.chunk_rows = int(chunk_rows)
        self.feature_begin = int(feature_begin)
        self.mean = np.asarray(mean[self.feature_begin :], dtype=np.float32)
        self.std = np.asarray(std[self.feature_begin :], dtype=np.float32)
        self._x_source: np.memmap | None = None

    def __len__(self) -> int:
        return (self.num_frames + self.chunk_rows - 1) // self.chunk_rows

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x_source"] = None
        return state

    def _ensure_open(self) -> np.memmap:
        if self._x_source is None:
            self._x_source = np.memmap(
                self.x_path,
                dtype=np.float32,
                mode="r",
                shape=(self.num_frames, 375),
            )
        return self._x_source

    def __getitem__(self, chunk_idx: int) -> tuple[int, torch.Tensor]:
        start = int(chunk_idx) * self.chunk_rows
        end = min(start + self.chunk_rows, self.num_frames)
        source = self._ensure_open()
        x = np.array(source[start:end, self.feature_begin :], dtype=np.float32, copy=True)
        np.subtract(x, self.mean, out=x)
        np.divide(x, self.std, out=x)
        return start, torch.from_numpy(x)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Precompute per-frame input-mode VQ target cache from memmap DB")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["traj_motion", "motion_only"],
        default="traj_motion",
        help="Used only to resolve mode-specific default checkpoint/cache paths",
    )
    parser.add_argument(
        "--network-arch",
        choices=["mlp", "skip_mlp", "skip_cat_mlp"],
        default="skip_cat_mlp",
        help="Used only to resolve architecture-specific default checkpoint/cache paths",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=script_dir / "database_100sty_375_memmap_nomirror",
        help="Input memmap directory",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_MODE_375to291_32_4_2048" / "best.pt",
        help="Frozen VQ checkpoint",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz with mean/std. Default: sibling of VQ checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "cache" / "vq_targets_vqvae_MODE_375to291_32_4_2048",
        help="Output cache directory",
    )
    parser.add_argument("--chunk-rows", type=int, default=4096, help="Frames encoded per chunk")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU workers for memmap read/normalization")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage normalized chunks in pinned memory before CUDA transfer",
    )
    parser.add_argument("--log-interval", type=int, default=25, help="Progress log interval in chunks")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA")
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default="highest",
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument('--verify-chunks', type=int, default=3, help='Same-device/precision/chunk deterministic readback checks')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parents[1]
    default_ckpt = script_dir / "checkpoints" / "vqvae_MODE_375to291_32_4_2048" / "best.pt"
    default_out_dir = script_dir / "cache" / "vq_targets_vqvae_MODE_375to291_32_4_2048"
    if args.vqvae_ckpt == default_ckpt:
        args.vqvae_ckpt = (
            script_dir
            / "checkpoints"
            / f"vqvae_{args.input_mode}_{args.network_arch}_375to291_32_4_2048"
            / "best.pt"
        )
    if args.out_dir == default_out_dir:
        args.out_dir = (
            script_dir
            / "cache"
            / f"vq_targets_vqvae_{args.input_mode}_{args.network_arch}_375to291_32_4_2048"
        )
    args.db = args.db.expanduser().resolve()
    args.vqvae_ckpt = args.vqvae_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    require_unmirrored_db(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"--vqvae-ckpt not found: {args.vqvae_ckpt}")
    if args.norm_stats is None:
        args.norm_stats = args.vqvae_ckpt.parent / "norm_stats.npz"
    args.norm_stats = args.norm_stats.expanduser().resolve()
    if args.chunk_rows < 1:
        raise ValueError(f"--chunk-rows must be >= 1, got {args.chunk_rows}")
    if args.verify_chunks < 0:
        raise ValueError('--verify-chunks must be non-negative')
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    if args.prefetch_factor < 1:
        raise ValueError(f"--prefetch-factor must be >= 1, got {args.prefetch_factor}")
    if args.log_interval < 1:
        raise ValueError(f"--log-interval must be >= 1, got {args.log_interval}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    mean, std = load_norm_stats(args.norm_stats)

    db_meta = load_motion375_metadata(args.db)
    num_frames = db_meta.num_frames
    feature_dim = db_meta.feature_dim
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames in {args.db / MEMMAP_META_FILE}: {num_frames}")
    if feature_dim != 375:
        raise ValueError(f"Invalid feature_dim in {args.db / MEMMAP_META_FILE}: {feature_dim}")

    vq_model = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)
    vq_model.eval()
    indices_dtype, indices_dtype_name, indices_file = resolve_vq_indices_dtype(vq_model.quantizer.num_embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices_path = args.out_dir / indices_file
    meta_path = args.out_dir / "meta.json"
    if not args.overwrite:
        for path in (indices_path, meta_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace.")

    fingerprints = artifact_fingerprints(args.db, args.vqvae_ckpt, args.norm_stats)
    import platform
    from importlib.metadata import version
    runtime = {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
               "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "cuda": torch.version.cuda, "vector_quantize_pytorch": version('vector-quantize-pytorch'),
               "matmul_precision": args.matmul_precision, "tf32": args.tf32,
               "chunk_rows": args.chunk_rows, "dtype": "float32"}
    incomplete = VQTargetCacheMeta(
        db=args.db, vqvae_ckpt=args.vqvae_ckpt, norm_stats=args.norm_stats,
        num_frames=num_frames, feature_dim=feature_dim, num_codebooks=vq_model.num_latents,
        code_dim=vq_model.code_dim, latent_dim=vq_model.latent_dim,
        num_embeddings=vq_model.quantizer.num_embeddings, indices_dtype=indices_dtype_name,
        indices_file=indices_file, fingerprints=fingerprints, runtime=runtime, complete=False)
    save_vq_target_cache_meta(args.out_dir, incomplete)
    partial_path = indices_path.with_suffix(indices_path.suffix + ".partial")
    indices_mm = np.memmap(
        partial_path,
        dtype=indices_dtype,
        mode="w+",
        shape=(num_frames, int(vq_model.num_latents)),
    )

    print(f"[Info] device          : {device}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] vq ckpt         : {args.vqvae_ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] out dir         : {args.out_dir}")
    print(
        f"[Info] cache layout    : num_frames={num_frames} num_codebooks={vq_model.num_latents} "
        f"code_dim={vq_model.code_dim} dtype={indices_dtype_name}"
    )
    feature_begin = vq_model.traj_dim if vq_model.input_mode == "motion_only" else 0
    chunk_dataset = _NormalizedFrameChunkDataset(
        x_path=args.db / MEMMAP_X_FILE,
        num_frames=num_frames,
        chunk_rows=args.chunk_rows,
        mean=mean,
        std=std,
        feature_begin=feature_begin,
    )
    loader_kwargs = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    chunk_loader = DataLoader(chunk_dataset, **loader_kwargs)
    print(
        f"[Info] input pipeline  : workers={args.num_workers} prefetch={args.prefetch_factor} "
        f"pin_memory={loader_kwargs['pin_memory']} feature_begin={feature_begin}"
    )

    with torch.inference_mode():
        for chunk_idx, (start, x_cpu) in enumerate(chunk_loader, start=1):
            start = int(start)
            end = start + x_cpu.size(0)
            x_t = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            _, indices_t, _ = vq_model.encode(x_t)
            indices_np = indices_t.detach().cpu().numpy().astype(indices_dtype, copy=False)
            indices_mm[start:end] = indices_np
            if chunk_idx % args.log_interval == 0 or end == num_frames:
                print(f"[Encode] rows {start}:{end} / {num_frames}")

    indices_mm.flush()
    del indices_mm
    partial_path.replace(indices_path)
    readback = np.memmap(indices_path, dtype=indices_dtype, mode='r', shape=(num_frames, vq_model.num_latents))
    checks = sorted(set(np.linspace(0, len(chunk_dataset) - 1, min(args.verify_chunks, len(chunk_dataset)), dtype=int).tolist()))
    with torch.inference_mode():
        for index in checks:
            start, x_cpu = chunk_dataset[index]
            _, indices, _ = vq_model.encode(x_cpu.to(device=device, dtype=torch.float32))
            if not np.array_equal(indices.cpu().numpy(), np.asarray(readback[start:start + len(x_cpu)])):
                raise RuntimeError(f'Same-device/chunk cache check failed at row {start}; metadata remains incomplete')
    runtime['verified_chunk_indices'] = checks
    del readback

    meta = VQTargetCacheMeta(
        db=args.db,
        vqvae_ckpt=args.vqvae_ckpt,
        norm_stats=args.norm_stats,
        num_frames=num_frames,
        feature_dim=feature_dim,
        num_codebooks=int(vq_model.num_latents),
        code_dim=int(vq_model.code_dim),
        latent_dim=int(vq_model.latent_dim),
        num_embeddings=int(vq_model.quantizer.num_embeddings),
        indices_dtype=indices_dtype_name,
        indices_file=indices_file,
        fingerprints=fingerprints,
        runtime=runtime,
        indices_sha256=file_sha256(indices_path),
    )
    save_vq_target_cache_meta(args.out_dir, meta)

    print(f"[Done] VQ target cache saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
