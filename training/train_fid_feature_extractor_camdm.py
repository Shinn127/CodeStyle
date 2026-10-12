#!/usr/bin/env python
"""Train a native-60fps, 240-frame style classifier with CAMDM's network family.

This is a faithful port of CAMDM (https://github.com/AIGAnimation/CAMDM)
`Evaluation/feature_extractor/train_classifier.py` + `process_train_data.py`, with the
training data read from this project's schema-v4 memmap database instead of CAMDM's
`data/100style_position_ori+ybot.pkl`.

What is kept identical to CAMDM
--------------------------------
* `Classifier` (here `Utils.fid_feature_extractor.MotionFeatureClassifier`): the same
  conv1/bn1 + conv2/bn2 residual stack, the same `residual1x1_*` shortcuts, dropout 0.5,
  fc1(128*window -> 1024)+bn3, fc2(1024 -> 512)+bn4 (the 512-dim penultimate activation is
  the FID feature), fc3(512 -> class_num).
* Input: the 15 `ybot_skeleton` joints in CAMDM's order (`YBOT15_JOINT_NAMES`) -> in_feat 45.
* Clip-local windows use `np.arange(0, frame_num - clip_len + 1, clip_offset)`.
* Training: CrossEntropyLoss, `torch.optim.Adam(lr=1e-3)`, batch 1024, shuffle, 50 epochs.
* Checkpoint keys: `state_dict`, `in_feat`, `acc`, `class_names` (plus extra provenance).

Deliberate deviations (documented, not silent)
----------------------------------------------
* **Clip-level, per-style stratified train/val split** (`--val-ratio`, default 0.1) and
  best-checkpoint selection on the validation split. CAMDM itself has no split; the split is
  used only for model selection, and `--val-ratio 0` restores the all-clips protocol.
* Higher dropout for robustness (`--dropout`, default 0.6 on the pre-FC stage, plus
  `--head-dropout`, default 0.2 between the hidden and the FID feature).
* Default windows are 240 native frames (4 s), without temporal resampling. CAMDM uses
  60 frames at 30 fps (2 s). Keep this classifier's metrics separate from legacy results.
* Mirrored segments are never used; the database itself is unmirrored (kept for the
  Mirrored clips are never used (mirrored segments distort the mesh).
* Local position coordinates follow this database, including Hips height, rather than
  CAMDM's additional XYZ Hips centering. Training and evaluation use the same local convention.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Utils.fid_feature_extractor import (
    JOINT_SET_SELECTED24,
    JOINT_SET_YBOT15,
    YBOT15_TO_SELECTED24_INDICES,
    MotionFeatureClassifier,
)
from Utils.runtime import configure_torch_runtime, resolve_device, set_seed, require_unmirrored_db


SCRIPT_DIR = Path(__file__).resolve().parents[1]

# CAMDM's hyperparameters (train_classifier.py)
CAMDM_WINDOW_SIZE = 60
LOCAL_WINDOW_SIZE = 240
LOCAL_FPS = 60.0
CAMDM_CLIP_OFFSET = 10
CAMDM_BATCH_SIZE = 1024
CAMDM_LR = 1e-3
CAMDM_EPOCHS = 50
CAMDM_NUM_WORKERS = 4

def plan_clip_split(
    segments: list[dict[str, object]],
    *,
    val_ratio: float,
    seed: int,
    num_styles: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Clip-level, per-style stratified train/val split (`--split-by clip`).

    Whole clips are the unit, so the validation clips are never seen during training. Note
    that every clip of a style is a different locomotion type (suffix), so a per-style clip
    holdout makes the validation set action-imbalanced across styles; `split_by='temporal'`
    or `'suffix'` keep the action composition matched instead.
    """
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    by_style: dict[int, list[dict[str, object]]] = {}
    for segment in segments:
        by_style.setdefault(int(segment["label"]), []).append(segment)
    rng = np.random.default_rng(int(seed))
    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    for label in range(int(num_styles)):
        clips = by_style.get(label, [])
        if float(val_ratio) <= 0.0 or len(clips) < 2:
            train.extend(clips)
            continue
        order = rng.permutation(len(clips))
        n_val = max(1, min(int(round(len(clips) * float(val_ratio))), len(clips) - 1))
        val_ids = set(int(i) for i in order[:n_val])
        for index, clip in enumerate(clips):
            (val if index in val_ids else train).append(clip)
    return train, val


def plan_temporal_windows(
    segment: dict[str, object],
    *,
    window_size: int,
    clip_offset: int,
    val_ratio: float,
    guard_frames: int | None = None,
) -> tuple[list[int], list[int]]:
    """Split one clip's windows into (train, val) by holding out its tail.

    Windows of one clip overlap heavily, so the two regions are separated by a guard band of
    `window_size` frames (default): train windows end at `L - val_frames - guard`, validation
    windows start at `L - val_frames`. Every (style, suffix) therefore contributes to both
    splits, which keeps the locomotion-type composition identical between them - a style's
    validation windows are never an action its training windows lack.
    """
    length = int(segment["num_frames"])
    start = int(segment["frame_start"])
    if length < 2 * window_size or val_ratio <= 0.0:
        return [start + int(offset) for offset in np.arange(0, max(length - window_size + 1, 0), clip_offset)], []
    guard = int(window_size if guard_frames is None else guard_frames)
    val_frames = max(int(window_size), int(round(length * float(val_ratio))))
    train_end = length - val_frames - guard
    val_start = length - val_frames
    train_offsets = np.arange(0, max(train_end - window_size + 1, 0), clip_offset)
    val_offsets = np.arange(0, max(length - window_size + 1, 0), clip_offset)
    val_offsets = val_offsets[val_offsets >= val_start]
    if train_offsets.size == 0 or val_offsets.size == 0:
        return [start + int(offset) for offset in np.arange(0, max(length - window_size + 1, 0), clip_offset)], []
    return [start + int(o) for o in train_offsets], [start + int(o) for o in val_offsets]


def plan_suffix_split(
    segments: list[dict[str, object]],
    *,
    val_suffixes: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Hold out whole suffixes for every style (`--split-by suffix`).

    Unseen clips with an identical action for every style, which removes the action-difficulty
    confound of a random per-style clip holdout. Each validation (style, suffix) group is
    absent from training by construction.
    """
    wanted = {str(s) for s in val_suffixes}
    if not wanted:
        raise ValueError("split_by='suffix' requires a non-empty val-suffix set")
    train, val = [], []
    for segment in segments:
        (val if str(segment.get("suffix", "")) in wanted else train).append(segment)
    return train, val


class CamdmWindowDataset(Dataset):
    """Windows of `window_size` at stride `clip_offset`, restricted to one split."""

    def __init__(
        self,
        *,
        db: Path,
        window_size: int,
        clip_offset: int,
        split: str = "all",
        val_ratio: float = 0.0,
        split_seed: int = 42,
        split_by: str = "temporal",
        val_suffixes: tuple[str, ...] = (),
        min_clip_frames: int | None = None,
    ) -> None:
        self.db = Path(db)
        meta = json.loads((self.db / "meta.json").read_text(encoding="utf-8"))
        self.num_frames = int(meta["num_frames"])
        self.style_names = [str(name) for name in json.loads((self.db / "StyleNames.json").read_text(encoding="utf-8"))]
        segments = json.loads((self.db / "segments.json").read_text(encoding="utf-8"))
        self.window_size = int(window_size)
        self.clip_offset = int(clip_offset)
        if self.window_size < 1 or self.clip_offset < 1:
            raise ValueError('window_size and clip_offset must be positive')
        if split not in ('all', 'train', 'val'):
            raise ValueError(f"split must be all|train|val, got {split!r}")
        if split_by not in ('temporal', 'clip', 'suffix'):
            raise ValueError(f"split_by must be temporal|clip|suffix, got {split_by!r}")
        self.split = split
        self.split_by = split_by
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)
        self.val_suffixes = tuple(val_suffixes)

        self._x = np.memmap(self.db / "X.f32", dtype=np.float32, mode="r").reshape(self.num_frames, 375)
        self._labels = np.memmap(self.db / "Labels.i32", dtype=np.int32, mode="r")

        covered = sum(int(segment["num_frames"]) for segment in segments)
        if covered != self.num_frames:
            raise ValueError(
                f"{self.db}: segments cover {covered} frames but meta.num_frames is {self.num_frames}; "
                "rebuild the database (its frame table is inconsistent)"
            )

        min_frames = max(self.window_size, int(min_clip_frames or self.window_size))
        if self.split_by == 'clip':
            train_clips, val_clips = plan_clip_split(
                segments, val_ratio=self.val_ratio, seed=self.split_seed, num_styles=len(self.style_names)
            )
            clip_groups = {'train': train_clips, 'val': val_clips}
        elif self.split_by == 'suffix':
            train_clips, val_clips = plan_suffix_split(segments, val_suffixes=self.val_suffixes)
            clip_groups = {'train': train_clips, 'val': val_clips}
        else:
            clip_groups = {'train': list(segments), 'val': list(segments)}
        self.clip_counts = {'train': len(clip_groups['train']), 'val': len(clip_groups['val'])}

        self.starts: list[int] = []
        self.motion_ids: list[int] = []
        self.windows_by_group: dict[str, int] = {'train': 0, 'val': 0}
        for segment in clip_groups.get(self.split, segments) if self.split != 'all' else segments:
            num_frames = int(segment["num_frames"])
            if num_frames < min_frames:
                continue
            if self.split_by == 'temporal' and self.split != 'all':
                train_starts, val_starts = plan_temporal_windows(
                    segment, window_size=self.window_size, clip_offset=self.clip_offset,
                    val_ratio=self.val_ratio,
                )
                self.windows_by_group['train'] += len(train_starts)
                self.windows_by_group['val'] += len(val_starts)
                chosen = train_starts if self.split == 'train' else val_starts
            else:
                frame_start = int(segment["frame_start"])
                offsets = np.arange(0, num_frames - self.window_size + 1, self.clip_offset)
                chosen = [int(frame_start + offset) for offset in offsets]
            self.starts.extend(chosen)
            self.motion_ids.extend([int(segment["label"])] * len(chosen))
        self.starts = np.asarray(self.starts, dtype=np.int64)
        self.motion_ids = np.asarray(self.motion_ids, dtype=np.int64)
        self.n_joints = len(YBOT15_TO_SELECTED24_INDICES)
        self.in_feat = self.n_joints * 3
        detail = ""
        if self.split_by == 'temporal' and self.split != 'all':
            detail = (f" temporal: train={self.windows_by_group['train']} val={self.windows_by_group['val']} "
                      f"windows over the same {self.clip_counts['train']} clips")
        print(
            f"[Data] db={self.db.name} styles={len(self.style_names)} split={self.split} "
            f"split_by={self.split_by} clips={len(clip_groups[self.split]) if self.split != 'all' else len(segments)} "
            f"(train={self.clip_counts['train']} val={self.clip_counts['val']}) "
            f"windows={len(self.starts)} in_feat={self.in_feat}{detail}"
        )

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        local_pos = np.asarray(self._x[start : start + self.window_size, 87:159], dtype=np.float32)
        ybot = local_pos.reshape(self.window_size, 24, 3)[:, YBOT15_TO_SELECTED24_INDICES, :].reshape(self.window_size, -1)
        return torch.from_numpy(ybot), torch.tensor(int(self.motion_ids[index]), dtype=torch.long)

    def style_frequency(self) -> np.ndarray:
        return np.bincount(self.motion_ids, minlength=len(self.style_names)).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAMDM's FID classifier on the full 100STYLE database")
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_fid_full100_nomirror")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "checkpoints" / "fid_feature_extractor_camdm100_60fps_240f_nomirror")
    parser.add_argument("--window-size", type=int, default=LOCAL_WINDOW_SIZE)
    parser.add_argument("--clip-offset", type=int, default=CAMDM_CLIP_OFFSET)
    parser.add_argument("--batch-size", type=int, default=CAMDM_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=CAMDM_EPOCHS)
    parser.add_argument("--lr", type=float, default=CAMDM_LR)
    parser.add_argument("--num-workers", type=int, default=CAMDM_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Clip-level per-style validation fraction; 0 restores CAMDM's all-clips protocol",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-by", choices=("temporal", "clip", "suffix"), default="temporal",
        help=(
            "temporal: hold out the tail of every clip (all locomotion types stay in both "
            "splits, guard band prevents window overlap); clip: hold out whole clips "
            "(unseen clips, action-imbalanced across styles); suffix: hold out whole "
            "suffixes for every style (unseen clips, identical action for all styles)"
        ),
    )
    parser.add_argument(
        "--val-suffixes", type=str, default="",
        help="comma-separated suffixes held out when --split-by suffix (e.g. SR,SW)",
    )
    parser.add_argument(
        "--temporal-guard-frames", type=int, default=0,
        help="frames skipped between the temporal train and val regions; 0 uses window_size",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout on the conv features (CAMDM uses 0.5)")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="Extra dropout before the FID feature")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def run_epoch(
    *,
    model: MotionFeatureClassifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    log_interval: int,
    training: bool = True,
) -> tuple[float, float]:
    model.train(training)
    sums = torch.zeros(2, device=device, dtype=torch.float32)
    total = 0
    tag = "train" if training else "val"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (inputs, labels) in enumerate(tqdm(loader, desc=f"{tag} {epoch}/{total_epochs}", leave=False), start=1):
            inputs = inputs.to(device=device, non_blocking=True).permute(0, 2, 1).float()
            labels = labels.to(device=device, non_blocking=True)
            if training:
                assert optimizer is not None, "training requires an optimizer"
                optimizer.zero_grad()
            _, logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            sums[0] += loss.detach() * int(labels.size(0))
            sums[1] += (logits.argmax(dim=1) == labels).sum()
            total += int(labels.size(0))
            if training and log_interval > 0 and step % log_interval == 0:
                print(f"[train] epoch {epoch}/{total_epochs} step {step}/{len(loader)} loss={float(loss.detach()):.4f}")
    loss_sum, correct = sums.cpu().tolist()
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def main() -> None:
    args = parse_args()
    args.db = Path(args.db).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    if args.batch_size < 2 or args.epochs < 1 or args.lr <= 0:
        raise ValueError('Classifier training requires batch_size >= 2, epochs >= 1 and lr > 0')
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError('Classifier training requires val_ratio in [0, 1)')
    if args.out_dir.exists() and (any(args.out_dir.glob('*.pt')) or any(args.out_dir.glob('*.pth'))
                                 or (args.out_dir / 'history.jsonl').exists()):
        raise ValueError('Classifier output already contains a run; choose a new directory')
    require_unmirrored_db(args.db)
    val_suffixes = tuple(part.strip() for part in str(args.val_suffixes).split(",") if part.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=True, matmul_precision='highest')

    dataset = CamdmWindowDataset(
        db=args.db,
        window_size=args.window_size,
        clip_offset=args.clip_offset,
        split='train' if args.val_ratio > 0 else 'all',
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
        val_suffixes=val_suffixes,
    )
    if len(dataset) < args.batch_size:
        raise ValueError('Too few windows for one complete classifier training batch')
    if (dataset.style_frequency() == 0).any():
        missing = [name for name, count in zip(dataset.style_names, dataset.style_frequency()) if count == 0]
        raise ValueError(f'No eligible classifier windows for styles: {missing}')
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_dataset = val_loader = None
    val_split_warning = None
    if args.val_ratio > 0:
        val_dataset = CamdmWindowDataset(
            db=args.db,
            window_size=args.window_size,
            clip_offset=args.clip_offset,
            split='val',
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            split_by=args.split_by,
            val_suffixes=val_suffixes,
        )
        if len(val_dataset) < 2:
            # Short clips cannot host the temporal guard band; fall back to the no-split
            # protocol with a loud warning instead of failing the run.
            print(f"[Warn] validation split is empty ({len(val_dataset)} windows): the database's "
                  "clips are too short for the requested split; best selection is disabled")
            val_dataset = val_loader = None
            val_split_warning = 'empty_val_split'
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=(device.type == "cuda"),
            )
    model = MotionFeatureClassifier(
        in_feat=dataset.in_feat,
        class_num=len(dataset.style_names),
        window_size=args.window_size,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Info] db            : {args.db}")
    print(f"[Info] out_dir       : {args.out_dir}")
    print(f"[Info] device        : {device}")
    print(f"[Info] styles        : {len(dataset.style_names)}")
    print(f"[Info] windows       : {len(dataset)}")
    print(f"[Info] input span    : {args.window_size} frames @ {LOCAL_FPS:g} fps = {args.window_size / LOCAL_FPS:g} s")
    print('[Info] precision     : FP32/highest, TF32 disabled')
    print(f"[Info] stride/offset : {args.clip_offset} (CAMDM default {CAMDM_CLIP_OFFSET})")
    print(f"[Info] optimizer     : Adam(lr={args.lr}) (CAMDM default {CAMDM_LR})")
    print(f"[Info] epochs        : {args.epochs} (CAMDM default {CAMDM_EPOCHS})")
    if val_loader is None:
        print("[Info] split         : none - every window is training data (CAMDM protocol)")
    else:
        print(f"[Info] split_by      : {args.split_by} "
              + (f"val_suffixes={list(val_suffixes)}" if args.split_by == 'suffix' else ""))
        print(f"[Info] split         : train={len(dataset)} windows "
              f"({dataset.clip_counts['train']} clips) / val={len(val_dataset)} windows "
              f"({val_dataset.clip_counts['val']} clips); val_ratio={args.val_ratio} seed={args.split_seed}")
    print(f"[Info] dropout       : conv={args.dropout} head={args.head_dropout}")

    history_path = args.out_dir / "history.jsonl"
    history = history_path.open("a", encoding="utf-8")
    train_acc = float("nan")
    best: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        val_record = None
        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model=model, loader=val_loader, device=device, optimizer=None,
                epoch=epoch, total_epochs=args.epochs, log_interval=0, training=False,
            )
            val_record = {"loss": val_loss, "acc": val_acc, "windows": len(val_dataset)}
            # Selection rule: highest val accuracy, ties broken by lower val loss, then the
            # earlier epoch (recorded in the checkpoint provenance as selection_rule).
            key = (-val_acc, val_loss)
            if best is None or key < best["key"]:
                best = {"key": key, "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
                torch.save(
                    {
                        "state_dict": {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()},
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                        "in_feat": int(dataset.in_feat), "class_names": list(dataset.style_names),
                        "class_num": len(dataset.style_names), "window_size": int(args.window_size),
                        "joint_set": JOINT_SET_YBOT15,
                        "dropout": float(args.dropout), "head_dropout": float(args.head_dropout),
                        "acc_split": "val",
                    },
                    args.out_dir / "best.pt",
                )
        record = {
            "epoch": epoch,
            "loss": loss,
            "acc": train_acc,
            "val": val_record,
            "lr": args.lr,
            "windows": len(dataset),
            "time_sec": time.time() - t0,
        }
        history.write(json.dumps(record) + "\n")
        history.flush()
        suffix = "" if val_record is None else f" val_loss={val_record['loss']:.6f} val_acc={val_record['acc']:.2f}%"
        print(f"[Epoch {epoch}] loss={loss:.6f} train_acc={train_acc:.2f}%{suffix} ({record['time_sec']:.1f}s)")
    history.close()

    final_state = None
    if best is not None:
        # The canonical file is the validation-selected model; the last epoch is kept as final.pt.
        final_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        selected = torch.load(args.out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        print(f"[Select] best epoch {best['epoch']} val_acc={best['val_acc']:.2f}% val_loss={best['val_loss']:.6f}")

    payload = {
        # CAMDM's own checkpoint keys
        "state_dict": model.state_dict(),
        "in_feat": int(dataset.in_feat),
        "acc": float(train_acc),
        "class_names": list(dataset.style_names),
        # provenance for this project
        "class_num": len(dataset.style_names),
        "window_size": int(args.window_size),
        "joint_set": JOINT_SET_YBOT15,
        "source": "train_fid_feature_extractor_camdm.py",
        "camdm_reference": "https://github.com/AIGAnimation/CAMDM Evaluation/feature_extractor/train_classifier.py",
        "db": str(args.db),
        "clip_offset": int(args.clip_offset),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "windows": len(dataset),
        "temporal_stride": 1,
        "fps": LOCAL_FPS,
        "window_seconds": args.window_size / LOCAL_FPS,
        "position_convention": "root_xz_facing_with_hips_height",
        "precision": "float32", "matmul_precision": "highest", "tf32": False,
        "dropout": float(args.dropout), "head_dropout": float(args.head_dropout),
        "val_ratio": float(args.val_ratio), "split_seed": int(args.split_seed),
        "split_by": str(args.split_by), "val_suffixes": list(val_suffixes),
        "temporal_guard_frames": int(args.temporal_guard_frames or args.window_size),
        "train_windows": len(dataset),
        "val_windows": len(val_dataset) if val_dataset is not None else 0,
        "selection_rule": "max_val_acc_then_min_val_loss" if best is not None else "no_val_split",
        "val_split_warning": val_split_warning,
        "best_epoch": None if best is None else int(best["epoch"]),
        "best_val_loss": None if best is None else float(best["val_loss"]),
        "best_val_acc": None if best is None else float(best["val_acc"]),
        "final_train_acc": float(train_acc),
        "acc_split": "val" if best is not None else "train",
        "split_protocol": (f"extractor_split_by_{args.split_by}" if best is not None else "camdm_all_clips"),
        "camdm_split_deviations": "train/val split and best-by-val selection are project additions",
    }
    selected_payload = dict(payload)
    selected_payload["acc"] = float(payload["best_val_acc"]) if best is not None else float(train_acc)
    final_path = args.out_dir / "100style_position_classifier.pth"
    torch.save(selected_payload, final_path)
    torch.save(selected_payload, args.out_dir / "latest.pt")
    if final_state is not None:
        final_payload = dict(payload)
        final_payload["state_dict"] = final_state
        final_payload["acc"] = float(train_acc)
        final_payload["acc_split"] = "train"
        torch.save(final_payload, args.out_dir / "final.pt")
    freq = dataset.style_frequency()
    (args.out_dir / "style_window_counts.json").write_text(
        json.dumps(
            {name: int(freq[idx]) for idx, name in enumerate(dataset.style_names)},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Done] final_train_acc={train_acc:.4f}% saved {final_path}")
    if best is not None:
        print(f"[Done] canonical file = best by val (epoch {best['epoch']}, val_acc={best['val_acc']:.4f}%)")
    else:
        print("[Done] note: CAMDM's recipe has no validation split, so there is no 'best' selection")


if __name__ == "__main__":
    main()
