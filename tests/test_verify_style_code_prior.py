from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from tests._memmap_fixtures import write_db

VERIFIER = NEWCODE_DIR / "tools" / "verify_style_code_prior.py"

NUM_SLOTS = 4
NUM_EMBEDDINGS = 16


def _segments(n_frames: int = 40) -> list[dict[str, object]]:
    out = []
    start = 0
    for label in (0, 1):
        out.append(
            {
                "frame_start": start,
                "num_frames": n_frames,
                "label": label,
                "style_name": f"style_{label}",
                "source_entry_id": label,
                "source_path": f"/tmp/{label}.bvh",
                "suffix": "BR",
                "mirror": False,
                "source_start": 0,
                "source_stop": n_frames - 1,
                "source_global_label": label,
                "split": "train" if label == 0 else "test",
            }
        )
        start += n_frames
    return out


def _build_fixture(root: Path, *, corrupt: bool = False) -> tuple[Path, Path, Path]:
    """Tiny DB + VQ target cache + a prior computed exactly the way the tool expects."""
    db = root / "db"
    db.mkdir()
    segments = _segments(40)
    write_db(db, num_frames=80, num_styles=2, segments=segments, seed=3)

    cache = root / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    # Concentrated code usage per style (a realistic prior is peaked, not uniform): train frames
    # (label 0) draw from codes 0..3, held-out frames (label 1) from codes 8..11.
    indices = np.empty((80, NUM_SLOTS), dtype=np.uint16)
    for slot in range(NUM_SLOTS):
        indices[:40, slot] = rng.choice([0, 1, 2, 3], size=40, p=[0.7, 0.1, 0.1, 0.1])
        indices[40:, slot] = rng.choice([8, 9, 10, 11], size=40, p=[0.1, 0.1, 0.1, 0.7])
    indices.tofile(cache / "indices.u16")
    (cache / "meta.json").write_text(
        json.dumps(
            {
                "format": "vq_target_cache_v1",
                "db": str(db),
                "vqvae_ckpt": str(root / "fake_vq.pt"),
                "norm_stats": str(root / "norm_stats.npz"),
                "num_frames": 80,
                "feature_dim": 375,
                "num_codebooks": NUM_SLOTS,
                "code_dim": 4,
                "latent_dim": NUM_SLOTS * 4,
                "num_embeddings": NUM_EMBEDDINGS,
                "indices_dtype": "uint16",
                "indices_file": "indices.u16",
            }
        ),
        encoding="utf-8",
    )

    labels = np.fromfile(db / "Labels.i32", dtype=np.int32)
    counts = np.zeros((2, NUM_SLOTS, NUM_EMBEDDINGS), dtype=np.int64)
    for label in (0, 1):  # the tool's split is "train" -> only segments with split == train
        for frame in np.flatnonzero(labels == label):
            if int(frame) >= 40:  # label 0 lives in the train segment only
                continue
            for slot in range(NUM_SLOTS):
                counts[label, slot, indices[frame, slot]] += 1
    if corrupt:
        counts[0, 0] = np.roll(counts[0, 0], 1)  # counts no longer match the raw cache

    smoothing = 1.0
    prior = counts.astype(np.float64) + smoothing
    prior /= prior.sum(axis=-1, keepdims=True)
    global_prior = counts.sum(axis=0).astype(np.float64) + smoothing
    global_prior /= global_prior.sum(axis=-1, keepdims=True)
    p = np.clip(prior, 1e-8, None)
    g = np.clip(global_prior[None, :, :], 1e-8, None)
    m = np.clip(0.5 * (p + g), 1e-8, None)
    js = 0.5 * (np.sum(p * (np.log(p) - np.log(m)), -1) + np.sum(g * (np.log(g) - np.log(m)), -1))
    slot_saliency = js.mean(axis=0)
    selected = np.sort(np.argsort(-slot_saliency, kind="stable")[:2])

    prior_path = root / "prior.npz"
    np.savez(
        prior_path,
        prior=prior.astype(np.float32),
        global_prior=global_prior.astype(np.float32),
        slot_saliency=slot_saliency.astype(np.float32),
        selected_slots=selected.astype(np.int64),
        counts=counts,
        style_names=np.asarray(["style_0", "style_1"]),
        split=np.asarray("train"),
        smoothing=np.asarray(smoothing, dtype=np.float32),
        vqvae_ckpt=np.asarray(str(root / "fake_vq.pt")),
        db=np.asarray(str(db)),
        num_codebooks=np.asarray(NUM_SLOTS),
    )
    return db, cache, prior_path


class VerifyStyleCodePriorTests(unittest.TestCase):
    def _run(self, db: Path, cache: Path, prior: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(VERIFIER), "--db", str(db), "--vq-target-cache", str(cache), "--prior", str(prior)],
            cwd=str(NEWCODE_DIR),
            capture_output=True,
            text=True,
        )

    def test_valid_prior_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, cache, prior = _build_fixture(Path(tmp))
            proc = self._run(db, cache, prior)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertIn("All checks passed", proc.stdout)
            self.assertIn("counts recomputed from indices/labels/segments", proc.stdout)

    def test_tampered_counts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, cache, prior = _build_fixture(Path(tmp), corrupt=True)
            proc = self._run(db, cache, prior)
            self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
            self.assertIn("[FAIL] 3.counts recomputed", proc.stdout)

    def test_tampered_selected_slots_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, cache, prior = _build_fixture(root)
            data = dict(np.load(prior, allow_pickle=True))
            data["selected_slots"] = np.asarray([0, 1], dtype=np.int64)  # not the top-2 by saliency
            np.savez(prior, **data)
            proc = self._run(db, cache, prior)
            self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
            self.assertIn("[FAIL] 3.selected_slots", proc.stdout)


if __name__ == "__main__":
    unittest.main()
