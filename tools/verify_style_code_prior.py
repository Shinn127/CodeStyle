#!/usr/bin/env python
"""Verify a style-code prior artifact against the database and its VQ target cache.

Independent recomputation: the prior is rebuilt from `indices.u16` + `Labels.i32` +
`segments.json` (the same inputs, but with this script's own code path) and compared with the
stored arrays. Any mismatch is a FAIL with a non-zero exit code.

    python verify_style_code_prior.py \
      --db database_100sty_375_memmap_nomirror \
      --vq-target-cache cache/vq_targets_vq150ep_nomirror_repair_v2 \
      --prior cache/style_code_prior_vq150ep_nomirror_repair_v2.npz \
      [--controller-ckpt checkpoints/flowmatch_vqvae_controller_prior_90st/best.pt]

Checks
  1  provenance: prior's recorded db / vqvae_ckpt / num_codebooks match the cache
  2  geometry: shapes, non-negativity, rows sum to 1, Laplace floor alpha/(n+alpha*K)
  3  recomputation: counts / prior / global_prior / slot_saliency / selected_slots
  4  split purity: train-derived vs val+test-derived prior (JS) and leakage check
  5  mirror sensitivity: unmirrored train frames vs all train frames
  6  degeneracy: per (style, slot) top-1 mass, near-uniform rows, dead codes
  7  entropy floor: E_Omega[H(pi_bar)] and the prior mass outside the selector's top-4
  8  selector agreement (needs --controller-ckpt): per-slot KL(prior || selector), top-4 overlap
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
EPS = 1e-8

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def info(label: str, detail: str) -> None:
    print(f"[info] {label}: {detail}")


def _js_rows(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """JS divergence with the (p+q)/2 mixture, matching the precompute script."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, None)
    m = np.clip(0.5 * (p + q), EPS, None)
    return 0.5 * (np.sum(p * (np.log(p) - np.log(m)), axis=axis)
                  + np.sum(q * (np.log(q) - np.log(m)), axis=axis))


def _segment_frame_ids(segments: list[dict], *, splits: tuple[str, ...], mirror: bool | None) -> dict[int, np.ndarray]:
    """style label -> frame ids, for the requested splits (and mirror filter when given)."""
    out: dict[int, list[np.ndarray]] = {}
    for segment in segments:
        if str(segment.get("split", "")) not in splits:
            continue
        if mirror is not None and bool(segment.get("mirror")) != mirror:
            continue
        start = int(segment["frame_start"])
        n = int(segment["num_frames"])
        if n <= 0:
            continue
        out.setdefault(int(segment["label"]), []).append(np.arange(start, start + n, dtype=np.int64))
    return {label: np.concatenate(chunks) for label, chunks in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a style-code prior artifact")
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument(
        "--vq-target-cache",
        type=Path,
        default=SCRIPT_DIR / "cache" / "vq_targets_vq150ep_nomirror_repair_v2",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=SCRIPT_DIR / "cache" / "style_code_prior_vq150ep_nomirror_repair_v2.npz",
    )
    parser.add_argument("--controller-ckpt", type=Path, default=None)
    parser.add_argument("--tolerance-counts", type=int, default=0, help="allowed count mismatch (exact by default)")
    parser.add_argument(
        "--expect-entropy-floor",
        type=float,
        default=None,
        help="when set, assert E_Omega[H(pi_bar)] matches this value (e.g. 4.834 for the 100STYLE 150ep prior)",
    )
    args = parser.parse_args()

    db = args.db.expanduser().resolve()
    cache_dir = args.vq_target_cache.expanduser().resolve()
    prior_path = args.prior.expanduser().resolve()

    data = np.load(prior_path, allow_pickle=True)
    prior = np.asarray(data["prior"], dtype=np.float64)
    global_prior = np.asarray(data["global_prior"], dtype=np.float64)
    slot_saliency = np.asarray(data["slot_saliency"], dtype=np.float64).reshape(-1)
    selected_slots = np.asarray(data["selected_slots"], dtype=np.int64).reshape(-1)
    counts = np.asarray(data["counts"], dtype=np.int64)
    style_names = [str(s) for s in np.asarray(data["style_names"]).tolist()]
    split = str(data["split"].item()) if "split" in data.files else "train"
    smoothing = float(np.asarray(data["smoothing"]).item()) if "smoothing" in data.files else 1.0

    cache_meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    db_meta = json.loads((db / "meta.json").read_text(encoding="utf-8"))
    segments = json.loads((db / "segments.json").read_text(encoding="utf-8"))
    labels = np.memmap(db / "Labels.i32", dtype=np.int32, mode="r")

    print("=" * 96)
    print(f"prior      : {prior_path}")
    print(f"db         : {db}")
    print(f"vq cache   : {cache_dir}")
    print("=" * 96)

    # ---------------------------------------------------------------- 1 provenance
    prov_ok = True
    for key, expected in (("vqvae_ckpt", cache_meta["vqvae_ckpt"]), ("db", cache_meta["db"])):
        recorded = str(data[key].item()) if key in data.files else None
        ok = recorded == str(expected)
        prov_ok &= ok
        check(ok, f"1.provenance {key}", f"prior={recorded} cache={expected}")
    num_codebooks = int(np.asarray(data["num_codebooks"]).item()) if "num_codebooks" in data.files else -1
    prov_ok &= check(num_codebooks == int(cache_meta["num_codebooks"]),
                     "1.provenance num_codebooks", f"prior={num_codebooks} cache={cache_meta['num_codebooks']}")
    prov_ok &= check(len(style_names) == int(db_meta["num_styles"]),
                     "1.provenance num_styles", f"prior={len(style_names)} db={db_meta['num_styles']}")

    # ---------------------------------------------------------------- 2 geometry
    num_styles, num_slots, num_embeddings = prior.shape
    check(prior.ndim == 3 and counts.shape == prior.shape, "2.shape prior==counts", f"{prior.shape}")
    check(global_prior.shape == (num_slots, num_embeddings), "2.shape global_prior", f"{global_prior.shape}")
    check(slot_saliency.shape == (num_slots,), "2.shape slot_saliency", f"{slot_saliency.shape}")
    check(int(selected_slots.min()) >= 0 and int(selected_slots.max()) < num_slots,
          "2.selected_slots in range", f"{selected_slots.tolist()}")
    check((counts >= 0).all(), "2.counts non-negative")
    row_sums = prior.sum(axis=-1)
    check(np.allclose(row_sums, 1.0, atol=1e-5), "2.prior rows sum to 1",
          f"max deviation {np.abs(row_sums - 1).max():.2e}")
    g_row_sums = global_prior.sum(axis=-1)
    check(np.allclose(g_row_sums, 1.0, atol=1e-5), "2.global_prior rows sum to 1",
          f"max deviation {np.abs(g_row_sums - 1).max():.2e}")
    # Laplace floor: every entry >= alpha / (n + alpha*K)
    denom = counts.sum(axis=-1, keepdims=True) + smoothing * num_embeddings
    floor = smoothing / np.clip(denom, 1e-12, None)
    check(bool((prior + 1e-9 >= floor).all()), "2.Laplace floor respected",
          f"smoothing={smoothing}, min entry {prior.min():.3e}")

    # ---------------------------------------------------------------- 3 recomputation
    indices = np.memmap(cache_dir / cache_meta["indices_file"], dtype=cache_meta["indices_dtype"],
                        mode="r", shape=(cache_meta["num_frames"], num_slots))
    frame_ids_by_label = _segment_frame_ids(segments, splits=(split,), mirror=None)
    recomputed = np.zeros_like(counts)
    for label, ids in frame_ids_by_label.items():
        chunk = np.asarray(indices[ids], dtype=np.int64)          # [n, slots]
        for slot in range(num_slots):
            recomputed[label, slot] = np.bincount(chunk[:, slot], minlength=num_embeddings)
    mismatch = int(np.abs(recomputed - counts).sum())
    check(mismatch <= args.tolerance_counts, "3.counts recomputed from indices/labels/segments",
          f"total absolute mismatch={mismatch} over {num_styles * num_slots * num_embeddings} entries")

    prior_re = counts.astype(np.float64) + smoothing
    prior_re /= np.clip(prior_re.sum(axis=-1, keepdims=True), 1.0, None)
    check(np.allclose(prior_re, prior, atol=1e-6), "3.prior recomputed",
          f"max abs diff {np.abs(prior_re - prior).max():.2e}")
    g_counts = counts.sum(axis=0)
    g_re = g_counts.astype(np.float64) + smoothing
    g_re /= np.clip(g_re.sum(axis=-1, keepdims=True), 1.0, None)
    check(np.allclose(g_re, global_prior, atol=1e-6), "3.global_prior recomputed",
          f"max abs diff {np.abs(g_re - global_prior).max():.2e}")
    sal_re = _js_rows(prior, global_prior[None, :, :]).mean(axis=0)
    check(np.allclose(sal_re, slot_saliency, atol=1e-5), "3.slot_saliency recomputed",
          f"max abs diff {np.abs(sal_re - slot_saliency).max():.2e}")
    order = np.argsort(-slot_saliency, kind="stable")
    selected_re = np.sort(order[: len(selected_slots)])
    check(np.array_equal(selected_re, selected_slots), "3.selected_slots = top-M by saliency",
          f"recomputed {selected_re.tolist()} stored {selected_slots.tolist()}")

    # ---------------------------------------------------------------- 4 split purity
    other_ids = _segment_frame_ids(segments, splits=tuple(s for s in ("train", "val", "test") if s != split), mirror=None)
    counts_other = np.zeros_like(counts)
    for label, ids in other_ids.items():
        chunk = np.asarray(indices[ids], dtype=np.int64)
        for slot in range(num_slots):
            counts_other[label, slot] = np.bincount(chunk[:, slot], minlength=num_embeddings)
    prior_other = counts_other.astype(np.float64) + smoothing
    prior_other /= np.clip(prior_other.sum(axis=-1, keepdims=True), 1.0, None)
    js_split = _js_rows(prior, prior_other)                       # [styles, slots]
    info("4.train-vs-eval prior JS", f"mean {js_split.mean():.4f} over (style,slot); "
                                     f"p50 {np.median(js_split):.4f}; p95 {np.percentile(js_split, 95):.4f}")
    sel_js = js_split[:, selected_slots]
    check(float(np.median(sel_js)) < 0.25, "4.train/held-out prior agreement on Omega",
          f"median JS {np.median(sel_js):.4f} (nats/2), mean {sel_js.mean():.4f}")

    # ---------------------------------------------------------------- 5 mirror sensitivity
    unmirrored_ids = _segment_frame_ids(segments, splits=(split,), mirror=False)
    counts_unmirrored = np.zeros_like(counts)
    for label, ids in unmirrored_ids.items():
        chunk = np.asarray(indices[ids], dtype=np.int64)
        for slot in range(num_slots):
            counts_unmirrored[label, slot] = np.bincount(chunk[:, slot], minlength=num_embeddings)
    prior_un = counts_unmirrored.astype(np.float64) + smoothing
    prior_un /= np.clip(prior_un.sum(axis=-1, keepdims=True), 1.0, None)
    js_mirror = _js_rows(prior, prior_un)[:, selected_slots]
    info("5.mirror sensitivity on Omega", f"mean JS {js_mirror.mean():.4f}, p95 {np.percentile(js_mirror, 95):.4f} "
                                          f"(0 = mirroring does not change the style code distribution)")

    # ---------------------------------------------------------------- 6 degeneracy
    top1 = prior.max(axis=-1)                                     # [styles, slots]
    scale = 1.0 / num_embeddings
    near_uniform = int((top1 < 2.0 * scale).sum())
    degenerate = int((top1 > 0.5).sum())
    info("6.top-1 mass per (style,slot)", f"mean {top1.mean():.4f} (uniform={scale:.1e}), "
                                         f"p05 {np.percentile(top1, 5):.4f}, p95 {np.percentile(top1, 95):.4f}")
    info("6.rows near uniform (<2x uniform)", f"{near_uniform} / {top1.size}")
    info("6.rows dominated by one code (>50%)", f"{degenerate} / {top1.size}")
    check(float(top1.mean()) > 2.0 * scale, "6.prior is informative overall",
          f"mean top-1 mass {top1.mean():.4f} vs uniform {scale:.1e} ({top1.mean() / scale:.0f}x)")
    active = counts.sum(axis=(0, 1)) > 0
    info("6.codes used anywhere", f"{int(active.sum())} / {num_embeddings}")

    # ---------------------------------------------------------------- 7 entropy floor
    clipped = np.clip(prior, 1e-12, None)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1)            # [styles, slots]
    floor_sel = float(entropy[:, selected_slots].mean())
    floor_all = float(entropy.mean())
    info("7.E_Omega[H(pi_bar)]", f"{floor_sel:.4f} nats (CE floor of the dense prior loss)")
    info("7.mean H over all slots", f"{floor_all:.4f} nats (log K = {np.log(num_embeddings):.4f})")
    outside = []
    for slot in selected_slots:
        p = np.sort(prior[:, slot], axis=-1)[:, ::-1]
        outside.append(1.0 - p[:, :4].sum(-1))
    outside = np.concatenate(outside)
    info("7.prior mass outside selector top-4", f"mean {outside.mean() * 100:.1f}% "
                                                f"(min {outside.min() * 100:.1f}%, max {outside.max() * 100:.1f}%)")
    if args.expect_entropy_floor is not None:
        check(abs(floor_sel - float(args.expect_entropy_floor)) < 0.05,
              f"7.entropy floor matches the expected {float(args.expect_entropy_floor):.3f} nats",
              f"measured {floor_sel:.4f}")
    else:
        check(floor_sel < float(np.log(num_embeddings)) - 0.1, "7.prior is peaked (floor < log K)",
              f"measured {floor_sel:.4f} nats vs log K = {np.log(num_embeddings):.4f}")

    # ---------------------------------------------------------------- 8 selector agreement
    if args.controller_ckpt is not None:
        import torch

        from Utils.motion_eval import load_motion_model

        ckpt = args.controller_ckpt.expanduser().resolve()
        model = load_motion_model(model_name="flowmatch_velocity", ckpt_path=ckpt, num_styles=num_styles)
        model.eval()
        selector = model.style_selector
        with torch.no_grad():
            out = selector(codebook_weight=model.vqvae.quantizer.embedding_weight.detach(),
                           style_label=torch.arange(num_styles))
        tau = float(getattr(selector, "temperature", 1.0))
        logits = out["style_logits"].double()
        dense = torch.softmax(logits / tau, dim=-1).numpy()        # what the dense loss supervises
        sparse = out["style_probs"].double().numpy()               # the TopK selection operator
        dense_c = np.clip(dense, 1e-12, None)
        sparse_c = np.clip(sparse, 1e-12, None)
        # the selector only parameterises the style slots Omega, so compare on those directly
        prior_sel = prior[:, selected_slots, :]
        kl_dense = (prior_sel * (np.log(prior_sel + 1e-12) - np.log(dense_c))).sum(axis=-1)
        kl_sparse = (prior_sel * (np.log(prior_sel + 1e-12) - np.log(sparse_c))).sum(axis=-1)
        kl_sel = kl_dense
        info("8.KL(prior || selector, dense softmax) on Omega",
             f"mean {kl_sel.mean():.4f} nats  =>  CE floor + KL = {floor_sel + kl_sel.mean():.4f} "
             f"(training log reported 5.176)")
        info("8.KL(prior || TopK-sparse pi) on Omega",
             f"mean {kl_sparse.mean():.4f} nats (this is the paper's literal Eq. 6; "
             f"large by construction, ~80.7% of prior mass sits outside top-4)")
        topk = 4
        prior_top = np.argsort(-prior, axis=-1)[..., :topk]
        sel_top = np.argsort(-sparse_c, axis=-1)[..., :topk]
        overlap = np.mean([len(set(prior_top[s, j]) & set(sel_top[s, local])) / topk
                           for s in range(num_styles)
                           for local, j in enumerate(selected_slots.tolist())])
        info("8.selector top-4 overlap with prior top-4",
             f"{overlap * 100:.1f}% of the prior's top-4 codes are actually selected")
        per_slot = kl_dense.mean(axis=0)
        worst = int(np.argmax(per_slot))
        check(float(kl_sel.mean()) < 1.0, "8.selector matches the prior on Omega (dense)",
              f"mean KL {kl_sel.mean():.4f} nats; worst slot {selected_slots[worst]} (KL {per_slot[worst]:.3f})")

    print("=" * 96)
    failed = [label for ok, label in RESULTS if not ok]
    print(f"SUMMARY: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
