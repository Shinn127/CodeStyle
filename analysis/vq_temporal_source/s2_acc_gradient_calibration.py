"""S2: measure the actual gradients of L_acc vs the base objective. NO optimizer.step.

Protocol (COMMON.md section 5, frozen before B1 existed):
  * model = ep90 baseline on an isolated copy, train mode with the STE graph kept;
    the library EMA/dead-code update is disabled with `freeze_codebook=True` so the
    calibration cannot pollute the source checkpoint.
  * 4 independent train batches B16 x T8 (seed 4242+i), windows drawn inside segments so
    no window crosses a clip boundary; sampling uses per-batch numpy RNG, and the model
    state hash is compared before/after (must be identical).
  * g_base = grad of the full base objective taken from `compute_loss` with acc_weight=0;
    g_acc  = grad of `acc_loss` alone (acc_weight=1).
  * Norms and cosine are grouped into encoder / decoder parameter sets.

Output: analysis/vq_temporal_source/S2/gradient_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from vq_temporal_common import (  # noqa: E402
    DB, OUT_ROOT, SRC_CKPT, environment_report, git_commit, json_dump,
)
from Utils.reproducibility import state_fingerprint  # noqa: E402
from Utils.runtime import load_norm_stats  # noqa: E402
from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: E402

BASE_WEIGHTS = dict(
    recon_type="l1", temporal_loss_type="l1",
    recon_weight=3.0, vq_weight=1.0, vel_weight=15.0, root_weight=3.0,
    code_align_weight=2.0, latent_l1_weight=0.01, latent_vel_weight=0.1,
    latent_margin_weight=0.0, roundtrip_weight=0.0,
)


def sample_batches(seed: int, batches: int, batch_size: int, seq_len: int):
    meta = json.loads((DB / "meta.json").read_text())
    segments = json.loads((DB / "segments.json").read_text())
    mean, std = load_norm_stats(SRC_CKPT.parent / "norm_stats.npz")
    x_source = np.memmap(DB / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))
    train = [s for s in segments if s.get("split") == "train" and not s.get("mirror", False)
             and int(s["num_frames"]) >= seq_len]
    for i in range(batches):
        rng = np.random.default_rng(seed + i)
        rows = np.empty((batch_size, seq_len, 375), dtype=np.float32)
        meta_rows = []
        for j in range(batch_size):
            s = train[int(rng.integers(len(train)))]
            slack = int(s["num_frames"]) - seq_len
            start = int(s["frame_start"]) + int(rng.integers(slack + 1))
            rows[j] = x_source[start:start + seq_len]
            meta_rows.append(dict(source_entry_id=int(s["source_entry_id"]), suffix=s["suffix"], start=start))
        normed = (rows - mean) / std
        yield torch.from_numpy(normed), meta_rows


def flat_grad(loss: torch.Tensor, parameters: list[torch.nn.Parameter], *,
              retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    pieces = [g.reshape(-1) for g in grads if g is not None]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=loss.device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=SRC_CKPT)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "S2" / "gradient_calibration.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_vqvae_from_checkpoint(args.ckpt, freeze=False).to(device).train()
    forward = model.quantizer.vq.forward
    model.quantizer.vq.forward = lambda *a, **kw: forward(*a, freeze_codebook=True, **kw)

    before = state_fingerprint(model.state_dict())
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    per_batch = []
    for i, (x, meta_rows) in enumerate(sample_batches(args.seed, args.batches, args.batch_size, args.seq_len)):
        x = x.to(device)
        target = x[..., 84:]
        model.zero_grad(set_to_none=True)
        base = model.compute_loss(x=x, target_motion=target, acc_weight=0.0, **BASE_WEIGHTS)
        g_base_e = flat_grad(base["loss"], encoder_params, retain_graph=True)
        g_base_d = flat_grad(base["loss"], decoder_params, retain_graph=False)

        model.zero_grad(set_to_none=True)
        acc = model.compute_loss(x=x, target_motion=target, acc_weight=1.0, **BASE_WEIGHTS)
        g_acc_e = flat_grad(acc["acc_loss"], encoder_params, retain_graph=True)
        g_acc_d = flat_grad(acc["acc_loss"], decoder_params, retain_graph=False)

        cos = torch.nn.functional.cosine_similarity
        entry = dict(
            batch=i, num_windows=args.batch_size, frames=args.seq_len,
            first_meta=meta_rows[0], last_meta=meta_rows[-1],
            loss_base=float(base["loss"].detach()), loss_acc_value=float(acc["acc_loss"].detach()),
            encoder=dict(
                norm_base=float(g_base_e.norm()), norm_acc=float(g_acc_e.norm()),
                cosine=float(cos(g_base_e, g_acc_e, dim=0)),
                base_per_window=float(g_base_e.norm()) / args.batch_size,
                acc_per_window=float(g_acc_e.norm()) / args.batch_size,
            ),
            decoder=dict(
                norm_base=float(g_base_d.norm()), norm_acc=float(g_acc_d.norm()),
                cosine=float(cos(g_base_d, g_acc_d, dim=0)),
                base_per_window=float(g_base_d.norm()) / args.batch_size,
                acc_per_window=float(g_acc_d.norm()) / args.batch_size,
            ),
        )
        per_batch.append(entry)
        print(json.dumps(entry))

    after = state_fingerprint(model.state_dict())
    if before != after:
        raise RuntimeError("calibration mutated the model state (EMA off expected); abort")

    def suggest(group: str) -> dict:
        norms_base = np.asarray([b[group]["norm_base"] for b in per_batch], dtype=np.float64)
        norms_acc = np.asarray([b[group]["norm_acc"] for b in per_batch], dtype=np.float64)
        cosines = np.asarray([b[group]["cosine"] for b in per_batch], dtype=np.float64)
        ratio = norms_acc / np.maximum(norms_base, 1e-12)
        w_for_10pct = 0.10 / ratio
        w_for_30pct = 0.30 / ratio
        return dict(
            norm_base_mean=float(norms_base.mean()), norm_base_std=float(norms_base.std()),
            norm_acc_mean=float(norms_acc.mean()), norm_acc_std=float(norms_acc.std()),
            ratio_mean=float(ratio.mean()), ratio_std=float(ratio.std()),
            weight_for_10pct_mean=float(w_for_10pct.mean()),
            weight_for_30pct_mean=float(w_for_30pct.mean()),
            cosine_mean=float(cosines.mean()), cosine_min=float(cosines.min()),
            stable=bool(ratio.std() / max(ratio.mean(), 1e-12) < 0.5),
        )

    encoder = suggest("encoder")
    decoder = suggest("decoder")
    lo = max(encoder["weight_for_10pct_mean"], decoder["weight_for_10pct_mean"])
    hi = min(encoder["weight_for_30pct_mean"], decoder["weight_for_30pct_mean"])
    risk_flags = []
    if encoder["cosine_min"] < 0 or decoder["cosine_min"] < 0:
        risk_flags.append("negative cosine in at least one batch (conflicting gradients)")
    if not (encoder["stable"] and decoder["stable"]):
        risk_flags.append("ratio spread too large across batches (unstable norm scale)")
    selection_rule = "intersection of the encoder/decoder 10-30% windows"
    if hi <= lo:
        # Registered fallback decision rule (COMMON.md section 5): when one weight cannot put
        # both groups inside the candidate band, pick the geometric midpoint of the gap between
        # the two windows -- balanced log-space deviation from the nearer edge on each side.
        # This is a documented engineering choice, not a hardcoded value; the risk is flagged.
        enc_hi = encoder["weight_for_30pct_mean"]
        dec_lo = decoder["weight_for_10pct_mean"]
        recommended = round(float(np.sqrt(enc_hi * dec_lo)), 4)
        risk_flags.append(
            "no ACC_WEIGHT satisfies 10-30% on both encoder and decoder "
            f"(gap=[{enc_hi:.3f},{dec_lo:.3f}]); using gap geometric midpoint {recommended}"
        )
        selection_rule = "geometric midpoint of the gap between the two 10-30% windows"
        achieved = dict(
            encoder_ratio_at_recommended=float(recommended * encoder["ratio_mean"]),
            decoder_ratio_at_recommended=float(recommended * decoder["ratio_mean"]),
        )
    else:
        recommended = round(float(np.sqrt(lo * hi)), 4)
        achieved = dict(
            encoder_ratio_at_recommended=float(recommended * encoder["ratio_mean"]),
            decoder_ratio_at_recommended=float(recommended * decoder["ratio_mean"]),
        )

    payload = dict(
        protocol="COMMON.md section 5: 4 train batches B16xT8, no optimizer.step, EMA frozen",
        checkpoint=str(Path(args.ckpt).resolve()),
        checkpoint_state_sha256=before, state_unchanged=True,
        base_weights=BASE_WEIGHTS, seed=args.seed, batch_size=args.batch_size,
        seq_len=args.seq_len, device=str(device), environment=environment_report(),
        git=git_commit(), per_batch=per_batch,
        encoder=encoder, decoder=decoder,
        overlap_window=dict(lo=lo, hi=hi),
        selection_rule=selection_rule, achieved_at_recommended=achieved,
        recommended_acc_weight=recommended, risk_flags=risk_flags,
    )
    json_dump(args.out, payload)
    print(f"[S2] encoder ratio={encoder['ratio_mean']:.4g} cosine={encoder['cosine_mean']:.3f} | "
          f"decoder ratio={decoder['ratio_mean']:.4g} cosine={decoder['cosine_mean']:.3f}")
    print(f"[S2] rule={selection_rule}; window=[{lo:.4g}, {hi:.4g}] -> ACC_WEIGHT={recommended}")
    print(f"[S2] achieved: encoder={achieved['encoder_ratio_at_recommended']:.3f} "
          f"decoder={achieved['decoder_ratio_at_recommended']:.3f} of base gradient norm")
    if risk_flags:
        print(f"[S2][RISK] {risk_flags}")
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
