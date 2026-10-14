"""V1 gradient calibration: decoder-only statistics for the physical-position acc loss.

Read-only (no optimizer.step). 4 independent train batches B16 x T8, EMA path isolated with
`freeze_codebook=True`, model state fingerprinted before/after. Only the decoder group is
reported because V1 freezes E/Q by construction.

  python analysis/vq_controller_recovery/V1/v1_gradient_calibration.py [--device cuda]
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
sys.path.insert(0, str(HERE.parent / "C0"))

import loop_lib as lib  # noqa: E402
from Utils.reproducibility import state_fingerprint  # noqa: E402
from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: E402
from Utils.runtime import load_norm_stats  # noqa: E402

VQ = lib.VQ
BASE_WEIGHTS = dict(
    recon_type="l1", temporal_loss_type="l1",
    recon_weight=3.0, vq_weight=1.0, vel_weight=15.0, root_weight=3.0,
    code_align_weight=2.0, latent_l1_weight=0.01, latent_vel_weight=0.1,
    latent_margin_weight=0.0, roundtrip_weight=0.0,
)
OUT = HERE / "gradient_calibration.json"


def sample_batches(seed: int, batches: int, batch_size: int, seq_len: int):
    meta = json.loads((lib.vt.DB / "meta.json").read_text())
    segments = json.loads((lib.vt.DB / "segments.json").read_text())
    mean, std = load_norm_stats(VQ / "norm_stats.npz")
    x_source = np.memmap(lib.vt.DB / "X.f32", mode="r", dtype=np.float32,
                         shape=(int(meta["num_frames"]), 375))
    train = [s for s in segments if s.get("split") == "train" and not s.get("mirror", False)
             and int(s["num_frames"]) >= seq_len]
    for i in range(batches):
        rng = np.random.default_rng(seed + i)
        rows = np.empty((batch_size, seq_len, 375), dtype=np.float32)
        for j in range(batch_size):
            s = train[int(rng.integers(len(train)))]
            start = int(s["frame_start"]) + int(rng.integers(int(s["num_frames"]) - seq_len + 1))
            rows[j] = x_source[start:start + seq_len]
        yield torch.from_numpy((rows - mean) / std), mean, std


def flat_grad(loss: torch.Tensor, parameters, *, retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    pieces = [g.reshape(-1) for g in grads if g is not None]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=loss.device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=VQ / "best.pt")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--beta-cm", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_vqvae_from_checkpoint(args.ckpt, freeze=False).to(device).train()
    model.set_representation_frozen(True)   # the V1 contract: decoder-only trainable
    forward = model.quantizer.vq.forward
    model.quantizer.vq.forward = lambda *a, **kw: forward(*a, freeze_codebook=True, **kw)

    before = state_fingerprint(model.state_dict())
    decoder = [p for p in model.decoder.parameters() if p.requires_grad]
    assert decoder, "decoder must be trainable"
    encoder_frozen = all(not p.requires_grad for p in model.encoder.parameters())

    per_batch = []
    for i, (x, mean, std) in enumerate(sample_batches(args.seed, args.batches, args.batch_size, args.seq_len)):
        x = x.to(device)
        target = x[..., 84:]
        model.zero_grad(set_to_none=True)
        base = model.compute_loss(
            x=x, target_motion=target, posacc_weight=0.0,
            motion_mean=torch.as_tensor(mean, device=device), motion_std=torch.as_tensor(std, device=device),
            **BASE_WEIGHTS)
        g_base = flat_grad(base["loss"], decoder, retain_graph=False)

        model.zero_grad(set_to_none=True)
        pos = model.compute_loss(
            x=x, target_motion=target, posacc_weight=1.0, posacc_beta_cm=args.beta_cm,
            motion_mean=torch.as_tensor(mean, device=device), motion_std=torch.as_tensor(std, device=device),
            **BASE_WEIGHTS)
        g_pos = flat_grad(pos["posacc_loss"], decoder, retain_graph=False)

        cos = float(torch.nn.functional.cosine_similarity(g_base, g_pos, dim=0))
        ratio = float(g_pos.norm() / g_base.norm().clamp_min(1e-12))
        entry = dict(batch=i, loss_base=float(base["loss"].detach()),
                     loss_posacc=float(pos["posacc_loss"].detach()),
                     posacc_abs_cm=float(pos["posacc_abs_cm"].detach()),
                     decoder=dict(norm_base=float(g_base.norm()), norm_posacc=float(g_pos.norm()),
                                  ratio=ratio, cosine=cos))
        per_batch.append(entry)
        print(json.dumps(entry))

    after = state_fingerprint(model.state_dict())
    # With E/Q frozen (set_representation_frozen) and freeze_codebook=True, the calibration
    # forward passes must leave every tensor bitwise unchanged.
    assert before == after, "calibration mutated the model state"

    ratios = np.asarray([b["decoder"]["ratio"] for b in per_batch], dtype=np.float64)
    cosines = np.asarray([b["decoder"]["cosine"] for b in per_batch], dtype=np.float64)
    w_lo = (0.10 / ratios)
    w_hi = (0.30 / ratios)
    lo, hi = float(np.max(w_lo)), float(np.min(w_hi))
    risk = []
    if float(cosines.min()) < 0:
        risk.append("negative cosine (conflicting gradients)")
    if ratios.std() / max(ratios.mean(), 1e-12) > 0.5:
        risk.append("ratio spread too large across batches")
    if hi <= lo:
        risk.append(f"no single weight places the term in 10-30% (gap [{hi:.4g},{lo:.4g}])")
        lam = float(np.sqrt(max(hi, 1e-12) * max(lo, 1e-12)))
        rule = "gap geometric midpoint"
    else:
        lam = float(np.sqrt(lo * hi))
        rule = "10-30% window geometric centre"

    payload = dict(
        protocol="V1/PROTOCOL.md section 3: 4 train batches B16xT8, decoder group only, "
                 "no optimizer.step, EMA frozen, E/Q frozen",
        checkpoint=str(Path(args.ckpt).resolve()), checkpoint_state_sha256=before,
        state_unchanged=True, encoder_frozen=encoder_frozen,
        base_weights=BASE_WEIGHTS, beta_cm=args.beta_cm, seed=args.seed,
        batch_size=args.batch_size, seq_len=args.seq_len,
        environment=lib.vt.environment_report(), git=lib.vt.git_commit(),
        per_batch=per_batch,
        ratio_mean=float(ratios.mean()), ratio_std=float(ratios.std()),
        cosine_mean=float(cosines.mean()), cosine_min=float(cosines.min()),
        weight_window=[lo, hi], selection_rule=rule,
        recommended_posacc_weight=round(lam, 4), risk_flags=risk,
    )
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"[V1] ratio={ratios.mean():.4g}+-{ratios.std():.2g} cosine={cosines.mean():.3f} "
          f"window=[{lo:.4g},{hi:.4g}] -> POSACC_WEIGHT={round(lam,4)} ({rule})")
    if risk:
        print("[V1][RISK]", risk)
    print("[Done]", OUT)


if __name__ == "__main__":
    main()
