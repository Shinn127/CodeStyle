"""C0 shared closed-loop diagnostic library (read-only; no training, no parameter updates).

Metric helpers are imported from analysis/vq_temporal_source/vq_temporal_common.py so the
C0 numbers share the exact aggregation semantics verified against the archived S1 audit
(joint-frame L2, cm, official Jitter identity, activity proxies, zero-block world layout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # .../vq_controller_recovery/C0
RECOVERY = HERE.parent                          # .../vq_controller_recovery
ROOT = RECOVERY.parents[1]                      # .../NewCode
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "vq_temporal_source"))

import vq_temporal_common as vt  # noqa: E402  (verified S1 metric helpers)
from Utils.interactive_control import InteractiveTrajectory  # noqa: E402
from Utils.motion_eval import MotionEvalConfig, compute_sequence_metric_sums  # noqa: E402
from Utils.motion375 import reconstruct_root_track_from_375  # noqa: E402
from Utils.reproducibility import case_noise_plan, file_sha256, state_fingerprint  # noqa: E402
from Utils.runtime import load_norm_stats  # noqa: E402
from Utils.motion_eval import load_motion_model  # noqa: E402

VQ = ROOT / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
CONTROLLER = ROOT / "checkpoints" / "flowmatch_label_history_obsnoise_tf100"
SEED = 42
NOISE_STD = 1.0
T_FRAMES = 241          # row 0 = GT seed, 240 generated transitions
T0_PERTURB = 120
PERTURB_AMPLITUDE = 0.5                     # in units of per-dim transport_std (L2 = amp*||transport_std||)
PERTURB_SEED = 4242
BUCKETS = ((1, 16), (17, 32), (33, 60), (61, 120), (121, 240))
PLANNER_SCHEDULE = (
    # (end_frame_inclusive, forward, side, turn, speed)
    (60, 1.0, 0.0, 0.0, 1.2),
    (120, 1.0, 0.0, 0.3, 1.2),
    (180, 0.8, 0.2, 0.0, 1.0),
    (240, 1.0, 0.0, -0.2, 1.2),
)


def planner_command(frame: int) -> tuple[float, float, float, float]:
    for end, forward, side, turn, speed in PLANNER_SCHEDULE:
        if frame <= end:
            return forward, side, turn, speed
    raise ValueError(f"frame {frame} outside the 240-frame schedule")


def c0_config() -> dict:
    return dict(
        seed=SEED, noise_std=NOISE_STD, frames=T_FRAMES,
        perturb=dict(t0=T0_PERTURB, amplitude=PERTURB_AMPLITUDE, seed=PERTURB_SEED),
        buckets=[list(b) for b in BUCKETS],
        planner_schedule=[list(row) for row in PLANNER_SCHEDULE],
        vq_ckpt=str(VQ / "best.pt"), vq_state_sha256=state_fingerprint(
            load_state(VQ / "best.pt")),
        controller_ckpt=str(CONTROLLER / "best.pt"), controller_sha256=file_sha256(CONTROLLER / "best.pt"),
        norm_sha256=file_sha256(VQ / "norm_stats.npz"),
    )


def load_state(path: Path):
    from Utils.runtime import sanitize_state_dict_keys
    return sanitize_state_dict_keys(torch.load(path, map_location="cpu", weights_only=False)["model"])


def load_controller(device: torch.device):
    model = load_motion_model("flowmatch_target", CONTROLLER / "best.pt", num_styles=90).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean, std = load_norm_stats(VQ / "norm_stats.npz")
    return model, mean, std


@torch.no_grad()
def style_feature_for(model, x_seed: torch.Tensor, label: int, device) -> torch.Tensor:
    return model.encode_conditions(
        x_curr=x_seed, style_label=torch.tensor([label], device=device))[2].detach()


@torch.no_grad()
def gt_latents(model, x_norm: torch.Tensor) -> torch.Tensor:
    return model.frame_latent(x_norm)   # Q(E(frame)) raw-space latents [T,128]


@torch.no_grad()
def sample_step(model, *, x_curr, prev_latent, history_latent, style_feature, z0, steps=4,
                requantize=True, noise_std=NOISE_STD):
    z_hat = model.sample_latent(
        x_curr=x_curr, style_label=None, style_feature=style_feature,
        prev_latent=prev_latent, history_latent=history_latent,
        z0=z0, num_steps=steps, noise_std=noise_std, solver="euler")
    motion, latent = model.feedback_state(z_hat, requantize=requantize, latent_clip=True)
    return motion, latent


@torch.no_grad()
def run_teacher_anchored(model, *, x_norm, latents_gt, style_feature, z0_plan, mode,
                         steps=4, chunk=256):
    """C0-A (mode='raw') / C0-B (mode='decoded') batched over frames; official output path."""
    transitions = x_norm.shape[0] - 1
    traj = x_norm[1:transitions + 1, :84]
    if mode == "raw":
        motion_token = x_norm[:transitions, 84:]
    elif mode == "decoded":
        motion_token = model.vqvae.decode(latents_gt[:transitions])
    else:
        raise ValueError(mode)
    x_curr = torch.cat((traj, motion_token), dim=-1)
    prev = latents_gt[:transitions]
    history = torch.cat((latents_gt[:1], latents_gt[:transitions - 1]), dim=0)
    z0 = z0_plan[:transitions]   # case_noise_plan already returns raw-space z0 (offset+scale included)
    # one style feature per source -> expand once, then slice per chunk
    style = style_feature.expand(transitions, -1)[:, :style_feature.shape[-1]]
    motions, latents = [], []
    for begin in range(0, transitions, chunk):
        end = min(begin + chunk, transitions)
        motion, latent = sample_step(
            model, x_curr=x_curr[begin:end], prev_latent=prev[begin:end],
            history_latent=history[begin:end], style_feature=style[begin:end],
            z0=z0[begin:end], steps=steps, requantize=True)
        motions.append(motion)
        latents.append(latent)
    generated = torch.cat(motions, dim=0)
    # predicted latents for the code_match diagnostic (frame t prediction at slot index t)
    predicted_latents = torch.cat((latents_gt[:1], torch.cat(latents, dim=0)), dim=0)
    # row 0 is the GT seed; rows 1..240 are the teacher-anchored outputs
    out = x_norm.clone()
    out[1:transitions + 1, 84:] = generated
    return out, predicted_latents, dict(invariant_violations=0)


@torch.no_grad()
def run_sequence(model, *, x_norm, label, style_feature, z0_plan,
                 mode="official", steps=4, planner=None, perturb=None):
    """Sequential rollout. mode in {official, continuous, planner}.

    Returns (generated_raw 375 [T,375], latents [T,128], diagnostics dict).
    Enforces the C0 state invariants on every step (motion == decode(carried latent)).
    `perturb=(direction, amplitude, t0)` injects the registered bounded perturbation at
    transition t0 only; the paired clean branch is produced by the same call with perturb=None.
    """
    device = x_norm.device
    transitions = x_norm.shape[0] - 1
    requantize = mode != "continuous"
    q = gt_latents(model, x_norm[0:1])
    current_motion = model.vqvae.decode(q)
    history_latent = q.clone()
    mean_t = torch.as_tensor(MEAN, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(STD, device=device, dtype=torch.float32)
    # raw rows: row 0 = raw GT seed frame; rows t = [traj used, generated motion] in raw units
    raw_rows = [(x_norm[0].detach().cpu() * std_t.cpu() + mean_t.cpu())]
    latents = [q[0].detach().cpu()]
    diagnostics = dict(invariant_violations=0, planner_speed=[], planner_dir_err_deg=[],
                       perturbed_at=None)
    planner_state = planner if isinstance(planner, InteractiveTrajectory) else None
    for t in range(1, transitions + 1):
        if planner_state is not None:
            raw_prev = current_motion[0] * std_t[84:] + mean_t[84:]
            planner_state.advance(raw_prev[:3].detach().cpu().numpy())
            forward, side, turn, speed = planner_command(t)
            traj_raw, _ = planner_state.condition(forward, side, turn, speed=speed)
            traj_norm = (torch.as_tensor(traj_raw, device=device, dtype=torch.float32) - mean_t[:84]) / std_t[:84]
            traj_t = traj_norm[None]
        else:
            traj_t = x_norm[t:t + 1, :84]
        injected = perturb is not None and t == perturb[2]
        q_before_injection = None
        if injected:
            q_before_injection = q.clone()
            direction, amplitude, _ = perturb
            delta = amplitude * model.latent_transport_std.to(direction) * direction
            q_perturbed_raw = q + delta[None]
            current_motion, q = model.feedback_state(q_perturbed_raw, requantize=requantize, latent_clip=True)
            diagnostics["perturbed_at"] = int(t)
            diagnostics["perturbation_l2_raw"] = float(delta.norm())
            if requantize:
                before_codes = code_indices(model, q_before_injection)
                after_codes = code_indices(model, q)
                diagnostics["perturbation_slots_changed"] = int((before_codes != after_codes).sum())
                diagnostics["perturbation_slots_total"] = int(before_codes.numel())
            # clip absorption: how much of the injected delta survives clipping (raw L2)
            diagnostics["perturbation_l2_after_clip"] = float(
                (model.clip_latent_to_data_bounds(q_perturbed_raw) - q_before_injection).norm())
        x_curr = torch.cat((traj_t, current_motion), dim=-1)
        motion_new, q_new = sample_step(
            model, x_curr=x_curr, prev_latent=q, history_latent=history_latent,
            style_feature=style_feature, z0=z0_plan[t - 1][None],
            steps=steps, requantize=requantize)
        # invariant 1: carried motion token equals decode(carried latent)
        if not torch.equal(model.vqvae.decode(q_new), motion_new):
            diagnostics["invariant_violations"] += 1
        # invariant 2: history is exactly the previous step's carried latent. At the injected
        # step the carried state was the perturbed observation q'; the NEXT step's history must
        # be the true (unperturbed) latent of t0-1, so the single-step injection does not
        # permanently rewrite the feedback history (documented in PROTOCOL section 5).
        history_latent = q_before_injection if injected else q
        q = q_new
        current_motion = motion_new
        if planner_state is not None:
            diagnostics["planner_speed"].append(float(np.linalg.norm(planner_state.velocity)))
            intent = planner_state.intent_velocity
            velocity = planner_state.velocity
            if np.linalg.norm(intent) > 1e-6 and np.linalg.norm(velocity) > 1e-6:
                cos = float(np.clip(np.dot(velocity, intent) / (np.linalg.norm(velocity) * np.linalg.norm(intent)), -1, 1))
                diagnostics["planner_dir_err_deg"].append(float(np.degrees(np.arccos(cos))))
            else:
                diagnostics["planner_dir_err_deg"].append(None)
        raw_rows.append((torch.cat((traj_t, current_motion), -1)[0] * std_t + mean_t).detach().cpu())
        latents.append(q[0].detach().cpu())
    return torch.stack(raw_rows), torch.stack(latents), diagnostics


MEAN: np.ndarray | None = None
STD: np.ndarray | None = None


def set_norm(mean, std) -> None:
    global MEAN, STD
    MEAN, STD = np.asarray(mean, np.float32), np.asarray(std, np.float32)


def z0_plan_for(model, start: int, label: int, device) -> torch.Tensor:
    """Deterministic per-(source, step) transport noise; identical across C0 groups."""
    plan = case_noise_plan(model, [int(start)], [int(label)], T_FRAMES, seed=SEED, noise_std=NOISE_STD)
    return plan[0].to(device)  # [T-1, 128] raw units


def perturb_direction(model, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(PERTURB_SEED)
    vector = torch.randn(model.latent_dim, generator=generator)
    return (vector / vector.norm()).to(device)


@torch.no_grad()
def code_indices(model, latents: torch.Tensor) -> torch.Tensor:
    """Nearest-code per slot for raw latents [T,128] -> [T,32] (diagnostic only)."""
    table = model.vqvae.quantizer.embedding_weight.detach().float().to(latents.device)
    if table.dim() == 3:
        table = table[0]
    x = latents.float().to(table.device).reshape(-1, model.vqvae.code_dim)
    out = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for begin in range(0, x.size(0), 8192):
        chunk = x[begin:begin + 8192]
        out[begin:begin + 8192] = torch.cdist(chunk, table).argmin(-1)
    return out.reshape(latents.shape[0], model.vqvae.num_latents)


def frame_metrics_local(gen_local: np.ndarray, gt_local: np.ndarray) -> dict:
    """Per-frame 24-joint local-position error + official Jitter values."""
    err = np.linalg.norm(gen_local - gt_local, axis=-1).reshape(-1)  # cm per joint-frame
    jitter = vt.jitter_values_cm(gen_local)
    return dict(local_mpjpe=err, jitter=jitter)


def root_velocity_series(raw_375: np.ndarray, *, dt: float = 1 / 60.0):
    positions, _ = reconstruct_root_track_from_375(raw_375, dt=dt, root_wy_in_degrees=True)
    velocity = np.diff(positions[:, [0, 2]], axis=0) / dt
    return positions, velocity


def velocity_dir_error_deg(gen_vel: np.ndarray, ref_vel: np.ndarray, *, min_speed_cm_s: float = 1.0) -> np.ndarray:
    speed = np.linalg.norm(ref_vel, axis=-1) * 100.0
    valid = speed >= min_speed_cm_s
    out = np.full(len(gen_vel), np.nan)
    for i in range(len(gen_vel)):
        if not valid[i]:
            continue
        a, b = gen_vel[i], ref_vel[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9:
            out[i] = np.nan
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
        out[i] = float(np.degrees(np.arccos(cos)))
    return out


def official_metrics(reference_raw: np.ndarray, prediction_raw: np.ndarray, cfg: MotionEvalConfig) -> dict:
    sums = compute_sequence_metric_sums(reference_375=reference_raw, prediction_375=prediction_raw, cfg=cfg)
    def mean_of(total, count):
        return float(total / count) if count else float("nan")
    return dict(
        mpjpe=mean_of(sums.mpjpe_sum_cm, sums.mpjpe_count),
        mpjpe_count=int(sums.mpjpe_count),
        root_ade=mean_of(sums.root_ade_sum_cm, sums.root_ade_count),
        jitter=mean_of(sums.jitter_sum_cm, sums.jitter_count),
        fs_local=mean_of(sums.foot_sliding_sum_cm, sums.foot_sliding_count),
        fs_local_count=int(sums.foot_sliding_count),
        fs_world=mean_of(sums.foot_sliding_world_sum_cm, sums.foot_sliding_world_count),
        fs_world_count=int(sums.foot_sliding_world_count),
        contact_possible=int(sums.foot_contact_possible_count),
    )


class StyleClassifier:
    """Frozen 100-class classifier on 240 generated frames (matched proxy only)."""

    def __init__(self, device):
        from Utils.controller_validation import classifier_label_mapping
        from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint
        names = json.loads((vt.DB / "StyleNames.json").read_text())
        self.classifier, self.meta = build_fid_feature_extractor_from_checkpoint(vt.CLASSIFIER_CKPT, map_location="cpu")
        if int(self.meta.get("window_size", 0)) != 240:
            raise ValueError(f"style classifier window must be 240, got {self.meta.get('window_size')}")
        self.classifier = self.classifier.to(device).eval().requires_grad_(False)
        self.label_map = classifier_label_mapping(names, self.meta, device=device)
        self.device = device

    @torch.no_grad()
    def classify(self, raw_241: np.ndarray, label: int) -> dict:
        from Utils.fid_feature_extractor import motion375_clip_to_classifier_input
        window = raw_241[1:241]  # exclude the GT seed frame, exactly 240 frames
        features = motion375_clip_to_classifier_input(window, joint_set=self.meta["joint_set"]).to(self.device)
        _, logits = self.classifier(features)
        prediction = int(logits.argmax(1).item())
        target = int(self.label_map[int(label)].item())
        return dict(predicted=prediction, target=target, correct=int(prediction == target),
                    confidence=float(torch.softmax(logits, 1).max().item()))


def bucket_slice(bucket: tuple[int, int]) -> slice:
    lo, hi = bucket
    return slice(lo - 1, hi)   # frame t maps to index t-1 in per-transition arrays


def summarise_source(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if not isinstance(value, (list, np.ndarray))}
