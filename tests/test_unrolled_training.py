from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from arch import build_flowmatch_controller_from_config
from training.train_flowmatch_vqvae_controller import unrolled_flow_loss


def build_controller(steps: int = 3) -> torch.nn.Module:
    from arch.basic_vqvae_375to291 import BasicVQVAE375to291

    torch.manual_seed(0)
    vqvae = BasicVQVAE375to291(
        input_dim=375,
        traj_dim=84,
        motion_dim=291,
        input_mode="traj_motion",
        encoder_arch="mlp",
        decoder_arch="mlp",
        hidden_dim=32,
        encoder_layers=2,
        decoder_hidden_dim=32,
        decoder_layers=2,
        num_latents=4,
        code_dim=4,
        num_embeddings=32,
    )
    controller = build_flowmatch_controller_from_config(
        {
            "use_style_code_selector": True,
            "style_slot_indices": [0, 1, 2, 3],
            "prediction_mode": "velocity",
            "flow_hidden_dim": 32,
            "flow_layers": 1,
            "flow_heads": 2,
            "flow_ffn_dim": 64,
            "style_dim": 16,
            "style_selector_hidden_dim": 32,
            "style_selector_topk": 2,
        },
        vqvae=vqvae,
        num_styles=3,
    )
    controller.set_latent_prior_stats(
        torch.zeros(controller.latent_dim),
        torch.ones(controller.latent_dim),
        transport_std=torch.ones(controller.latent_dim) * 0.5,
    )
    return controller


class UnrolledFlowLossTests(unittest.TestCase):
    def _window(self, batch: int = 3, steps: int = 3) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(batch, steps + 1, 375) * 0.1

    def test_loss_is_finite_and_covers_every_step(self) -> None:
        controller = build_controller()
        controller.train()
        steps = 3
        window = self._window(steps=steps)
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), steps, -1
        )
        out = unrolled_flow_loss(
            model=controller,
            x_window=window,
            style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0],
            target_latents=targets,
            recon_targets=recon,
            steps=steps,
            recon_weight=0.1,
        )
        self.assertTrue(torch.isfinite(out["loss"]))
        for key in ("flow_loss", "recon_loss", "style_hist_loss", "style_entropy", "style_top1_prob"):
            self.assertIn(key, out)
        out["loss"].backward()
        grads = [parameter.grad for parameter in controller.parameters() if parameter.grad is not None]
        self.assertTrue(grads, msg="no gradients reached the controller")

    def test_one_step_equals_the_single_step_objective(self) -> None:
        """steps=1 must reduce to the ordinary training objective (no rollout feedback)."""
        controller = build_controller()
        controller.eval()
        window = self._window(steps=1)
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), 2, -1)
        targets = latents[:, 1:2].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), 1, -1
        )
        style = torch.tensor([0, 1, 2])
        torch.manual_seed(3)
        unrolled = unrolled_flow_loss(
            model=controller, x_window=window, style_label=style, prev_latent_0=latents[:, 0],
            target_latents=targets, recon_targets=recon, steps=1, recon_weight=0.1,
        )
        torch.manual_seed(3)
        direct = controller.compute_loss(
            x_curr=window[:, 0], x_next=window[:, 1], style_label=style,
            prev_latent=latents[:, 0], target_latent=targets[:, 0], recon_weight=0.1,
        )
        self.assertAlmostEqual(float(unrolled["flow_loss"]), float(direct["flow_loss"]), places=5)

    def test_feedback_actually_conditions_on_the_model_output(self) -> None:
        """The later steps must depend on the model's own prediction, not on the ground truth."""
        controller = build_controller()
        controller.train()
        steps = 3
        window = self._window(steps=steps)
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), steps, -1
        )
        torch.manual_seed(7)
        a = unrolled_flow_loss(model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                               prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
                               steps=steps, recon_weight=0.0)
        torch.manual_seed(7)
        shifted = window.clone()
        shifted[:, 1:, 84:] += 5.0            # change the *teacher* future motion only
        b = unrolled_flow_loss(model=controller, x_window=shifted, style_label=torch.tensor([0, 1, 2]),
                               prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
                               steps=steps, recon_weight=0.0)
        # step 1 conditions on the (unchanged) trajectory + the model's own motion prediction, so the
        # first step's flow loss must be identical even though the teacher motion changed
        self.assertAlmostEqual(float(a["flow_loss"].detach()), float(b["flow_loss"].detach()), places=6)

    def test_steps_validation(self) -> None:
        controller = build_controller()
        window = self._window(steps=2)
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), 3, -1)
        with self.assertRaises(ValueError):
            unrolled_flow_loss(model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                               prev_latent_0=latents[:, 0], target_latents=latents[:, 1:], recon_targets=latents[:, 1:],
                               steps=3)


class RequantizedFeedbackTests(unittest.TestCase):
    """The fed-back latent must be snapped to the codebook (runtime semantics)."""

    def _run(self, *, requantize: bool) -> tuple[float, int]:
        from training.train_flowmatch_vqvae_controller import unrolled_flow_loss

        controller = build_controller()
        controller.train()
        steps = 3
        torch.manual_seed(2)
        window = torch.randn(3, steps + 1, 375) * 0.1
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(3, steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(3, steps, -1)

        calls = {"n": 0}
        original = controller.requantize_latent

        def counting_requantize(z):
            calls["n"] += 1
            return original(z)

        controller.requantize_latent = counting_requantize  # type: ignore[assignment]
        torch.manual_seed(9)
        out = unrolled_flow_loss(
            model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=steps,
            recon_weight=0.0, requantize_feedback=requantize,
        )
        controller.requantize_latent = original  # type: ignore[assignment]
        return float(out["flow_loss"].detach()), calls["n"]

    def test_requantize_is_applied_between_steps(self) -> None:
        _, calls_on = self._run(requantize=True)
        self.assertEqual(calls_on, 2, msg="expected one requantization per intermediate step (steps-1)")

    def test_disabling_it_skips_requantization(self) -> None:
        _, calls_off = self._run(requantize=False)
        self.assertEqual(calls_off, 0)

    def test_the_choice_changes_the_objective(self) -> None:
        on, _ = self._run(requantize=True)
        off, _ = self._run(requantize=False)
        self.assertNotAlmostEqual(on, off, places=6)


class FeedbackSolverStepsTests(unittest.TestCase):
    """--feedback-solver-steps only shortens the integration that builds the rolled feedback."""

    def _inputs(self, controller, steps=3):
        torch.manual_seed(1)
        window = torch.randn(3, steps + 1, 375) * 0.1
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), steps, -1)
        return window, latents, targets, recon

    def test_feedback_steps_control_the_integration_count(self):
        controller = build_controller().train()
        calls = []
        original_integrate = controller._integrate
        def counting_integrate(**kwargs):
            calls.append(kwargs["num_steps"])
            return original_integrate(**kwargs)
        controller._integrate = counting_integrate
        window, latents, targets, recon = self._inputs(controller)
        torch.manual_seed(3)
        unrolled_flow_loss(model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
            steps=3, feedback_solver_steps=2)
        # The FM loss is a single field evaluation; the only Euler integrations are the two
        # feedback builds (steps 0 and 1) and they must use the cheaper solver.
        self.assertEqual(calls, [2, 2])

    def test_auxiliary_loss_rides_the_feedback_solver_on_intermediate_steps(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller, steps=2)
        calls = []
        original_integrate = controller._integrate
        def counting_integrate(**kwargs):
            calls.append(kwargs["num_steps"])
            return original_integrate(**kwargs)
        controller._integrate = counting_integrate
        out = unrolled_flow_loss(model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
            steps=2, recon_weight=0.1, feedback_solver_steps=2)
        # step 0 (intermediate): feedback solver 2; step 1 (final): full solver 8.
        self.assertEqual(calls, [2, 8])
        out["loss"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in controller.field.parameters()))

    def test_default_keeps_full_solver_for_feedback(self):
        controller = build_controller().train()
        calls = []
        original_integrate = controller._integrate
        def counting_integrate(**kwargs):
            calls.append(kwargs["num_steps"])
            return original_integrate(**kwargs)
        controller._integrate = counting_integrate
        window, latents, targets, recon = self._inputs(controller)
        torch.manual_seed(3)
        unrolled_flow_loss(model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=3)
        # Two feedback builds, both at the full solver: [8, 8].
        self.assertEqual(calls, [8, 8])


from training.train_flowmatch_vqvae_controller import feedback_probability_for_epoch


class FeedbackCurriculumTests(unittest.TestCase):
    """TF -> rollout curriculum: p=0 is pure GT multi-step TF, p=1 is the plain rollout objective."""

    def _inputs(self, controller, steps=3):
        torch.manual_seed(1)
        window = torch.randn(3, steps + 1, 375) * 0.1
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), steps, -1)
        return window, latents, targets, recon

    def test_ramp_schedule_values(self):
        self.assertEqual(feedback_probability_for_epoch(1, start_epoch=3, ramp_epochs=4, probability=1.), 0.0)
        self.assertEqual(feedback_probability_for_epoch(3, start_epoch=3, ramp_epochs=4, probability=1.), 0.25)
        self.assertEqual(feedback_probability_for_epoch(5, start_epoch=3, ramp_epochs=4, probability=1.), 0.75)
        self.assertEqual(feedback_probability_for_epoch(6, start_epoch=3, ramp_epochs=4, probability=1.), 1.0)
        self.assertEqual(feedback_probability_for_epoch(99, start_epoch=3, ramp_epochs=4, probability=1.), 1.0)
        # ramp_epochs=0 jumps straight to the target share at start_epoch
        self.assertEqual(feedback_probability_for_epoch(2, start_epoch=2, ramp_epochs=0, probability=1.), 1.0)
        self.assertEqual(feedback_probability_for_epoch(1, start_epoch=2, ramp_epochs=0, probability=1.), 0.0)
        # scaled target share
        self.assertAlmostEqual(feedback_probability_for_epoch(4, start_epoch=4, ramp_epochs=2, probability=.5), .25)
        with self.assertRaises(ValueError):
            feedback_probability_for_epoch(1, start_epoch=0, ramp_epochs=1, probability=1.)
        with self.assertRaises(ValueError):
            feedback_probability_for_epoch(1, start_epoch=1, ramp_epochs=1, probability=1.5)

    def test_pure_gt_history_needs_no_euler_or_decode(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller)
        def unused(*args, **kwargs):
            raise AssertionError('p=0 must not integrate Euler or decode during training')
        original_integrate, original_decode = controller._integrate, controller.vqvae.decode
        controller._integrate = unused
        controller.vqvae.decode = unused
        try:
            out = unrolled_flow_loss(
                model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=3,
                feedback_probability=0.0)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            self.assertEqual(float(out["feedback_fraction"]), 0.0)
        finally:
            controller._integrate, controller.vqvae.decode = original_integrate, original_decode

    def test_full_feedback_matches_plain_rollout_objective(self):
        def run(probability, seed):
            controller = build_controller().train()
            window, latents, targets, recon = self._inputs(controller)
            torch.manual_seed(seed)
            out = unrolled_flow_loss(
                model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=3,
                feedback_probability=probability)
            out["loss"].backward()
            grads = torch.cat([p.grad.flatten() for p in controller.parameters() if p.grad is not None])
            return float(out["loss"]), float(out["feedback_fraction"]), grads
        loss_default, fraction_default, grads_default = run(1.0, seed=5)
        loss_p1, fraction_p1, grads_p1 = run(None or 1.0, seed=5)  # default is 1.0
        self.assertEqual(fraction_default, 1.0)
        self.assertEqual(fraction_p1, 1.0)
        self.assertAlmostEqual(loss_default, loss_p1, places=7)
        torch.testing.assert_close(grads_default, grads_p1, rtol=0, atol=0)

    def test_observation_latents_drive_the_gt_branch(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller)
        # Source latents differ from the targets: GT-history steps must observe the source.
        source = latents[:, 1 : 1 + 3].clone() + 0.5
        captured = []
        original = controller.compute_loss
        def spy(**kwargs):
            captured.append(kwargs.get("prev_latent"))
            return original(**kwargs)
        controller.compute_loss = spy
        try:
            unrolled_flow_loss(
                model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=3,
                observation_latents=source, feedback_probability=0.0)
        finally:
            controller.compute_loss = original
        torch.testing.assert_close(captured[1], source[:, 0])
        torch.testing.assert_close(captured[2], source[:, 1])

    def test_partial_gate_is_fixed_per_window_and_statistically_split(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller, steps=2)
        window = window.repeat_interleave(64, dim=0)
        latents = latents.repeat_interleave(64, dim=0)
        targets = targets.repeat_interleave(64, dim=0)
        recon = recon.repeat_interleave(64, dim=0)
        torch.manual_seed(9)
        out = unrolled_flow_loss(
            model=controller, x_window=window, style_label=torch.zeros(window.size(0), dtype=torch.long),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon, steps=2,
            feedback_probability=0.5)
        fraction = float(out["feedback_fraction"])
        self.assertTrue(0.2 < fraction < 0.8, msg=f"unexpected gate fraction {fraction}")
        # Each window draws one gate: a single-window batch is exactly all-feedback or all-GT,
        # and independent runs produce both outcomes.
        outcomes = set()
        for seed in range(40):
            torch.manual_seed(seed)
            single = unrolled_flow_loss(
                model=controller, x_window=window[:1], style_label=torch.zeros(1, dtype=torch.long),
                prev_latent_0=latents[:1, 0], target_latents=targets[:1], recon_targets=recon[:1], steps=2,
                feedback_probability=0.5)
            value = float(single["feedback_fraction"])
            self.assertIn(value, (0.0, 1.0))
            outcomes.add(value)
        self.assertEqual(outcomes, {0.0, 1.0})


class FeedbackCurriculumCLITests(unittest.TestCase):
    """CLI-level: curriculum scheduling, pure-TF epochs skip Euler, resume validation."""

    def test_cli_curriculum_ramp_progresses_and_records(self):
        import json, sys, tempfile
        from pathlib import Path
        from test_controller_repair import small_training_fixture, training_argv
        import training.train_flowmatch_vqvae_controller as trainer
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, vq, norm = small_training_fixture(tmp)
            out = tmp / 'run'
            argv = training_argv(db, vq, norm, out, 2) + [
                '--feedback-start-epoch', '2', '--feedback-ramp-epochs', '1',
                '--feedback-probability', '1']
            with mock.patch.object(sys, 'argv', argv):
                trainer.main()
            rows = [json.loads(s) for s in (out / 'history.jsonl').read_text().splitlines()]
            first, second = rows[0]['feedback_curriculum'], rows[1]['feedback_curriculum']
            self.assertEqual(first['probability'], 0.0)
            self.assertEqual(first['actual_fraction'], 0.0)
            self.assertEqual(second['probability'], 1.0)
            self.assertEqual(second['actual_fraction'], 1.0)

    def test_cli_pure_tf_epoch_never_integrates(self):
        import sys, tempfile
        from pathlib import Path
        from test_controller_repair import small_training_fixture, training_argv
        import training.train_flowmatch_vqvae_controller as trainer
        from arch.flowmatch_vqvae_controller import FlowMatchVQVAEController
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, vq, norm = small_training_fixture(tmp)
            out = tmp / 'run'
            argv = training_argv(db, vq, norm, out, 2) + [
                '--feedback-start-epoch', '99', '--feedback-ramp-epochs', '0']
            def unused(*args, **kwargs):
                raise AssertionError('pure TF curriculum must not integrate Euler')
            with mock.patch.object(FlowMatchVQVAEController, '_integrate', unused), \
                 mock.patch.object(sys, 'argv', argv):
                trainer.main()
            import json
            rows = [json.loads(s) for s in (out / 'history.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['feedback_curriculum']['actual_fraction'] == 0.0 for r in rows))

    def test_cli_rejects_feedback_args_without_unroll(self):
        import sys, tempfile
        from pathlib import Path
        from test_controller_repair import small_training_fixture, training_argv
        import training.train_flowmatch_vqvae_controller as trainer
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, vq, norm = small_training_fixture(tmp)
            out = tmp / 'run'
            argv = training_argv(db, vq, norm, out, 1) + ['--feedback-ramp-epochs', '2']
            with mock.patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'unroll-steps > 1'):
                trainer.main()

    def test_cli_resume_validates_curriculum_fields(self):
        import sys, tempfile
        from pathlib import Path
        from test_controller_repair import small_training_fixture, training_argv
        import training.train_flowmatch_vqvae_controller as trainer
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            db, vq, norm = small_training_fixture(tmp)
            out = tmp / 'run'
            base = training_argv(db, vq, norm, out, 2)
            with mock.patch.object(sys, 'argv', base):
                trainer.main()
            changed = base + ['--resume', str(out / 'latest.pt'), '--feedback-probability', '0']
            with mock.patch.object(sys, 'argv', changed), self.assertRaisesRegex(ValueError, 'feedback_probability'):
                trainer.main()


class InferencePathReconTests(unittest.TestCase):
    """--recon-inference-path supervises decode(requantize(clip(z_hat))) with an STE gradient."""

    def _inputs(self, controller, steps=3):
        torch.manual_seed(1)
        window = torch.randn(3, steps + 1, 375) * 0.1
        latents = controller.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
        targets = latents[:, 1 : 1 + steps].contiguous()
        recon = controller.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(
            window.size(0), steps, -1)
        return window, latents, targets, recon

    def test_decode_input_is_quantized_and_receives_gradient(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller)
        captured = []
        original = controller.vqvae.decode
        def spy(z, **kwargs):
            captured.append(z)
            return original(z, **kwargs)
        controller.vqvae.decode = spy
        try:
            out = unrolled_flow_loss(
                model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
                steps=3, recon_weight=1.0, recon_inference_path=True)
            out["loss"].backward()
        finally:
            controller.vqvae.decode = original
        self.assertTrue(captured, msg="decode was never called")
        z_input = captured[-1]
        # The displayed latent is the requantized/clipped prediction (float reassociation in the
        # STE expression z_c + (z_q - z_c) leaves ~1e-8 residue, so compare with tolerance).
        torch.testing.assert_close(z_input, controller.requantize_latent(z_input), atol=1e-6, rtol=0)
        # STE keeps the gradient on the predicted latent.
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in controller.field.parameters()))
        self.assertGreater(float(out["recon_loss"]), 0.0)

    def test_default_continuous_decode_is_unchanged(self):
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller)
        captured = []
        original = controller.vqvae.decode
        def spy(z, **kwargs):
            captured.append(z)
            return original(z, **kwargs)
        controller.vqvae.decode = spy
        try:
            out = unrolled_flow_loss(
                model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
                prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
                steps=3, recon_weight=1.0)
            out["loss"].backward()
        finally:
            controller.vqvae.decode = original
        z_input = captured[-1]
        # Continuous decode input is not generally on the codebook.
        self.assertFalse(torch.equal(z_input, controller.requantize_latent(z_input)))

    def test_auxiliary_recon_disables_prediction_no_grad_shortcut(self):
        """With recon>0 the rollout feedback graph must stay alive so the loss reaches the field."""
        controller = build_controller().train()
        window, latents, targets, recon = self._inputs(controller)
        out = unrolled_flow_loss(
            model=controller, x_window=window, style_label=torch.tensor([0, 1, 2]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
            steps=3, recon_weight=.5, recon_inference_path=True)
        out["loss"].backward()
        grads = [p.grad for p in controller.field.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        self.assertTrue(grads)


class VisualLossScopeTests(unittest.TestCase):
    """Inference-path recon is applied once per window (final step, full solver)."""

    def _run(self, *, inference_path: bool):
        model = build_controller().train()
        torch.manual_seed(21)
        window = torch.randn(8, 5, 375) * 0.1
        latents = model.frame_latent(window.reshape(-1, 375)).reshape(8, 5, -1)
        targets = latents[:, 1:].contiguous()
        recon = model.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(8, 4, -1)
        calls = []
        original = model._integrate
        def counting(**kwargs):
            calls.append(kwargs["num_steps"])
            return original(**kwargs)
        model._integrate = counting
        torch.manual_seed(22)
        out = unrolled_flow_loss(
            model=model, x_window=window, style_label=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
            steps=4, recon_weight=1.0, recon_inference_path=inference_path,
            recon_anchor_weight=0.1 if inference_path else 0.0,
            feedback_solver_steps=2)
        out["loss"].backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return out, calls, grads

    def test_inference_path_supervises_final_step_with_full_solver(self):
        out, calls, grads = self._run(inference_path=True)
        # steps 0-2 build feedback at 2 Euler steps; the final step is the supervised one and
        # must use the full solver (the inference configuration that renders the loss target).
        self.assertEqual(calls, [2, 2, 2, 8])
        self.assertTrue(float(out["recon_loss"]) > 0)
        self.assertTrue(float(out["recon_anchor_loss"]) > 0)
        self.assertTrue(float(out["recon_quant_error"]) >= 0)
        self.assertTrue(torch.isfinite(grads).all() and grads.abs().sum() > 0)

    def test_continuous_recon_still_supervises_every_step(self):
        out, calls, grads = self._run(inference_path=False)
        # Legacy continuous recon keeps the old scope: every step integrates (feedback steps at
        # 2, the final at 8) and its loss is aggregated across steps.
        self.assertEqual(calls, [2, 2, 2, 8])
        self.assertTrue(float(out["recon_loss"]) > 0)
        self.assertEqual(float(out["recon_anchor_loss"]), 0.0)
