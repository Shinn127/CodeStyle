"""Decoder-only calibration and frozen-teacher alignment (docs/VQ_DECODE_CONTINUITY_PLAN.md).

Contract under test:
* `set_representation_frozen` freezes encoder parameters AND keeps the quantizer's EMA/dead-code
  path in eval even after `train()` (parameter freezing alone does not freeze the EMA buffer).
* The trained model's representation (encoder+quantizer) is bit-identical before/after a
  forward/backward in decoder-only mode; q and code indices do not move.
* The teacher alignment term pulls D(q) toward D0(E0(x)) and gives gradients to the decoder only;
  the teacher module never enters the trained state_dict.
* Decoder-only runs reject objectives that would need the frozen parts.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_basic_vqvae as vq_fixtures
import training.train_basic_vqvae_375to291 as trainer


def _model():
    return vq_fixtures.BoundaryMarginAndRoundtripTests._build()


def _fingerprint(model):
    return trainer.state_dict_fingerprint(model.state_dict(), prefix=('encoder.', 'quantizer.'))


def test_decoder_only_freezes_encoder_params_and_ema_and_keeps_q_stable():
    model = _model()
    x = torch.randn(2, 3, 375)
    model.eval()
    with torch.no_grad():
        _, idx_before, _ = model.encode(x)
        q_before = model.quantizer.embedding_weight.clone()
    model.set_representation_frozen(True)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    # The critical part: train() must not re-enable the EMA path.
    model.train()
    assert not model.encoder.training and not model.quantizer.training and model.decoder.training
    rep_before = _fingerprint(model)
    out = model.compute_loss(x, teacher_align_weight=0.0)
    out['loss'].backward()
    rep_after = _fingerprint(model)
    assert rep_before == rep_after
    with torch.no_grad():
        _, idx_after, _ = model.encode(x)
    assert torch.equal(idx_before, idx_after)
    assert torch.equal(q_before, model.quantizer.embedding_weight)
    # Gradients reached the decoder only.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())


def test_teacher_alignment_targets_the_teacher_decode_and_gives_decoder_gradients():
    student = _model()
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 3, 375)
    student.train()
    target = trainer._teacher_target(teacher, x)
    assert target.shape == (2, 3, 291)
    with torch.no_grad():
        continuous = teacher.decode(teacher.encode_pre_vq(x))
    torch.testing.assert_close(target, continuous, atol=0, rtol=0)
    out = student.compute_loss(x, teacher_align_weight=1.0, teacher_motion=target)
    assert float(out['teacher_align_loss']) >= 0.0
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.decoder.parameters())
    # Teacher never accumulates gradients.
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_alignment_requires_a_matching_target():
    model = _model()
    x = torch.randn(2, 3, 375)
    with pytest.raises(ValueError, match='teacher_motion'):
        model.compute_loss(x, teacher_align_weight=1.0)
    with pytest.raises(ValueError, match='shape mismatch'):
        model.compute_loss(x, teacher_align_weight=1.0, teacher_motion=torch.randn(2, 3, 290))


def test_teacher_never_enters_the_trained_state_dict():
    student = _model()
    teacher = copy.deepcopy(student).eval()
    student_keys = set(student.state_dict())
    teacher_keys = set(teacher.state_dict())
    assert student_keys == teacher_keys
    # A teacher is a separate module; nothing named teacher.* exists in the trained model.
    assert not any(k.startswith('teacher') for k in student_keys)


def test_decoder_only_cli_rejects_frozen_part_objectives(tmp_path, monkeypatch):
    base = ['vq', '--db', str(tmp_path / 'db'), '--out-dir', str(tmp_path / 'out'),
            '--device', 'cpu', '--epochs', '1', '--decoder-only', '--init-ckpt', str(tmp_path / 'x.pt')]

    def parsed(extra):
        monkeypatch.setattr(sys, 'argv', base + extra)
        return trainer.parse_args()

    # Margin/roundtrip train the frozen encoder: hard error.
    for extra, needle in ((['--latent-margin-weight', '1'], 'latent-margin-weight'),
                          (['--roundtrip-weight', '1'], 'roundtrip-weight')):
        with pytest.raises(ValueError, match=needle):
            trainer.validate_args(parsed(extra))
    # Frozen-representation regularizers are auto-zeroed instead (wrapper defaults would
    # otherwise make the mode unusable).
    args = parsed(['--vq-weight', '1', '--code-align-weight', '2'])
    trainer.validate_args(args)
    assert args.vq_weight == 0.0 and args.code_align_weight == 0.0
    # Decoder-only without --init-ckpt is also rejected.
    monkeypatch.setattr(sys, 'argv', ['vq', '--db', str(tmp_path / 'db'),
                                      '--out-dir', str(tmp_path / 'out'), '--decoder-only'])
    with pytest.raises(ValueError, match='init-ckpt'):
        trainer.validate_args(trainer.parse_args())


def test_teacher_flags_are_mutually_required():
    import argparse
    def parsed(argv):
        old = sys.argv
        sys.argv = ['vq', *argv]
        try:
            return trainer.parse_args()
        finally:
            sys.argv = old
    with pytest.raises(ValueError, match='teacher-ckpt'):
        trainer.validate_args(parsed(['--teacher-align-weight', '2']))
    with pytest.raises(ValueError, match='teacher-align-weight'):
        trainer.validate_args(parsed(['--teacher-ckpt', '/tmp/x.pt']))


def test_validate_teacher_match_rejects_a_different_representation():
    a = _model()
    b = _model()
    with torch.no_grad():
        next(b.encoder.parameters()).add_(0.01)
    with pytest.raises(ValueError, match='alignment path'):
        trainer.validate_teacher_match(a, b)


def test_teacher_fingerprint_detects_codebook_drift():
    a = _model()
    b = copy.deepcopy(a)
    assert trainer.state_dict_fingerprint(a.state_dict(), prefix=('encoder.', 'quantizer.')) == \
        trainer.state_dict_fingerprint(b.state_dict(), prefix=('encoder.', 'quantizer.'))
    with torch.no_grad():
        b.quantizer.vq._codebook.embed.add_(1.0)
    assert trainer.state_dict_fingerprint(a.state_dict(), prefix=('encoder.', 'quantizer.')) != \
        trainer.state_dict_fingerprint(b.state_dict(), prefix=('encoder.', 'quantizer.'))


def test_resume_guard_covers_decoder_only_and_teacher():
    args = SimpleNamespace(code_align_weight=0., latent_margin_weight=0., roundtrip_weight=0.,
                           decoder_only=True, teacher_align_weight=2.0, teacher_fingerprint='abc',
                           latent_margin_mode='nearest_boundary', latent_margin_target=.05)
    old = {'args': dict(code_align_weight=0., latent_margin_weight=0., roundtrip_weight=0.,
                        decoder_only=False)}
    with pytest.raises(ValueError, match='decoder_only'):
        trainer.validate_stability_resume(args, old)
    old = {'args': dict(code_align_weight=0., latent_margin_weight=0., roundtrip_weight=0.,
                        decoder_only=True, teacher_align_weight=1.0, teacher_fingerprint='abc')}
    with pytest.raises(ValueError, match='teacher_align_weight'):
        trainer.validate_stability_resume(args, old)
    old['args']['teacher_align_weight'] = 2.0
    old['args']['teacher_fingerprint'] = 'different'
    with pytest.raises(ValueError, match='Resume teacher differs'):
        trainer.validate_stability_resume(args, old)
    old['args']['teacher_fingerprint'] = 'abc'
    trainer.validate_stability_resume(args, old)


def test_teacher_fingerprint_covers_the_decoder():
    """The alignment target is D0(E0(x)); a different teacher decoder must be detected."""
    a = _model()
    b = copy.deepcopy(a)
    assert trainer.state_dict_fingerprint(a.state_dict(), prefix=trainer.TEACHER_FINGERPRINT_PREFIXES) == \
        trainer.state_dict_fingerprint(b.state_dict(), prefix=trainer.TEACHER_FINGERPRINT_PREFIXES)
    # Change ONLY the decoder: encoder+quantizer stay identical, the target would change.
    with torch.no_grad():
        next(b.decoder.parameters()).add_(0.01)
    assert trainer.state_dict_fingerprint(a.state_dict(), prefix=('encoder.', 'quantizer.')) == \
        trainer.state_dict_fingerprint(b.state_dict(), prefix=('encoder.', 'quantizer.'))
    assert trainer.state_dict_fingerprint(a.state_dict(), prefix=trainer.TEACHER_FINGERPRINT_PREFIXES) != \
        trainer.state_dict_fingerprint(b.state_dict(), prefix=trainer.TEACHER_FINGERPRINT_PREFIXES)
    with pytest.raises(ValueError, match='alignment path'):
        trainer.validate_teacher_match(a, b)


def test_teacher_build_does_not_perturb_global_rng(monkeypatch):
    """Building the teacher must not consume the global RNG stream (paired A/B sampling)."""
    args = SimpleNamespace(seed=7, model_type='x')
    calls = {'n': 0}

    def fake_build(args_, device):
        calls['n'] += 1
        return build_model_only()

    def build_model_only():
        return _model()

    monkeypatch.setattr(trainer, 'build_model', lambda args, device: _model())
    monkeypatch.setattr(trainer, 'load_initial_vq_weights', lambda m, c: False)
    monkeypatch.setattr(trainer, 'load_checkpoint', lambda p, map_location=None: {'model': _model().state_dict(), 'args': {}})
    device = torch.device('cpu')
    torch.manual_seed(1234)
    expected = torch.randn(4).clone()
    torch.manual_seed(1234)
    trainer.build_frozen_teacher(SimpleNamespace(seed=7, teacher_ckpt=Path('/tmp/x.pt')), device)
    after = torch.randn(4)
    assert torch.equal(after, expected), 'teacher build consumed global RNG'


def test_loader_generator_flags_produce_deterministic_order():
    from Utils.training import _loader_generator
    off = SimpleNamespace(loader_generator_seed=0)
    assert _loader_generator(off, 'train') is None
    on = SimpleNamespace(loader_generator_seed=999)
    g1, g2 = _loader_generator(on, 'train'), _loader_generator(on, 'train')
    assert isinstance(g1, torch.Generator) and torch.equal(
        torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))
    # Splits get decorrelated but reproducible streams.
    gv = _loader_generator(on, 'val')
    gt = _loader_generator(on, 'test')
    assert not torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=gv))
    assert not torch.equal(torch.randperm(10, generator=gv), torch.randperm(10, generator=gt))


def _full_args(extra=(), tmp_path=Path('/tmp')):
    old = sys.argv
    sys.argv = ['vq', '--db', str(tmp_path / 'db'), '--out-dir', str(tmp_path / 'out'), *extra]
    try:
        return trainer.parse_args()
    finally:
        sys.argv = old


def test_decoder_only_resume_allows_init_and_resume_together(tmp_path):
    """Decoder-only resume needs both flags; a full run still rejects the combination."""
    args = _full_args(['--decoder-only', '--init-ckpt', '/tmp/init.pt',
                       '--resume', '/tmp/latest.pt', '--norm-stats', '/tmp/norm.npz'], tmp_path)
    trainer.validate_args(args)  # must not raise for decoder-only
    args.decoder_only = False
    with pytest.raises(ValueError, match='init-ckpt and --resume'):
        trainer.validate_args(args)


def test_decoder_only_without_norm_stats_warns(tmp_path, capsys):
    args = _full_args(['--decoder-only', '--init-ckpt', '/tmp/init.pt',
                       '--vq-weight', '1', '--code-align-weight', '2'], tmp_path)
    trainer.validate_args(args)
    captured = capsys.readouterr()
    assert 'norm-stats' in captured.out
    assert args.vq_weight == 0.0 and args.code_align_weight == 0.0
