import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from yaslowfast.checkpoint import atomic_save, capture_rng_state, restore_rng_state
from yaslowfast.config import Config, config_from_dict
from yaslowfast.schedule import WarmupCosine
from yaslowfast.train import backoff_scaler, weighted_classification_loss


def test_configuration_rejects_misspelled_settings():
    with pytest.raises(ValueError, match="Unknown TrainConfig"):
        config_from_dict({"train": {"learning_reate": 1}})
    assert asdict(config_from_dict(asdict(Config()))) == asdict(Config())


def test_rng_roundtrip_is_weights_only_safe(tmp_path):
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(2))
    atomic_save(state, tmp_path / "rng.pt")
    restore_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)


def test_scheduler_resume_preserves_updates_and_rejects_changed_schedule():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = WarmupCosine(optimizer, 10, 2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    scheduler.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    state = scheduler.state_dict()
    expected = []
    for _ in range(7):
        scheduler.step()
        expected.append(optimizer.param_groups[0]["lr"])
    other = torch.optim.SGD([parameter], lr=0.1)
    restored = WarmupCosine(other, 10, 2)
    restored.load_state_dict(state)
    actual = []
    for _ in range(7):
        restored.step()
        actual.append(other.param_groups[0]["lr"])
    assert actual == expected
    assert actual[-1] == pytest.approx(0.0)
    changed = WarmupCosine(torch.optim.SGD([parameter], lr=0.1), 20, 2)
    with pytest.raises(ValueError, match="total_steps"):
        changed.load_state_dict(state)


def test_video_weighting_preserves_video_mean_and_microbatch_gradients():
    # Video A has 3 windows, B has 1. Equal video influence requires weights 2/3 and 2.
    logits = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 0, 0, 1])
    weights = torch.tensor([2 / 3, 2 / 3, 2 / 3, 2.0])
    individual = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    expected = (individual[:3].mean() + individual[3]) / 2
    result = weighted_classification_loss(logits, targets, weights)
    torch.testing.assert_close(result, expected)
    result.backward()
    expected_grad = logits.grad.clone()
    logits.grad = None
    for indices in (slice(0, 2), slice(2, 4)):
        (
            weighted_classification_loss(logits[indices], targets[indices], weights[indices]) / 2
        ).backward()
    torch.testing.assert_close(logits.grad, expected_grad)


def test_synchronized_overflow_resets_scaler_growth_counter():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=2)
    scaler.scale(parameter.square()).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    # A normal update just preceded an overflow on another rank.
    scaler.scale(parameter.square()).backward()
    scaler.unscale_(optimizer)
    backoff_scaler(scaler)
    assert scaler.get_scale() == 4
    assert scaler.state_dict()["_growth_tracker"] == 0
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == 4  # Must not immediately undo the backoff.
