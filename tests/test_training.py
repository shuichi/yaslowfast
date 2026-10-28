import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

import yaslowfast.train as training
from yaslowfast.checkpoint import atomic_save, capture_rng_state, restore_rng_state
from yaslowfast.config import Config, config_from_dict
from yaslowfast.distributed import DistributedContext
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


class _TrackingClassifier(torch.nn.Module):
    def __init__(self, overflow_sample=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.2], [-0.3]]))
        self.seen = []
        if overflow_sample is not None:
            # Produce a finite loss but overflow one accumulated gradient, just
            # as AMP can do. Use the real GradScaler/backoff path on CPU.
            self.weight.register_hook(
                lambda gradient: (
                    torch.full_like(gradient, float("inf"))
                    if self.seen[-1] == overflow_sample
                    else gradient
                )
            )

    def forward(self, slow, fast):
        self.seen.extend(int(index) for index in slow[:, 0])
        return fast @ self.weight.T


def _training_batch(index):
    return {
        "slow": torch.tensor([[float(index)]]),
        "fast": torch.tensor([[(index + 1) / 10]]),
        "target": torch.tensor([index % 2]),
        "weight": torch.ones(1),
        "key": [str(index)],
    }


def test_epoch_drains_only_after_accumulated_overflow_retry(monkeypatch):
    config = Config()
    config.train.accumulation_steps = 2
    model = _TrackingClassifier(overflow_sample=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = WarmupCosine(optimizer, 6, 2)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=100)
    batches = [_training_batch(index) for index in range(8)]
    optimizer_steps = []
    optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(1))
    drained = []

    def finish(loader, iterator):
        assert loader is batches
        assert model.seen == list(range(6))
        assert len(optimizer_steps) == scheduler.step_index == 2
        # Stand in for already-prefetched batches. They must be consumed only
        # by cleanup, without forward/backward or optimizer/scheduler updates.
        drained.extend(batch["key"][0] for batch in iterator)

    monkeypatch.setattr(training, "finish_training_epoch", finish)
    metrics, global_step = training.train_epoch(
        model,
        model,
        batches,
        optimizer,
        scheduler,
        scaler,
        config,
        DistributedContext(torch.device("cpu")),
        "fp32",
        epoch=0,
        steps_per_epoch=2,
        global_step=7,
    )
    assert model.seen == list(range(6))
    assert drained == ["6", "7"]
    assert len(optimizer_steps) == scheduler.step_index == 2
    assert global_step == 9
    assert metrics["optimizer_steps"] == 2
    assert metrics["overflow_skips"] == 1
    assert metrics["sample_visits"] == 4
    assert scaler.get_scale() == 4

    # The update before the overflow and its retry use exactly the same
    # samples/gradients as the former unbounded stream. Skipped samples 2/3
    # and drained samples 6/7 must contribute nothing to momentum or weights.
    reference = _TrackingClassifier()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    reference_scheduler = WarmupCosine(reference_optimizer, 6, 2)
    for indices in ((0, 1), (4, 5)):
        reference_optimizer.zero_grad(set_to_none=True)
        for index in indices:
            batch = batches[index]
            logits = reference(batch["slow"], batch["fast"])
            (weighted_classification_loss(logits, batch["target"], batch["weight"]) / 2).backward()
        reference_optimizer.step()
        reference_scheduler.step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[model.weight]["momentum_buffer"],
        reference_optimizer.state[reference.weight]["momentum_buffer"],
        rtol=0,
        atol=0,
    )
    assert scheduler.state_dict() == reference_scheduler.state_dict()


def test_epoch_cleanup_failure_does_not_publish_checkpoint(monkeypatch, tmp_path):
    config = Config()
    config.train.device = "cpu"
    config.train.epochs = 1
    config.train.warmup_epochs = 0
    config.train.steps_per_epoch = 1
    config.train.accumulation_steps = 2
    config.train.output_dir = str(tmp_path)
    config.data.normalization = "config"
    metadata = {
        "class_to_idx": {"zero": 0, "one": 1},
        "videos": {
            "a": {"class_id": 0, "split": "train"},
            "b": {"class_id": 1, "split": "train"},
        },
        "splits": {"train": {"num_samples": 2}, "val": {"num_samples": 1}},
    }
    model = _TrackingClassifier()
    monkeypatch.setattr(training, "read_dataset", lambda path: metadata)
    monkeypatch.setattr(training, "dataset_fingerprint", lambda path: "test-dataset")
    monkeypatch.setattr(training, "build_model", lambda *args: model)
    monkeypatch.setattr(
        training, "build_loader", lambda *args, **kwargs: [_training_batch(i) for i in range(4)]
    )

    def fail_cleanup(loader, iterator):
        assert model.seen == [0, 1]  # The last optimizer update already completed.
        assert next(iterator)["key"] == ["2"]
        raise RuntimeError("worker failed while draining the epoch")

    def unexpected_validation(*args, **kwargs):
        pytest.fail("Validation must not run after failed worker cleanup")

    monkeypatch.setattr(training, "finish_training_epoch", fail_cleanup)
    monkeypatch.setattr(training, "evaluate_model", unexpected_validation)
    with pytest.raises(RuntimeError, match="worker failed while draining"):
        training.run_training(config)
    for name in ("latest.pt", "best.pt", "metrics.jsonl", "validation.json"):
        assert not (tmp_path / name).exists()
