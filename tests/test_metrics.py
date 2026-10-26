import math

import pytest

from yaslowfast.metrics import VideoClassificationAccumulator


@pytest.mark.parametrize("num_classes", [0, 1, 3, 400, 2.0, True])
def test_metrics_reject_nonbinary_class_counts(num_classes):
    with pytest.raises(ValueError, match="num_classes=2"):
        VideoClassificationAccumulator(num_classes)


def test_video_metrics_count_videos_not_windows():
    accumulator = VideoClassificationAccumulator(2)
    accumulator.update(
        [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2], [0.1, 0.9]],
        [0, 0, 1, 1],
        ["a", "a", "b", "c"],
        ["a0", "a1", "b0", "c0"],
        loss_sum=4.0,
    )
    result = accumulator.compute()
    assert result["task"] == "binary_classification"
    assert result["num_classes"] == 2
    assert result["num_videos"] == 3
    assert result["num_windows"] == 4
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    assert result["loss"] == 1.0
    assert result["video_loss"] == pytest.approx(
        (-math.log(0.8) - math.log(0.2) - math.log(0.9)) / 3
    )
    assert result["predictions"][0]["probabilities"] == pytest.approx([0.8, 0.2])


def test_merge_sums_probabilities_before_dividing_across_ranks():
    rank0 = VideoClassificationAccumulator(2)
    rank0.update([[0.8, 0.2]] * 3, [0, 0, 0], ["a"] * 3, ["a0", "a1", "a2"], loss_sum=3.0)
    rank1 = VideoClassificationAccumulator(2)
    rank1.update([[0.1, 0.9]], [0], ["a"], ["a3"], loss_sum=2.0)
    combined = VideoClassificationAccumulator(2)
    combined.merge(rank0.state_dict())
    combined.merge(rank1.state_dict())
    result = combined.compute()
    assert result["predictions"][0]["probabilities"] == pytest.approx([0.625, 0.375])
    assert result["accuracy"] == 1.0
    assert result["loss"] == pytest.approx(1.25)


def test_duplicate_keys_rejected_locally_and_across_ranks():
    first = VideoClassificationAccumulator(2)
    first.update([[0.8, 0.2]], [0], ["a"], ["a0"], loss_sum=0.2)
    with pytest.raises(ValueError, match="Duplicate"):
        first.update([[0.8, 0.2]], [0], ["a"], ["a0"], loss_sum=0.2)
    second = VideoClassificationAccumulator(2)
    second.merge(first.state_dict())
    with pytest.raises(ValueError, match="Duplicate"):
        second.merge(first.state_dict())


def test_conflicting_video_labels_rejected():
    accumulator = VideoClassificationAccumulator(2)
    accumulator.update([[0.8, 0.2]], [0], ["a"], ["a0"], loss_sum=0.2)
    with pytest.raises(ValueError, match="Conflicting labels"):
        accumulator.update([[0.1, 0.9]], [1], ["a"], ["a1"], loss_sum=0.2)


def test_empty_evaluation_is_an_error_and_absent_classes_have_zero_f1():
    accumulator = VideoClassificationAccumulator(2)
    with pytest.raises(ValueError, match="no videos"):
        accumulator.compute()
    accumulator.update([[0.8, 0.2]], [0], ["a"], ["a0"], loss_sum=0.2)
    result = accumulator.compute()
    assert result["macro_f1"] == 0.5
    assert result["balanced_accuracy"] == 1.0
    assert result["per_class"][1]["support"] == 0


@pytest.mark.parametrize("probabilities", [[[float("nan"), 0.1]], [[0.2, 0.2]], [[1.1, -0.1]]])
def test_invalid_probability_vectors_rejected(probabilities):
    with pytest.raises(ValueError, match="Probabilit|probabilit"):
        VideoClassificationAccumulator(2).update(probabilities, [0], ["a"], ["a0"], loss_sum=1.0)
