"""Metrics for binary classification, with exactly one vote per video."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .labels import NUM_CLASSES


@dataclass
class VideoClassificationAccumulator:
    """Accumulate window probabilities, rejecting duplicate window identifiers.

    Accumulators can be merged across ranks even when a video's windows occur on
    multiple ranks. Probabilities are summed before dividing by the window count;
    taking the mean of rank-local video means would bias the result.
    """

    num_classes: int = NUM_CLASSES
    videos: dict[str, dict[str, Any]] = field(default_factory=dict)
    keys: set[str] = field(default_factory=set)
    loss_sum: float = 0.0
    num_windows: int = 0

    def __post_init__(self) -> None:
        if type(self.num_classes) is not int or self.num_classes != NUM_CLASSES:
            raise ValueError("Binary classification requires num_classes=2")

    def update(
        self,
        probabilities: Sequence[Sequence[float]] | np.ndarray,
        targets: Sequence[int] | np.ndarray,
        video_ids: Sequence[str],
        keys: Sequence[str],
        *,
        loss_sum: float,
    ) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        targets = np.asarray(targets)
        count = len(video_ids)
        if probabilities.shape != (count, self.num_classes):
            raise ValueError("Probability shape must be (number of windows, num_classes)")
        if targets.shape != (count,) or len(keys) != count:
            raise ValueError("Mismatched prediction, target, video ID, and key counts")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Targets must be integer class IDs")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Probabilities must be finite and non-negative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("Each probability vector must sum to one")
        if not np.isfinite(loss_sum):
            raise ValueError("Evaluation loss must be finite")
        for probs, target_value, video_id, key in zip(
            probabilities, targets, video_ids, keys, strict=True
        ):
            target = int(target_value)
            if not 0 <= target < self.num_classes:
                raise ValueError(f"Class ID {target} is outside the model's class range")
            if not isinstance(video_id, str) or not video_id:
                raise ValueError("video_id must be a non-empty string")
            if not isinstance(key, str) or not key:
                raise ValueError("Window key must be a non-empty string")
            if key in self.keys:
                raise ValueError(f"Duplicate evaluation window key: {key}")
            self.keys.add(key)
            if video_id not in self.videos:
                self.videos[video_id] = {
                    "target": target,
                    "probability_sum": np.zeros(self.num_classes, dtype=np.float64),
                    "num_windows": 0,
                }
            record = self.videos[video_id]
            if record["target"] != target:
                raise ValueError(f"Conflicting labels for video {video_id!r}")
            record["probability_sum"] += probs
            record["num_windows"] += 1
        self.loss_sum += float(loss_sum)
        self.num_windows += count

    def state_dict(self) -> dict[str, Any]:
        """Return plain Python data suitable for distributed object collection."""
        return {
            "num_classes": self.num_classes,
            "videos": {
                key: {
                    "target": value["target"],
                    "probability_sum": value["probability_sum"].tolist(),
                    "num_windows": value["num_windows"],
                }
                for key, value in self.videos.items()
            },
            "keys": sorted(self.keys),
            "loss_sum": self.loss_sum,
            "num_windows": self.num_windows,
        }

    def merge(self, state: dict[str, Any]) -> None:
        if state["num_classes"] != self.num_classes:
            raise ValueError("Cannot merge predictions with different class counts")
        other_keys = set(state["keys"])
        duplicate = self.keys.intersection(other_keys)
        if duplicate or len(other_keys) != len(state["keys"]):
            example = min(duplicate) if duplicate else "within collected rank state"
            raise ValueError(f"Duplicate evaluation window key: {example}")
        if len(other_keys) != state["num_windows"]:
            raise ValueError("Collected rank state has inconsistent window counts")
        self.keys.update(other_keys)
        for video_id, other in state["videos"].items():
            if video_id not in self.videos:
                self.videos[video_id] = {
                    "target": other["target"],
                    "probability_sum": np.asarray(other["probability_sum"], dtype=np.float64),
                    "num_windows": other["num_windows"],
                }
            else:
                record = self.videos[video_id]
                if record["target"] != other["target"]:
                    raise ValueError(f"Conflicting labels for video {video_id!r}")
                record["probability_sum"] += np.asarray(other["probability_sum"], dtype=np.float64)
                record["num_windows"] += other["num_windows"]
        self.loss_sum += state["loss_sum"]
        self.num_windows += state["num_windows"]

    def compute(self) -> dict[str, Any]:
        if not self.videos:
            raise ValueError("Evaluation contains no videos")
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        predictions = []
        video_loss_sum = 0.0
        for video_id in sorted(self.videos):
            record = self.videos[video_id]
            probabilities = record["probability_sum"] / record["num_windows"]
            prediction = int(np.argmax(probabilities))
            target = record["target"]
            confusion[target, prediction] += 1
            video_loss_sum -= float(np.log(max(probabilities[target], np.finfo(np.float64).tiny)))
            predictions.append(
                {
                    "video_id": video_id,
                    "target": target,
                    "prediction": prediction,
                    "probabilities": probabilities.tolist(),
                    "num_windows": record["num_windows"],
                }
            )

        true_positives = confusion.diagonal().astype(np.float64)
        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        precision = np.divide(
            true_positives, predicted, out=np.zeros(self.num_classes), where=predicted != 0
        )
        recall = np.divide(
            true_positives, support, out=np.zeros(self.num_classes), where=support != 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(self.num_classes),
            where=(precision + recall) != 0,
        )
        num_videos = len(self.videos)
        return {
            "task": "binary_classification",
            "num_classes": NUM_CLASSES,
            "accuracy": float(true_positives.sum() / num_videos),
            "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "loss": self.loss_sum / self.num_windows,
            "window_loss": self.loss_sum / self.num_windows,
            "video_loss": video_loss_sum / num_videos,
            "num_videos": num_videos,
            "num_windows": self.num_windows,
            "confusion_matrix": confusion.tolist(),
            "per_class": [
                {
                    "class_id": index,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                    "predicted_count": int(predicted[index]),
                }
                for index in range(self.num_classes)
            ],
            "predictions": predictions,
            "metric_notes": {
                "aggregation": "Mean window probabilities, then one prediction per video",
                "macro_f1": "Arithmetic mean across all configured classes; zero for undefined class F1",
                "confusion_matrix": "Rows are true labels; columns are predicted labels",
                "loss": "Mean cross entropy over windows (unweighted)",
                "video_loss": "Mean negative log likelihood after averaging probabilities per video",
            },
        }
