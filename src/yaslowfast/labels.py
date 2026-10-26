"""Shared label contract for mutually exclusive binary video classification."""

from collections.abc import Mapping

NUM_CLASSES = 2


def validate_class_to_idx(mapping, *, context: str = "class_to_idx") -> None:
    """Require two named classes with unambiguous integer IDs zero and one."""
    if not isinstance(mapping, Mapping) or len(mapping) != NUM_CLASSES:
        raise ValueError(f"{context} must define exactly two classes for binary classification")
    if any(not isinstance(name, str) or not name.strip() for name in mapping):
        raise ValueError(f"{context} must use two non-empty string class names")
    if any(type(index) is not int for index in mapping.values()) or set(mapping.values()) != {0, 1}:
        raise ValueError(f"{context} must map the two class names to integer IDs 0 and 1")
