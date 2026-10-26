import pytest

from yaslowfast.config import DataConfig, apply_dataset_normalization, config_from_dict


def statistics(**changes):
    return {
        "mean": [0.12, 0.34, 0.56],
        "std": [0.11, 0.22, 0.33],
        "split": "train",
        **changes,
    }


def test_automatic_normalization_overrides_existing_yaml_defaults():
    config = config_from_dict({"data": {"mean": [0.45] * 3, "std": [0.225] * 3}})
    assert apply_dataset_normalization(config.data, {"normalization": statistics()})
    assert config.data.mean == pytest.approx((0.12, 0.34, 0.56))
    assert config.data.std == pytest.approx((0.11, 0.22, 0.33))
    # Checkpoint deserialization preserves resolved values without needing dataset access.
    restored = config_from_dict(config.to_dict())
    assert restored.data.mean == config.data.mean
    assert restored.data.std == config.data.std


def test_explicit_config_normalization_preserves_custom_values():
    config = DataConfig(normalization="config", mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    assert not apply_dataset_normalization(config, {"normalization": statistics()})
    assert config.mean == (0.1, 0.2, 0.3)
    assert config.std == (0.4, 0.5, 0.6)


def test_legacy_dataset_preserves_configured_normalization():
    config = DataConfig(mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    assert not apply_dataset_normalization(config, {})
    assert config.mean == (0.1, 0.2, 0.3)
    assert config.std == (0.4, 0.5, 0.6)


@pytest.mark.parametrize(
    "report",
    [
        None,
        [],
        {},
        statistics(split="val"),
        statistics(mean=[0.1, 0.2]),
        statistics(mean=[float("nan"), 0.2, 0.3]),
        statistics(std=[float("inf"), 0.2, 0.3]),
        statistics(std=[0, 0.2, 0.3]),
        statistics(std=[-0.1, 0.2, 0.3]),
    ],
)
def test_invalid_dataset_statistics_fail_instead_of_silent_fallback(report):
    config = DataConfig()
    with pytest.raises(ValueError):
        apply_dataset_normalization(config, {"normalization": report})


def test_statistics_must_belong_to_the_selected_training_split():
    config = DataConfig(train_split="alternate")
    with pytest.raises(ValueError):
        apply_dataset_normalization(config, {"normalization": statistics()})


def test_unknown_normalization_source_is_rejected():
    with pytest.raises(ValueError, match="normalization"):
        config_from_dict({"data": {"normalization": "datset"}})
