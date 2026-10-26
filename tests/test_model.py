import pytest
import torch

from yaslowfast.config import ModelConfig
from yaslowfast.models import SlowFast, build_model, pack_pathways


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pack_pathways_matches_reference_sampling():
    frames = torch.arange(32).reshape(1, 32, 1, 1)
    slow, fast = pack_pathways(frames)
    assert slow.flatten().tolist() == [0, 4, 8, 13, 17, 22, 26, 31]
    assert fast is frames
    batched = frames.unsqueeze(0).expand(2, -1, -1, -1, -1)
    slow, _ = pack_pathways(batched)
    assert slow.shape == (2, 1, 8, 1, 1)
    with pytest.raises(ValueError, match="multiple"):
        pack_pathways(torch.randn(3, 7, 2, 2))


def test_binary_r50_parameter_count_and_initialization():
    model = SlowFast()
    assert model.num_classes == 2
    assert sum(p.numel() for p in model.parameters()) == 33_649_098
    final_bn = [m for m in model.modules() if getattr(m, "block_final_bn", False)]
    assert len(final_bn) == 32
    assert all(torch.count_nonzero(m.weight) == 0 for m in final_bn)
    assert torch.count_nonzero(model.blocks[6].proj.bias) == 0
    assert model.blocks[6].proj.in_features == 2304
    assert model.blocks[6].proj.out_features == 2


def test_tiny_model_backward_has_finite_gradients():
    torch.manual_seed(1)
    model = SlowFast(dropout=0.0, width_per_group=8, stage_depths=(1, 1, 1, 1))
    slow, fast = pack_pathways(torch.randn(2, 3, 8, 32, 32))
    logits = model(slow, fast)
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.blocks[6].proj.weight.grad) > 0


@pytest.mark.parametrize("num_classes", [0, 1, 3, 400, True, 2.0])
def test_binary_model_rejects_invalid_class_count(num_classes):
    with pytest.raises(ValueError, match="num_classes=2"):
        SlowFast(num_classes)
    with pytest.raises(ValueError, match="num_classes=2"):
        build_model(ModelConfig(), num_classes)


def test_builder_defaults_to_two_output_head():
    model = build_model(ModelConfig(width_per_group=8, stage_depths=(1, 1, 1, 1)))
    assert model.num_classes == 2
    assert model.blocks[6].proj.out_features == 2


def test_adaptive_matches_reference_pooling_at_standard_input():
    torch.manual_seed(2)
    adaptive = SlowFast(2, width_per_group=8, stage_depths=(1, 1, 1, 1)).eval()
    reference = SlowFast(
        2, pooling="reference", width_per_group=8, stage_depths=(1, 1, 1, 1)
    ).eval()
    reference.load_state_dict(adaptive.state_dict(), strict=True)
    slow, fast = pack_pathways(torch.randn(1, 3, 32, 224, 224))
    with torch.inference_mode():
        torch.testing.assert_close(
            adaptive(slow, fast), reference(slow, fast), rtol=1e-5, atol=1e-6
        )


def test_reject_mismatched_pathway_lengths():
    model = SlowFast(2, width_per_group=8, stage_depths=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="four times"):
        model(torch.randn(1, 3, 3, 32, 32), torch.randn(1, 3, 8, 32, 32))
