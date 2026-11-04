"""Shape and freezing checks for Team B models (no pretrained download, no data)."""
import pytest
import torch

from src.models import ARCHS, build_model, gradcam_target_layer, head_name, set_backbone_frozen


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet18"])
def test_one_logit_output(arch):
    m = build_model(arch, pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 1)


@pytest.mark.parametrize("arch", ["efficientnet_b0", "resnet18"])
def test_freeze_only_head_trainable(arch):
    m = build_model(arch, pretrained=False)
    set_backbone_frozen(m, arch, True)
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {head_name(arch)}
    set_backbone_frozen(m, arch, False)
    assert all(p.requires_grad for p in m.parameters())


def test_gradcam_layer_exists():
    m = build_model("efficientnet_b0", pretrained=False)
    assert isinstance(gradcam_target_layer(m, "efficientnet_b0"), torch.nn.Module)
    assert "efficientnet_b0" in ARCHS
