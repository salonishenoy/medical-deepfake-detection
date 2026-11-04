"""Team B model builders: EfficientNet-B0 and ResNet, one-logit binary heads.

Output is a single logit per image: sigmoid(logit) = P(manipulated).
"""
import torch.nn as nn
from torchvision import models

ARCHS = ("efficientnet_b0", "resnet18", "resnet34", "resnet50")


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features  # 1280
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return m
    if arch in ("resnet18", "resnet34", "resnet50"):
        weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        }[arch] if pretrained else None
        m = getattr(models, arch)(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
        return m
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


def head_name(arch: str) -> str:
    return "classifier" if arch.startswith("efficientnet") else "fc"


def set_backbone_frozen(model: nn.Module, arch: str, frozen: bool) -> None:
    """Freeze/unfreeze everything except the classification head."""
    head = head_name(arch)
    for name, p in model.named_parameters():
        if not name.startswith(head + "."):
            p.requires_grad = not frozen


def backbone_modules(model: nn.Module, arch: str):
    head = head_name(arch)
    return [m for name, m in model.named_children() if name != head]


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last conv block; give this to Team C for Grad-CAM."""
    return model.features[-1] if arch.startswith("efficientnet") else model.layer4[-1]
