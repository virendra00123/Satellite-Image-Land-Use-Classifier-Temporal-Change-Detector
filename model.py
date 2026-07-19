"""
Module: transfer-learning backbone wrapper.

Wraps torchvision ResNet-18 or EfficientNet-B0 and exposes:
  - freeze_backbone() / unfreeze_last_n_blocks(n)  -> the two-phase strategy
  - forward(x, return_embedding=False) -> logits, or 512-d embedding when
    the classifier head is stripped (used by embeddings.py for Module 2)
"""
import torch
import torch.nn as nn
from torchvision import models


class TransferNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.embedding_dim = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.backbone = net
            self.blocks = [net.layer1, net.layer2, net.layer3, net.layer4]  # coarse->fine

        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.embedding_dim = net.classifier[1].in_features  # 1280
            net.classifier = nn.Identity()
            self.backbone = net
            self.blocks = list(net.features)  # sequential feature blocks

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head = nn.Linear(self.embedding_dim, num_classes)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Phase 2: unfreeze only the last n blocks of the backbone,
        leaving earlier (more generic) layers frozen."""
        for block in self.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x, return_embedding: bool = False):
        emb = self.backbone(x)
        if emb.dim() > 2:
            emb = torch.flatten(emb, 1)
        if return_embedding:
            return emb
        return self.head(emb)

    def trainable_param_groups(self, lr: float):
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr}]
