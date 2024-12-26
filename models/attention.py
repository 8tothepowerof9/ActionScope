from torch import nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_feats = self.max_pool(x)
        avg_feats = self.avg_pool(x)

        max_feats = torch.flatten(max_feats, 1)
        avg_feats = torch.flatten(avg_feats, 1)

        max_feats = self.mlp(max_feats)
        avg_feats = self.mlp(avg_feats)

        output = (
            self.sigmoid(max_feats + avg_feats).unsqueeze(2).unsqueeze(3).expand_as(x)
        )

        return output * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)

        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()

        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out


class AttentionModel(nn.Module):
    def __init__(
        self,
        reduction_ratio,
        kernel_size,
        finetune=False,
        dropout_rate=0.5,
        in_channels=[256, 128, 8],
        name="attention",
    ):
        super().__init__()

        self.name = name
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze backbone if finetune
        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        n_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Add attention modules to the backbone and replace the fully connected layer
        self.att = CBAM(n_feats, reduction_ratio, kernel_size)
        self.sa = ChannelAttention(n_feats, reduction_ratio)

        # Pooling
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveAvgPool2d(1)

        self.action_fc = nn.Sequential(
            nn.Linear(n_feats, in_channels[0]),
            nn.BatchNorm1d(in_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels[0], in_channels[1]),
            nn.BatchNorm1d(in_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.person_fc = nn.Sequential(
            nn.Linear(n_feats, in_channels[2]),
            nn.BatchNorm1d(in_channels[2]),
            nn.ReLU(),
        )

        # Classification layers
        self.action_out = nn.Linear(in_channels[1], 40)
        self.person_out = nn.Linear(in_channels[2], 1)

    def forward(self, x):
        x = self.backbone(x)

        action = self.att(x)
        action = torch.flatten(self.pool_1(action), 1)
        action = self.action_fc(action)
        action = self.action_out(action)

        person = self.sa(x)
        person = torch.flatten(self.pool_2(person), 1)
        person = self.person_fc(person)
        person = self.person_out(person)
        return action, person