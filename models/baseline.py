from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import init


class BaselineModel(nn.Module):
    def __init__(
        self, in_channels=[128], dropout_rate=0.5, finetune=False, name="baseline"
    ):
        super().__init__()
        self.name = name
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze backbone if finetune
        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace fully connected layers
        n_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Classification layer with 2 outputs
        self.fc = nn.Sequential(
            nn.Linear(n_feats, in_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.action_out = nn.Linear(in_channels[0], 40)  # multi-class
        self.person_out = nn.Linear(in_channels[0], 1)  # binary

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        action = self.action_out(x)
        person = self.person_out(x)

        return action, person
