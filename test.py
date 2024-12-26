import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

model_ft = resnet50(weights=ResNet50_Weights.DEFAULT)
feature_extractor_1 = nn.Sequential(*list(model_ft.children())[:-1])
feature_extractor_2 = nn.Sequential(*list(model_ft.children())[:-2])

x = torch.randn(1, 3, 224, 224)

output_1 = feature_extractor_1(x)
output_2 = feature_extractor_2(x)

print(output_1.shape)
print(output_2.shape)
print(model_ft)
