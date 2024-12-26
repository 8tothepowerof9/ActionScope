import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

model_ft = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
feature_extractor_1 = nn.Sequential(*list(model_ft.children())[:-1])
feature_extractor_2 = nn.Sequential(*list(model_ft.children())[:-2])

x = torch.randn(1, 3, 224, 224)

output_1 = feature_extractor_1(x)
output_2 = feature_extractor_2(x)

print(output_1.shape)
print(output_2.shape)
print(model_ft)
print(model_ft.classifier[0].in_features)
