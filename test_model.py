import torch
from src.model import ImageRestorationCNN

model = ImageRestorationCNN()

dummy = torch.randn(1, 1, 128, 128)

output = model(dummy)

print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)