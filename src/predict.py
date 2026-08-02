import os
import numpy as np
import torch

from src.model import ImageRestorationCNN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageRestorationCNN().to(device)
model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()

# Create output folder
os.makedirs("outputs", exist_ok=True)

test_folder = "dataset/NoisyLR"

files = sorted(os.listdir(test_folder))

print("Total Test Images:", len(files))

with torch.no_grad():

    for file in files:

        img = np.load(os.path.join(test_folder, file)).astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        output = model(img)

        output = output.squeeze().cpu().numpy()

        np.save(os.path.join("outputs", file), output)

print("\nPrediction Completed!")
print("Outputs saved in outputs/")