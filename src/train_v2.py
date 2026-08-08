import torch
import torch.nn as nn
import torch.optim as optim

from src.model_v2 import ImageRestorationCNNV2
from src.dataset_loader import ImageDataset

dataset = ImageDataset("dataset")

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageRestorationCNNV2().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 8

for epoch in range(epochs):

    running_loss = 0.0

    for noisy, gt in loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        output = model(noisy)

        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.6f}"
    )

torch.save(model.state_dict(), "models/model_v2.pth")

print("\nV2 Training Completed!")
print("Model saved to models/model_v2.pth")