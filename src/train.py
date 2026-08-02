import torch
import torch.nn as nn
import torch.optim as optim

from src.model import ImageRestorationCNN
from src.dataset_loader import ImageDataset

dataset = ImageDataset("dataset")

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageRestorationCNN().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):

    running_loss = 0

    for noisy, gt in loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        output = model(noisy)

        loss = criterion(output, gt)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}  Loss: {running_loss/len(loader):.6f}"
    )

torch.save(model.state_dict(), "models/model.pth")

print("\nTraining Completed!")
print("Model saved to models/model.pth")