import numpy as np
import matplotlib.pyplot as plt

# Load images
gt = np.load("dataset/train/GT/000000.npy")
noisy = np.load("dataset/train/NoisyLR/000000.npy")

# Print information
print("GT Shape:", gt.shape)
print("Noisy Shape:", noisy.shape)

# Display images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(noisy, cmap="gray")
plt.title("NoisyLR")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()