import numpy as np
import matplotlib.pyplot as plt
import os

# Select image number
image_name = "000000.npy"

# Paths
noisy_path = os.path.join("dataset", "NoisyLR", image_name)
output_path = os.path.join("outputs", image_name)
gt_path = os.path.join("dataset", "train", "GT", image_name)

# Load images
noisy = np.load(noisy_path)
output = np.load(output_path)
gt = np.load(gt_path)

# Display
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Input")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(output, cmap="gray")
plt.title("Restored Output")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()