import numpy as np
import matplotlib.pyplot as plt

img = np.load("dataset/NoisyLR/000000.npy")

print("Shape:", img.shape)
print("Data Type:", img.dtype)

plt.imshow(img, cmap="gray")
plt.title("GT Image")
plt.axis("off")
plt.show()