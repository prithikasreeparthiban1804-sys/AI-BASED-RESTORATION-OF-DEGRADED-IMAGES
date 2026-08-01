import numpy as np

# Load one training pair
gt = np.load("dataset/train/GT/000000.npy")
noisy = np.load("dataset/train/NoisyLR/000000.npy")

print("Before Normalization")
print("--------------------")
print("GT")
print("Min :", gt.min())
print("Max :", gt.max())

print("\nNoisy")
print("Min :", noisy.min())
print("Max :", noisy.max())

# Normalize to range 0–1
gt = (gt - gt.min()) / (gt.max() - gt.min())
noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())

print("\nAfter Normalization")
print("--------------------")
print("GT")
print("Min :", gt.min())
print("Max :", gt.max())

print("\nNoisy")
print("Min :", noisy.min())
print("Max :", noisy.max())