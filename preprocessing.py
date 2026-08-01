import os
import numpy as np

# Dataset paths
gt_path = "dataset/train/GT"
noisy_path = "dataset/train/NoisyLR"

# Get all filenames
gt_files = sorted(os.listdir(gt_path))
noisy_files = sorted(os.listdir(noisy_path))

# Remove hidden files like .DS_Store
gt_files = [f for f in gt_files if f.endswith(".npy")]
noisy_files = [f for f in noisy_files if f.endswith(".npy")]

print("=" * 40)
print("DATASET INFORMATION")
print("=" * 40)

print(f"Ground Truth Images : {len(gt_files)}")
print(f"Noisy Images        : {len(noisy_files)}")

# Check if numbers match
if len(gt_files) == len(noisy_files):
    print("✅ Number of images match")
else:
    print("❌ Number of images do NOT match")

print("\nChecking first 5 image pairs...\n")

for i in range(5):
    gt = np.load(os.path.join(gt_path, gt_files[i]))
    noisy = np.load(os.path.join(noisy_path, noisy_files[i]))

    print(f"Pair {i+1}")
    print(f"GT    : {gt_files[i]}  Shape: {gt.shape}")
    print(f"Noisy : {noisy_files[i]}  Shape: {noisy.shape}")
    print("-" * 30)