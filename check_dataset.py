import os

print("Contents of train folder:")
print(os.listdir("dataset/train"))

print("\nContents of NoisyLR folder:")
print(os.listdir("dataset/NoisyLR")[:10])  # Shows first 10 files