from src.dataset_loader import ImageDataset

dataset = ImageDataset("dataset")

print("Total Images:", len(dataset))

noisy, gt = dataset[0]

print("Noisy Shape:", noisy.shape)
print("Ground Truth Shape:", gt.shape)