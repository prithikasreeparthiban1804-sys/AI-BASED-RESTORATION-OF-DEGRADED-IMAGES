import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataset_path):
        self.gt_path = os.path.join(dataset_path, "train", "GT")
        self.noisy_path = os.path.join(dataset_path, "train", "NoisyLR")

        self.gt_files = sorted(os.listdir(self.gt_path))
        self.noisy_files = sorted(os.listdir(self.noisy_path))

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        gt = np.load(os.path.join(self.gt_path, self.gt_files[index]))
        noisy = np.load(os.path.join(self.noisy_path, self.noisy_files[index]))

        gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)

        return noisy, gt