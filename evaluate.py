import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

gt_folder = "dataset/train/GT"
output_folder = "train_outputs"

files = sorted(os.listdir(gt_folder))

psnr_scores = []
ssim_scores = []

for file in files:

    gt = np.load(os.path.join(gt_folder, file))
    output = np.load(os.path.join(output_folder, file))

    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    psnr = peak_signal_noise_ratio(gt, output, data_range=1.0)
    ssim = structural_similarity(gt, output, data_range=1.0)

    psnr_scores.append(psnr)
    ssim_scores.append(ssim)

print("================================")
print("Average PSNR :", np.mean(psnr_scores))
print("Average SSIM :", np.mean(ssim_scores))
print("================================")