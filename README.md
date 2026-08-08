# AI-Based Restoration of Degraded Images

##  Project Overview

**AI-Based Restoration of Degraded Images** is a deep learning project designed to restore degraded/noisy low-resolution images into cleaner and higher-resolution images.

The system takes a **128×128 degraded image** as input and generates a **256×256 restored image** using a CNN-based image restoration model.

The project includes a baseline **V1 model** and an improved **V2 model**, which are evaluated using quantitative metrics and visual comparisons.

---

##  Objectives

* Restore degraded/noisy images using deep learning.
* Convert 128×128 degraded images into 256×256 restored images.
* Preserve important image structures and details.
* Compare a baseline CNN with an improved CNN architecture.
* Evaluate restoration performance using **PSNR and SSIM**.
* Analyze the visual quality of the restored images.

---

##  Dataset

The dataset contains paired degraded and ground-truth images.

| Dataset      | Number of Images |     Resolution |
| ------------ | ---------------: | -------------: |
| Training     |      3,200 pairs | Input: 128×128 |
| Test         |       400 images | Input: 128×128 |
| Ground Truth |     3,200 images |        256×256 |

### Dataset Structure

```text
dataset/
│
├── train/
│   ├── GT/
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...
│   │
│   └── NoisyLR/
│       ├── 000000.npy
│       ├── 000001.npy
│       └── ...
│
└── NoisyLR/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

---

##  Preprocessing

The following preprocessing steps were performed:

1. Dataset structure verification.
2. Pairing degraded images with their corresponding ground-truth images.
3. Verification of image dimensions.
4. Normalization of image values to the **0–1 range**.
5. Conversion of NumPy arrays into PyTorch tensors.
6. Loading the dataset using PyTorch DataLoader.

### Data Flow

```text
Raw Dataset
     ↓
Dataset Verification
     ↓
Image Pairing
     ↓
Normalization
     ↓
PyTorch Tensor
     ↓
DataLoader
```

---

#  V1 Model — Baseline CNN

The V1 model was developed as the baseline image restoration network.

### Architecture

```text
128×128 Input
      ↓
Conv2D (1 → 32)
      ↓
ReLU
      ↓
Conv2D (32 → 64)
      ↓
ReLU
      ↓
2× Bilinear Upsampling
      ↓
Conv2D (64 → 32)
      ↓
ReLU
      ↓
Conv2D (32 → 1)
      ↓
Sigmoid
      ↓
256×256 Output
```

### Training Configuration

* Framework: **PyTorch**
* Loss Function: **MSELoss**
* Optimizer: **Adam**
* Learning Rate: **0.001**
* Batch Size: **8**

---

#  V2 Model — Improved CNN

V2 was developed to increase the feature-extraction capacity of the baseline model.

### Architecture

```text
128×128 Input
      ↓
Conv2D (1 → 64)
      ↓
ReLU
      ↓
Conv2D (64 → 64)
      ↓
ReLU
      ↓
Conv2D (64 → 128)
      ↓
ReLU
      ↓
Conv2D (128 → 128)
      ↓
ReLU
      ↓
2× Bilinear Upsampling
      ↓
Conv2D (128 → 64)
      ↓
ReLU
      ↓
Conv2D (64 → 32)
      ↓
ReLU
      ↓
Conv2D (32 → 1)
      ↓
Sigmoid
      ↓
256×256 Output
```

V2 training was performed using **Google Colab** to reduce the training time compared with CPU-based local training.

---

#  Overall Workflow

```text
              Dataset
                 ↓
       Data Preprocessing
                 ↓
          Normalization
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
     V1 Model           V2 Model
        ↓                 ↓
     Training           Training
        ↓                 ↓
     Inference          Inference
        └────────┬────────┘
                 ↓
          Image Evaluation
                 ↓
       ┌─────────┴─────────┐
       ↓                   ↓
      PSNR                SSIM
       ↓                   ↓
       └─────────┬─────────┘
                 ↓
         Visual Comparison
                 ↓
          Final Restoration
```

---

#  Evaluation

The models are evaluated using:

### PSNR — Peak Signal-to-Noise Ratio

PSNR measures the reconstruction quality between the restored image and the ground-truth image.

Higher PSNR generally indicates better reconstruction quality.

### SSIM — Structural Similarity Index

SSIM measures structural similarity between the restored image and the ground-truth image.

Higher SSIM generally indicates better structural preservation.

---

##  V1 Baseline Results

| Metric |          V1 |
| ------ | ----------: |
| PSNR   | **22.8270** |
| SSIM   |  **0.6825** |

### V2 Results

| Metric         |            V2 |
| -------------- | ------------: |
| PSNR           | To be updated |
| SSIM           | To be updated |
| Inference Time | To be updated |

> V2 metrics will be updated after the final evaluation.

---

#  Visual Comparison

The restoration results are compared using:

```text
Degraded Input
      ↓
V1 Restored Output
      ↓
V2 Restored Output
      ↓
Ground Truth
```



#  Technologies Used

* **Python**
* **PyTorch**
* **NumPy**
* **Google Colab**
* **VS Code**
* **Git**
* **GitHub**

---

#  Project Structure

```text
AI-BASED-RESTORATION-OF-DEGRADED-IMAGE/
│
├── dataset/
│
├── models/
│   └── model.pth
│
├── src/
│   ├── dataset_loader.py
│   ├── model.py
│   ├── model_v2.py
│   ├── train.py
│   ├── train_v2.py
│   ├── predict.py
│   └── utils.py
│
├── outputs/
├── outputs_v2/
├── validation_input/
├── validation_v2/
│
├── preprocessing.py
├── normalize_data.py
├── evaluate.py
├── inference_time.py
├── visual_compare.py
├── validation_v2_infer.py
├── competition_v2.py
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-BASED-RESTORATION-OF-DEGRADED-IMAGE
```

Install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib
```

---

#  Running the Project

### Train V1

```bash
python -m src.train
```

### Generate Predictions

```bash
python -m src.predict
```

### Evaluate Results

```bash
python evaluate.py
```

### V2 Training

```bash
python -m src.train_v2
```

> V2 training was primarily performed using Google Colab because of the computational requirements of local CPU training.

---

#  Key Features

* Supervised image restoration
* CNN-based architecture
* 2× image upscaling
* V1 baseline model
* Improved V2 model
* PSNR and SSIM evaluation
* Visual comparison of restoration quality
* Git/GitHub-based project collaboration

---

# Future Scope

Possible improvements include:

* Advanced image restoration architectures.
* Residual and attention-based networks.
* Perceptual loss functions.
* GPU-based training.
* Improved preservation of fine image details.
* Faster inference.
* Deployment as a web-based image restoration application.

---

#  Team

**Team Members:**

1. PRITHIKA SREE P
2. DHANUSHYASHREE R
3. KAILASH B



---

## Conclusion

This project demonstrates a deep learning approach for restoring degraded low-resolution images. A baseline V1 CNN was first developed and evaluated, followed by an improved V2 architecture with increased feature-extraction capacity.

The models are evaluated using **PSNR, SSIM, and visual comparisons** to determine restoration quality and identify improvements between the two approaches.
