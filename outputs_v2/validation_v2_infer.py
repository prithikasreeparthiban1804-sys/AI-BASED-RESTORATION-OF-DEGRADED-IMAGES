import json
import shutil
import subprocess
from pathlib import Path

# ============================================================
# V2 VALIDATION INFERENCE
# ============================================================

ROOT = Path(__file__).resolve().parent

# Actual paths in your project
SPLIT_FILE = ROOT / "splits" / "v2_split.json"
NOISY_DIR = ROOT / "dataset" / "train" / "NoisyLR"

TEMP_INPUT = ROOT / "validation_input"
OUTPUT_DIR = ROOT / "validation_v2"

CHECKPOINT = ROOT / "models" / "v2_best.pt"

# ============================================================
# CHECK FILES
# ============================================================

print("\nV2 VALIDATION INFERENCE")
print("=" * 50)

print("Project:", ROOT)
print("Split  :", SPLIT_FILE)
print("Noisy  :", NOISY_DIR)
print("Model  :", CHECKPOINT)

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_FILE}"
    )

if not NOISY_DIR.exists():
    raise FileNotFoundError(
        f"Noisy image folder not found:\n{NOISY_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"V2 checkpoint not found:\n{CHECKPOINT}"
    )

# ============================================================
# READ SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

print("\nSplit JSON loaded.")
print("JSON keys:", list(split.keys()))

# Find validation IDs safely
if "val" in split:
    validation_ids = split["val"]

elif "validation" in split:
    validation_ids = split["validation"]

elif "val_ids" in split:
    validation_ids = split["val_ids"]

elif "validation_ids" in split:
    validation_ids = split["validation_ids"]

else:
    raise KeyError(
        "Could not find validation IDs in split JSON."
    )

print("Validation images:", len(validation_ids))

# ============================================================
# CLEAN TEMPORARY FOLDER
# ============================================================

if TEMP_INPUT.exists():
    shutil.rmtree(TEMP_INPUT)

TEMP_INPUT.mkdir()

# ============================================================
# COPY VALIDATION IMAGES
# ============================================================

print("\nPreparing validation images...")

copied = 0

for image_id in validation_ids:

    image_id = str(image_id).replace(".npy", "")

    source = NOISY_DIR / f"{image_id}.npy"
    destination = TEMP_INPUT / f"{image_id}.npy"

    if source.exists():
        shutil.copy2(source, destination)
        copied += 1

print("Validation images prepared:", copied)

if copied != len(validation_ids):
    print(
        "WARNING: Some validation images were missing."
    )

# ============================================================
# RUN EXISTING V2 INFERENCE
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir()

command = [
    "python",
    str(ROOT / "competition_v2.py"),
    "infer",
    "--checkpoint",
    str(CHECKPOINT),
    "--input-dir",
    str(TEMP_INPUT),
    "--output-dir",
    str(OUTPUT_DIR),
]

print("\nRunning V2 inference...")
print("-" * 50)

subprocess.run(
    command,
    cwd=ROOT,
    check=True
)

# ============================================================
# FINAL CHECK
# ============================================================

outputs = list(OUTPUT_DIR.glob("*.npy"))

print("\n" + "=" * 50)
print("V2 VALIDATION INFERENCE COMPLETE")
print("=" * 50)

print("Validation IDs :", len(validation_ids))
print("Images copied  :", copied)
print("V2 outputs     :", len(outputs))
print("Output folder  :", OUTPUT_DIR)

if len(outputs) == len(validation_ids):
    print("\nSUCCESS: All 640 validation predictions created.")

else:
    print(
        "\nWARNING: Output count does not match validation count."
    )