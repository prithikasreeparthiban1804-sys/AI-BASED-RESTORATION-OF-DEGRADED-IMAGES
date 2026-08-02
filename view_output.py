import os
import numpy as np
import matplotlib.pyplot as plt

output_folder = "outputs"

files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])

if len(files) == 0:
    print("No output files found!")
else:
    print("Total Output Files:", len(files))
    print("Showing:", files[0])

    img = np.load(os.path.join(output_folder, files[0]))

    plt.imshow(img, cmap="gray")
    plt.title(files[0])
    plt.axis("off")
    plt.show()