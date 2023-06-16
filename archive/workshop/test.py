import numpy as np
import matplotlib.pyplot as plt

x_train = np.load("data/train_format_gp.npy")
BAND_COLORS = ["C4", "C2", "C3", "C1", "k", "C5"]

for i in range(100):
    fig, ax = plt.subplots(figsize=[5, 4])
    for band in range(6):
        ax.scatter(
            np.arange(1094), x_train[i, :, band], color=BAND_COLORS[band], marker="."
        )
    fig.savefig(f"test/test_{i}.png")
    plt.close()
