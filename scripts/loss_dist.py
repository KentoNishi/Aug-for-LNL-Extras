# %%

import torch
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import shutil
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# %%
RATIO = 90
LOSS_DIR = f"../Augmentation-for-LNL/checkpoints/c1m/AugDesc-WS/SAW/save_losses/loss"
FRAME_PATH = "./img/loss_dist/frames"
N = 350
SKIP_LOADING = False
FPS = 5

# %%


def create_dataframe(all):

    df = pd.DataFrame(all, columns=["loss"])
    df = df.assign(label="loss")

    df["loss"] = df["loss"] / df["loss"].max()

    return df


# %%


def get_epoch(filename):
    num_epoch = int(filename.split("epoch")[1].split(".")[0])
    return num_epoch


items = list(filter(lambda f: f.endswith(".pth.tar"), os.listdir(LOSS_DIR)))
items.sort(key=get_epoch)
items = items[:N]

fig = plt.figure()

if not SKIP_LOADING:
    for root, dirs, files in os.walk(FRAME_PATH):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    for filename in items:
        if filename.endswith(".pth.tar"):
            print(f"Loading {filename}...")
            losses = torch.load(os.path.join(LOSS_DIR, filename))

            items = [item.item() for item in losses]

            print(f"Loaded {filename}\n\n")

            gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
            gmm.fit(losses.reshape(-1, 1))

            mean_1 = gmm.means_[0][0]
            mean_2 = gmm.means_[1][0]
            std_1 = np.sqrt(gmm.covariances_[0][0])[0]
            std_2 = np.sqrt(gmm.covariances_[1][0])[0]

            x = np.linspace(0, 1, num=1000)

            sns.set(font_scale=1)

            ax = sns.histplot(
                create_dataframe(items),
                x="loss",
                hue="label",
                multiple="layer",
                hue_order=["loss"],
                bins=100,
                palette=list(sns.color_palette("tab10")[:1]),
            )
            ax.grid(False)
            ax2 = ax.twinx()
            ax2.grid(False)

            y = norm.pdf(x, mean_1, std_1)
            ax2.plot(x, y, c="blue")

            y = norm.pdf(x, mean_2, std_2)
            ax2.plot(x, y, c="orange")

            epoch = get_epoch(filename)
            ax.set_title(f"Loss Distribution ({RATIO}% Sym. Noise, Epoch {epoch})")

            fig.savefig(f"{FRAME_PATH}/epoch{epoch}.png")
            plt.cla()
            ax.cla()
            ax2.cla()

# %%

images = []
items = list(filter(lambda f: f.endswith(".png"), os.listdir(FRAME_PATH)))
items.sort(key=get_epoch)
for file_name in items:
    if file_name.endswith(".png"):
        epoch = get_epoch(file_name)
        if epoch > N:
            continue
        file_path = os.path.join(FRAME_PATH, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(f"{FRAME_PATH}/../animated_{RATIO}.gif", images, fps=FPS)

# %%
