# %%

import torch
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio


# %%
LOSS_DIR = "../Augmentation-for-LNL/checkpoints/c10/80sym/AugDesc-WS/loss"
IS_NOISY_PATH = (
    "../Augmentation-for-LNL/checkpoints/c10/80sym/AugDesc-WS/saved/is_noisy.npy"
)
FRAME_PATH = "./img/train_loss_plot/frames"
N = 100
SKIP_LOADING = True
FPS = 10

# %%


def create_dataframe(clean, dirty):

    dfc = pd.DataFrame(clean, columns=["loss"])
    dfc = dfc.assign(label="correct")

    dfd = pd.DataFrame(dirty, columns=["loss"])
    dfd = dfd.assign(label="incorrect")

    df = pd.concat([dfd, dfc])
    df["loss"] = df["loss"] / df["loss"].max()

    return df


# %%


is_noisy = np.load(IS_NOISY_PATH)


def get_epoch(filename):
    num_epoch = int(filename.split("epoch")[1].split(".")[0])
    return num_epoch


items = list(filter(lambda f: f.endswith(".pth.tar"), os.listdir(LOSS_DIR)))
items.sort(key=get_epoch)
items = items[:N]

fig = plt.figure()

if not SKIP_LOADING:
    for filename in items:
        if filename.endswith(".pth.tar"):
            print(f"Loading {filename}...")
            losses = torch.load(os.path.join(LOSS_DIR, filename))[0].tolist()

            clean_items, dirty_items = [], []
            for i in range(50000):
                (dirty_items if is_noisy[i] else clean_items).append(losses[i])

            print(f"Loaded {filename}\n\n")

            sns.set(font_scale=1)

            ax = sns.histplot(
                create_dataframe(clean_items, dirty_items),
                x="loss",
                hue="label",
                multiple="layer",
                bins=100,
                palette=sns.color_palette("tab10")[:2],
            )
            epoch = get_epoch(filename)
            ax.set_title(f"Loss Distribution (80% Sym. Noise, Epoch {epoch})")

            fig.savefig(f"{FRAME_PATH}/epoch{epoch}.png")
            plt.cla()

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
imageio.mimsave(f"{FRAME_PATH}/../animated.gif", images, fps=FPS)

# %%
