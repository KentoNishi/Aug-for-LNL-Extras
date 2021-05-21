# %%

import torch
import os
import json
import numpy as np

# %%
LOSS_DIR = "../Augmentation-for-LNL/checkpoints/c10/50sym/AugDesc-WS/loss"
IS_NOISY_PATH = (
    "../Augmentation-for-LNL/checkpoints/c10/50sym/AugDesc-WS/saved/is_noisy.npy"
)

# %%

is_noisy = np.load(IS_NOISY_PATH)

for filename in os.listdir(LOSS_DIR):
    if filename.endswith(".pth.tar"):
        print(f"Loading {filename}...")
        losses = torch.load(os.path.join(LOSS_DIR, filename))[0].tolist()

        clean_items, dirty_items = [], []
        for i in range(50000):
            (dirty_items if is_noisy[i] else clean_items).append(losses[i])

        print(f"Loaded {filename}\n\n")


# %%
