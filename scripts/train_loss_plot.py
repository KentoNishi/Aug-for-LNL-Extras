# %%

import torch
import os

# %%
LOSS_DIR = "../Augmentation-for-LNL/checkpoints/c10/50sym/AugDesc-WS/loss"

# %%

for filename in os.listdir(LOSS_DIR):
    if filename.endswith(".pth.tar"):
        print(f"Loading {filename}...")
        losses = torch.load(os.path.join(LOSS_DIR, filename))
        print(f"Loaded {filename}\n")

