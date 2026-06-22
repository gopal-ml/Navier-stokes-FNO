# ==================================
# IMPORTS
# ==================================

import torch
import matplotlib.pyplot as plt

from Navier.burgers_eq.dataset import BurgersDataset
from Navier.core.fno import FNO1d

# ==================================
# DEVICE
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

# ==================================
# LOAD CHECKPOINT
# ==================================

checkpoint = torch.load(
    "best_fno.pt",
    map_location=device
)

print("Checkpoint Loaded...")

# ==================================
# LOAD MODEL
# ==================================

model = FNO1d(
    modes=checkpoint["modes"],
    width=checkpoint["width"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Loaded...")

# ==================================
# LOAD DATASET
# ==================================

dataset = BurgersDataset(
    "burgers_dataset.pt"
)

print("Dataset Loaded...")

# ==================================
# PICK SAMPLE
# ==================================

idx = torch.randint(
    len(dataset),
    (1,)
).item()

sample = dataset[idx]

print(f"Sample Index: {idx}")

# ==================================
# PREPARE INPUTS
# ==================================

a = sample["a"].unsqueeze(0).to(device)
target = sample["u"].unsqueeze(0).to(device)

# ==================================
# INFERENCE
# ==================================

with torch.no_grad():

    pred = model(a)

print("Inference Complete...")

# ==================================
# DENORMALIZE
# ==================================

a_mean = checkpoint["a_mean"].item()
a_std  = checkpoint["a_std"].item()

u_mean = checkpoint["u_mean"].item()
u_std  = checkpoint["u_std"].item()

initial = a[0, 0].cpu()
initial = initial * a_std + a_mean

pred = pred * u_std + u_mean
target = target * u_std + u_mean

pred = pred[0, 0].cpu()
target = target[0, 0].cpu()

# ==================================
# ERROR METRICS
# ==================================

rel_l2 = (
    torch.norm(pred - target)
    /
    torch.norm(target)
)

mse = torch.mean(
    (pred - target) ** 2
)

mae = torch.mean(
    torch.abs(pred - target)
)

print(f"Relative L2 : {rel_l2:.6f}")
print(f"MSE         : {mse:.6f}")
print(f"MAE         : {mae:.6f}")

# ==================================
# GRID
# ==================================

N = len(pred)

x = torch.linspace(
    0,
    1,
    N
)

# ==================================
# PLOTS
# ==================================

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Initial condition
ax[0].plot(x, initial)
ax[0].set_title("Initial Condition a(x)")
ax[0].grid(True)

# Prediction vs truth
ax[1].plot(
    x,
    target,
    label="Ground Truth"
)

ax[1].plot(
    x,
    pred,
    "--",
    label="FNO Prediction"
)

ax[1].set_title(f"Burgers Solution (sample{idx})")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
