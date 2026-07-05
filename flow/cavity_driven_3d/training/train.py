# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tqdm import tqdm

from flow.cavity_driven_3d.dataset.dataset import Cavity3DDataset 
from core.fno import FNO3d
from datetime import datetime

# ============================================================
# LOGS
# ============================================================
LOG_FILE = Path("flow/cavity_driven_3d/training/logs.txt")

def log(message):

    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


run_name = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

# ============================================================
# DEVICE
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device}")

# ==================================
# DATASET
# ==================================
dataset = Cavity3DDataset(
    "flow/cavity_driven_3d/data/processed/dataset.pt",
    normalize=True,
)

print('Dataset Loaded...')

# ==================================
# Train/Val Split
# ==================================

batch_size = 4

train_nu = {
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
}

val_nu = {
    0.045,
    0.050,
}

train_idx = [
    i
    for i, nu in enumerate(dataset.sample_nu)
    if round(float(nu), 3) in train_nu
]

val_idx = [
    i
    for i, nu in enumerate(dataset.sample_nu)
    if round(float(nu), 3) in val_nu
]

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
)

print('Train, Val sets created...')

# ==================================
# MODEL
# ==================================
modesx = 12
modesy = 12
modesz = 12
width = 32

model = FNO3d(
    modesx=modesx,
    modesy=modesy,
    modesz=modesz,
    width=width,
    in_channels=7,
    out_channels=3,
    n_layers= 8,
).to(device)

print('Model Defined...')


# ==================================
# OPTIMIZER AND TRAIN PARAMETERS
# ==================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

CHECKPOINT_DIR = Path("flow/cavity_driven_3d/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

best_loss = float("inf")
epochs = 10

train_history = []
val_history = []


print('Optimiser and Scheduler defined...')

# ==================================
# TRAINING
# ==================================

print('Training Starts: ')

log("="*70)
log(f"{run_name} - FNO3D TRAINING")
log("="*70)

log(f"Device              : {device}")
log(f"Training Samples    : {len(train_dataset)}")
log(f"Validation Samples  : {len(val_dataset)}")

log(f"Epochs              : {epochs}")
log(f"Batch Size          : {batch_size}")
log(f"Learning Rate       : {optimizer.param_groups[0]['lr']}")

log("")

training_start = time.time()

for epoch in range(epochs):

    epoch_start = time.time()
    # ==========================
    # TRAIN
    # ==========================

    model.train()
    train_loss = 0.0

    for X,Y in tqdm(train_loader):

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(pred,Y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ==========================
    # VALIDATION
    # ==========================
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X,Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)

            val_loss += criterion(
                pred,
                Y,
            ).item()

    val_loss /= len(val_loader)

    train_history.append(train_loss)
    val_history.append(val_loss)

    # ==========================
    # SAVE BEST
    # ==========================
    
    if val_loss < best_loss:

        best_loss = val_loss
        checkpoint = {

            "model_state_dict": model.state_dict(),

            "train_history": train_history,
            "val_history": val_history,

            # architecture
            "modesx": modesx,
            "modesy": modesy,
            "modesz": modesz,
            "width": width,

            "in_channels": 7,
            "out_channels": 3,

            # normalization
            "X_mean": dataset.X_mean,
            "X_std": dataset.X_std,
            "Y_mean": dataset.Y_mean,
            "Y_std": dataset.Y_std,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR/"best_fno_8.pt"
        )

    if epoch % 1 == 0:

        print(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Time: {time.time() - epoch_start : .2f}s"
        )
        log(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Time: {time.time() - epoch_start : .2f}s"
        )

print(f"Total Time taken : {time.time() - training_start: .2f}s")
log(f"Total Time taken : {time.time() - training_start: .2f}s")

torch.save(
    {
        "train_loss": train_history,
        "val_loss": val_history,
    },
    CHECKPOINT_DIR / "loss_history_8.pt",
)
