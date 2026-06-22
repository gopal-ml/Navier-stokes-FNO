# ==================================
# IMPORTS
# ==================================
from pathlib import Path
import sys
from turtle import width

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from Navier.flow.poiseuille.dataset import PoiseuilleDataset
from Navier.core.fno import FNO1d
import time

# ==================================
# DEVICE
# ==================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using:", device)

# ==================================
# DATASET
# ==================================
dataset = PoiseuilleDataset(
    "data/poiseuille_dataset_20.pt"
)

print('Dataset Loaded...')

n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
batch_size = 32

train_set, val_set = random_split(
    dataset,
    [n_train, n_val]
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size
)

print('Train, Val sets created...')


# ==================================
# MODEL
# ==================================
modes = 32
width = 64
model = FNO1d(
    modes=modes,
    width=width,
    in_channels=4,
    out_channels=1
).to(device)

print('Model Defined...')
# ==================================
# RELATIVE R2 LOSS
# ==================================

def relative_l2(pred, target):

    num = torch.norm(
        pred - target,
        p=2,
        dim=(-2, -1)
    )

    den = torch.norm(
        target,
        p=2,
        dim=(-2, -1)
    )
    

    return torch.mean(num / (den+1e-8))

# ==================================
# OPTIMIZER AND TRAIN PARAMETERS
# ==================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-5
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)

epochs = 50
best_val = float("inf")

print('Optimiser and Scheduler defined...')

# ==================================
# TRAINING
# ==================================

print('Training Starts: ')

train_history = []
val_history = []
training_start = time.time()
for epoch in range(epochs):

    epoch_start = time.time()
    model.train()
    train_loss = 0.0

    for batch in train_loader:

        a = batch["a"].to(device)
        u = batch["u"].to(device)

        optimizer.zero_grad()
        pred = model(a)
        loss = torch.mean((pred - u)**2)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item() * a.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()

    val_loss = 0.0
    with torch.no_grad():

        for batch in val_loader:
            a = batch["a"].to(device)
            u = batch["u"].to(device)

            pred = model(a)
            loss = relative_l2(pred, u)

            val_loss += loss.item() * a.size(0)

    val_loss /= len(val_loader.dataset)

    train_history.append(train_loss)
    val_history.append(val_loss)

    scheduler.step()

    if val_loss < best_val:

        best_val = val_loss

        checkpoint = {

            "model_state_dict": model.state_dict(),

            "train_history": train_history,
            "val_history": val_history,

            "modes": modes,
            "width": width,

            "in_channels": 4,
            "out_channels": 1,

            "X_mean": dataset.X_mean,
            "X_std": dataset.X_std,

            "Y_mean": dataset.Y_mean,
            "Y_std": dataset.Y_std,

            "nu_mean": dataset.nu_mean,
            "nu_std": dataset.nu_std,

            "G_mean": dataset.G_mean,
            "G_std": dataset.G_std
        }

        torch.save(
            checkpoint,
            "checkpoints/best_fno.pt"
        )

    if epoch % 1 == 0:

        print(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Time: {time.time() - epoch_start : .2f}s"
        )

print(f"Total Time taken : {time.time() - training_start: .2f}s")
