# ==================================
# IMPORTS
# ==================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

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
# DATASET
# ==================================
dataset = BurgersDataset("burgers_dataset.pt")
print('Dataset Loaded...')

n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train

train_set, val_set = random_split(
    dataset,
    [n_train, n_val]
)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=32
)

print('Train, Val sets created...')


# ==================================
# MODEL
# ==================================

model = FNO1d(
    modes=12,
    width=32
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
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)

epochs = 200
best_val = float("inf")

print('Optimiser and Scheduler defined...')

# ==================================
# TRAINING
# ==================================

print('Training Starts: ')
for epoch in range(epochs):

    model.train()
    train_loss = 0.0

    for batch in train_loader:

        a = batch["a"].to(device)
        u = batch["u"].to(device)

        optimizer.zero_grad()
        pred = model(a)
        loss = relative_l2(pred, u)

        loss.backward()
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

    scheduler.step()

    if val_loss < best_val:

        best_val = val_loss

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "modes": 12,
            "width": 32,

            "a_mean": dataset.a_mean,
            "a_std": dataset.a_std,

            "u_mean": dataset.u_mean,
            "u_std": dataset.u_std,
        }

        torch.save(
            checkpoint,
            "best_fno.pt"
        )

    if epoch % 10 == 0:

        print(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )
