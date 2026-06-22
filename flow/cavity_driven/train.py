# ==================================
# IMPORTS
# ==================================
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from flow.cavity_driven.dataset import CavityDrivenDataset
from core.fno import FNO1d, FNO2d
import time

# ==================================
# DEVICE
# ==================================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using:", device)

# ==================================
# DATASET
# ==================================
dataset = CavityDrivenDataset(
    "flow/cavity_driven/data/cavity_dataset_200_no_subsampling.pt"
)

print('Dataset Loaded...')

n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
batch_size = 16

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
modesx = 32
modesy = 32
width = 64
model = FNO2d(
    modesx= modesx,
    modesy= modesy,
    width=width,
    in_channels=5,
    out_channels=2,
    n_layers=8
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

criterion = nn.MSELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)

epochs = 50
best_val = float("inf")

train_history = []
val_history = []

print('Optimiser and Scheduler defined...')

# ==================================
# TRAINING
# ==================================

print('Training Starts: ')

training_start = time.time()

for epoch in range(epochs):

    epoch_start = time.time()
    # ==========================
    # TRAIN
    # ==========================
    model.train()
    train_loss = 0.0

    for batch in train_loader:

        a = batch["a"].to(device)
        u = batch["u"].to(device)

        optimizer.zero_grad()
        pred = model(a)
        loss = criterion(
            pred,
            u
        )

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(),
        #     1.0
        # )

        optimizer.step()

        train_loss += loss.item() * a.size(0)

    train_loss /= len(train_loader.dataset)

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    val_loss = 0.0
    with torch.no_grad():

        for batch in val_loader:
            a = batch["a"].to(device)
            u = batch["u"].to(device)

            pred = model(a)
            loss = criterion(
                pred,
                u
            )

            val_loss += loss.item() * a.size(0)

    val_loss /= len(val_loader.dataset)

    train_history.append(train_loss)
    val_history.append(val_loss)

    # scheduler.step()

    # ==========================
    # SAVE BEST
    # ==========================

    if val_loss < best_val:

        best_val = val_loss

        checkpoint = {

            "model_state_dict": model.state_dict(),

            "train_history": train_history,
            "val_history": val_history,

            # architecture
            "modesx": modesx,
            "modesy": modesy,
            "width": width,

            "in_channels": 5,
            "out_channels": 2,

            # normalization
            "u_mean": dataset.u_mean,
            "u_std": dataset.u_std,

            "v_mean": dataset.v_mean,
            "v_std": dataset.v_std,

            "nu_mean": dataset.nu_mean,
            "nu_std": dataset.nu_std,
        }

        torch.save(
            checkpoint,
            "flow/cavity_driven/checkpoints/best_fno_64_no_sub.pt"
        )

    if epoch % 1 == 0:

        print(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Time: {time.time() - epoch_start : .2f}s"
        )

print(f"Total Time taken : {time.time() - training_start: .2f}s")
