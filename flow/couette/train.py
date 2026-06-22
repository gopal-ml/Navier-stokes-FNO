# ==================================
# IMPORTS
# ==================================
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from flow.couette.dataset import CouetteDataset
from core.fno import FNO1d
import time

CURR_DIR = "flow/couette/"
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
dataset = CouetteDataset(
    CURR_DIR + "data/couette_dataset_100.pt"
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
    in_channels=3,
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

# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=100
# )

epochs = 20
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

    # scheduler.step()

    if val_loss < best_val:

        best_val = val_loss

        checkpoint = {

            "model_state_dict": model.state_dict(),

            "train_history": train_history,
            "val_history": val_history,

            "modes": modes,
            "width": width,

            "in_channels": 3,
            "out_channels": 1,

            "X_mean": dataset.X_mean,
            "X_std": dataset.X_std,

            "Y_mean": dataset.Y_mean,
            "Y_std": dataset.Y_std,

            "nu_mean": dataset.nu_mean,
            "nu_std": dataset.nu_std
        }

        torch.save(
            checkpoint,
            CURR_DIR +
            "checkpoints/best_fno_100.pt"
        )

        print("Model saved...")

    if epoch % 1 == 0:

        print(
            f"Epoch: {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Time: {time.time() - epoch_start : .2f}s"
        )

print(f"Total Time taken : {time.time() - training_start: .2f}s")
