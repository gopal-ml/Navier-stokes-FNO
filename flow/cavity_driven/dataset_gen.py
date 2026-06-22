# ============================================================
# IMPORTS
# ============================================================
import torch
from flow.cavity_driven.solver import solve_cavity_driven
from tqdm import tqdm

# ============================================================
# Device
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f'using: {device}')



# ============================================================
# PARAMETERS
# ============================================================

NU_VALUES = [
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.02,
    0.03,
    0.05
]

PREDICT_AHEAD = 10

# N_KEEP = 500

SAVE_EVERY = 20

# ============================================================
# STORAGE
# ============================================================

all_X_u = []
all_X_v = []

all_Y_u = []
all_Y_v = []

all_nu = []

# ============================================================
# GENERATE DATA
# ============================================================

for nu in tqdm(NU_VALUES):

    print(f"Generating nu={nu}")

    u_hist, v_hist, p_hist, x, y = solve_cavity_driven(
        nu=nu,
        T_final=5.0,
        save_every=SAVE_EVERY,
        device= 'cpu',
        verbose=False
    )

    u_hist = torch.stack(u_hist)
    v_hist = torch.stack(v_hist)

    if torch.isnan(u_hist).any():
        print(f"NaNs detected for nu={nu}")
        continue

    if torch.isnan(v_hist).any():
        print(f"NaNs detected for nu={nu}")
        continue

    # --------------------------------------------------------
    # Keep fixed number of snapshots
    # --------------------------------------------------------

    # n_keep = min(
    #     N_KEEP,
    #     len(u_hist)
    # )

    # idx = torch.linspace(
    #     0,
    #     len(u_hist) - 1,
    #     n_keep,
    #     dtype=torch.long
    # )

    # u_hist = torch.stack()
    # v_hist = v_hist[start_indx:]

    # --------------------------------------------------------
    # Autoregressive pairs
    # --------------------------------------------------------

    X_u = u_hist[:-PREDICT_AHEAD]
    X_v = v_hist[:-PREDICT_AHEAD]

    Y_u = u_hist[PREDICT_AHEAD:]
    Y_v = v_hist[PREDICT_AHEAD:]

    all_X_u.append(X_u)
    all_X_v.append(X_v)

    all_Y_u.append(Y_u)
    all_Y_v.append(Y_v)

    all_nu.append(
        torch.full(
            (X_u.shape[0], 1),
            nu
        )
    )

# ============================================================
# CONCATENATE
# ============================================================

X_u = torch.cat(all_X_u)
X_v = torch.cat(all_X_v)

Y_u = torch.cat(all_Y_u)
Y_v = torch.cat(all_Y_v)

nu = torch.cat(all_nu)

# ============================================================
# SANITY CHECKS
# ============================================================

assert not torch.isnan(X_u).any()
assert not torch.isnan(X_v).any()

assert not torch.isnan(Y_u).any()
assert not torch.isnan(Y_v).any()

assert len(X_u) == len(nu)

# ============================================================
# SAVE
# ============================================================

torch.save(
    {
        "X_u": X_u,
        "X_v": X_v,

        "Y_u": Y_u,
        "Y_v": Y_v,

        "nu": nu,

        "x": x,
        "y": y,

        "predict_ahead": PREDICT_AHEAD,
        "save_every": SAVE_EVERY,

        "nu_values": NU_VALUES
    },
    f"data/cavity_dataset_{PREDICT_AHEAD * SAVE_EVERY}_no_subsampling.pt"
)

print()

print("Dataset Saved")

print("Samples:", len(X_u))

print("Shape U:", X_u.shape)
print("Shape V:", X_v.shape)
