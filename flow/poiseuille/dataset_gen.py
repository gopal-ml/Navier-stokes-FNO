# ============================================================
# IMPORTS
# ============================================================
import torch
from Navier.flow.poiseuille.solver import solve_poiseuille
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

# ==================================
# AutoRegressive Dataset Generation 
# ==================================

PREDICT_AHEAD = 1
SAVE_EVERY = 20
# ==================================
# PARAMETERS
# ==================================

nu_values = [
    0.005,
    0.01,
    0.02,
    0.05
]                 # viscosity

G_values = [
    0.5,
    1.0,
    2.0
]

Ny = 501           # grid points
T_final = 1.0


all_X = []
all_Y = []
all_nu = []
all_G = []

N_SNAPSHOTS = 1000
for G in tqdm(G_values):
    for nu in nu_values:
        snapshots = solve_poiseuille(nu, G, 
                                     Ny= Ny,
                                     T_final= T_final)
        snapshots = torch.stack(snapshots)

        N_keep = min(N_SNAPSHOTS, len(snapshots))

        indices = torch.linspace(
            0,
            len(snapshots)-1,
            N_keep,
            dtype=torch.long
        )

        snapshots = snapshots[indices]
        
        X = snapshots[:-PREDICT_AHEAD]
        Y = snapshots[PREDICT_AHEAD:]

        all_X.append(X)
        all_Y.append(Y)

        all_nu.append(torch.full((X.shape[0],1),nu))
        all_G.append(torch.full((X.shape[0],1),G))


X = torch.cat(all_X)
Y = torch.cat(all_Y)

nu = torch.cat(all_nu)
G = torch.cat(all_G)
# ==================================
# SANITY CHECKS
# ==================================

assert not torch.isnan(X).any()
assert not torch.isnan(Y).any()

assert X.shape == Y.shape
assert len(X) == len(nu)

assert not torch.isnan(G).any()
assert len(G) == len(X)

# ==================================
# SAVE
# ==================================

torch.save(
    {
        "X": X,
        "Y": Y,
        "nu": nu,
        "G": G,
        "y": torch.linspace(0, 1, Ny),

        "Ny": Ny,
        "T_final": T_final,
        "save_every": SAVE_EVERY,
        "predict_ahead": PREDICT_AHEAD,
        "nu_values": nu_values,
        "G_values": G_values
    },
    f"data/poiseuille_dataset_{PREDICT_AHEAD * SAVE_EVERY}.pt"
)
