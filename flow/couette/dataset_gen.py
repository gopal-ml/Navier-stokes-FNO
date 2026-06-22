# ============================================================
# IMPORTS
# ============================================================
import torch

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using: {device}')


# ============================================================
# Couette Forward Euler solver
# ============================================================
PREDICT_AHEAD = 5
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

Ny = 501           # grid points

T_final = 5.0     # final Time


# ==================================
# GRID
# ==================================

y = torch.linspace(0, 1, Ny)

dy = y[1] - y[0]


def solve_couette(nu):
    # ==================================
    # STABLE TIME STEP
    # ==================================

    dt = 0.4 * dy**2 / nu

    Nt = int(T_final / dt)

    r = nu * dt / dy**2

    print(f"dt = {dt:.6e}")
    print(f"Nt = {Nt}")
    print(f"r  = {r:.4f}")


    # ==================================
    # INITIAL CONDITION
    # ==================================

    u_field = torch.zeros(Ny)

    # ==================================
    # BOUNDARY CONDITIONS
    # ==================================

    u_field[0] = 0.0
    u_field[-1] = 1.0


    # ==================================
    # STORAGE
    # ==================================

    snapshots = []

    snapshots.append(
        u_field.clone()
    )


    # ==================================
    # TIME INTEGRATION
    # ==================================

    for step in range(Nt):

        u_new = u_field.clone()

        u_new[1:-1] = (
            u_field[1:-1]
            +
            r * (
                u_field[2:]
                - 2.0 * u_field[1:-1]
                + u_field[:-2]
            )
        )

        # boundary conditions

        u_new[0] = 0.0

        u_new[-1] = 1.0

        u_field = u_new

        # save every 20 steps

        if step % SAVE_EVERY == 0:

            snapshots.append(
                u_field.clone()
            )
        
    return snapshots

# for nu in nu_values:

#     snapshots = solve_couette(nu)
#     snapshots = torch.stack(snapshots)

#     X = snapshots[:-PREDICT_AHEAD]
#     Y = snapshots[PREDICT_AHEAD:]

#     print(
#         f"nu={nu:.3f}",
#         "samples=", X.shape[0]
#     )

# ==================================
# AutoRegressive Dataset Generation 
# ==================================


all_X = []
all_Y = []
all_nu = []
for nu in nu_values:
    snapshots = solve_couette(nu)
    snapshots = torch.stack(snapshots)
    
    X = snapshots[:-PREDICT_AHEAD]
    Y = snapshots[PREDICT_AHEAD:]

    all_X.append(X)
    all_Y.append(Y)

    all_nu.append(torch.full((X.shape[0],1),nu))


X = torch.cat(all_X)

Y = torch.cat(all_Y)

nu = torch.cat(all_nu)

# ==================================
# SANITY CHECKS
# ==================================

assert not torch.isnan(X).any()
assert not torch.isnan(Y).any()

assert X.shape == Y.shape
assert len(X) == len(nu)

# ==================================
# SAVE
# ==================================

torch.save(
    {
        "X": X,
        "Y": Y,
        "nu": nu,
        "y": y,

        "Ny": Ny,
        "T_final": T_final,
        "save_every": SAVE_EVERY,
        "predict_ahead": PREDICT_AHEAD,
        "nu_values": nu_values
    },
    f"flow/couette/data/couette_dataset_{PREDICT_AHEAD * SAVE_EVERY}.pt"
)
