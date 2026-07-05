# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import torch
from tqdm import tqdm

from flow.cavity_driven_3d.solver import CavitySolver3D


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("flow/cavity_driven_3d/data/raw/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

Nx = 32
Ny = 32
Nz = 32

dt = 1e-3
T_final = 1.5

save_every = 5

pressure_iterations = 100

nu_values = torch.linspace(
    0.005,
    0.05,
    10,
).tolist()


# ============================================================
# GENERATE DATA
# ============================================================

for nu in tqdm(nu_values):

    solver = CavitySolver3D(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        nu=nu,
        dt=dt,
        T_final=T_final,
        save_every=save_every,
        pressure_iterations=pressure_iterations,
        verbose=False,
        device='cpu',
    )

    solver.solve()

    filename = OUTPUT_DIR / f"cavity3d_nu_{nu:.3f}.pt"

    solver.save(filename)

    print(f"Saved {filename}")

print("\nFinished generating raw simulations.")
