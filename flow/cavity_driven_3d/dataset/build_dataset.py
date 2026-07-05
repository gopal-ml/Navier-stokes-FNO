# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import torch
from tqdm import tqdm


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    raw_dir,
    save_path,
):

    raw_dir = Path(raw_dir)
    save_path = Path(save_path)

    files = sorted(raw_dir.glob("*.pt"))

    X = []
    Y = []
    sample_nu = []

    x = y = z = None

    for file in tqdm(files):

        data = torch.load(file, map_location="cpu")

        u_hist = data["u_hist"]
        v_hist = data["v_hist"]
        w_hist = data["w_hist"]

        nu = float(data["nu"])

        x = data["x"]
        y = data["y"]
        z = data["z"]

        Nx = len(x)
        Ny = len(y)
        Nz = len(z)

        # ----------------------------------------------------
        # Coordinate channels
        # ----------------------------------------------------

        X_grid, Y_grid, Z_grid = torch.meshgrid(
            x,
            y,
            z,
            indexing="ij",
        )

        nu_channel = torch.full(
            (Nx, Ny, Nz),
            nu,
        )

        # ----------------------------------------------------
        # Consecutive pairs
        # ----------------------------------------------------
        

        for i in range(len(u_hist) - 1):

            u = u_hist[i].float()
            v = v_hist[i].float()
            w = w_hist[i].float()

            u_next = u_hist[i + 1].float()
            v_next = v_hist[i + 1].float()
            w_next = w_hist[i + 1].float()

            input_tensor = torch.stack(
                [
                    u,
                    v,
                    w,
                    nu_channel,
                    X_grid,
                    Y_grid,
                    Z_grid,
                ],
                dim=0,
            )

            target_tensor = torch.stack(
                [
                    u_next,
                    v_next,
                    w_next,
                ],
                dim=0,
            )

            X.append(input_tensor)
            Y.append(target_tensor)
            sample_nu.append(nu)

    X = torch.stack(X)
    Y = torch.stack(Y)

    print(f"Dataset size : {len(X)}")
    print(f"Input shape  : {X.shape}")
    print(f"Target shape : {Y.shape}")

    torch.save(
        {
            "X": X.float(),
            "Y": Y.float(),
            "x": x.float(),
            "y": y.float(),
            "z": z.float(),
            "nu": nu,
            "sample_nu": sample_nu,
            "input_channels": [
                "u",
                "v",
                "w",
                "nu",
                "x",
                "y",
                "z",
            ],
            "output_channels": [
                "u",
                "v",
                "w",
            ],
        },
        save_path,
    )

    print(f"\nSaved dataset to {save_path}")


if __name__ == "__main__":

    build_dataset(
        raw_dir="flow/cavity_driven_3d/data/raw/",
        save_path="flow/cavity_driven_3d/data/processed/dataset.pt",
    )
