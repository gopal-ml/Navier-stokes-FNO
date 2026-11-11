"""Dataset reader for prepared author-provided SAW simulation snapshots.

Expected torch file keys:
    X: float tensor [samples, input_channels, Nx, Ny, Nz]
    Y: float tensor [samples, output_channels, Nx, Ny, Nz]
    case_id: sequence with one case identifier per sample
    input_channels: optional sequence of input channel names
    output_channels: optional sequence of output channel names
    x, y, z: optional coordinate vectors

Each sample should map a current flow state and its conditioning fields in X
to the corresponding next saved flow state in Y. For this project, Y should
contain velocity components (u, v, w), with pressure optional. Temperature is
not a predicted target. If supplied and needed by the chosen flow model,
temperature may be included among the conditioning channels in X. This reader
does not synthesize or modify physical data.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TandemSAWDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Prepared author dataset not found: {path}")

        data = torch.load(path, map_location="cpu")
        required = {"X", "Y", "case_id"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"Dataset is missing required keys: {sorted(missing)}")

        self.X = torch.as_tensor(data["X"], dtype=torch.float32)
        self.Y = torch.as_tensor(data["Y"], dtype=torch.float32)
        self.case_id = [str(v) for v in data["case_id"]]
        self.input_channels = [str(name) for name in data.get(
            "input_channels", [f"input_{i}" for i in range(self.X.shape[1])]
        )]
        self.output_channels = [str(name).strip().lower() for name in data.get(
            "output_channels", [f"output_{i}" for i in range(self.Y.shape[1])]
        )]
        self.coordinates = {name: data[name] for name in ("x", "y", "z") if name in data}

        if self.X.ndim != 5 or self.Y.ndim != 5:
            raise ValueError("X and Y must have shape [samples, channels, Nx, Ny, Nz]")
        if self.X.shape[0] != self.Y.shape[0] or self.X.shape[0] != len(self.case_id):
            raise ValueError("X, Y, and case_id must contain the same sample count")
        if self.X.shape[-3:] != self.Y.shape[-3:]:
            raise ValueError("Input and target fields must use the same spatial grid")
        if len(self.input_channels) != self.X.shape[1]:
            raise ValueError("input_channels length must match X channel count")
        if len(self.output_channels) != self.Y.shape[1]:
            raise ValueError("output_channels length must match Y channel count")
        target_names = {name.strip().lower() for name in self.output_channels}
        missing_velocity = {"u", "v", "w"} - target_names
        if missing_velocity:
            raise ValueError(
                "Y must contain the canonical velocity targets u, v, and w; "
                f"missing {sorted(missing_velocity)}"
            )
        non_flow = target_names - {"u", "v", "w", "p"}
        if non_flow:
            raise ValueError(
                "Y may contain only u, v, w, and optionally p for flow-only "
                f"training; unexpected targets: {sorted(non_flow)}"
            )
        if not torch.isfinite(self.X).all() or not torch.isfinite(self.Y).all():
            raise ValueError("Dataset contains NaN or Inf values")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
