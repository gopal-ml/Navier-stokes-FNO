"""Train the existing FNO3d to predict flow fields from author-provided SAW data.

Prepare Y with u, v, w (and optionally pressure). Do not include temperature
as a target when the task is fluid-flow prediction only.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.fno import FNO3d
from flow.tandem_saw.dataset import TandemSAWDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared author-field .pt file")
    parser.add_argument(
        "--val-cases",
        nargs="+",
        required=True,
        help="Whole case IDs reserved for validation",
    )
    parser.add_argument("--output-dir", default="flow/tandem_saw/runs/default")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--modes", type=int, nargs=3, default=(12, 12, 12))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def channel_stats(tensor, indices):
    values = tensor[indices]
    mean = values.mean(dim=(0, 2, 3, 4), keepdim=True)
    std = values.std(dim=(0, 2, 3, 4), keepdim=True, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def relative_l2(prediction, target):
    dims = tuple(range(1, prediction.ndim))
    numerator = torch.linalg.vector_norm(prediction - target, dim=dims)
    denominator = torch.linalg.vector_norm(target, dim=dims).clamp_min(1e-8)
    return (numerator / denominator).mean()


def evaluate(model, loader, device, x_mean, x_std, y_mean, y_std):
    model.eval()
    totals = {"mse": 0.0, "mae": 0.0, "relative_l2": 0.0, "count": 0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model((x - x_mean.to(device)) / x_std.to(device))
            prediction = prediction * y_std.to(device) + y_mean.to(device)
            n = x.shape[0]
            totals["mse"] += torch.mean((prediction - y) ** 2).item() * n
            totals["mae"] += torch.mean(torch.abs(prediction - y)).item() * n
            totals["relative_l2"] += relative_l2(prediction, y).item() * n
            totals["count"] += n
    if totals["count"] == 0:
        raise ValueError("Validation split contains no samples")
    count = totals.pop("count")
    return {key: value / count for key, value in totals.items()}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = TandemSAWDataset(args.data)
    validation_cases = set(args.val_cases)
    all_cases = set(dataset.case_id)
    unknown = validation_cases - all_cases
    if unknown:
        raise ValueError(
            f"Unknown validation case IDs: {sorted(unknown)}; available: {sorted(all_cases)}"
        )

    train_idx = [
        i for i, case in enumerate(dataset.case_id) if case not in validation_cases
    ]
    val_idx = [i for i, case in enumerate(dataset.case_id) if case in validation_cases]
    if not train_idx or not val_idx:
        raise ValueError("Both train and validation splits must contain samples")

    x_mean, x_std = channel_stats(dataset.X, train_idx)
    y_mean, y_std = channel_stats(dataset.Y, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = FNO3d(
        modesx=args.modes[0],
        modesy=args.modes[1],
        modesz=args.modes[2],
        width=args.width,
        in_channels=dataset.X.shape[1],
        out_channels=dataset.Y.shape[1],
        n_layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_norm = (x - x_mean.to(device)) / x_std.to(device)
            y_norm = (y - y_mean.to(device)) / y_std.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_norm)
            loss = torch.mean((prediction - y_norm) ** 2)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * x.shape[0]
            train_count += x.shape[0]

        val_metrics = evaluate(model, val_loader, device, x_mean, x_std, y_mean, y_std)
        row = {
            "epoch": epoch,
            "train_normalized_mse": train_total / train_count,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "modes": list(args.modes),
                "width": args.width,
                "layers": args.layers,
                "in_channels": dataset.X.shape[1],
                "out_channels": dataset.Y.shape[1],
                "input_channels": dataset.input_channels,
                "output_channels": dataset.output_channels,
                "X_mean": x_mean,
                "X_std": x_std,
                "Y_mean": y_mean,
                "Y_std": y_std,
                "train_cases": sorted(all_cases - validation_cases),
                "validation_cases": sorted(validation_cases),
                "best_validation_metrics": val_metrics,
            }
            torch.save(checkpoint, out_dir / "best_fno3d.pt")

        with (out_dir / "history.jsonl").open("w") as stream:
            for item in history:
                stream.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
