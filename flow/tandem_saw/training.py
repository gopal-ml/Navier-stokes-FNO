"""Train FNO3d on author-provided tandem SAW flow snapshots.

Targets must contain u, v, w and may additionally contain pressure p.
Temperature can be supplied as an input channel, but is never a target here.
"""

import argparse
import json
import logging
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.fno import FNO3d
from flow.tandem_saw.dataset import TandemSAWDataset


def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _run_logger(run_dir):
    logger = logging.getLogger(f"tandem_saw.{run_dir.name}.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _stats_for_log(mean, std, names):
    return {
        name: {"mean": float(mean[0, i, 0, 0, 0]), "std": float(std[0, i, 0, 0, 0])}
        for i, name in enumerate(names)
    }


def train_model(
    data_path,
    val_cases,
    runs_dir="flow/tandem_saw/runs",
    run_name=None,
    epochs=100,
    batch_size=2,
    modes=(12, 12, 12),
    width=32,
    layers=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
):
    """Train one case-held-out run and return its paths, metrics, and history."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(modes) != 3 or any(mode < 1 for mode in modes):
        raise ValueError("modes must contain three positive integers")
    if width < 1 or layers < 1:
        raise ValueError("width and layers must be positive")
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _run_logger(run_dir)

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        dataset = TandemSAWDataset(data_path)
        validation_cases = {str(case) for case in val_cases}
        all_cases = set(dataset.case_id)
        unknown = validation_cases - all_cases
        if unknown:
            raise ValueError(
                f"Unknown validation cases {sorted(unknown)}; available cases: {sorted(all_cases)}"
            )

        train_idx = [
            i for i, case in enumerate(dataset.case_id) if case not in validation_cases
        ]
        val_idx = [
            i for i, case in enumerate(dataset.case_id) if case in validation_cases
        ]
        if not train_idx or not val_idx:
            raise ValueError("Both training and validation splits must contain samples")

        x_mean, x_std = channel_stats(dataset.X, train_idx)
        y_mean, y_std = channel_stats(dataset.Y, train_idx)
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
        )
        device = select_device()

        model = FNO3d(
            modesx=modes[0],
            modesy=modes[1],
            modesz=modes[2],
            width=width,
            in_channels=dataset.X.shape[1],
            out_channels=dataset.Y.shape[1],
            n_layers=layers,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        config = {
            "run_name": run_name,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_path": str(Path(data_path).resolve()),
            "dataset_samples": len(dataset),
            "input_shape_per_sample": list(dataset.X.shape[1:]),
            "target_shape_per_sample": list(dataset.Y.shape[1:]),
            "input_channels": dataset.input_channels,
            "output_channels": dataset.output_channels,
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "train_cases": sorted(all_cases - validation_cases),
            "validation_cases": sorted(validation_cases),
            "architecture": {
                "name": "FNO3d",
                "modes": list(modes),
                "width": width,
                "layers": layers,
                "parameters": parameter_count,
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "normalization_X": _stats_for_log(x_mean, x_std, dataset.input_channels),
            "normalization_Y": _stats_for_log(y_mean, y_std, dataset.output_channels),
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        logger.info("Starting training run %s", run_name)
        logger.info("Configuration:\n%s", json.dumps(config, indent=2))

        history = []
        best_val = float("inf")
        checkpoint_path = run_dir / "best_fno3d.pt"
        history_path = run_dir / "history.jsonl"
        training_started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
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

            val_metrics = evaluate(
                model, val_loader, device, x_mean, x_std, y_mean, y_std
            )
            row = {
                "epoch": epoch,
                "train_normalized_mse": train_total / train_count,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "epoch_seconds": time.perf_counter() - epoch_started,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            logger.info("Epoch %d/%d | %s", epoch, epochs, json.dumps(row))

            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "modes": list(modes),
                    "width": width,
                    "layers": layers,
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
                    "run_name": run_name,
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info("Saved best checkpoint: %s", checkpoint_path)

        summary = {
            "run_name": run_name,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": time.perf_counter() - training_started,
            "best_validation_metrics": checkpoint["best_validation_metrics"],
            "checkpoint": str(checkpoint_path),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info("Run completed:\n%s", json.dumps(summary, indent=2))
        return {
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "config": config,
            "summary": summary,
            "history": history,
        }
    except Exception:
        logger.exception("Training run failed")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared author-field .pt file")
    parser.add_argument("--val-cases", nargs="+", required=True)
    parser.add_argument("--runs-dir", default="flow/tandem_saw/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--modes", type=int, nargs=3, default=(12, 12, 12))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        data_path=args.data,
        val_cases=args.val_cases,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        modes=args.modes,
        width=args.width,
        layers=args.layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
