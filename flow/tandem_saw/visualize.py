"""Load FNO predictions and plot comparisons for tandem SAW flow fields."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from core.fno import FNO3d
from flow.tandem_saw.dataset import TandemSAWDataset
from flow.tandem_saw.training import select_device


def predict_sample(dataset, checkpoint_path, sample_index, device=None):
    """Return one sample's physical-unit target and FNO prediction."""
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"Sample index must be in [0, {len(dataset) - 1}]")
    device = device or select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = FNO3d(
        modesx=checkpoint["modes"][0], modesy=checkpoint["modes"][1],
        modesz=checkpoint["modes"][2], width=checkpoint["width"],
        in_channels=checkpoint["in_channels"], out_channels=checkpoint["out_channels"],
        n_layers=checkpoint["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, target = dataset[sample_index]
    x_mean = checkpoint["X_mean"].to(device)
    x_std = checkpoint["X_std"].to(device)
    y_mean = checkpoint["Y_mean"].to(device)
    y_std = checkpoint["Y_std"].to(device)
    with torch.no_grad():
        normalized_x = (x.unsqueeze(0).to(device) - x_mean) / x_std
        prediction = model(normalized_x)
        prediction = (prediction * y_std + y_mean).squeeze(0).cpu()

    return {
        "case_id": dataset.case_id[sample_index],
        "sample_index": sample_index,
        "fields": dataset.output_channels,
        "target": target,
        "prediction": prediction,
        "checkpoint": checkpoint,
    }


def plot_prediction(result, field="u", output_path=None):
    """Plot target, prediction, and signed error on three central slices."""
    if field not in result["fields"]:
        raise ValueError(f"Field {field!r} not found. Available: {result['fields']}")
    channel = result["fields"].index(field)
    truth = result["target"][channel]
    pred = result["prediction"][channel]
    error = pred - truth
    mid = [n // 2 for n in truth.shape]
    views = [
        ("XY mid-plane", truth[:, :, mid[2]], pred[:, :, mid[2]], error[:, :, mid[2]]),
        ("XZ mid-plane", truth[:, mid[1], :], pred[:, mid[1], :], error[:, mid[1], :]),
        ("YZ mid-plane", truth[mid[0], :, :], pred[mid[0], :, :], error[mid[0], :, :]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, (plane, actual, estimate, residual) in enumerate(views):
        vmax = max(actual.abs().max().item(), estimate.abs().max().item(), 1e-12)
        images = [
            axes[row, 0].imshow(
                actual.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
            ),
            axes[row, 1].imshow(
                estimate.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
            ),
        ]
        error_limit = max(residual.abs().max().item(), 1e-12)
        images.append(
            axes[row, 2].imshow(
                residual.T, origin="lower", cmap="coolwarm",
                vmin=-error_limit, vmax=error_limit,
            )
        )
        axes[row, 0].set_ylabel(plane)
        for ax, title, image in zip(
            axes[row],
            ("Author target", "FNO prediction", "Prediction error"),
            images,
        ):
            ax.set_title(title)
            fig.colorbar(image, ax=ax, shrink=0.78)

    fig.suptitle(
        f"Case {result['case_id']} | {field} | sample {result['sample_index']}"
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    return fig, axes


def plot_training_history(history):
    """Plot normalized train loss and physical-unit validation metrics."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_normalized_mse"] for row in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Normalized MSE")
    axes[0].grid(True, alpha=0.3)
    for metric in ("mse", "mae", "relative_l2"):
        axes[1].plot(epochs, [row[f"val_{metric}"] for row in history], label=metric)
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Physical-unit metric")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--field", required=True, help="Output channel name")
    parser.add_argument("--output", default="flow/tandem_saw/prediction.png")
    args = parser.parse_args()

    dataset = TandemSAWDataset(args.data)
    result = predict_sample(dataset, args.checkpoint, args.sample)
    plot_prediction(result, args.field, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
