"""Plot truth, FNO prediction, and error on the three mid-plane slices."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from core.fno import FNO3d
from flow.tandem_saw.dataset import TandemSAWDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--field", required=True, help="Output channel name")
    parser.add_argument("--output", default="flow/tandem_saw/prediction.png")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    dataset = TandemSAWDataset(args.data)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.field not in dataset.output_channels:
        raise ValueError(
            f"Field {args.field!r} not found. Available: {dataset.output_channels}"
        )
    if args.field not in checkpoint["output_channels"]:
        raise ValueError(f"Checkpoint has no output field {args.field!r}")
    if not 0 <= args.sample < len(dataset):
        raise IndexError(f"Sample index must be in [0, {len(dataset) - 1}]")

    model = FNO3d(
        modesx=checkpoint["modes"][0],
        modesy=checkpoint["modes"][1],
        modesz=checkpoint["modes"][2],
        width=checkpoint["width"],
        in_channels=checkpoint["in_channels"],
        out_channels=checkpoint["out_channels"],
        n_layers=checkpoint["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, y = dataset[args.sample]
    x_mean, x_std = checkpoint["X_mean"].to(device), checkpoint["X_std"].to(device)
    y_mean, y_std = checkpoint["Y_mean"].to(device), checkpoint["Y_std"].to(device)
    with torch.no_grad():
        prediction = model(((x.unsqueeze(0).to(device) - x_mean) / x_std))
        prediction = (prediction * y_std + y_mean).squeeze(0).cpu()

    channel = dataset.output_channels.index(args.field)
    truth = y[channel]
    pred = prediction[channel]
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
        im0 = axes[row, 0].imshow(
            actual.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        im1 = axes[row, 1].imshow(
            estimate.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        emax = max(residual.abs().max().item(), 1e-12)
        im2 = axes[row, 2].imshow(
            residual.T, origin="lower", cmap="coolwarm", vmin=-emax, vmax=emax
        )
        axes[row, 0].set_ylabel(plane)
        for ax, title in zip(
            axes[row], ("Author target", "FNO prediction", "Prediction error")
        ):
            ax.set_title(title)
        for ax, image in zip(axes[row], (im0, im1, im2)):
            fig.colorbar(image, ax=ax, shrink=0.78)

    fig.suptitle(
        f"Case {dataset.case_id[args.sample]} | {args.field} | sample {args.sample}"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
