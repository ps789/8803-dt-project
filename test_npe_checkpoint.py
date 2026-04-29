from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sbi.inference import NPE

from train_npe_phillip import MASS_DIM, build_prior


THETA_COMPONENTS = (
    ("x1", 0),
    ("vx1", 1),
    ("x2", 2),
    ("vx2", 3),
    ("x3", 4),
    ("vx3", 5),
    ("y1", 6),
    ("vy1", 7),
    ("y2", 8),
    ("vy2", 9),
    ("y3", 10),
    ("vy3", 11),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained NPE checkpoint, evaluate one sample from a saved batch, "
            "and print predicted vs true positions and velocities."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "npe_1time",
        help="Directory containing npe_density_estimator.pt, theta.pt, and x.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_1_time",
        help="Directory containing raw theta_batch_*.npy and x_batch_*.npy files.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=9,
        help="Batch index to inspect, e.g. 9 for theta_batch_09.npy.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=398,
        help="Sample row within the selected batch.",
    )
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples used to form the posterior mean prediction.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Torch device for inference, e.g. "cpu", "cuda", or "cuda:0".',
    )
    return parser.parse_args()


def load_training_artifacts(
    artifact_dir: Path, device: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    theta = torch.load(artifact_dir / "theta.pt", map_location=device)
    x = torch.load(artifact_dir / "x.pt", map_location=device)
    checkpoint = torch.load(
        artifact_dir / "npe_density_estimator.pt",
        map_location=device,
        weights_only=False,
    )
    return theta, x, checkpoint


def build_posterior(
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    checkpoint: dict[str, torch.Tensor],
    device: str,
):
    prior = build_prior(device)
    inference = NPE(prior=prior, device=device, show_progress_bars=False)
    inference.append_simulations(theta_train, x_train)

    if not hasattr(inference, "_build_neural_net"):
        raise RuntimeError(
            "This sbi version does not expose NPE._build_neural_net, so the "
            "saved checkpoint cannot be reconstructed with this script."
        )

    density_estimator = inference._build_neural_net(theta_train, x_train)
    density_estimator.load_state_dict(checkpoint["density_estimator_state_dict"])
    density_estimator = density_estimator.to(device)
    density_estimator.eval()
    return inference.build_posterior(density_estimator)


def load_test_sample(
    data_dir: Path, batch_index: int, sample_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_label = f"{batch_index:02d}"
    theta_raw = np.load(data_dir / f"theta_batch_{batch_label}.npy").astype(
        np.float32, copy=False
    )
    x_raw = np.load(data_dir / f"x_batch_{batch_label}.npy").astype(
        np.float32, copy=False
    )

    if sample_index < 0 or sample_index >= theta_raw.shape[0]:
        raise IndexError(
            f"sample_index={sample_index} is out of range for batch {batch_label} "
            f"with {theta_raw.shape[0]} rows"
        )
    if theta_raw.shape[0] != x_raw.shape[0]:
        raise ValueError(
            f"Batch {batch_label} length mismatch: {theta_raw.shape[0]} theta rows vs "
            f"{x_raw.shape[0]} x rows"
        )

    theta_true = theta_raw[sample_index, MASS_DIM:]
    masses = theta_raw[sample_index, :MASS_DIM]
    x_obs = np.concatenate([masses, x_raw[sample_index]], axis=0)

    return torch.from_numpy(theta_true), torch.from_numpy(x_obs)


def posterior_mean_prediction(
    posterior, x_obs: torch.Tensor, num_samples: int, device: str
) -> torch.Tensor:
    x_obs = x_obs.to(device)
    with torch.no_grad():
        samples = posterior.sample((num_samples,), x=x_obs)
    return samples.mean(dim=0).detach().cpu()


def print_comparison(theta_true: torch.Tensor, theta_pred: torch.Tensor) -> None:
    print("Predicted vs true positions and velocities")
    print()
    for name, index in THETA_COMPONENTS:
        true_value = theta_true[index].item()
        pred_value = theta_pred[index].item()
        error = pred_value - true_value
        print(
            f"{name:>3}: predicted={pred_value: .6f}  true={true_value: .6f}  "
            f"error={error: .6f}"
        )

    print()
    print("Grouped by body")
    print(
        "body 1: "
        f"pos=({theta_pred[0].item(): .6f}, {theta_pred[6].item(): .6f}) "
        f"true=({theta_true[0].item(): .6f}, {theta_true[6].item(): .6f})"
    )
    print(
        "body 1 vel: "
        f"pred=({theta_pred[1].item(): .6f}, {theta_pred[7].item(): .6f}) "
        f"true=({theta_true[1].item(): .6f}, {theta_true[7].item(): .6f})"
    )
    print(
        "body 2: "
        f"pos=({theta_pred[2].item(): .6f}, {theta_pred[8].item(): .6f}) "
        f"true=({theta_true[2].item(): .6f}, {theta_true[8].item(): .6f})"
    )
    print(
        "body 2 vel: "
        f"pred=({theta_pred[3].item(): .6f}, {theta_pred[9].item(): .6f}) "
        f"true=({theta_true[3].item(): .6f}, {theta_true[9].item(): .6f})"
    )
    print(
        "body 3: "
        f"pos=({theta_pred[4].item(): .6f}, {theta_pred[10].item(): .6f}) "
        f"true=({theta_true[4].item(): .6f}, {theta_true[10].item(): .6f})"
    )
    print(
        "body 3 vel: "
        f"pred=({theta_pred[5].item(): .6f}, {theta_pred[11].item(): .6f}) "
        f"true=({theta_true[5].item(): .6f}, {theta_true[11].item(): .6f})"
    )


def print_final_state_reference(x_obs: torch.Tensor) -> None:
    final_state = x_obs[MASS_DIM:]

    print()
    print("Final positions and velocities used as x_obs")
    print(
        "body 1: "
        f"pos=({final_state[0].item(): .6f}, {final_state[1].item(): .6f}) "
        f"vel=({final_state[6].item(): .6f}, {final_state[7].item(): .6f})"
    )
    print(
        "body 2: "
        f"pos=({final_state[2].item(): .6f}, {final_state[3].item(): .6f}) "
        f"vel=({final_state[8].item(): .6f}, {final_state[9].item(): .6f})"
    )
    print(
        "body 3: "
        f"pos=({final_state[4].item(): .6f}, {final_state[5].item(): .6f}) "
        f"vel=({final_state[10].item(): .6f}, {final_state[11].item(): .6f})"
    )


def main() -> None:
    args = parse_args()
    theta_train, x_train, checkpoint = load_training_artifacts(
        args.artifact_dir, args.device
    )
    posterior = build_posterior(theta_train, x_train, checkpoint, args.device)
    theta_true, x_obs = load_test_sample(
        args.data_dir, args.batch_index, args.sample_index
    )
    theta_pred = posterior_mean_prediction(
        posterior, x_obs, args.num_posterior_samples, args.device
    )

    print(
        f"Batch {args.batch_index:02d}, sample {args.sample_index}, "
        f"{args.num_posterior_samples} posterior samples"
    )
    print(f"x_obs shape: {tuple(x_obs.shape)}")
    print(f"theta_true shape: {tuple(theta_true.shape)}")
    print()
    print_comparison(theta_true, theta_pred)
    print_final_state_reference(x_obs.cpu())


if __name__ == "__main__":
    main()
