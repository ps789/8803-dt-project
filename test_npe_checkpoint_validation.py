from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import numpy as np
import torch
from sbi.inference import NPE

from train_npe_phillip import MASS_DIM, VALIDATION_BATCH_INDEX, build_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained NPE checkpoint, evaluate all validation trajectories, "
            "and print relative error for each trajectory."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "npe_1time",
        help="Directory containing the trained checkpoint and saved tensors.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_1_time",
        help="Directory containing raw theta_batch_*.npy and x_batch_*.npy files.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="npe_density_estimator.pt",
        help="Checkpoint filename inside --artifact-dir.",
    )
    parser.add_argument(
        "--validation-batch-index",
        type=int,
        default=int(VALIDATION_BATCH_INDEX),
        help="Validation batch index to evaluate.",
    )
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples used to form each posterior mean prediction.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Torch device for inference, e.g. "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on the number of validation trajectories to evaluate.",
    )
    return parser.parse_args()


def load_training_artifacts(
    artifact_dir: Path, checkpoint_name: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    theta = torch.load(artifact_dir / "theta.pt", map_location=device)
    x = torch.load(artifact_dir / "x.pt", map_location=device)
    checkpoint = torch.load(
        artifact_dir / checkpoint_name,
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


def load_validation_batch(
    data_dir: Path, batch_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_label = f"{batch_index:02d}"
    theta_raw = np.load(data_dir / f"theta_batch_{batch_label}.npy").astype(
        np.float32, copy=False
    )
    x_raw = np.load(data_dir / f"x_batch_{batch_label}.npy").astype(
        np.float32, copy=False
    )

    if theta_raw.shape[0] != x_raw.shape[0]:
        raise ValueError(
            f"Batch {batch_label} length mismatch: {theta_raw.shape[0]} theta rows vs "
            f"{x_raw.shape[0]} x rows"
        )

    theta_true = theta_raw[:, MASS_DIM:]
    masses = theta_raw[:, :MASS_DIM]
    x_obs = np.concatenate([masses, x_raw], axis=1)
    return torch.from_numpy(theta_true), torch.from_numpy(x_obs)


def posterior_mean_prediction(
    posterior, x_obs: torch.Tensor, num_samples: int, device: str
) -> torch.Tensor:
    x_obs = x_obs.to(device)
    with torch.no_grad():
        samples = posterior.sample((num_samples,), x=x_obs)
    return samples.mean(dim=0).detach().cpu()


def relative_l2_error(
    theta_true: torch.Tensor, theta_pred: torch.Tensor, eps: float = 1e-12
) -> float:
    numerator = torch.linalg.vector_norm(theta_pred - theta_true).item()
    denominator = torch.linalg.vector_norm(theta_true).item()
    return numerator / max(denominator, eps)


def main() -> None:
    args = parse_args()
    theta_train, x_train, checkpoint = load_training_artifacts(
        args.artifact_dir, args.checkpoint_name, args.device
    )
    posterior = build_posterior(theta_train, x_train, checkpoint, args.device)
    theta_val, x_val = load_validation_batch(args.data_dir, args.validation_batch_index)

    total_trajectories = theta_val.shape[0]
    if args.max_trajectories is None:
        num_trajectories = total_trajectories
    else:
        num_trajectories = min(args.max_trajectories, total_trajectories)

    print(
        f"Evaluating validation batch {args.validation_batch_index:02d} "
        f"with {num_trajectories}/{total_trajectories} trajectories"
    )
    print(
        f"Using checkpoint {args.artifact_dir / args.checkpoint_name} "
        f"and {args.num_posterior_samples} posterior samples per trajectory"
    )
    print()

    relative_errors: list[float] = []
    for idx in range(num_trajectories):
        theta_true = theta_val[idx]
        x_obs = x_val[idx]
        theta_pred = posterior_mean_prediction(
            posterior, x_obs, args.num_posterior_samples, args.device
        )
        rel_error = relative_l2_error(theta_true, theta_pred)
        relative_errors.append(rel_error)
        print(f"trajectory {idx:05d}: relative_l2_error={rel_error:.6e}")

    relative_errors_np = np.asarray(relative_errors, dtype=np.float64)
    print()
    print("Summary")
    print(f"mean_relative_l2_error={relative_errors_np.mean():.6e}")
    print(f"median_relative_l2_error={np.median(relative_errors_np):.6e}")
    print(f"min_relative_l2_error={relative_errors_np.min():.6e}")
    print(f"max_relative_l2_error={relative_errors_np.max():.6e}")
    print(f"std_relative_l2_error={relative_errors_np.std():.6e}")


if __name__ == "__main__":
    main()
