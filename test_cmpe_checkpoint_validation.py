from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr
from sbijax import CMPE

from train_cmpe_phillip import (
    MASS_DIM,
    VALIDATION_BATCH_INDEX,
    build_density_estimator,
    build_prior_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load trained CMPE params, evaluate all validation trajectories, "
            "and print relative error for each trajectory."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "cmpe_1time",
        help="Directory containing cmpe params and saved arrays.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_1_time",
        help="Directory containing raw theta_batch_*.npy and x_batch_*.npy files.",
    )
    parser.add_argument(
        "--params-name",
        default="cmpe_params.pkl",
        help="CMPE parameter filename inside --artifact-dir.",
    )
    parser.add_argument(
        "--metadata-name",
        default="cmpe_metadata.json",
        help="Metadata filename inside --artifact-dir.",
    )
    parser.add_argument(
        "--validation-batch-index",
        type=int,
        default=int(VALIDATION_BATCH_INDEX),
        help="Validation batch index to evaluate when loading from --data-dir.",
    )
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples used to form each posterior mean prediction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for posterior sampling.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on the number of validation trajectories to evaluate.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Validation trajectory index to start evaluation from.",
    )
    return parser.parse_args()


def load_cmpe_artifacts(
    artifact_dir: Path, params_name: str, metadata_name: str
) -> tuple[dict, dict]:
    with (artifact_dir / params_name).open("rb") as f:
        params = pickle.load(f)
    with (artifact_dir / metadata_name).open("r", encoding="ascii") as f:
        metadata = json.load(f)
    return params, metadata


def load_validation_batch(
    artifact_dir: Path, data_dir: Path, batch_index: int
) -> tuple[np.ndarray, np.ndarray]:
    theta_val_path = artifact_dir / "theta_val.npy"
    x_val_path = artifact_dir / "x_val.npy"

    if theta_val_path.exists() and x_val_path.exists():
        theta_val = np.load(theta_val_path).astype(np.float32, copy=False)
        x_val = np.load(x_val_path).astype(np.float32, copy=False)
        return theta_val, x_val

    batch_label = f"{batch_index:02d}"
    theta_raw = np.load(data_dir / f"theta_batch_{batch_label}.npy").astype(
        np.float32, copy=False
    )
    x_raw = np.load(data_dir / f"x_batch_{batch_label}.npy").astype(np.float32, copy=False)

    if theta_raw.shape[0] != x_raw.shape[0]:
        raise ValueError(
            f"Batch {batch_label} length mismatch: {theta_raw.shape[0]} theta rows vs "
            f"{x_raw.shape[0]} x rows"
        )

    masses = theta_raw[:, :MASS_DIM]
    theta_val = theta_raw[:, MASS_DIM:]
    x_val = np.concatenate([masses, x_raw], axis=1)
    return theta_val, x_val


def build_cmpe_model(theta_dim: int, metadata: dict) -> CMPE:
    class _Args:
        pass

    args = _Args()
    args.cm_n_layers = int(metadata["cm_n_layers"])
    args.cm_hidden_size = int(metadata["cm_hidden_size"])
    args.cm_dropout_rate = float(metadata["cm_dropout_rate"])
    args.cm_t_min = float(metadata["cm_t_min"])
    args.cm_t_max = float(metadata["cm_t_max"])
    args.cm_sigma_data = float(metadata["cm_sigma_data"])

    prior_fn = build_prior_fn()
    density_estimator = build_density_estimator(args, theta_dim)
    return CMPE((prior_fn, None), density_estimator)


def posterior_mean_prediction(
    model: CMPE,
    params,
    x_obs: np.ndarray,
    num_samples: int,
    key: jax.Array,
    theta_dim: int,
) -> np.ndarray:
    posterior_samples, _ = model.sample_posterior(
        key,
        params,
        observable=jnp.asarray(x_obs),
        n_samples=num_samples,
    )
    theta_samples = np.asarray(posterior_samples.posterior["theta"].values)
    theta_samples = theta_samples.reshape(-1, theta_dim).astype(np.float32, copy=False)
    return theta_samples.mean(axis=0)


def relative_l2_error(theta_true: np.ndarray, theta_pred: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.linalg.norm(theta_pred - theta_true)
    denominator = np.linalg.norm(theta_true)
    return float(numerator / max(denominator, eps))


def main() -> None:
    args = parse_args()

    params, metadata = load_cmpe_artifacts(
        args.artifact_dir, args.params_name, args.metadata_name
    )
    theta_val, x_val = load_validation_batch(
        args.artifact_dir, args.data_dir, args.validation_batch_index
    )

    theta_dim = theta_val.shape[1]
    model = build_cmpe_model(theta_dim, metadata)

    total_trajectories = theta_val.shape[0]
    start_index = args.start_index
    if start_index < 0 or start_index >= total_trajectories:
        raise ValueError(
            f"--start-index must be in [0, {total_trajectories - 1}], got {start_index}"
        )
    available = total_trajectories - start_index
    if args.max_trajectories is None:
        num_trajectories = available
    else:
        num_trajectories = min(args.max_trajectories, available)
    end_index = start_index + num_trajectories

    print(
        f"Evaluating validation set trajectories [{start_index}, {end_index}) "
        f"({num_trajectories}/{total_trajectories})"
    )
    print(
        f"Using params {args.artifact_dir / args.params_name} "
        f"and {args.num_posterior_samples} posterior samples per trajectory"
    )
    print()

    relative_errors: list[float] = []
    for idx in range(start_index, end_index):
        theta_true = theta_val[idx]
        x_obs = x_val[idx]
        key = jr.PRNGKey(args.seed + idx)
        theta_pred = posterior_mean_prediction(
            model, params, x_obs, args.num_posterior_samples, key, theta_dim
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
