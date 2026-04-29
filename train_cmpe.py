from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

# Restrict JAX to one visible GPU by default to avoid multi-GPU topology setup.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr
from sbijax import CMPE
from tensorflow_probability.substrates.jax import distributions as tfd

from sbijax.nn import make_cm

from rebound_simulator import simulate, summary_statistics


PRIOR_LOW = np.array(
    [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0],
    dtype=np.float32,
)
PRIOR_HIGH = np.array(
    [5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
    dtype=np.float32,
)

MASS_DIM = 3
TRAIN_BATCH_INDICES = tuple(f"{i:02d}" for i in range(9))
VALIDATION_BATCH_INDEX = "09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an sbijax CMPE model to infer theta from x batches."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_1_time",
        help="Directory containing theta_batch_*.npy and x_batch_*.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "cmpe_1time",
        help="Directory to write trained artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size passed through to sbijax.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate passed through to optax Adam.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of the training set reserved internally by sbijax for "
            "early stopping. Batch 09 is still loaded and saved separately for "
            "external validation."
        ),
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=int,
        default=50,
        help="Early-stop patience passed through to sbijax.",
    )
    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=500,
        help="Maximum training iterations passed through to sbijax.",
    )
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples to draw when evaluating an observed x.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for training and posterior sampling.",
    )
    parser.add_argument(
        "--x-obs-file",
        type=Path,
        default=None,
        help=(
            "Optional .npy file containing a single observed x vector "
            "with shape (15,) or (1, 15)."
        ),
    )
    parser.add_argument(
        "--use-notebook-example-x",
        action="store_true",
        help="Use the notebook's example observation generated from rebound_simulator.",
    )
    parser.add_argument(
        "--cm-n-layers",
        type=int,
        default=8,
        help="Number of residual blocks in the CMPE consistency model.",
    )
    parser.add_argument(
        "--cm-hidden-size",
        type=int,
        default=128,
        help="Hidden size used in each CMPE residual block.",
    )
    parser.add_argument(
        "--cm-dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate used in the CMPE residual network.",
    )
    parser.add_argument(
        "--cm-t-min",
        type=float,
        default=1e-3,
        help="Minimum consistency-model time value.",
    )
    parser.add_argument(
        "--cm-t-max",
        type=float,
        default=50.0,
        help="Maximum consistency-model time value.",
    )
    parser.add_argument(
        "--cm-sigma-data",
        type=float,
        default=1.0,
        help="Sigma-data hyperparameter for the consistency model.",
    )
    return parser.parse_args()


def build_prior_fn():
    low = jnp.asarray(PRIOR_LOW)
    high = jnp.asarray(PRIOR_HIGH)

    def prior_fn():
        return tfd.JointDistributionNamed(
            {
                "theta": tfd.Independent(
                    tfd.Uniform(low=low, high=high),
                    reinterpreted_batch_ndims=1,
                )
            }
        )

    return prior_fn


def build_density_estimator(args: argparse.Namespace, theta_dim: int):
    return make_cm(
        n_dimension=theta_dim,
        n_layers=args.cm_n_layers,
        hidden_size=args.cm_hidden_size,
        dropout_rate=args.cm_dropout_rate,
        t_min=args.cm_t_min,
        t_max=args.cm_t_max,
        sigma_data=args.cm_sigma_data,
    )


def find_batch_indices(data_dir: Path) -> list[str]:
    theta_files = {
        p.name.replace("theta_batch_", "").replace(".npy", "")
        for p in data_dir.glob("theta_batch_*.npy")
    }
    x_files = {
        p.name.replace("x_batch_", "").replace(".npy", "")
        for p in data_dir.glob("x_batch_*.npy")
    }
    indices = sorted(theta_files & x_files)
    missing_theta = sorted(x_files - theta_files)
    missing_x = sorted(theta_files - x_files)
    if missing_theta or missing_x:
        raise ValueError(
            f"Mismatched batch files in {data_dir}: "
            f"missing theta={missing_theta}, missing x={missing_x}"
        )
    if not indices:
        raise FileNotFoundError(f"No matching batch files found in {data_dir}")
    return indices


def transform_batch(
    theta_chunk: np.ndarray, x_chunk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta_chunk = theta_chunk.astype(np.float32, copy=False)
    x_chunk = x_chunk.astype(np.float32, copy=False)

    masses = theta_chunk[:, :MASS_DIM]
    theta_without_masses = theta_chunk[:, MASS_DIM:]
    x_with_masses = np.concatenate([masses, x_chunk], axis=1)
    return theta_without_masses, x_with_masses


def load_selected_batches(
    data_dir: Path, selected_indices: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    theta_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    available_indices = set(find_batch_indices(data_dir))

    for idx in selected_indices:
        if idx not in available_indices:
            raise FileNotFoundError(f"Missing batch {idx} in {data_dir}")
        theta_path = data_dir / f"theta_batch_{idx}.npy"
        x_path = data_dir / f"x_batch_{idx}.npy"
        theta_chunk = np.load(theta_path)
        x_chunk = np.load(x_path)
        if theta_chunk.shape[0] != x_chunk.shape[0]:
            raise ValueError(
                f"Batch {idx} length mismatch: {theta_chunk.shape[0]} "
                f"theta rows vs {x_chunk.shape[0]} x rows"
            )
        theta_without_masses, x_with_masses = transform_batch(theta_chunk, x_chunk)
        theta_chunks.append(theta_without_masses)
        x_chunks.append(x_with_masses)

    theta = np.concatenate(theta_chunks, axis=0)
    x = np.concatenate(x_chunks, axis=0)
    return theta, x


def load_train_validation_batches(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_train, x_train = load_selected_batches(data_dir, TRAIN_BATCH_INDICES)
    theta_val, x_val = load_selected_batches(data_dir, (VALIDATION_BATCH_INDEX,))
    return theta_train, x_train, theta_val, x_val


def load_observed_x(args: argparse.Namespace) -> np.ndarray | None:
    if args.x_obs_file is not None:
        x_obs = np.load(args.x_obs_file).astype(np.float32, copy=False)
        x_obs = np.squeeze(x_obs)
        if x_obs.shape != (15,):
            raise ValueError(f"Expected observed x to have shape (15,), got {x_obs.shape}")
        return x_obs

    if args.use_notebook_example_x:
        masses = np.array([1, 1, 1], dtype=np.float32)
        sim = simulate(1, 1, 1, 0, 0, 1, 2, 3, 0.6, 0, 0, 1, 1, 0.9, 0.3)
        stats = summary_statistics(sim).astype(np.float32, copy=False)
        return np.concatenate([masses, stats]).astype(np.float32, copy=False)

    return None


def extract_theta_samples(posterior_samples, theta_dim: int) -> np.ndarray:
    theta_samples = np.asarray(posterior_samples.posterior["theta"].values)
    return theta_samples.reshape(-1, theta_dim).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    theta_train, x_train, theta_val, x_val = load_train_validation_batches(args.data_dir)
    theta_dim = theta_train.shape[1]

    print(
        f"Loaded train theta with shape {tuple(theta_train.shape)} from "
        f"batches {TRAIN_BATCH_INDICES[0]}-{TRAIN_BATCH_INDICES[-1]} in {args.data_dir}"
    )
    print(
        f"Loaded train x with shape {tuple(x_train.shape)} from "
        f"batches {TRAIN_BATCH_INDICES[0]}-{TRAIN_BATCH_INDICES[-1]} in {args.data_dir}"
    )
    print(
        f"Loaded validation theta with shape {tuple(theta_val.shape)} from "
        f"batch {VALIDATION_BATCH_INDEX} in {args.data_dir}"
    )
    print(
        f"Loaded validation x with shape {tuple(x_val.shape)} from "
        f"batch {VALIDATION_BATCH_INDEX} in {args.data_dir}"
    )
    print(
        "CMPE training uses sbijax's internal validation split from the training "
        f"data with validation_fraction={args.validation_fraction:.3f}; "
        "batch 09 is saved separately for downstream evaluation."
    )

    prior_fn = build_prior_fn()
    density_estimator = build_density_estimator(args, theta_dim)
    model = CMPE((prior_fn, None), density_estimator)

    data = {
        "theta": jnp.asarray(theta_train),
        "y": jnp.asarray(x_train),
    }
    train_key = jr.PRNGKey(args.seed)
    params, losses = model.fit(
        train_key,
        data=data,
        optimizer=optax.adam(args.learning_rate),
        n_iter=args.max_num_epochs,
        batch_size=args.batch_size,
        percentage_data_as_validation_set=args.validation_fraction,
        n_early_stopping_patience=args.stop_after_epochs,
    )

    losses_np = np.asarray(jax.device_get(losses))
    with (args.output_dir / "cmpe_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    np.save(args.output_dir / "theta.npy", theta_train)
    np.save(args.output_dir / "x.npy", x_train)
    np.save(args.output_dir / "theta_val.npy", theta_val)
    np.save(args.output_dir / "x_val.npy", x_val)
    np.save(args.output_dir / "training_losses.npy", losses_np)
    with (args.output_dir / "cmpe_metadata.json").open("w", encoding="ascii") as f:
        json.dump(
            {
                "theta_shape": list(theta_train.shape),
                "x_shape": list(x_train.shape),
                "validation_theta_shape": list(theta_val.shape),
                "validation_x_shape": list(x_val.shape),
                "prior_low": PRIOR_LOW.tolist(),
                "prior_high": PRIOR_HIGH.tolist(),
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "validation_fraction": args.validation_fraction,
                "stop_after_epochs": args.stop_after_epochs,
                "max_num_epochs": args.max_num_epochs,
                "cm_n_layers": args.cm_n_layers,
                "cm_hidden_size": args.cm_hidden_size,
                "cm_dropout_rate": args.cm_dropout_rate,
                "cm_t_min": args.cm_t_min,
                "cm_t_max": args.cm_t_max,
                "cm_sigma_data": args.cm_sigma_data,
            },
            f,
            indent=2,
        )
    print(f"Saved artifacts to {args.output_dir}")
    print(f"Loss history shape: {tuple(losses_np.shape)}")

    x_obs = load_observed_x(args)
    if x_obs is None:
        return

    posterior_key = jr.PRNGKey(args.seed + 1)
    posterior_samples, diagnostics = model.sample_posterior(
        posterior_key,
        params,
        observable=jnp.asarray(x_obs),
        n_samples=args.num_posterior_samples,
    )
    theta_samples = extract_theta_samples(posterior_samples, theta_dim)
    posterior_mean_estimate = theta_samples.mean(axis=0)

    np.save(args.output_dir / "x_obs.npy", x_obs)
    np.save(args.output_dir / "posterior_theta_samples.npy", theta_samples)
    np.save(args.output_dir / "posterior_mean_estimate.npy", posterior_mean_estimate)
    with (args.output_dir / "posterior_diagnostics.pkl").open("wb") as f:
        pickle.dump(jax.device_get(diagnostics), f)

    print(f"Observed x shape: {tuple(x_obs.shape)}")
    print(f"Posterior theta samples shape: {tuple(theta_samples.shape)}")
    print(f"Posterior mean estimate shape: {tuple(posterior_mean_estimate.shape)}")


if __name__ == "__main__":
    main()
