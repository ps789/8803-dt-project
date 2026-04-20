from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi.inference import NPE
from tqdm.auto import tqdm

from rebound_simulator import simulate, summary_statistics


PRIOR_LOW = torch.tensor(
    [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0],
    dtype=torch.float32,
)
PRIOR_HIGH = torch.tensor(
    [10, 10, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an sbi NPE model to infer theta from x batches."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing theta_batch_*.npy and x_batch_*.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "npe",
        help="Directory to write trained artifacts.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device for training, e.g. "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size passed through to sbi.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate passed through to sbi.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Validation split passed through to sbi.",
    )
    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=200,
        help="Maximum training epochs passed through to sbi.",
    )
    parser.add_argument(
        "--num-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples to draw when evaluating an observed x.",
    )
    parser.add_argument(
        "--x-obs-file",
        type=Path,
        default=None,
        help=(
            "Optional .npy file containing a single observed x vector "
            "with shape (12,) or (1, 12)."
        ),
    )
    parser.add_argument(
        "--use-notebook-example-x",
        action="store_true",
        help="Use the notebook's example observation generated from rebound_simulator.",
    )
    return parser.parse_args()


def build_prior(device: str) -> sbi_utils.BoxUniform:
    return sbi_utils.BoxUniform(
        low=PRIOR_LOW.to(device),
        high=PRIOR_HIGH.to(device),
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


def load_batches(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    theta_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []

    for idx in find_batch_indices(data_dir):
        theta_path = data_dir / f"theta_batch_{idx}.npy"
        x_path = data_dir / f"x_batch_{idx}.npy"
        theta_chunk = np.load(theta_path)
        x_chunk = np.load(x_path)
        if theta_chunk.shape[0] != x_chunk.shape[0]:
            raise ValueError(
                f"Batch {idx} length mismatch: {theta_chunk.shape[0]} "
                f"theta rows vs {x_chunk.shape[0]} x rows"
            )
        theta_chunks.append(theta_chunk.astype(np.float32, copy=False))
        x_chunks.append(x_chunk.astype(np.float32, copy=False))

    theta = torch.from_numpy(np.concatenate(theta_chunks, axis=0))
    x = torch.from_numpy(np.concatenate(x_chunks, axis=0))
    return theta, x


def load_observed_x(args: argparse.Namespace) -> torch.Tensor | None:
    if args.x_obs_file is not None:
        x_obs = np.load(args.x_obs_file).astype(np.float32, copy=False)
        x_obs = np.squeeze(x_obs)
        if x_obs.shape != (12,):
            raise ValueError(f"Expected observed x to have shape (12,), got {x_obs.shape}")
        return torch.from_numpy(x_obs)

    if args.use_notebook_example_x:
        sim = simulate(1, 1, 1, 0, 0, 1, 2, 3, 0.6, 0, 0, 1, 1, 0.9, 0.3)
        return torch.tensor(summary_statistics(sim), dtype=torch.float32)

    return None


def train_with_loss_tqdm(
    inference: NPE,
    *,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    max_num_epochs: int,
) -> torch.nn.Module:
    result: dict[str, object] = {}

    def _run_training() -> None:
        try:
            result["density_estimator"] = inference.train(
                training_batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
                max_num_epochs=max_num_epochs,
                show_train_summary=False,
            )
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    last_seen = 0
    with tqdm(total=max_num_epochs, desc="NPE training", unit="epoch") as pbar:
        while thread.is_alive():
            summary = getattr(inference, "_summary", {})
            training_loss = summary.get("training_loss", [])
            validation_loss = summary.get("validation_loss", [])
            current = len(training_loss)

            if current > last_seen:
                pbar.update(current - last_seen)
                last_seen = current
                postfix: dict[str, str] = {
                    "train_loss": f"{training_loss[-1]:.6f}",
                }
                if validation_loss:
                    postfix["val_loss"] = f"{validation_loss[-1]:.6f}"
                pbar.set_postfix(postfix)

            time.sleep(0.2)

        thread.join()

        summary = getattr(inference, "_summary", {})
        training_loss = summary.get("training_loss", [])
        validation_loss = summary.get("validation_loss", [])
        current = len(training_loss)
        if current > last_seen:
            pbar.update(current - last_seen)
        if training_loss:
            postfix = {"train_loss": f"{training_loss[-1]:.6f}"}
            if validation_loss:
                postfix["val_loss"] = f"{validation_loss[-1]:.6f}"
            pbar.set_postfix(postfix)

    if "error" in result:
        raise result["error"]  # type: ignore[misc]

    return result["density_estimator"]  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    theta, x = load_batches(args.data_dir)
    theta = theta.to(args.device)
    x = x.to(args.device)
    print(f"Loaded theta with shape {tuple(theta.shape)} from {args.data_dir}")
    print(f"Loaded x with shape {tuple(x.shape)} from {args.data_dir}")

    prior = build_prior(args.device)
    inference = NPE(prior=prior, device=args.device, show_progress_bars=False)
    inference.append_simulations(theta, x)
    density_estimator = train_with_loss_tqdm(
        inference,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        max_num_epochs=args.max_num_epochs,
    )
    posterior = inference.build_posterior(density_estimator)

    torch.save(
        {
            "density_estimator_state_dict": density_estimator.state_dict(),
            "theta_shape": tuple(theta.shape),
            "x_shape": tuple(x.shape),
            "prior_low": PRIOR_LOW,
            "prior_high": PRIOR_HIGH,
        },
        args.output_dir / "npe_density_estimator.pt",
    )
    torch.save(theta, args.output_dir / "theta.pt")
    torch.save(x, args.output_dir / "x.pt")
    print(f"Saved artifacts to {args.output_dir}")

    x_obs = load_observed_x(args)
    if x_obs is None:
        return

    x_obs = x_obs.to(args.device)
    posterior = posterior.set_default_x(x_obs)
    map_estimate = posterior.map()
    samples = posterior.sample((args.num_posterior_samples,), x=x_obs)

    torch.save(x_obs, args.output_dir / "x_obs.pt")
    torch.save(map_estimate, args.output_dir / "map_estimate.pt")
    torch.save(samples, args.output_dir / "posterior_samples.pt")
    print(f"Observed x shape: {tuple(x_obs.shape)}")
    print(f"MAP estimate shape: {tuple(map_estimate.shape)}")
    print(f"Posterior samples shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
