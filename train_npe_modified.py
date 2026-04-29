from __future__ import annotations

import argparse
from copy import deepcopy
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from torch.utils import data
from tqdm.auto import tqdm

from rebound_simulator import simulate, summary_statistics


PRIOR_LOW = torch.tensor(
    [0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0],
    dtype=torch.float32,
)
PRIOR_HIGH = torch.tensor(
    [5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
    dtype=torch.float32,
)

MASS_DIM = 3
TRAIN_BATCH_INDICES = tuple(f"{i:02d}" for i in range(9))
VALIDATION_BATCH_INDEX = "09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an sbi NPE model to infer theta from x batches."
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
        default=Path(__file__).resolve().parent / "artifacts" / "npe_1time",
        help="Directory to write trained artifacts.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
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
        help="Unused with the fixed split; batch 09 is always the validation set.",
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=int,
        default=50,
        help="Early-stop after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=500,
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
            "with shape (15,) or (1, 15)."
        ),
    )
    parser.add_argument(
        "--use-notebook-example-x",
        action="store_true",
        help="Use the notebook's example observation generated from rebound_simulator.",
    )
    parser.add_argument(
        "--use-alternative-network",
        action="store_true",
        help="Use a larger custom posterior network instead of sbi's default estimator.",
    )
    return parser.parse_args()


def build_prior(device: str) -> sbi_utils.BoxUniform:
    return sbi_utils.BoxUniform(
        low=PRIOR_LOW.to(device),
        high=PRIOR_HIGH.to(device),
    )


def build_density_estimator(args: argparse.Namespace):
    if not args.use_alternative_network:
        return "maf"

    return posterior_nn(
        model="zuko_nsf",
        hidden_features=128,
        num_transforms=8,
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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    theta = torch.from_numpy(np.concatenate(theta_chunks, axis=0))
    x = torch.from_numpy(np.concatenate(x_chunks, axis=0))
    return theta, x


def load_train_validation_batches(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_train, x_train = load_selected_batches(data_dir, TRAIN_BATCH_INDICES)
    theta_val, x_val = load_selected_batches(data_dir, (VALIDATION_BATCH_INDEX,))
    return theta_train, x_train, theta_val, x_val


def load_observed_x(args: argparse.Namespace) -> torch.Tensor | None:
    if args.x_obs_file is not None:
        x_obs = np.load(args.x_obs_file).astype(np.float32, copy=False)
        x_obs = np.squeeze(x_obs)
        if x_obs.shape != (15,):
            raise ValueError(f"Expected observed x to have shape (15,), got {x_obs.shape}")
        return torch.from_numpy(x_obs)

    if args.use_notebook_example_x:
        masses = np.array([1, 1, 1], dtype=np.float32)
        sim = simulate(1, 1, 1, 0, 0, 1, 2, 3, 0.6, 0, 0, 1, 1, 0.9, 0.3)
        stats = summary_statistics(sim).astype(np.float32, copy=False)
        return torch.from_numpy(np.concatenate([masses, stats]))

    return None


def train_with_loss_tqdm(
    inference: NPE,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_val: torch.Tensor,
    *,
    batch_size: int,
    learning_rate: float,
    stop_after_epochs: int,
    max_num_epochs: int,
) -> torch.nn.Module:
    result: dict[str, object] = {}

    def _run_training() -> None:
        try:
            density_estimator = inference._build_neural_net(theta_train, x_train)
            density_estimator = density_estimator.to(theta_train.device)

            train_dataset = data.TensorDataset(theta_train, x_train)
            val_dataset = data.TensorDataset(theta_val, x_val)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=min(batch_size, len(train_dataset)),
                shuffle=True,
                drop_last=False,
            )
            val_loader = data.DataLoader(
                val_dataset,
                batch_size=min(batch_size, len(val_dataset)),
                shuffle=False,
                drop_last=False,
            )

            optimizer = torch.optim.Adam(
                list(density_estimator.parameters()), lr=learning_rate
            )
            best_val_loss = float("inf")
            best_model_state_dict = deepcopy(density_estimator.state_dict())
            epochs_since_last_improvement = 0
            summary = {
                "training_loss": [],
                "validation_loss": [],
            }
            inference._summary = summary

            for epoch in range(max_num_epochs):
                density_estimator.train()
                train_loss_sum = 0.0
                train_count = 0
                for theta_batch, x_batch in train_loader:
                    optimizer.zero_grad()
                    train_losses = density_estimator.loss(theta_batch, x_batch)
                    train_loss = torch.mean(train_losses)
                    train_loss.backward()
                    optimizer.step()

                    train_loss_sum += train_losses.sum().item()
                    train_count += theta_batch.shape[0]

                train_loss_average = train_loss_sum / max(train_count, 1)

                density_estimator.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for theta_batch, x_batch in val_loader:
                        val_losses = density_estimator.loss(theta_batch, x_batch)
                        val_loss_sum += val_losses.sum().item()
                        val_count += theta_batch.shape[0]

                val_loss_average = val_loss_sum / max(val_count, 1)
                summary["training_loss"].append(train_loss_average)
                summary["validation_loss"].append(val_loss_average)

                if val_loss_average < best_val_loss:
                    best_val_loss = val_loss_average
                    best_model_state_dict = deepcopy(density_estimator.state_dict())
                    epochs_since_last_improvement = 0
                else:
                    epochs_since_last_improvement += 1
                    if epochs_since_last_improvement >= stop_after_epochs:
                        break

            density_estimator.load_state_dict(best_model_state_dict)
            result["density_estimator"] = density_estimator
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

    theta_train, x_train, theta_val, x_val = load_train_validation_batches(args.data_dir)
    theta_train = theta_train.to(args.device)
    x_train = x_train.to(args.device)
    theta_val = theta_val.to(args.device)
    x_val = x_val.to(args.device)
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

    prior = build_prior(args.device)
    density_estimator = build_density_estimator(args)
    inference = NPE(
        prior=prior,
        density_estimator=density_estimator,
        device=args.device,
        show_progress_bars=False,
    )
    inference.append_simulations(theta_train, x_train)
    density_estimator = train_with_loss_tqdm(
        inference,
        theta_train,
        x_train,
        theta_val,
        x_val,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
    )
    posterior = inference.build_posterior(density_estimator)

    torch.save(
        {
            "density_estimator_state_dict": density_estimator.state_dict(),
            "theta_shape": tuple(theta_train.shape),
            "x_shape": tuple(x_train.shape),
            "validation_theta_shape": tuple(theta_val.shape),
            "validation_x_shape": tuple(x_val.shape),
            "prior_low": PRIOR_LOW,
            "prior_high": PRIOR_HIGH,
        },
        args.output_dir / "npe_density_estimator.pt",
    )
    torch.save(theta_train, args.output_dir / "theta.pt")
    torch.save(x_train, args.output_dir / "x.pt")
    torch.save(theta_val, args.output_dir / "theta_val.pt")
    torch.save(x_val, args.output_dir / "x_val.pt")
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
