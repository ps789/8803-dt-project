from __future__ import annotations

from pathlib import Path

import numpy as np
import rebound
import torch
from joblib import Parallel, delayed
from sbi import utils as sbi_utils
from tqdm import tqdm

from rebound_simulator import (
    simulator_single_with_timeout,
)


def main() -> None:
    total_simulations = 100000
    batch_size = 10000
    n_jobs = 20
    timeout_seconds = 10

    # Notebook cell 4
    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")
    sim.add(m=1)
    sim.add(m=2e-3, a=1.0)
    sim.add(m=1e-2, a=2.0, e=0.1)
    sim.add(m=1e-3, a=1.5, e=0.2)
    sim.integrate(10000.0)

    # Notebook cell 5
    for particle in sim.particles:
        print(particle.x, particle.y, particle.z)
    for orbit in sim.orbits():
        print(orbit)

    # Notebook cell 7
    prior = sbi_utils.BoxUniform(
        low=torch.tensor(
            [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0]
        ),
        high=torch.tensor([10, 10, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10]),
    )

    # Notebook cells 9-10
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for batch_index, start in enumerate(range(0, total_simulations, batch_size)):
        current_batch_size = min(batch_size, total_simulations - start)
        all_params = prior.sample((current_batch_size,)).numpy()
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(simulator_single_with_timeout)(all_params[i], timeout_seconds)
            for i in tqdm(range(current_batch_size), desc=f"batch {batch_index}")
        )

        valid_indices = [i for i, result in enumerate(results) if result is not None]
        skipped = current_batch_size - len(valid_indices)
        if not valid_indices:
            print(f"Skipped batch {batch_index:02d}: all simulations exceeded timeout")
            continue

        theta = torch.tensor(all_params[valid_indices], dtype=torch.float32)
        x = torch.from_numpy(
            np.stack([results[i] for i in valid_indices]).astype(np.float32)
        )
        np.save(data_dir / f"theta_batch_{(batch_index):02d}.npy", theta, allow_pickle=True)
        np.save(data_dir / f"x_batch_{(batch_index):02d}.npy", x, allow_pickle=True)
        print(
            f"Saved batch {(batch_index):02d}: "
            f"{len(valid_indices)} simulations to {data_dir}"
        )
        if skipped:
            print(
                f"Skipped {skipped} timed-out simulations in batch {(batch_index):02d}"
            )


if __name__ == "__main__":
    main()
