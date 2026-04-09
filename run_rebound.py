from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rebound
import torch
from joblib import Parallel, delayed
from sbi import utils as sbi_utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm import tqdm

from rebound_simulator import (
    simulator_for_sbi,
    simulator_single,
    simulate,
    summary_statistics,
)


def main() -> None:
    total_simulations = 100000
    batch_size = 10000
    n_jobs = 10

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
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(simulator_single)(all_params[i])
            for i in tqdm(range(current_batch_size), desc=f"batch {batch_index}")
        )

        theta = torch.tensor(all_params, dtype=torch.float32)
        x = torch.from_numpy(np.stack(results).astype(np.float32))
        np.save(data_dir / f"theta_batch_{batch_index:02d}.npy", theta, allow_pickle=True)
        np.save(data_dir / f"x_batch_{batch_index:02d}.npy", x, allow_pickle=True)
        print(
            f"Saved batch {batch_index:02d}: "
            f"{current_batch_size} simulations to {data_dir}"
        )


if __name__ == "__main__":
    main()
