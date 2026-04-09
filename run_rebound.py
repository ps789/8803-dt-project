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

    # Notebook cell 9
    all_params = prior.sample((100000,)).numpy()
    results = Parallel(n_jobs=10, backend="multiprocessing", verbose=10)(
        delayed(simulator_single)(all_params[i]) for i in tqdm(range(100000))
    )

    theta = torch.tensor(all_params, dtype=torch.float32)
    x = torch.from_numpy(np.stack(results).astype(np.float32))

    # Notebook cell 10
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "theta.npy", theta, allow_pickle=True)
    np.save(data_dir / "x.npy", x, allow_pickle=True)



if __name__ == "__main__":
    main()
