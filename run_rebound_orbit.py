from __future__ import annotations

from pathlib import Path

import numpy as np
import rebound
import torch
from joblib import Parallel, delayed
from sbi import utils as sbi_utils
from tqdm import tqdm

from rebound_simulator import (
    simulator_single_with_timeout_orbit,
)


def main() -> None:
    total_simulations = 100000
    batch_size = 10000
    n_jobs = 20
    timeout_seconds = 100


    class MixedPrior2(torch.distributions.Distribution):
        arg_constraints = {}
        has_rsample = True

        def __init__(self):
            super().__init__(batch_shape=torch.Size([]), event_shape=torch.Size([6]))
            self.log_a_dist = torch.distributions.Independent(
                torch.distributions.Uniform(
                    low=torch.log(torch.tensor([0.5, 0.5, 0.5])),
                    high=torch.log(torch.tensor([5.0, 10.0, 15.0]))
                ), 1)
            self.ecc_dist = torch.distributions.Independent(
                torch.distributions.Uniform(
                    low=torch.tensor([0.0, 0.0, 0.0]),
                    high=torch.tensor([0.3, 0.3, 0.3])
                ), 1)

        def rsample(self, sample_shape=torch.Size([])):
            a   = torch.exp(self.log_a_dist.rsample(sample_shape))
            ecc = self.ecc_dist.rsample(sample_shape)
            return torch.cat([a, ecc], dim=-1)

        def log_prob(self, x):
            a   = x[..., :3]
            ecc = x[..., 3:]
            a_lp   = self.log_a_dist.log_prob(torch.log(a)) - torch.log(a).sum(-1)
            ecc_lp = self.ecc_dist.log_prob(ecc)
            return a_lp + ecc_lp

        @property
        def support(self):
            return torch.distributions.constraints.independent(
                torch.distributions.constraints.real, 1)
    prior = MixedPrior2()

    # Notebook cells 9-10
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for batch_index, start in enumerate(range(0, total_simulations, batch_size)):
        current_batch_size = min(batch_size, total_simulations - start)
        all_params = prior.sample((current_batch_size,)).numpy()
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(simulator_single_with_timeout_orbit)(all_params[i], timeout_seconds)
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
