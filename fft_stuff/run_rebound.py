from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from rebound_simulator import simulator_single

class MixedPrior(torch.distributions.Distribution):
    arg_constraints = {}
    has_rsample = True
    def __init__(self):
        # We now have exactly 6 dimensions (3 masses, 3 semi-major axes)
        super().__init__(batch_shape=torch.Size([]), event_shape=torch.Size([6]))
        
        self.log_mass_dist = torch.distributions.Uniform(
            low=torch.log(torch.tensor([1e-6, 1e-6, 1e-6])),
            high=torch.log(torch.tensor([1e-3, 1e-3, 1e-3]))
        )
        self.log_a_dist = torch.distributions.Uniform(
            low=torch.log(torch.tensor([0.5, 3.0, 6.0])),
            high=torch.log(torch.tensor([3.0, 6.0, 9.0]))
        )

    def rsample(self, sample_shape=torch.Size([])):
        masses = torch.exp(self.log_mass_dist.rsample(sample_shape))
        a_raw = torch.exp(self.log_a_dist.rsample(sample_shape))
        
        # Strictly sort the semi-major axes to break permutation symmetry
        # a_sorted, _ = torch.sort(a_raw, dim=-1)
        
        # Returns a clean 6D vector
        # return torch.cat([masses, a_sorted], dim=-1)
        return torch.cat([masses, a_raw], dim=-1)

    def log_prob(self, value):
        masses = value[..., 0:3]
        a_vals = value[..., 3:6]
        
        m_lp = self.log_mass_dist.log_prob(torch.log(masses)).sum(dim=-1)
        a_lp = self.log_a_dist.log_prob(torch.log(a_vals)).sum(dim=-1)
        
        return m_lp + a_lp

    @property
    def support(self):
        return torch.distributions.constraints.independent(
            torch.distributions.constraints.real, 1)

def main() -> None:
    # Scaled up to 5,000 simulations to ensure a robust posterior
    total_simulations = 10000 
    batch_size = 500
    n_jobs = 10

    prior = MixedPrior()
    data_dir = Path("time_series_data_dirty_case_a555_e0.05_shorttraj")
    data_dir.mkdir(exist_ok=True)

    for batch_index, start in enumerate(range(0, total_simulations, batch_size)):
        current_batch_size = min(batch_size, total_simulations - start)
        
        theta = torch.stack([prior.rsample() for _ in range(current_batch_size)])
        all_params = [theta[i].numpy() for i in range(current_batch_size)]
        
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=0)(
            delayed(simulator_single)(all_params[i])
            for i in tqdm(range(current_batch_size), desc=f"batch {batch_index}")
        )

        # list of numpy ndarray to tensor was inefficient based on the warning
        all_params_np = np.array(all_params, dtype=np.float32)
        theta_tensor = torch.from_numpy(all_params_np)
        x_tensor = torch.from_numpy(np.stack(results).astype(np.float32))
        
        np.save(data_dir / f"theta_batch_{batch_index:02d}.npy", theta_tensor.numpy())
        np.save(data_dir / f"x_batch_{batch_index:02d}.npy", x_tensor.numpy())
        
        print(f"Batch {batch_index} Saved | Theta: {theta_tensor.shape} | X: {x_tensor.shape}")

if __name__ == "__main__":
    main()