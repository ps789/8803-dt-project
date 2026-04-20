from pathlib import Path
from tqdm import tqdm

import numpy as np

def extract_power_spectrum(obs_data_raw):
    obs_reshaped = obs_data_raw.reshape(3, 3, -1)
    
    # if 100 timesteps, we get 51 bins
    fft_coeffs = np.fft.rfft(obs_reshaped, axis=2)
    power_spectrum = np.abs(fft_coeffs) ** 2
    
    # remove the first bin to get the wobbly bins
    wobble_power = power_spectrum[:, :, 1:] 
    log_power = np.log10(wobble_power + 1e-12)
    
    return log_power.flatten()

data_dir = Path("./time_series_data_dirty_case_a555_e0.05_shorttraj/")
# data_dir = Path("./time_series_data_dirty_case_a369_e0.05_shorttraj/")
# data_dir = Path("./time_series_data_dirty_case_e0.05_shorttraj/")
# data_dir = Path("./time_series_data_dirty_case_e0.05/")
n_batches = 20 # adjust to your number of saved batches

all_x_fft = []

print("Extracting FFT Power Spectrums...")

for i in tqdm(range(n_batches)):
    x_raw_batch = np.load(data_dir / f"x_batch_{i:02d}.npy")
    N_sims = x_raw_batch.shape[0]
    
    x_fft_batch = np.zeros((N_sims, 9000))
    # x_fft_batch = np.zeros((N_sims, 450))
    for j in range(N_sims):
        x_fft_batch[j] = extract_power_spectrum(x_raw_batch[j])
        
    np.save(data_dir / f"x_batch_fft_{i:02d}.npy", x_fft_batch)

print(f"Success!")