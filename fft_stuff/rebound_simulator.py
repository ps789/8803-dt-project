import numpy as np
import rebound

def simulate_timeseries(
    m1, m2, m3, 
    a1, a2, a3, 
    t_end=1000., num_steps=2000
):
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    
    # central star; sun
    sim.add(m=1.0) 
    
    # hardcoded the "noise" variables to perfect circles and flat planes
    # testing setting; we can now play and change them to more noisy cases
    e_fixed = 0.05
    inc_fixed = 0.0
    angle_fixed = 0.0
    
    sim.add(m=m1, a=a1, e=e_fixed, inc=inc_fixed, Omega=angle_fixed, omega=angle_fixed, f=angle_fixed)
    sim.add(m=m2, a=a2, e=e_fixed, inc=inc_fixed, Omega=angle_fixed, omega=angle_fixed, f=angle_fixed)
    sim.add(m=m3, a=a3, e=e_fixed, inc=inc_fixed, Omega=angle_fixed, omega=angle_fixed, f=angle_fixed)
    
    sim.integrator = "ias15"
    min_period = min(a1, a2, a3) ** 1.5
    sim.dt = 0.05 * min_period
    
    times = np.linspace(0, t_end, num_steps)
    obs_data = np.zeros((3, 3, num_steps))
    
    for i, t in enumerate(times):
        sim.integrate(t)
        for p_idx in range(1, 4): 
            obs_data[p_idx-1, 0, i] = sim.particles[p_idx].a
            obs_data[p_idx-1, 1, i] = sim.particles[p_idx].e
            obs_data[p_idx-1, 2, i] = sim.particles[p_idx].inc
            
    return obs_data.flatten()

def simulator_single(params: np.ndarray) -> np.ndarray:
    # unpack exactly 6 parameters (m1, m2, m3, a1, a2, a3)
    return simulate_timeseries(*params, t_end=1000., num_steps=2000)