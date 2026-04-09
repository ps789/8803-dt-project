from __future__ import annotations

import numpy as np
import rebound
import torch


def simulate(
    m1,
    m2,
    m3,
    x1,
    vx1,
    x2,
    vx2,
    x3,
    vx3,
    y1,
    vy1,
    y2,
    vy2,
    y3,
    vy3,
    t_end: float = 200.0,
):
    sim = rebound.Simulation()
    sim.add(m=m1, x=x1, vx=vx1, y=y1, vy=vy1)
    sim.add(m=m2, x=x2, vx=vx2, y=y2, vy=vy2)
    sim.add(m=m3, x=x3, vx=vx3, y=y3, vy=vy3)
    sim.integrator = "ias15"
    sim.integrate(t_end)
    return sim


def summary_statistics(sim) -> np.ndarray:
    particles = sim.particles
    x1, y1 = particles[0].x, particles[0].y
    x2, y2 = particles[1].x, particles[1].y
    x3, y3 = particles[2].x, particles[2].y
    vx1, vy1 = particles[0].vx, particles[0].vy
    vx2, vy2 = particles[1].vx, particles[1].vy
    vx3, vy3 = particles[2].vx, particles[2].vy
    return np.array([x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3])


def simulator_single(params: torch.Tensor) -> np.ndarray:
    if isinstance(params, torch.Tensor):
        params = params.numpy()
    m1, m2, m3, x1, vx1, x2, vx2, x3, vx3, y1, vy1, y2, vy2, y3, vy3 = params
    sim = simulate(
        m1, m2, m3, x1, vx1, x2, vx2, x3, vx3, y1, vy1, y2, vy2, y3, vy3
    )
    stats = summary_statistics(sim)
    del sim
    return stats


def simulator_for_sbi(params: torch.Tensor) -> torch.Tensor:
    if params.ndim == 1:
        return torch.tensor(simulator_single(params), dtype=torch.float32)
    return torch.stack(
        [torch.tensor(simulator_single(p), dtype=torch.float32) for p in params]
    )
