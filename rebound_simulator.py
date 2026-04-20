from __future__ import annotations

from multiprocessing import get_context

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
    t_end: float = 1.0,
):
    sim = rebound.Simulation()
    sim.add(m=m1, x=x1, vx=vx1, y=y1, vy=vy1)
    sim.add(m=m2, x=x2, vx=vx2, y=y2, vy=vy2)
    sim.add(m=m3, x=x3, vx=vx3, y=y3, vy=vy3)
    sim.integrator = "ias15"
    sim.integrate(t_end)
    return sim


def simulate_orbit(m1, m2, m3, a1, a2, a3, e1, e2, e3, t_end=100000.):
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=1)
    sim.add(m=m1, a=a1, e=e1)
    sim.add(m=m2, a=a2, e=e2)
    sim.add(m=m3, a=a3, e=e3)
    sim.integrator = "whfast"
    min_period = min(a1, a2, a3) ** 1.5
    sim.dt = 0.05 * min_period
    sim.integrate(t_end)
    return sim

def summary_statistics_orbit(sim):
    particles = sim.particles
    a1, e1, m1= particles[1].a, particles[1].e, particles[1].m
    a2, e2, m2= particles[2].a, particles[2].e, particles[2].m
    a3, e3, m3= particles[3].a, particles[3].e, particles[3].m
    return np.array([m1, m2, m3, a1, a2, a3, e1, e2, e3])


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

def simulator_single_orbit(params: torch.Tensor) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = params.numpy()
    m1, m2, m3, a1, a2, a3, e1, e2, e3 = params
    sim = simulate_orbit(m1, m2, m3, a1, a2, a3, e1, e2, e3)
    stats = summary_statistics_orbit(sim)
    del sim
    return stats

def simulator_for_sbi_orbit(params: torch.Tensor) -> torch.Tensor:
    if params.ndim == 1:
        return torch.tensor(simulator_single_orbit(params), dtype=torch.float32)
    else:
        return torch.stack([torch.tensor(simulator_single_orbit(p), dtype=torch.float32) for p in params])

def _simulator_single_worker(params, queue) -> None:
    try:
        queue.put(("ok", simulator_single(params)))
    except Exception as exc:  # pragma: no cover - defensive worker wrapper
        queue.put(("error", repr(exc)))


def _simulator_single_worker_orbit(params, queue) -> None:
    try:
        queue.put(("ok", simulator_single_orbit(params)))
    except Exception as exc:  # pragma: no cover - defensive worker wrapper
        queue.put(("error", repr(exc)))

def simulator_single_with_timeout(
    params: torch.Tensor, timeout_seconds: int
) -> np.ndarray | None:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_simulator_single_worker, args=(params, queue))
    process.start()
    try:
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return None
        if queue.empty():
            return None
        status, payload = queue.get()
        if status == "ok":
            return payload
        return None
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()


def simulator_single_with_timeout_orbit(
    params: torch.Tensor, timeout_seconds: int
) -> np.ndarray | None:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_simulator_single_worker_orbit, args=(params, queue))
    process.start()
    try:
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return None
        if queue.empty():
            return None
        status, payload = queue.get()
        if status == "ok":
            return payload
        return None
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()

def simulator_for_sbi(params: torch.Tensor) -> torch.Tensor:
    if params.ndim == 1:
        return torch.tensor(simulator_single(params), dtype=torch.float32)
    return torch.stack(
        [torch.tensor(simulator_single(p), dtype=torch.float32) for p in params]
    )
