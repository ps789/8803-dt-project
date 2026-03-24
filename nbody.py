from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


G = 1.0  # Gravitational constant (arbitrary units)

@dataclass
class Body:
    """A point mass with 3D position and velocity vectors."""

    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        """Normalize position and velocity inputs to float NumPy arrays."""
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    def copy(self) -> "Body":
        """Return a deep copy of the body state."""
        return Body(
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )


def compute_accelerations(bodies: Sequence[Body], softening: float = 1e-9) -> List[np.ndarray]:
    """Compute the gravitational acceleration on each body from all others."""
    accelerations: List[np.ndarray] = [np.zeros(3, dtype=float) for _ in bodies]

    for i, body_i in enumerate(bodies):
        for j, body_j in enumerate(bodies):
            if i == j:
                continue

            displacement = body_j.position - body_i.position
            distance = np.linalg.norm(displacement)
            scale = G * body_j.mass / (distance ** 3)
            accelerations[i] = accelerations[i] + displacement * scale
    return accelerations


def propagate(
    bodies: Iterable[Body],
    dt: float,
    steps: int = 1,
    softening: float = 1e-9,
) -> List[Body]:
    """Advance the system forward in time using explicit Euler integration."""
    state = [body.copy() for body in bodies]

    for _ in range(steps):
        accelerations = compute_accelerations(state, softening=softening)

        for body, acceleration in zip(state, accelerations):
            body.velocity = body.velocity + acceleration * dt
            body.position = body.position + body.velocity * dt

    return state


def propagate_trajectory(
    bodies: Iterable[Body],
    dt: float,
    steps: int,
    softening: float = 1e-9,
) -> List[List[Body]]:
    """Return the full time history of the system over the requested steps."""
    state = [body.copy() for body in bodies]
    trajectory: List[List[Body]] = [[body.copy() for body in state]]

    for _ in range(steps):
        state = propagate(state, dt=dt, steps=1, softening=softening)
        trajectory.append([body.copy() for body in state])

    return trajectory


def create_example_bodies() -> List[Body]:
    """Create three toy-scale bodies with visibly changing trajectories."""
    return [
        Body(
            mass=1,
            position=[-1, 0, 0],
            velocity=[0.347111,
                0.532728, 0],
        ),
        Body(
            mass=1,
            position=[1, 0, 0],
            velocity=[0.347111,
                0.532728, 0],
        ),
        Body(
            mass=1,
            position=[0, 0, 0],
            velocity=[-0.694222,
                -1.065456, 0],
        ),
    ]


def _format_body(index: int, body: Body) -> str:
    """Format one body state for console output."""
    return (
        f"Body {index}: mass={body.mass:.3e}, "
        f"position={body.position.tolist()}, velocity={body.velocity.tolist()}"
    )


def animate_trajectory(trajectory: Sequence[Sequence[Body]], dt: float) -> FuncAnimation:
    """Build a matplotlib 3D animation for a simulated trajectory."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    line_artists = []
    point_artists = []

    all_positions = [
        body.position
        for frame in trajectory
        for body in frame
    ]
    x_values = [position[0] for position in all_positions]
    y_values = [position[1] for position in all_positions]
    z_values = [position[2] for position in all_positions]

    def _axis_limits(values: Sequence[float]) -> tuple[float, float]:
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            padding = max(1.0, abs(minimum) * 0.1 + 1.0)
        else:
            padding = (maximum - minimum) * 0.1
        return minimum - padding, maximum + padding

    ax.set_xlim(_axis_limits(x_values))
    ax.set_ylim(_axis_limits(y_values))
    ax.set_zlim(_axis_limits(z_values))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    for index in range(len(trajectory[0])):
        color = colors[index % len(colors)]
        line, = ax.plot([], [], [], color=color, linewidth=1.5)
        point, = ax.plot([], [], [], marker="o", color=color, markersize=8)
        line_artists.append(line)
        point_artists.append(point)

    title = ax.set_title("N-body simulation")

    def update(frame_index: int):
        frame = trajectory[frame_index]

        for body_index, body in enumerate(frame):
            history = np.array(
                [step[body_index].position for step in trajectory[: frame_index + 1]],
                dtype=float,
            )
            x_history = history[:, 0]
            y_history = history[:, 1]
            z_history = history[:, 2]

            line_artists[body_index].set_data(x_history, y_history)
            line_artists[body_index].set_3d_properties(z_history)
            point_artists[body_index].set_data([body.position[0]], [body.position[1]])
            point_artists[body_index].set_3d_properties([body.position[2]])

        title.set_text(f"N-body simulation at t = {frame_index * dt:.1f} s")
        return [*line_artists, *point_artists, title]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=50,
        blit=False,
        repeat=True,
    )
    fig._nbody_animation = animation
    return animation


def save_animation(animation: FuncAnimation, output_path: str = "nbody.gif", fps: int = 20) -> None:
    """Save an animation to disk as a GIF using matplotlib's Pillow writer."""
    writer = PillowWriter(fps=fps)
    animation.save(output_path, writer=writer)


def main() -> None:
    """Run the example simulation, save the animation, and show the animation."""
    bodies = create_example_bodies()
    dt = 0.02
    steps = 300
    output_path = "nbody.gif"

    print("Initial state:")
    for index, body in enumerate(bodies, start=1):
        print(_format_body(index, body))

    trajectory = propagate_trajectory(bodies, dt=dt, steps=steps)
    final_state = trajectory[-1]

    print(f"\nState after {steps} steps with dt={dt} seconds:")
    for index, body in enumerate(final_state, start=1):
        print(_format_body(index, body))

    animation = animate_trajectory(trajectory, dt=dt)
    save_animation(animation, output_path=output_path)
    print(f"\nSaved animation to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
