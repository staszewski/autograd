import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from autograd.simulations.object_following_demo import ObjectFollower
from autograd.drone_problems.simple_drone import SimpleDrone


def simulate_object_following():
    """Simulate drone following a moving target with visualization"""

    target_x, target_y = 10, 10
    target_vx, target_vy = 2, 0
    following_distance = 4.0

    drone = SimpleDrone(x=5, y=10, speed=5.0)
    follower = ObjectFollower(following_distance=following_distance, kp=0.5)

    dt = 0.1
    max_time = 15.0
    t = 0.0

    target_positions = []
    drone_positions = []
    times = []

    print("Starting object following simulation...")
    print("Target: moving right at 2 m/s")
    print(f"Drone: following {following_distance}m behind")

    while t < max_time:
        target_x += target_vx * dt
        target_y += target_vy * dt
        target_pos = (target_x, target_y)
        target_vel = (target_vx, target_vy)

        follower.update_target(target_pos, target_vel)

        follower.update_drone(drone.pos)

        vx, vy = follower.get_control_command()
        drone.set_velocity(vx, vy)
        drone.update(dt)

        target_positions.append(target_pos)
        drone_positions.append(drone.pos)
        times.append(t)

        t += dt

    print("Following simulation complete!")
    animate_following(drone_positions, target_positions, times)

    return drone, target_positions, drone_positions


def animate_following(drone_positions, target_positions, times):
    """Animate the following behavior"""

    fig, ax = plt.subplots(figsize=(12, 8))

    target_x = [pos[0] for pos in target_positions]
    target_y = [pos[1] for pos in target_positions]
    ax.plot(target_x, target_y, "r--", alpha=0.5, linewidth=2, label="Target path")

    (drone_line,) = ax.plot([], [], "b-", linewidth=2.5, label="Drone path")
    (target_dot,) = ax.plot([], [], "ro", markersize=12, label="Target")
    (drone_dot,) = ax.plot([], [], "bo", markersize=12, label="Drone")

    following_circle = plt.Circle(
        (0, 0),
        4.0,
        fill=False,
        color="green",
        linestyle=":",
        alpha=0.3,
        linewidth=2,
        label="Following distance",
    )
    ax.add_patch(following_circle)

    all_x = target_x + [pos[0] for pos in drone_positions]
    all_y = target_y + [pos[1] for pos in drone_positions]
    margin = 3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax.set_title("Object Following: Drone Tracking Moving Target", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    total_frames = len(drone_positions)
    skip_frames = max(1, total_frames // 200)

    def init():
        drone_line.set_data([], [])
        target_dot.set_data([], [])
        drone_dot.set_data([], [])
        info_text.set_text("")
        return drone_line, target_dot, drone_dot, following_circle, info_text

    def update(frame):
        idx = frame * skip_frames
        if idx >= total_frames:
            idx = total_frames - 1

        drone_x = [pos[0] for pos in drone_positions[: idx + 1]]
        drone_y = [pos[1] for pos in drone_positions[: idx + 1]]
        drone_line.set_data(drone_x, drone_y)

        target_x, target_y = target_positions[idx]
        drone_x, drone_y = drone_positions[idx]

        target_dot.set_data([target_x], [target_y])
        drone_dot.set_data([drone_x], [drone_y])

        following_circle.center = (target_x, target_y)

        current_time = times[idx]
        distance = ((drone_x - target_x) ** 2 + (drone_y - target_y) ** 2) ** 0.5
        info_text.set_text(
            f"Time: {current_time:.1f}s\n"
            f"Distance: {distance:.2f}m\n"
            f"Target: ({target_x:.1f}, {target_y:.1f})\n"
            f"Drone: ({drone_x:.1f}, {drone_y:.1f})"
        )

        return drone_line, target_dot, drone_dot, following_circle, info_text

    num_frames = (total_frames + skip_frames - 1) // skip_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim


if __name__ == "__main__":
    simulate_object_following()
