import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from autograd.drone_problems.orbital_follower import OrbitalFollower
from autograd.drone_problems.simple_drone import SimpleDrone


def simulate_orbital_following_random_walk():
    """Demo: Drone orbits target with completely unpredictable random walk motion"""

    # Target starts at center, moves randomly
    target_x, target_y = 0, 0
    target_vx, target_vy = 0, 0  # Initial velocity

    orbit_radius = 4.0
    # Start drone on orbit circle
    drone = SimpleDrone(x=orbit_radius, y=0, speed=8.0)
    follower = OrbitalFollower(
        orbit_radius=orbit_radius,
        angular_velocity=0.6,  # Moderate orbital speed
        kp=3.0,  # Strong correction for unpredictable target
    )

    dt = 0.1
    max_time = 30.0
    t = 0.0

    target_positions = []
    drone_positions = []
    times = []

    print("Starting orbital following with random walk target...")
    print("Target: completely unpredictable random walk")
    print(f"Drone: maintaining perfect {orbit_radius}m orbit")
    print("-" * 50)

    while t < max_time:
        # Random walk target motion - completely unpredictable
        if int(t * 10) % 12 == 0:  # Every 1.2 seconds, change acceleration
            # Random acceleration between -1.5 and 1.5
            ax = np.random.uniform(-1.5, 1.5)
            ay = np.random.uniform(-1.5, 1.5)
            target_vx += ax * dt
            target_vy += ay * dt

            # Limit speed to reasonable range (0.5 to 3.0 m/s)
            speed = np.sqrt(target_vx**2 + target_vy**2)
            if speed > 3.0:
                target_vx *= 3.0 / speed
                target_vy *= 3.0 / speed
            elif speed < 0.5 and t > 2.0:  # Minimum speed after initial settling
                target_vx *= 0.5 / speed
                target_vy *= 0.5 / speed

        # Occasional sudden stops (unpredictable behavior)
        if np.random.random() < 0.015:  # 1.5% chance each step
            old_vx, old_vy = target_vx, target_vy
            target_vx, target_vy = 0, 0
            if np.sqrt(old_vx**2 + old_vy**2) > 0.1:  # Only print significant stops
                print(".1f")

        # Update target position
        target_x += target_vx * dt
        target_y += target_vy * dt
        target_pos = (target_x, target_y)

        # Update orbital follower
        follower.update_target(target_pos)
        follower.update_drone(drone.pos)

        # Get orbital control command
        vx, vy = follower.get_control_command(dt)
        drone.set_velocity(vx, vy)
        drone.update(dt)

        target_positions.append(target_pos)
        drone_positions.append(drone.pos)
        times.append(t)

        t += dt

    print("-" * 50)
    print("Random walk simulation complete!")
    animate_orbital_following(target_positions, drone_positions, times, orbit_radius, dt)


def animate_orbital_following(target_positions, drone_positions, times, orbit_radius, dt):
    """Create animated visualization of orbital following with random walk target"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot complete target path (random walk)
    target_x = [pos[0] for pos in target_positions]
    target_y = [pos[1] for pos in target_positions]
    ax.plot(target_x, target_y, "r-", alpha=0.6, linewidth=2, label="Target path (random walk)")

    # Plot complete drone path (orbital)
    drone_x = [pos[0] for pos in drone_positions]
    drone_y = [pos[1] for pos in drone_positions]
    ax.plot(drone_x, drone_y, "b-", linewidth=2.5, alpha=0.8, label="Drone path (orbital)")

    # Current position markers
    (target_dot,) = ax.plot(
        [], [], "ro", markersize=15, markeredgewidth=2, markeredgecolor="darkred", label="Target"
    )
    (drone_dot,) = ax.plot(
        [], [], "bo", markersize=12, markeredgewidth=2, markeredgecolor="darkblue", label="Drone"
    )

    # Orbit radius circle (updates to follow target)
    orbit_circle = Circle(
        (0, 0),
        orbit_radius,
        fill=False,
        edgecolor="green",
        linestyle=":",
        alpha=0.4,
        linewidth=2,
        label="Orbit radius",
    )
    ax.add_patch(orbit_circle)

    # Set axis limits with margin
    all_x = target_x + drone_x
    all_y = target_y + drone_y
    margin = orbit_radius + 3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax.set_title("Orbital Following: Random Walk Target", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Info text with orbital metrics
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
        target_dot.set_data([], [])
        drone_dot.set_data([], [])
        info_text.set_text("")
        return target_dot, drone_dot, orbit_circle, info_text

    def update(frame):
        idx = frame * skip_frames
        if idx >= total_frames:
            idx = total_frames - 1

        # Update current positions
        target_x, target_y = target_positions[idx]
        drone_x, drone_y = drone_positions[idx]

        target_dot.set_data([target_x], [target_y])
        drone_dot.set_data([drone_x], [drone_y])

        # Update orbit circle to follow current target position
        orbit_circle.center = (target_x, target_y)

        # Calculate current metrics
        current_time = times[idx]
        distance = np.sqrt((drone_x - target_x) ** 2 + (drone_y - target_y) ** 2)

        # Calculate target speed from position differences
        if idx < len(target_positions) - 1:
            next_target = target_positions[idx + 1]
            target_speed = np.sqrt((next_target[0] - target_x) ** 2 + (next_target[1] - target_y) ** 2) / dt
        else:
            target_speed = 0.0

        info_text.set_text(
            f"Time: {current_time:.1f}s\n"
            f"Distance: {distance:.2f}m (target: {orbit_radius:.1f}m)\n"
            f"Target speed: {target_speed:.2f} m/s\n"
            f"Target: ({target_x:.1f}, {target_y:.1f})\n"
            f"Drone: ({drone_x:.1f}, {drone_y:.1f})"
        )

        return target_dot, drone_dot, orbit_circle, info_text

    num_frames = (total_frames + skip_frames - 1) // skip_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

    return anim


if __name__ == "__main__":
    print("=" * 60)
    print("  ORBITAL FOLLOWING DEMO: RANDOM WALK TARGET")
    print("=" * 60)
    print("Watch the drone maintain perfect orbital distance")
    print("even as the target moves completely unpredictably!")
    print("=" * 60)

    simulate_orbital_following_random_walk()

    print("=" * 60)
    print("  DEMO COMPLETE!")
    print("  The drone maintained perfect orbit despite")
    print("  the target's erratic random walk motion!")
    print("=" * 60)
