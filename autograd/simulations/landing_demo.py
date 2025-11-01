import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from autograd.drone_problems.landing_controller import LandingController
from autograd.drone_problems.simple_drone import SimpleDrone3D


def run_landing_demo(pad_position=(12, 15, 0), start_position=(10, 12, 5), Kp=0.5, max_time=30.0):
    """
    Run landing demonstration with visualization.

    Args:
        pad_position: (x, y, z) landing pad position
        start_position: (x, y, z) starting drone position
        Kp: Controller gain
        max_time: Maximum simulation time

    Returns:
        drone, controller, success info
    """
    # Initialize
    drone = SimpleDrone3D(*start_position)
    controller = LandingController(pad_position, Kp=Kp)

    dt = 0.1
    t = 0.0

    # Store trajectory for animation
    trajectory = [start_position]
    controller_states = [
        {
            "centered": False,  # Initially not centered
            "landed": False,
            "time": 0.0,
        }
    ]
    times = [0.0]

    print("Starting precision landing demo...")
    print(f"Pad at: {pad_position}")
    print(f"Start at: {start_position}")
    print("-" * 50)

    # Simulation loop
    while t < max_time and not controller.landed:
        # Update controller
        controller.set_drone_position((drone.x, drone.y, drone.z))
        vx, vy, vz = controller.update(dt)

        # Update drone
        drone.set_velocity(vx, vy, vz)
        drone.update(dt)

        # Store data for visualization
        trajectory.append((drone.x, drone.y, drone.z))
        controller_states.append({"centered": controller.centered, "landed": controller.landed, "time": t})
        times.append(t)

        # Progress logging
        if int(t * 10) % 10 == 0:  # Every second
            status = controller.get_status()
            print(
                f"t={t:5.1f}s: Centered={status['centered']}, "
                f"Alt={status['altitude']:.1f}m, "
                f"Pos=({drone.x:.1f},{drone.y:.1f},{drone.z:.1f})"
            )

        t += dt

    # Final results
    print("-" * 50)
    if controller.landed:
        final_error = np.sqrt((drone.x - pad_position[0]) ** 2 + (drone.y - pad_position[1]) ** 2)
        print("✅ SUCCESSFUL LANDING!")
        print(".1f")
        print(".1f")
        print(".3f")
        success = True
    else:
        print("❌ Landing timeout")
        success = False

    return drone, controller, trajectory, controller_states, times, success


def animate_landing(drone, controller, trajectory, controller_states, times):
    """
    Create animated visualization of the landing sequence.
    """
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)  # Top-down view
    ax2 = fig.add_subplot(122)  # Altitude vs time

    # === Setup Top-Down View ===

    # Plot landing pad
    pad_x, pad_y, pad_z = controller.pad_pos
    ax1.scatter([pad_x], [pad_y], c="red", s=300, marker="X", linewidths=3, label="Landing Pad", zorder=5)

    # Plot centering threshold
    threshold_circle = Circle(
        (pad_x, pad_y),
        controller.centering_threshold,
        fill=False,
        edgecolor="red",
        linestyle=":",
        alpha=0.5,
        linewidth=2,
        label="Centering Zone",
    )
    ax1.add_patch(threshold_circle)

    # Set axis limits
    all_x = [pos[0] for pos in trajectory] + [pad_x]
    all_y = [pos[1] for pos in trajectory] + [pad_y]
    margin = 3
    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Trajectory line (initially empty)
    (traj_line,) = ax1.plot([], [], "b-", linewidth=2.5, alpha=0.7, label="Flight Path")

    # Drone marker
    (drone_dot,) = ax1.plot(
        [], [], "bo", markersize=12, markeredgewidth=2, markeredgecolor="darkblue", label="Drone"
    )

    ax1.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax1.set_title("Precision Landing - Top View", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Status text
    status_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # === Setup Altitude Plot ===

    altitudes = [pos[2] for pos in trajectory]
    (alt_line,) = ax2.plot([], [], "g-", linewidth=2.5, label="Altitude")

    # Ground reference
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Ground")

    # Centering zone marker (when descent starts)
    centering_line = ax2.axvline(x=0, color="orange", linestyle=":", linewidth=2, alpha=0, label="Centered")

    ax2.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Altitude (m)", fontsize=12, fontweight="bold")
    ax2.set_title("Landing Sequence", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(times) if times else 1)
    ax2.set_ylim(-0.5, max(altitudes) + 1)

    # Mission metrics
    metrics_text = ax2.text(
        0.02,
        0.02,
        "",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle("Autonomous Precision Landing", fontsize=16, fontweight="bold", y=0.98)

    # === Animation ===

    total_frames = len(trajectory)
    skip_frames = max(1, total_frames // 200)  # Smooth playback

    def init():
        traj_line.set_data([], [])
        drone_dot.set_data([], [])
        alt_line.set_data([], [])
        status_text.set_text("")
        metrics_text.set_text("")
        return traj_line, drone_dot, alt_line, status_text, metrics_text

    def update(frame):
        idx = frame * skip_frames
        if idx >= total_frames:
            idx = total_frames - 1

        # Update trajectory
        traj_x = [pos[0] for pos in trajectory[: idx + 1]]
        traj_y = [pos[1] for pos in trajectory[: idx + 1]]
        traj_line.set_data(traj_x, traj_y)

        # Update drone position
        drone_x, drone_y, drone_z = trajectory[idx]
        drone_dot.set_data([drone_x], [drone_y])

        # Update altitude plot
        alt_x = times[: idx + 1]
        alt_y = [pos[2] for pos in trajectory[: idx + 1]]
        alt_line.set_data(alt_x, alt_y)

        # Update centering indicator
        if controller_states[idx]["centered"]:
            centering_time = times[idx]
            centering_line.set_xdata([centering_time, centering_time])
            centering_line.set_alpha(1)

        # Update status text
        state = controller_states[idx]
        current_time = times[idx]
        status_text.set_text(
            f"Time: {current_time:.1f}s\n"
            f"Centered: {state['centered']}\n"
            f"Landed: {state['landed']}\n"
            f"Altitude: {drone_z:.2f}m\n"
            f"Position: ({drone_x:.1f}, {drone_y:.1f})"
        )

        # Update metrics
        if idx == total_frames - 1 and controller.landed:
            final_error = np.sqrt((drone_x - pad_x) ** 2 + (drone_y - pad_y) ** 2)
            metrics_text.set_text(f"Final Error: {final_error:.3f}m\nLanding Time: {current_time:.1f}s")

        return traj_line, drone_dot, alt_line, status_text, metrics_text

    num_frames = (total_frames + skip_frames - 1) // skip_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return anim


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PRECISION LANDING DEMO")
    print("=" * 60 + "\n")

    # Run the demo
    drone, controller, trajectory, states, times, success = run_landing_demo(
        pad_position=(12, 15, 0), start_position=(10, 12, 5), Kp=0.5
    )

    if success:
        print("\n" + "=" * 60)
        print("  Creating visualization...")
        print("=" * 60 + "\n")

        animate_landing(drone, controller, trajectory, states, times)

        print("\n" + "=" * 60)
        print("  DEMO COMPLETE!")
        print("=" * 60 + "\n")
    else:
        print("Demo failed - no visualization available")
