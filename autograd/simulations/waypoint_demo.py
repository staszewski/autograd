import numpy as np
from autograd.drone_problems.simple_drone import SimpleDrone
from autograd.drone_problems.pursuit import pure_pursuit
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def calculate_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        pos1: Tuple (x, y)
        pos2: Tuple (x, y)

    Returns:
        float: Distance between points
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx**2 + dy**2)


class MissionState(IntEnum):
    """Mission states for the waypoint navigator"""

    IDLE = 0
    FLYING_TO_WAYPOINT = 1
    RETURNING_HOME = 2
    LANDING = 3
    MISSION_COMPLETE = 4


class WaypointNavigator:
    """
    Manages waypoint navigation mission with state machine.

    Responsibilities:
    - Track current mission state
    - Determine which waypoint to fly to
    - Detect waypoint arrival
    - Calculate path metrics
    """

    def __init__(self, waypoints, home=(0, 0), threshold: float = 2.0):
        """
        Args:
            waypoints: List of (x, y) tuples
            home: Home position (x, y)
            threshold: Distance threshold for "reached" (meters)
        """
        self.waypoints = waypoints
        self.home = home
        self.threshold = threshold

        self.current_state = MissionState.IDLE
        self.current_waypoint_index = 0

        self.mission_path = [home]
        self.mission_distance = 0.0
        self.mission_start_time = 0.0
        self.mission_time = 0.0

        self.waypoint_times = []

    def start_mission(self):
        if self.current_state == MissionState.IDLE:
            if len(self.waypoints) > 0:
                self.current_state = MissionState.FLYING_TO_WAYPOINT
            else:
                self.current_state = MissionState.RETURNING_HOME

    def get_current_target(self):
        if self.current_state == MissionState.FLYING_TO_WAYPOINT:
            return self.waypoints[self.current_waypoint_index]
        elif self.current_state == MissionState.RETURNING_HOME:
            return self.home
        else:
            return self.home

    def get_actual_distance(self):
        return self.mission_distance

    def update(self, drone_pos, dt=0.1):
        self.mission_time += dt

        waypoint_reached = False

        if len(self.mission_path) > 0:
            prev_pos = self.mission_path[-1]
            step_distance = calculate_distance(prev_pos, drone_pos)
            self.mission_distance += step_distance

        self.mission_path.append(drone_pos)

        if self.current_state == MissionState.FLYING_TO_WAYPOINT:
            waypoint_reached = self._update_flying_to_waypoint(drone_pos)

        elif self.current_state == MissionState.RETURNING_HOME:
            self._update_returning_home(drone_pos)

        elif self.current_state == MissionState.LANDING:
            self._update_landing()

        return waypoint_reached

    def is_mission_complete(self):
        return self.current_state == MissionState.MISSION_COMPLETE

    def calculate_ideal_path_distance(self):
        if len(self.waypoints) == 0:
            return 0.0

        total = 0.0

        distance_from_home_to_first_waypoint = calculate_distance(self.home, self.waypoints[0])

        total += distance_from_home_to_first_waypoint

        for i in range(len(self.waypoints) - 1):
            total += calculate_distance(self.waypoints[i], self.waypoints[i + 1])

        total += calculate_distance(self.waypoints[-1], self.home)

        return total

    def calculate_efficiency(self):
        ideal = self.calculate_ideal_path_distance()
        actual = self.get_actual_distance()

        if actual == 0.0:
            return 100.0

        efficiency = (ideal / actual) * 100
        return efficiency

    def _update_flying_to_waypoint(self, drone_pos):
        target = self.get_current_target()
        distance = calculate_distance(drone_pos, target)

        if distance < self.threshold:
            self.waypoint_times.append(self.mission_time)
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoints):
                self.current_state = MissionState.RETURNING_HOME

            return True

        return False

    def _update_returning_home(self, drone_pos):
        distance = calculate_distance(drone_pos, self.home)

        if distance < self.threshold:
            self.current_state = MissionState.LANDING

    def _update_landing(self):
        self.current_state = MissionState.MISSION_COMPLETE


def run_waypoint_mission(waypoints, home=(0, 0), drone_speed=5.0, max_time=60.0):
    drone = SimpleDrone(x=home[0], y=home[1], speed=drone_speed)
    navigator = WaypointNavigator(waypoints, home=home, threshold=2.0)

    navigator.start_mission()

    dt = 0.1
    t = 0.0

    print("Starting waypoint mission")

    while t < max_time and not navigator.is_mission_complete():
        current_target = navigator.get_current_target()
        vx, vy = pure_pursuit(drone.pos, drone_speed, current_target, (0, 0), 0.5)

        drone.set_velocity(vx, vy)
        drone.update(dt)
        navigator.update(drone.pos, dt)
        t += dt
        if int(t * 10) % 10 == 0:
            state_name = MissionState(navigator.current_state).name
            distance_to_target = calculate_distance(drone.pos, current_target)
            print(
                f"t={t:5.1f}s: State={state_name:20s} "
                + f"Pos=({drone.x:6.1f},{drone.y:6.1f}) "
                + f"Dist={distance_to_target:5.2f}m"
            )

    print("-" * 60)
    if navigator.is_mission_complete():
        print("MISSION COMPLETE!")
        print(f"Total time: {navigator.mission_time:.1f}s")
        print(f"Ideal distance: {navigator.calculate_ideal_path_distance():.1f}m")
        print(f"Actual distance: {navigator.get_actual_distance():.1f}m")
        print(f"Efficiency: {navigator.calculate_efficiency():.1f}%")
        print("Threshold → Corner cutting → Shorter actual path → Efficiency > 100%")
    else:
        print(f"Mission timeout after {max_time}s")

    return drone, navigator


def animate_mission(drone, navigator, waypoints):
    """
    Create animated visualization of waypoint mission.

    Args:
        drone: Drone object with path_x, path_y
        navigator: Navigator with metrics
        waypoints: List of waypoint positions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # === Left Plot: Path Visualization ===

    # Draw ideal path (dashed lines)
    path_points = [navigator.home] + waypoints + [navigator.home]
    for i in range(len(path_points) - 1):
        ax1.plot(
            [path_points[i][0], path_points[i + 1][0]],
            [path_points[i][1], path_points[i + 1][1]],
            "k--",
            alpha=0.3,
            linewidth=2,
            label="Ideal path" if i == 0 else "",
        )

    # Draw threshold circles around waypoints
    for i, wp in enumerate(waypoints):
        circle = Circle(
            wp,
            navigator.threshold,
            fill=False,
            edgecolor="red",
            linestyle=":",
            alpha=0.3,
            linewidth=2,
            label="Threshold zone" if i == 0 else "",
        )
        ax1.add_patch(circle)

    # Plot waypoints
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax1.scatter(wp_x, wp_y, c="red", s=200, marker="x", linewidths=3, label="Waypoints", zorder=5)

    # Add waypoint numbers
    for i, (x, y) in enumerate(waypoints):
        ax1.text(x, y + 2, f"WP{i + 1}", ha="center", fontsize=10, fontweight="bold")

    # Plot home
    ax1.scatter(
        [navigator.home[0]],
        [navigator.home[1]],
        c="green",
        s=300,
        marker="H",
        edgecolors="darkgreen",
        linewidths=2,
        label="Home",
        zorder=5,
    )

    # Animated elements
    (actual_line,) = ax1.plot([], [], "b-", linewidth=2.5, label="Actual path", alpha=0.8)
    (drone_dot,) = ax1.plot([], [], "bo", markersize=15, markeredgewidth=2, markeredgecolor="darkblue")

    # Set plot limits with margin
    all_x = [navigator.home[0]] + wp_x + drone.path_x
    all_y = [navigator.home[1]] + wp_y + drone.path_y
    margin = 5
    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax1.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax1.set_title("Waypoint Navigation Path", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axis("equal")

    # Mission info box
    info_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # === Right Plot: Metrics Dashboard ===

    # Distance over time
    times = [i * 0.1 for i in range(len(navigator.mission_path))]
    distances = []
    cumulative_dist = 0
    for i in range(len(navigator.mission_path)):
        if i > 0:
            cumulative_dist += calculate_distance(navigator.mission_path[i - 1], navigator.mission_path[i])
        distances.append(cumulative_dist)

    ax2.plot(times, distances, "b-", linewidth=2.5, label="Actual distance")

    # Ideal distance as reference line
    ideal_dist = navigator.calculate_ideal_path_distance()
    ax2.axhline(
        y=ideal_dist, color="gray", linestyle="--", linewidth=2, alpha=0.7, label=f"Ideal: {ideal_dist:.1f}m"
    )

    # Animate progress dot
    (distance_dot,) = ax2.plot([], [], "bo", markersize=10)

    ax2.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Distance Traveled (m)", fontsize=12, fontweight="bold")
    ax2.set_title("Distance vs Time", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(0, max(times) * 1.05)
    ax2.set_ylim(0, max(distances) * 1.1)

    # Add metrics text
    efficiency = navigator.calculate_efficiency()
    metrics_text = (
        f"📊 Mission Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total Time: {navigator.mission_time:.1f}s\n"
        f"Ideal Distance: {ideal_dist:.1f}m\n"
        f"Actual Distance: {navigator.get_actual_distance():.1f}m\n"
        f"Efficiency: {efficiency:.1f}%"
    )
    if efficiency > 100:
        metrics_text += "\n⚡ Smart corner cutting!"

    ax2.text(
        0.95,
        0.95,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # Overall title
    fig.suptitle("Autonomous Waypoint Navigation Mission", fontsize=16, fontweight="bold", y=0.98)

    # === Animation ===

    total_frames = len(drone.path_x)
    skip_frames = max(1, total_frames // 200)  # Limit to ~200 frames for smooth playback

    def init():
        actual_line.set_data([], [])
        drone_dot.set_data([], [])
        distance_dot.set_data([], [])
        info_text.set_text("")
        return actual_line, drone_dot, distance_dot, info_text

    def update(frame):
        idx = frame * skip_frames
        if idx >= total_frames:
            idx = total_frames - 1

        # Update actual path
        actual_line.set_data(drone.path_x[: idx + 1], drone.path_y[: idx + 1])

        # Update drone position
        drone_dot.set_data([drone.path_x[idx]], [drone.path_y[idx]])

        # Update distance plot
        if idx < len(times):
            distance_dot.set_data([times[idx]], [distances[idx]])

        # Update info text
        current_time = idx * 0.1
        current_dist = distances[idx] if idx < len(distances) else distances[-1]
        state_name = MissionState.MISSION_COMPLETE.name
        if idx < len(navigator.mission_path) - 1:
            # Approximate state based on time
            if current_time < navigator.mission_time * 0.8:
                state_name = MissionState.FLYING_TO_WAYPOINT.name
            elif current_time < navigator.mission_time * 0.95:
                state_name = MissionState.RETURNING_HOME.name
            else:
                state_name = MissionState.LANDING.name

        info_text.set_text(
            f"Time: {current_time:.1f}s\n"
            f"State: {state_name}\n"
            f"Position: ({drone.path_x[idx]:.1f}, {drone.path_y[idx]:.1f})\n"
            f"Distance: {current_dist:.1f}m"
        )

        return actual_line, drone_dot, distance_dot, info_text

    num_frames = (total_frames + skip_frames - 1) // skip_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return anim


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WAYPOINT NAVIGATION DEMO")
    print("=" * 70 + "\n")

    waypoints = [
        (20, 10),
        (30, 30),
        (10, 40),
        (-10, 20),
    ]
    home = (0, 0)

    drone, navigator = run_waypoint_mission(waypoints=waypoints, home=home, drone_speed=5.0)
    animate_mission(drone, navigator, waypoints)
    print("DEMO COMPLETE!")
