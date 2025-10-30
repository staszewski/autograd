import numpy as np


def generate_straight_line(
    num_points=20, start=(0.0, 0.0), velocity=(2.0, 1.0), noise=0.1
):
    """
    Generate straight-line trajectory with optional noise.

    Math: x(t) = x0 + vx*t
          y(t) = y0 + vy*t

    Args:
        num_points: Number of positions in trajectory
        start: Starting (x, y) position
        velocity: Constant (vx, vy) velocity
        noise: Random noise to add (std deviation) to make it realistic

    Returns:
        positions: array of shape (num_points, 2)
    """
    positions = []
    x, y = start
    vx, vy = velocity

    for t in range(num_points):
        noise_x = np.random.normal(0, noise)
        noise_y = np.random.normal(0, noise)

        positions.append([x + vx * t + noise_x, y + vy * t + noise_y])

    return np.array(positions)


def generate_circular_loiter(
    num_points=20, center=(10.0, 10.0), radius=5.0, angular_velocity=0.3, noise=0.1
):
    """
    Generate circular trajectory (drone loitering).

    Math: x(t) = cx + r*cos(ω*t)
          y(t) = cy + r*sin(ω*t)

    Args:
        num_points: Number of positions
        center: Circle center (cx, cy)
        radius: Circle radius
        angular_velocity: How fast to rotate (radians per step)
        noise: Random noise

    Returns:
        positions: array of shape (num_points, 2)
    """
    positions = []
    cx, cy = center

    for t in range(num_points):
        angle = angular_velocity * t

        noise_x = np.random.normal(0, noise)
        noise_y = np.random.normal(0, noise)

        positions.append(
            [
                cx + radius * np.cos(angle) + noise_x,
                cy + radius * np.sin(angle) + noise_y,
            ]
        )

    return np.array(positions)


def generate_waypoint_path(waypoints, points_per_segment=5, noise=0.1):
    """
    Generate trajectory that moves between waypoints.

    Math: Linear interpolation between consecutive waypoints
          p(t) = p_start + (p_end - p_start) * t

    Args:
        waypoints: List of (x, y) positions to visit
        points_per_segment: How many points between each waypoint
        noise: Random noise

    Returns:
        positions: array of shape (num_points, 2)
    """
    positions = []

    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])

        for t in range(points_per_segment):
            # t goes from 0 to 1 over the segment
            alpha = t / points_per_segment

            # Linear interpolation
            pos = start + (end - start) * alpha

            noise_x = np.random.normal(0, noise)
            noise_y = np.random.normal(0, noise)

            positions.append([pos[0] + noise_x, pos[1] + noise_y])

    positions.append(waypoints[-1])

    return np.array(positions)


def create_sliding_window_samples(trajectory, input_len=5, output_len=3):
    """
    Convert single trajectory into multiple training samples using sliding window.

    Example:
        trajectory: [p0, p1, p2, ..., p19] (20 positions)
        Creates samples like:
          X: [p0,p1,p2,p3,p4] → y: [p5,p6,p7]
          X: [p1,p2,p3,p4,p5] → y: [p6,p7,p8]
          etc.

    Args:
        trajectory: array of shape (num_points, 2)
        input_len: How many past positions to use (default 5)
        output_len: How many future positions to predict (default 3)

    Returns:
        X: array of shape (num_samples, input_len * 2)
        y: array of shape (num_samples, output_len * 2)
    """
    X = []
    y = []

    num_points = len(trajectory)
    window_size = input_len + output_len

    for i in range(num_points - window_size + 1):
        # Input: flatten last 5 positions into single vector
        input_positions = trajectory[i : i + input_len]  # shape (5, 2)
        X.append(input_positions.flatten())  # shape (10,)

        # Output: flatten next 3 positions
        output_positions = trajectory[i + input_len : i + window_size]  # shape (3, 2)
        y.append(output_positions.flatten())  # shape (6,)

    return np.array(X), np.array(y)


def generate_dataset(num_trajectories=1000, trajectory_length=20):
    """
    Generate complete training dataset with multiple trajectory types.

    Mix of:
    - 40% straight lines (various velocities)
    - 30% circular loiters (various radii)
    - 30% waypoint paths (various patterns)

    Args:
        num_trajectories: How many trajectories to generate
        trajectory_length: Points per trajectory

    Returns:
        X_train: Input sequences, shape (num_samples, 10)
        y_train: Output sequences, shape (num_samples, 6)
    """
    all_X = []
    all_y = []

    for i in range(num_trajectories):
        # Randomly choose trajectory type
        traj_type = np.random.choice(
            ["straight", "circular", "waypoint"], p=[0.4, 0.3, 0.3]
        )

        if traj_type == "straight":
            # Random velocity
            velocity = (np.random.uniform(-3, 3), np.random.uniform(-3, 3))
            trajectory = generate_straight_line(
                num_points=trajectory_length, velocity=velocity, noise=0.1
            )

        elif traj_type == "circular":
            # Random radius and angular velocity
            radius = np.random.uniform(3, 10)
            angular_velocity = np.random.uniform(0.1, 0.5)
            trajectory = generate_circular_loiter(
                num_points=trajectory_length,
                radius=radius,
                angular_velocity=angular_velocity,
                noise=0.1,
            )

        else:  # waypoint
            # Random waypoints forming a path
            num_waypoints = 5
            waypoints = []
            for _ in range(num_waypoints):
                waypoints.append((np.random.uniform(0, 20), np.random.uniform(0, 20)))

            points_per_segment = trajectory_length // (num_waypoints - 1)
            trajectory = generate_waypoint_path(
                waypoints, points_per_segment=points_per_segment, noise=0.1
            )[:trajectory_length]  # Trim to exact length

        X, y = create_sliding_window_samples(trajectory)
        all_X.append(X)
        all_y.append(y)

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    return X_train, y_train


def visualize_samples():
    """Quick visualization to verify trajectories look realistic"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Straight line
    straight = generate_straight_line(num_points=20, velocity=(2, 1))
    axes[0].plot(straight[:, 0], straight[:, 1], "bo-")
    axes[0].set_title("Straight Line")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True)

    # Circular
    circular = generate_circular_loiter(num_points=30, radius=5, angular_velocity=0.2)
    axes[1].plot(circular[:, 0], circular[:, 1], "ro-")
    axes[1].set_title("Circular Loiter")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].grid(True)
    axes[1].axis("equal")

    # Waypoints
    waypoints = [(0, 0), (10, 5), (15, 15), (5, 20)]
    waypoint_traj = generate_waypoint_path(waypoints, points_per_segment=8)
    axes[2].plot(waypoint_traj[:, 0], waypoint_traj[:, 1], "go-")
    # Mark waypoints
    wp_array = np.array(waypoints)
    axes[2].plot(wp_array[:, 0], wp_array[:, 1], "r*", markersize=15, label="Waypoints")
    axes[2].set_title("Waypoint Path")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("trajectory_samples.png", dpi=150)
    print("Saved visualization to trajectory_samples.png")
    plt.show()


if __name__ == "__main__":
    # Test the generators
    print("Testing trajectory generators...\n")

    # Test 1: Straight line
    straight = generate_straight_line(num_points=10, velocity=(2, 1))
    print(f"Straight line trajectory shape: {straight.shape}")
    print(f"First 3 positions:\n{straight[:3]}\n")

    # Test 2: Circular
    circular = generate_circular_loiter(num_points=10, radius=5)
    print(f"Circular trajectory shape: {circular.shape}")
    print(f"First 3 positions:\n{circular[:3]}\n")

    # Test 3: Waypoints
    waypoints = [(0, 0), (5, 5), (10, 5), (10, 0)]
    waypoint_traj = generate_waypoint_path(waypoints, points_per_segment=5)
    print(f"Waypoint trajectory shape: {waypoint_traj.shape}")
    print(f"First 3 positions:\n{waypoint_traj[:3]}\n")

    # Test 4: Sliding window
    X, y = create_sliding_window_samples(straight, input_len=5, output_len=3)
    print("Sliding window samples:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  First sample - Input: {X[0]}")
    print(f"  First sample - Output: {y[0]}\n")

    # Test 5: Full dataset
    X_train, y_train = generate_dataset(num_trajectories=100, trajectory_length=20)
    print("Full dataset generated:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Total training samples: {len(X_train)}")

    visualize_samples()

