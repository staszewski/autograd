class OrbitalFollower:
    def __init__(self, orbit_radius=5.0, angular_velocity=0.3, kp=0.8) -> None:
        self.orbit_radius = orbit_radius
        self.angular_velocity = angular_velocity
        self.kp = kp

        # State
        self.current_angle = 0.0
        self.target_pos = None
        self.drone_pos = None

    def update_target(self, target_pos):
        """Update target position"""
        self.target_pos = target_pos

    def update_drone(self, drone_pos):
        """Update drone position"""
        self.drone_pos = drone_pos

    def get_control_command(self, dt):
        """Calculate proper orbital velocity (tangential + radial correction)"""
        if self.target_pos is None or self.drone_pos is None:
            return (0, 0)

        target_x, target_y = self.target_pos
        drone_x, drone_y = self.drone_pos

        radial_x = drone_x - target_x
        radial_y = drone_y - target_y
        radial_distance = (radial_x**2 + radial_y**2) ** 0.5

        if radial_distance < 0.1:
            return (0, 0)

        radial_error = radial_distance - self.orbit_radius
        radial_correction_magnitude = self.kp * radial_error
        radial_vx = -radial_correction_magnitude * (radial_x / radial_distance)
        radial_vy = -radial_correction_magnitude * (radial_y / radial_distance)

        tangential_vx = -self.angular_velocity * radial_y  # Counter-clockwise
        tangential_vy = self.angular_velocity * radial_x

        total_vx = radial_vx + tangential_vx
        total_vy = radial_vy + tangential_vy

        return total_vx, total_vy
