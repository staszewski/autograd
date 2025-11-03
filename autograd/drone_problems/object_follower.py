from math import sqrt


def calculate_following_position(target_pos, target_vel, distance):
    """
    Calculate where drone should position to follow target from behind.

    Args:
        target_pos: (x, y) target position
        target_vel: (vx, vy) target velocity
        distance: following distance behind target

    Returns:
        (follow_x, follow_y): optimal drone position
    """
    x_t, y_t = target_pos
    v_tx, v_ty = target_vel
    speed = sqrt(v_tx**2 + v_ty**2)

    if speed == 0:
        return x_t, y_t - distance

    follow_x = x_t - distance * (v_tx / speed)
    follow_y = y_t - distance * (v_ty / speed)

    return follow_x, follow_y


class ObjectFollower:
    """
    Autonomous object following system.

    Tracks a moving target and maintains optimal following position.
    """

    def __init__(self, following_distance=5.0, kp=0.5):
        """
        Initialize object follower.

        Args:
            following_distance: Distance to maintain behind target (meters)
            kp: Proportional gain for pursuit control
        """
        self.following_distance = following_distance
        self.Kp = kp

        # State
        self.target_pos = None
        self.target_vel = None
        self.drone_pos = None

    def update_target(self, target_pos, target_vel):
        self.target_pos = target_pos
        self.target_vel = target_vel

    def update_drone(self, drone_pos):
        self.drone_pos = drone_pos

    def calculate_desired_position(self):
        if self.target_pos is None or self.target_vel is None:
            return None

        follow_x, follow_y = calculate_following_position(
            self.target_pos, self.target_vel, self.following_distance
        )

        return (follow_x, follow_y)

    def get_control_command(self):
        desired_pos = self.calculate_desired_position()

        if desired_pos is None or self.drone_pos is None:
            return (0, 0)

        desired_x, desired_y = desired_pos
        drone_x, drone_y = self.drone_pos

        error_x = desired_x - drone_x
        error_y = desired_y - drone_y

        vx = self.Kp * error_x
        vy = self.Kp * error_y

        return vx, vy
