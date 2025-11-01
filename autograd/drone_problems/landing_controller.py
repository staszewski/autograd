import numpy as np


class LandingController:
    def __init__(self, pad_position, Kp=0.5, Kd=0.3, centering_threshold=0.2):
        self.pad_pos = pad_position

        self.Kp = Kp
        self.Kd = Kd
        self.centering_threshold = centering_threshold

        self.drone_pos = None
        self.centered = False
        self.landed = False
        self.vertical_threshold = 0.05

    def set_drone_position(self, drone_pos):
        self.drone_pos = drone_pos

    def calculate_errors(self):
        if self.drone_pos is None:
            return 0, 0, 0, 0

        x_d, y_d, z_d = self.drone_pos
        x_p, y_p, z_p = self.pad_pos

        e_x = x_p - x_d
        e_y = y_p - y_d

        e_horizontal = np.sqrt(e_x**2 + e_y**2)
        e_z = z_d

        return e_x, e_y, e_horizontal, e_z

    def compute_velocities(self, e_x, e_y, e_z):
        v_x = self.Kp * e_x
        v_y = self.Kp * e_y

        if self.centered:
            v_z = -self.Kd * e_z
        else:
            v_z = 0.0

        return v_x, v_y, v_z

    def is_centered(self, e_x, e_y):
        horizontal_distance = np.sqrt(e_x**2 + e_y**2)
        return horizontal_distance < self.centering_threshold

    def update(self, dt=0.1):
        e_x, e_y, e_horizontal, e_z = self.calculate_errors()
        self.centered = self.is_centered(e_x, e_y)

        _, _, z_d = self.drone_pos
        if self.drone_pos and z_d <= self.vertical_threshold:
            self.landed = True

        v_x, v_y, v_z = self.compute_velocities(e_x, e_y, e_z)

        return v_x, v_y, v_z

    def get_status(self):
        e_x, e_y, e_horizontal, e_z = self.calculate_errors()

        return {
            "centered": self.centered,
            "landed": self.landed,
            "horizontal_error": e_horizontal,
            "altitude": e_z,
        }
