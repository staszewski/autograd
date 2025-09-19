from math import cos, sin


class Drone:
    GRAVITY = 0.1 # not realistic, but for testing purposes
    def __init__(self, x, y, vx, vy, angle):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.angular_velocity = 0.0

    def update(self, thrust, torque, dt = 0.1):
        """Update the drone's state for a given time step.
        1. Calculate forces from thrust
        2. Calculate accelerations (including gravity)
        3. Update velocities 
        4. Update positions
        5. Handle any limits/boundaries 
        """
        force_x = thrust * sin(self.angle)
        force_y = thrust * cos(self.angle)

        acceleration_x = force_x
        acceleration_y = force_y - self.GRAVITY

        angular_acceleration = torque

        self.vx += acceleration_x * dt
        self.vy += acceleration_y * dt
        self.angular_velocity =  self.angular_velocity + angular_acceleration * dt

        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.angle = self.angle + self.angular_velocity * dt

    def get_state(self):
        return self.x, self.y, self.vx, self.vy, self.angle