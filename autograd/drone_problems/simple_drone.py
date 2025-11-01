class SimpleDrone:
    """Simplified 2D drone (reused from pursuit_demo)"""

    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.vx = 0.0
        self.vy = 0.0
        self.path_x = [x]
        self.path_y = [y]

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def update(self, dt=0.1):
        """Move drone based on current velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.path_x.append(self.x)
        self.path_y.append(self.y)

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def vel(self):
        return (self.vx, self.vy)


class SimpleDrone3D:
    """Simple 3D drone"""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def set_velocity(self, vx, vy, vz):
        self.vx, self.vy, self.vz = vx, vy, vz

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
