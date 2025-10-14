from math import cos, pi

class StepLR:
    def __init__(self, opt, step_size: int = 1, gamma: float = 0.0) -> None:
        self.opt = opt
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.t = 0

        if self.step_size <= 0:
            raise ZeroDivisionError("Step size must be >= 1")
        
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("Gamma should be between 0 and 1")

    def step(self):
        if self.t > 0 and self.t % self.step_size == 0:
            self.opt.lr *= self.gamma

        self.t += 1

class CosineAnnealingLR:
    def __init__(self, opt, T_max, eta_min: float = 0.0) -> None:
        self.opt = opt
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.base_lr = float(opt.lr)
        self.t = 0

    def step(self):
        if self.t < self.T_max:
            self.t += 1
        
        theta = pi * self.t / self.T_max
        self.opt.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + cos(theta))