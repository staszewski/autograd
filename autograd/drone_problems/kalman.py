import numpy as np

class Kalman2D:
    r"""
    Constant-velocity 2D Kalman filter with explicit (predict/update) steps
    and notation: \hat{x}^-, P^- for prediction; \hat{x}, P after update.
    State x = [x, y, vx, vy]^T
    """

    def __init__(self, dt: float = 1.0, q: float = 0.5, sigma_meas: float = 3.0,
                 x0=None, A=None, H=None, Q=None, R=None) -> None:
        self.dt = float(dt)
        self.q = float(q)
        self.sigma_meas = float(sigma_meas)

        # Default matrices for constant-velocity
        if A is None:
            dt = self.dt
            self.A = np.array([[1,0,dt,0],
                               [0,1,0,dt],
                               [0,0, 1,0],
                               [0,0, 0,1]], dtype=np.float32)
        else:
            self.A = np.array(A, dtype=np.float32)

        if H is None:
            self.H = np.array([[1,0,0,0],
                               [0,1,0,0]], dtype=np.float32)
        else:
            self.H = np.array(H, dtype=np.float32)

        if Q is None:
            dt = self.dt
            dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
            self.Q = self.q * np.array([[dt4/4,   0.0, dt3/2,  0.0],
                                        [  0.0, dt4/4,  0.0, dt3/2],
                                        [dt3/2,   0.0,  dt2,  0.0],
                                        [  0.0, dt3/2,  0.0,  dt2]], dtype=np.float32)
        else:
            self.Q = np.array(Q, dtype=np.float32)

        if R is None:
            r2 = self.sigma_meas ** 2
            self.R = np.array([[r2, 0.0],
                               [0.0, r2]], dtype=np.float32)
        else:
            self.R = np.array(R, dtype=np.float32)

        self.I = np.eye(4, dtype=np.float32)

        if x0 is None:
            self.hat_x = None  # \hat{x}_{k-1}
            self.P = None      # P_{k-1}
        else:
            self.hat_x = np.array(x0, dtype=np.float32).reshape(4,)
            self.P = np.diag([100.0, 100.0, 25.0, 25.0]).astype(np.float32)

        self.hat_x_minus = None  # \hat{x}^-
        self.P_minus = None      # P^-

    def _ensure_initialized(self, z: np.ndarray):
        if self.hat_x is None:
            cx, cy = float(z[0]), float(z[1])
            self.hat_x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
            self.P = np.diag([100.0, 100.0, 25.0, 25.0]).astype(np.float32)

    def predict(self):
        r"""Time update: (\hat{x}^-, P^-) = (A \hat{x}, A P A^T + Q)."""
        if self.hat_x is None:
            # nothing to predict
            return None
        self.hat_x_minus = self.A @ self.hat_x
        self.P_minus = self.A @ self.P @ self.A.T + self.Q
        return self.hat_x_minus[:2].copy()

    def update(self, z):
        r"""
        Measurement update:
          K = P^- H^T (H P^- H^T + R)^{-1}
          \hat{x} = \hat{x}^- + K (z - H \hat{x}^-)
          P = P^- - K H P^-
        If z is None, skips update and promotes predicted values.
        """
        if z is None:
            # Promote prediction to current estimate
            if self.hat_x_minus is not None:
                self.hat_x = self.hat_x_minus
                self.P = self.P_minus
            return

        z = np.asarray(z, dtype=np.float32).reshape(2,)
        self._ensure_initialized(z)

        # If predict wasn't called, do an implicit prediction
        if self.hat_x_minus is None or self.P_minus is None:
            self.predict()

        S = self.H @ self.P_minus @ self.H.T + self.R
        K = self.P_minus @ self.H.T @ np.linalg.inv(S)
        innovation = z - (self.H @ self.hat_x_minus)

        self.hat_x = self.hat_x_minus + (K @ innovation)
        self.P = self.P_minus - K @ self.H @ self.P_minus

        # Clear minus buffers (next cycle should call predict again)
        self.hat_x_minus = None
        self.P_minus = None

    def get_state(self):
        r"""Return current estimate \hat{x} = [x, y, vx, vy] (or None if uninitialized)."""
        if self.hat_x is None:
            return None
        return tuple(map(float, self.hat_x))