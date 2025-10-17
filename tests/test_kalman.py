import numpy as np
from autograd.drone_problems.kalman import Kalman2D

def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

def test_kalman_noise_reduction():
    T = 50
    ground_truth = []
    measurements = []
    kf = Kalman2D(dt=1.0, q=0.5, sigma_meas=3.0, x0=(0.0, 0.0, 1.0, 0.5))
    x, y, vx, vy = 0.0, 0.0, 1.0, 0.5
    rng = np.random.RandomState(0)
    estimates = []

    for _ in range(T):
        x += vx; y += vy
        ground_truth.append([x, y])

        noisy_measurements = np.array([x, y]) + rng.randn(2) * kf.sigma_meas
        measurements.append(noisy_measurements.tolist())

        kf.predict()
        kf.update(noisy_measurements)
        cx, cy, _, _ = kf.get_state()
        estimates.append([cx, cy])

    rmse_meas = _rmse(measurements, ground_truth)
    rmse_kf = _rmse(estimates, ground_truth)
    assert rmse_kf < rmse_meas * 0.7

def test_kalman_handles_missed_detections():
    T = 60
    kf = Kalman2D(dt=1.0, q=0.5, sigma_meas=2.0, x0=(0.0, 0.0, 1.0, 0.0))
    x, y, vx, vy = 0.0, 0.0, 1.0, 0.0
    rng = np.random.RandomState(1)
    estimates_with_measurements = []
    estimates_without_measurements = []

    for t in range(T):
        x += vx; y += vy
        noisy_measurements = np.array([x, y]) + rng.randn(2) * kf.sigma_meas

        kf.predict()
        if 20 <= t < 30:
            # simulate occlusion: skip update
            cx, cy, _, _ = kf.get_state()
            estimates_without_measurements.append(np.hypot(cx - x, cy - y))
        else:
            kf.update(noisy_measurements)
            cx, cy, _, _ = kf.get_state()
            estimates_with_measurements.append(np.hypot(cx - x, cy - y))

    # error should be small on average
    assert np.mean(estimates_with_measurements) < 2.5
    # estimates shouldn't blow up
    assert np.mean(estimates_without_measurements) < 10.0