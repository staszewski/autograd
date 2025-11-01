import numpy as np
from autograd.drone_problems.landing_controller import LandingController
from autograd.drone_problems.simple_drone import SimpleDrone3D


def test_error_calculation():
    """Test horizontal and vertical error calculation"""
    controller = LandingController(pad_position=(12, 15, 0))
    controller.set_drone_position((10, 12, 5))

    error_x, error_y, error_horizontal, error_z = controller.calculate_errors()

    assert error_x == 2.0
    assert error_y == 3.0
    assert error_z == 5.0
    expected_horizontal = np.sqrt(2**2 + 3**2)
    assert abs(error_horizontal - expected_horizontal) < 0.01


def test_proportional_control():
    """Test P controller velocity calculation"""
    controller = LandingController(pad_position=(0, 0, 0), Kp=0.5)
    controller.set_drone_position((10, 12, 5))  # Far from pad

    vx, vy, vz = controller.compute_velocities(2.0, 3.0, 5.0)

    assert vx == 1.0
    assert vy == 1.5
    assert vz == 0.0


def test_centering_detection():
    """Test when drone is considered centered above pad"""
    controller = LandingController(pad_position=(0, 0, 0), centering_threshold=0.2)

    assert controller.is_centered(0.1, 0.1)

    assert not controller.is_centered(0.3, 0.4)


def test_descent_control():
    """Test vertical velocity during descent"""
    controller = LandingController(pad_position=(0, 0, 0), Kd=0.3)

    # When not centered: hover
    controller.centered = False
    vx, vy, vz = controller.compute_velocities(0, 0, 5.0)
    assert vz == 0.0

    # When centered: descend
    controller.centered = True
    vx, vy, vz = controller.compute_velocities(0, 0, 5.0)
    assert vz == -1.5  # -0.3 * 5


def test_full_landing_sequence():
    """Test complete landing mission from start to finish"""
    drone = SimpleDrone3D(10, 12, 5)
    controller = LandingController(pad_position=(12, 15, 0), Kp=0.5)

    dt = 0.1
    max_time = 30.0
    t = 0.0

    while t < max_time and not controller.landed:
        controller.set_drone_position((drone.x, drone.y, drone.z))
        vx, vy, vz = controller.update(dt)

        drone.set_velocity(vx, vy, vz)
        drone.update(dt)

        t += dt

    assert controller.landed
    assert t < max_time

    final_error = np.sqrt((drone.x - 12) ** 2 + (drone.y - 15) ** 2)
    assert final_error < 0.1
