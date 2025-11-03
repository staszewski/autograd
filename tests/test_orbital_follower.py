import numpy as np

from autograd.drone_problems.orbital_follower import OrbitalFollower
from autograd.drone_problems.simple_drone import SimpleDrone

def test_basic_orbital_motion():
    """Test drone moves in circle around stationary target"""
    R = 5.0
    omega = 0.5
    kp = 2.0
    dt = 0.1
    T = 12.5
    steps = int(T / dt)
    delta0 = 0.5

    follower = OrbitalFollower(orbit_radius=R, angular_velocity=omega, kp=kp)
    drone = SimpleDrone(x = R + delta0, y = 0, speed = 10.0)  # Start on circle at angle=0
    
    positions = []
    
    for _ in range(steps):
        follower.update_target((0, 0))
        follower.update_drone(drone.pos)
        
        vx, vy = follower.get_control_command(dt)
        drone.set_velocity(vx, vy)
        drone.update(dt)
        positions.append(drone.pos)
    
    positions = np.array(positions)
    
    distances_from_center = [np.sqrt(x**2 + y**2) for x, y in positions]
    avg_distance = np.mean(distances_from_center)
    assert 4.5 < avg_distance < 5.5, f"Expected radius 5, got {avg_distance}"
    
    total_path_length = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        step_distance = np.sqrt(dx**2 + dy**2)
        total_path_length += step_distance
    
    # Should travel at least half the circumference: π * radius ≈ 15.7 units
    assert total_path_length > 15, f"Drone barely moved: {total_path_length}"
    
    unique_positions = set()
    for pos in positions[::5]:
        rounded_pos = (round(pos[0], 1), round(pos[1], 1))
        unique_positions.add(rounded_pos)
    
    # Should visit at least 8 different 0.1-precision positions around the circle
    assert len(unique_positions) > 8, f"Drone didn't orbit: only {len(unique_positions)} unique positions"
   

def test_radius_maintenance():
    """Test orbit radius is maintained despite starting offset"""
    
    follower = OrbitalFollower(orbit_radius=4.0, angular_velocity=0.3, kp=3.0)
    drone = SimpleDrone(x=0, y=0, speed=8.0)  # Start at center (worst case)
    
    dt = 0.1
    target_pos = (10, 10)
    
    for _ in range(100):
        follower.update_target(target_pos)
        follower.update_drone(drone.pos)
        
        vx, vy = follower.get_control_command(dt)
        drone.set_velocity(vx, vy)
        drone.update(dt)
    
    distance_from_target = np.sqrt((drone.x - target_pos[0])**2 + (drone.y - target_pos[1])**2)
    assert 3.5 < distance_from_target < 4.5, f"Expected radius 4, got {distance_from_target}"


def test_moving_target_orbit():
    """Test orbit follows moving target"""
    
    follower = OrbitalFollower(orbit_radius=3.0, angular_velocity=0.4, kp=2.5)
    drone = SimpleDrone(x=3, y=0, speed=7.0)  # Start on orbit
    
    dt = 0.1
    target_positions = []
    drone_positions = []
    
    # Target moves in straight line: (0,0) → (10,0)
    for step in range(80):  # 8 seconds
        target_x = step * 0.1  # Moves 1 unit/second right
        target_y = 0
        
        follower.update_target((target_x, target_y))
        follower.update_drone(drone.pos)
        
        vx, vy = follower.get_control_command(dt)
        drone.set_velocity(vx, vy)
        drone.update(dt)
        
        target_positions.append((target_x, target_y))
        drone_positions.append(drone.pos)
    
    for i in range(40, 80):  # Check latter half (after settling)
        target_pos = target_positions[i]
        drone_pos = drone_positions[i]
        
        distance = np.sqrt((drone_pos[0] - target_pos[0])**2 + (drone_pos[1] - target_pos[1])**2)
        assert 2.5 < distance < 3.5, f"Lost orbit at step {i}: distance {distance}"
