import numpy as np
from autograd.drone_problems.pursuit import pure_pursuit

def test_stationary_target():
    """If target isn't moving, fly straight at it"""
    pursuer_pos = (0, 0)
    pursuer_speed = 5.0
    target_pos = (3, 4)
    target_vel = (0, 0)
    
    result_vel = pure_pursuit(pursuer_pos, pursuer_speed, target_pos, target_vel)
    
    # Should fly toward (3, 4) at speed 5
    # Unit direction = (3/5, 4/5) = (0.6, 0.8)
    # Velocity = (0.6*5, 0.8*5) = (3, 4)
    expected_vx, expected_vy = 3.0, 4.0

    assert result_vel[0]  == expected_vx
    assert result_vel[1]  == expected_vy

def test_moving_target_lead():
    """Should aim ahead of moving target"""
    pursuer_pos = (0, 0)
    pursuer_speed = 5.0
    target_pos = (10, 0)  # On x-axis
    target_vel = (0, 5)   # Moving up
    lookahead = 1.0
    
    # After 1 second, target at (10, 5)
    # Direction to (10, 5) from (0, 0)
    result_vel = pure_pursuit(pursuer_pos, pursuer_speed, target_pos, target_vel, lookahead)
    
    # Predicted position: (10, 0) + (0, 5)*1 = (10, 5)
    # Direction: (10, 5), norm = sqrt(125) ≈ 11.18
    # Unit: (10/11.18, 5/11.18) ≈ (0.894, 0.447)
    # Velocity: (0.894*5, 0.447*5) ≈ (4.47, 2.24)
    
    expected_vx = 10 / np.sqrt(125) * 5
    expected_vy = 5 / np.sqrt(125) * 5
    
    assert result_vel[0]  == expected_vx
    assert result_vel[1]  == expected_vy

def test_pursuit_speed_correct():
    """Result velocity magnitude should equal pursuer speed"""
    pursuer_pos = (0, 0)
    pursuer_speed = 10.0
    target_pos = (7, 7)
    target_vel = (1, -1)
    
    result_vel = pure_pursuit(pursuer_pos, pursuer_speed, target_pos, target_vel)
    
    speed = np.sqrt(result_vel[0]**2 + result_vel[1]**2)
    assert speed - pursuer_speed == 0