import numpy as np

def pure_pursuit(pursuer_pos, pursuer_speed, target_pos, target_velocity, lookahead_time=1.0):
    """
    Pure pursuit: fly toward predicted target position.
    
    Args:
        pursuer_pos: (x, y) current position
        pursuer_speed: scalar speed
        target_pos: (x, y) target position
        target_vel: (vx, vy) target velocity
        lookahead_time: how far ahead to predict (seconds)
    
    Returns:
        (vx, vy): velocity vector for pursuer
    """
    pursuer_position = np.array(pursuer_pos)
    target_position = np.array(target_pos)
    target_velocity_arr = np.array(target_velocity)
    # 1. Predict future target position
    future_target_position = target_position + target_velocity_arr * lookahead_time
    # 2. Compute direction vector
    direction_vector = future_target_position - pursuer_position
    # 3. Normalize
    norm = np.linalg.norm(direction_vector)

    if norm < 1e-6:
        return (0.0, 0.0)

    unit_direction = direction_vector / norm
    # 4. Scale by pursuer speed
    pursuer_velocity = unit_direction * pursuer_speed

    return (float(pursuer_velocity[0]), float(pursuer_velocity[1]))