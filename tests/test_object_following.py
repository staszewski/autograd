from autograd.simulations.object_following_demo import calculate_following_position, ObjectFollower
from autograd.drone_problems.simple_drone import SimpleDrone


# Calculate following distance
def test_calculate_following_position():
    target_pos = (20, 15)
    target_vel = (0, 3)
    following_distance = 4.0

    follow_x, follow_y = calculate_following_position(target_pos, target_vel, following_distance)

    assert follow_x == 20.0
    assert follow_y == 11.0


def test_zero_speed_following():
    target_pos = (10, 10)
    target_vel = (0, 0)
    following_distance = 3.0

    follow_x, follow_y = calculate_following_position(target_pos, target_vel, following_distance)

    assert follow_x == target_pos[0]
    assert follow_y == target_pos[1] - following_distance


# ObjectFollower
def test_object_follower_basic():
    """Test basic object follower functionality"""

    follower = ObjectFollower(following_distance=4.0, kp=0.5)

    follower.update_target((10, 10), (2, 0))
    follower.update_drone((8, 10))

    desired = follower.calculate_desired_position()
    assert desired == (6, 10)

    vx, vy = follower.get_control_command()
    assert vx == 0.5 * (6 - 8)
    assert vy == 0


def test_complete_following_system():
    """Test the full object following system"""

    follower = ObjectFollower(following_distance=4.0, kp=0.5)
    drone = SimpleDrone(8, 10, 5)

    target_pos = (10, 10)
    target_vel = (2, 0)

    follower.update_target(target_pos, target_vel)
    follower.update_drone(drone.pos)

    vx, vy = follower.get_control_command()
    drone.set_velocity(vx, vy)
    drone.update(dt=0.1)

    assert abs(drone.x - 7.9) < 0.01
    assert drone.y == 10
