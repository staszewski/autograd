import numpy as np
from autograd.simulations.waypoint_demo import (
    calculate_distance,
    MissionState,
    WaypointNavigator,
)


def test_distance_calculation():
    """Basic Euclidean distance should work correctly for sanity"""
    pos1 = (0, 0)
    pos2 = (3, 4)
    assert calculate_distance(pos1, pos2) == 5.0

    assert calculate_distance((5, 5), (5, 5)) == 0.0

    assert calculate_distance((-3, -4), (0, 0)) == 5.0

    distance = calculate_distance((0, 0), (10, 10))
    expected = np.sqrt(200)
    assert abs(distance - expected) < 0.01


# State Machine Transitions
def test_state_machine_initial_state():
    """Navigator should start in IDLE state"""
    waypoints = [(10, 0), (10, 10), (0, 10)]
    nav = WaypointNavigator(waypoints, home=(0, 0))

    assert nav.current_state == MissionState.IDLE
    assert nav.current_waypoint_index == 0


def test_state_machine_start_mission():
    """Starting mission should transition to first waypoint"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0))

    nav.start_mission()

    assert nav.current_state == MissionState.FLYING_TO_WAYPOINT
    assert nav.current_waypoint_index == 0
    assert nav.get_current_target() == (10, 0)


def test_state_machine_waypoint_reached():
    """Should transition to next waypoint when current is reached"""
    waypoints = [(10, 0), (10, 10), (0, 10)]
    nav = WaypointNavigator(waypoints, home=(0, 0), threshold=2.0)
    nav.start_mission()

    drone_pos = (10.5, 0.5)

    reached = nav.update(drone_pos)

    assert nav.current_waypoint_index == 1
    assert nav.get_current_target() == (10, 10)
    assert reached


def test_state_machine_all_waypoints_reached():
    """After all waypoints, should transition to RETURNING_HOME"""
    waypoints = [(10, 0), (10, 10)]
    nav = WaypointNavigator(waypoints, home=(0, 0), threshold=2.0)
    nav.start_mission()

    nav.update((10, 0))
    assert nav.current_state == MissionState.FLYING_TO_WAYPOINT
    assert nav.current_waypoint_index == 1

    nav.update((10, 10))
    assert nav.current_state == MissionState.RETURNING_HOME
    assert nav.get_current_target() == (0, 0)  # Home


def test_state_machine_return_home():
    """Should transition to LANDING when home is reached"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0), threshold=2.0)
    nav.start_mission()

    nav.update((10, 0))

    nav.update((0.5, 0.5))

    assert nav.current_state == MissionState.LANDING


def test_state_machine_mission_complete():
    """Landing should transition to MISSION_COMPLETE"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0), threshold=2.0)
    nav.start_mission()

    nav.update((10, 0))
    nav.update((0, 0))
    nav.update((0, 0))

    assert nav.current_state in [MissionState.LANDING, MissionState.MISSION_COMPLETE]


def test_landing_completes_mission():
    """LANDING state should eventually transition to MISSION_COMPLETE"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0), threshold=2.0)
    nav.start_mission()

    nav.update((10, 0))
    assert nav.current_state == MissionState.RETURNING_HOME

    nav.update((0, 0))
    assert nav.current_state == MissionState.LANDING

    nav.update((0, 0))
    assert nav.current_state == MissionState.MISSION_COMPLETE


# Path metrics
def test_actual_path_tracking():
    """Should track actual path as drone moves"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0))
    nav.start_mission()

    nav.update((2, 0))
    nav.update((4, 0))
    nav.update((6, 0))
    nav.update((8, 0))

    actual_distance = nav.get_actual_distance()
    assert actual_distance == 8.0


def test_ideal_path_calculation():
    """Calculate ideal (straight-line) path distance"""
    waypoints = [(10, 0), (10, 10), (0, 10)]
    nav = WaypointNavigator(waypoints, home=(0, 0))

    ideal_distance = nav.calculate_ideal_path_distance()

    # (0,0)->(10,0): 10
    # (10,0)->(10,10): 10
    # (10,10)->(0,10): 10
    # (0,10)->(0,0): 10
    expected = 10 + 10 + 10 + 10
    assert ideal_distance == expected


def test_path_efficiency_perfect():
    """Perfect straight-line flight should give ~100% efficiency"""
    waypoints = [(10, 0)]
    nav = WaypointNavigator(waypoints, home=(0, 0))
    nav.start_mission()

    # Simulate perfect straight-line flight
    for x in range(0, 11):
        nav.update((x, 0))

    # Reach waypoint, then return home perfectly
    for x in range(10, -1, -1):
        nav.update((x, 0))

    efficiency = nav.calculate_efficiency()

    # Should be very close to 100% (might have tiny rounding errors)
    assert efficiency > 99.0
    assert efficiency <= 100.0
