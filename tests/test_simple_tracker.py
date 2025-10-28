from autograd.drone_problems.simple_tracker import SimpleTracker
from autograd.drone_problems.tracking_data import generate_straight_line_sequence


def test_tracker_follows_straight_line():
    """Core test: Does tracker follow a moving object?"""
    frames, ground_truth = generate_straight_line_sequence(
        num_frames=10, 
        start=(10.0, 10.0), 
        velocity=(2.0, 1.0)
    )
    
    tracker = SimpleTracker()
    
    for t, detections in enumerate(frames):
        tracked_box = tracker.update(detections, current_time=t)
        
        assert tracked_box is not None
        
        tracked_cx = (tracked_box[0] + tracked_box[2]) / 2
        tracked_cy = (tracked_box[1] + tracked_box[3]) / 2
        true_cx, true_cy = ground_truth[t]

        assert tracked_cx == true_cx
        assert tracked_cy == true_cy

def test_tracker_estimates_velocity():
    """Does velocity calculation work"""
    object_velocity = (3.0, 2.0)
    frames, _ = generate_straight_line_sequence(
        num_frames=5,
        start=(10.0, 10.0),
        velocity=object_velocity,
        box_size=5
    )
    
    tracker = SimpleTracker()
    
    tracker.update(frames[0], current_time=0)
    tracker.update(frames[1], current_time=1)
    
    vx, vy = tracker.velocity
    object_vx, object_vy = object_velocity

    assert vx == object_vx
    assert vy == object_vy


def test_tracker_picks_closest_detection():
    """With multiple detections, should pick closest (not highest score)"""
    tracker = SimpleTracker()
    
    tracker.update([([10, 10, 15, 15], 0.9)], current_time=0)
    close_lower_score = ([12, 11, 17, 16], 0.85)

    detections = [
        ([50, 50, 55, 55], 0.95), 
        close_lower_score
    ]
    result = tracker.update(detections, current_time=1)
    
    assert result == close_lower_score[0]