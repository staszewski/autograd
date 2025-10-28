def generate_straight_line_sequence(num_frames=10, start=(10.0, 10.0), velocity=(2.0, 1.0), box_size=5):
    """
    Generate synthetic frames with a single object moving in a straight line.
    
    Args:
        num_frames: Number of frames to generate
        start: (x, y) starting position of object center
        velocity: (vx, vy) velocity in pixels per frame
        box_size: Size of bounding box (square)
    
    Returns:
        frames: List of detections per frame, each frame is [(box, score), ...]
        ground_truth: List of true center positions [(x, y), ...]
    """
    frames = []
    ground_truth = []

    start_x, start_y = start
    vx, vy = velocity

    for t in range(num_frames):
        center_x = start_x + vx * t
        center_y = start_y + vy * t
        ground_truth.append((center_x, center_y))

        half_size = box_size / 2.0
        box = [
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size
        ]

        score = 0.95
        detections = [(box, score)]

        frames.append(detections)

    return frames, ground_truth