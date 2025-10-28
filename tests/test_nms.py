from autograd.operations.bbox_ops import compute_iou, non_max_suppression

def test_nms_no_overlap():
    """Three boxes with no overlap should keep all"""
    detections = [
        ([0, 0, 10, 10], 0.9),
        ([20, 20, 30, 30], 0.8),
        ([40, 40, 50, 50], 0.7),
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.5)
    
    assert len(kept) == 3

def test_nms_complete_overlap():
    """Two identical boxes should keep only highest score"""
    highest_detection = ([0, 0, 10, 10], 0.9)
    detections = [
        highest_detection,
        ([0, 0, 10, 10], 0.8),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.5)
    
    assert len(kept) == 1
    assert kept[0] == highest_detection 

def test_nms_partial_overlap_below_threshold():
    """Overlapping boxes below threshold should both be kept"""
    detections = [
        ([0, 0, 10, 10], 0.9),
        ([8, 8, 18, 18], 0.8),
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.5)
    
    assert len(kept) == 2

def test_nms_partial_overlap_above_threshold():
    """Overlapping boxes above threshold should suppress lower score"""
    highest_detection = ([0, 0, 10, 10], 0.9)
    detections = [
        highest_detection
        ([5, 5, 15, 15], 0.8),
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.1)
    
    assert len(kept) == 1
    assert kept[0] == highest_detection 

def test_nms_multiple_groups():
    """Multiple groups of overlapping boxes"""
    detections = [
        ([0, 0, 10, 10], 0.95),    # Group 1, best
        ([2, 2, 12, 12], 0.90),    # Group 1, overlaps first
        ([50, 50, 60, 60], 0.85),  # Group 2, best
        ([52, 52, 62, 62], 0.80),  # Group 2, overlaps third
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.3)
    
    # Should keep one from each group
    assert len(kept) == 2
    assert kept[0] == ([0, 0, 10, 10], 0.95)
    assert kept[1] == ([50, 50, 60, 60], 0.85)

def test_nms_already_sorted():
    """Input already sorted by score (descending)"""
    detections = [
        ([0, 0, 10, 10], 0.9),
        ([5, 5, 15, 15], 0.8),
        ([20, 20, 30, 30], 0.7),
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.1)
    
    # First and third should be kept (second overlaps first)
    assert len(kept) == 2
    assert kept[0] == ([0, 0, 10, 10], 0.9)
    assert kept[1] == ([20, 20, 30, 30], 0.7)

def test_nms_unsorted_input():
    """Input not sorted - algorithm should handle it"""
    detections = [
        ([0, 0, 10, 10], 0.7),     # Low score but listed first
        ([5, 5, 15, 15], 0.9), 
        ([20, 20, 30, 30], 0.8),
    ]
    
    kept = non_max_suppression(detections, iou_threshold=0.1) 
    
    # Should keep the 0.9 and 0.8 boxes (0.7 suppressed)
    assert len(kept) == 2
    assert kept[0] == ([5, 5, 15, 15], 0.9)
    assert kept[1] == ([20, 20, 30, 30], 0.8)