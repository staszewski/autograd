from autograd.operations.bbox_ops import compute_iou

def test_iou_perfect_overlap():
    """Two identical boxes should have IoU = 1.0"""
    
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    
    iou = compute_iou(box1, box2)
    assert abs(iou - 1.0) < 1e-6

def test_iou_no_overlap():
    """Two boxes that don't touch should have IoU = 0.0"""
    
    box1 = [0, 0, 5, 5]
    box2 = [10, 10, 15, 15]
    
    iou = compute_iou(box1, box2)
    assert abs(iou - 0.0) < 1e-6

def test_iou_partial_overlap():
    """Boxes with 25% overlap"""
    
    box1 = [0, 0, 4, 4] 
    box2 = [2, 2, 6, 6]

    iou = compute_iou(box1, box2)
    expected = 4.0 / 28.0
    assert abs(iou - expected) < 1e-6

def test_iou_one_inside_another():
    """Small box completely inside large box"""
    
    box1 = [0, 0, 10, 10]
    box2 = [2, 2, 8, 8] 
    
    iou = compute_iou(box1, box2)
    expected = 36.0 / 100.0
    assert abs(iou - expected) < 1e-6

def test_iou_edge_touching():
    """Boxes touching at edge (zero overlap)"""
    
    box1 = [0, 0, 5, 5]
    box2 = [5, 0, 10, 5]
    
    iou = compute_iou(box1, box2)
    assert abs(iou - 0.0) < 1e-6