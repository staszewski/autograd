def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] - first box
        box2: [x1, y1, x2, y2] - second box
    
    Returns:
        float: IoU value between 0 and 1
    """
    # 1. Compute intersection coordinates
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    # 2. Compute intersection area (remember: can be 0!)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_height * inter_width
    # 3. Compute areas of both boxes
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    # 4. Compute union
    union = area1 + area2 - inter_area
    # 5. Return IoU
    return inter_area / union