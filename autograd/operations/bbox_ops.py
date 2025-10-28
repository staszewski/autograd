def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] - first box
        box2: [x1, y1, x2, y2] - second box
    
    Returns:
        float: IoU value between 0 and 1
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_height * inter_width

    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = area1 + area2 - inter_area

    iou = inter_area / union

    return iou

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of (box, score) tuples
                   where box = [x1, y1, x2, y2]
                   and score = confidence score (float)
        iou_threshold: IoU threshold for suppression (0.0 to 1.0)
    
    Returns:
        List of (box, score) tuples after NMS
    """
    sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
    keep = []

    while sorted_detections:
        highest_detection = sorted_detections.pop(0)
        keep.append(highest_detection)

        i = 0
        while i < len(sorted_detections):
            current_box = sorted_detections[i][0]
            highest_box = highest_detection[0]

            iou = compute_iou(highest_box, current_box)

            if iou > iou_threshold:
                del sorted_detections[i]
            else:
                i += 1

    return keep