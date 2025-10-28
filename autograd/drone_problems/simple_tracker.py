import math

class SimpleTracker:
    def __init__(self):
        self.last_box = None
        self.last_time = 0
        self.velocity = (0, 0)
        
    def update(self, detections, current_time):
        """
        Update tracker with new detections.
        
        Args:
            detections: [(box, score), ...] from detector
            current_time: frame number or timestamp
            
        Returns:
            tracked_box or None if lost
        """
        if not detections:
            self.last_time = current_time
            return None

        if self.last_box is None:
            self.last_box, _ = detections[0]
            self.last_time = current_time
            self.velocity = (0.0, 0.0)
            return self.last_box

        closest_detection = min(detections, key=lambda det: self._compute_distance(self.last_box, det[0]))
        new_box, _ = closest_detection

        dt = current_time - self.last_time
        if dt > 0:
            old_cx = (self.last_box[0] + self.last_box[2]) / 2
            old_cy = (self.last_box[1] + self.last_box[3]) / 2
            new_cx = (new_box[0] + new_box[2]) / 2
            new_cy = (new_box[1] + new_box[3]) / 2

            vx = (new_cx - old_cx) / dt
            vy = (new_cy - old_cy) / dt
            self.velocity = (vx, vy)
        
        self.last_box = new_box
        self.last_time = current_time

        return self.last_box

    
    def _compute_distance(self, box1, box2):
        """Compute Euclidean distance between box centers"""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
    
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2

        dx = cx1 - cx2
        dy = cy1 - cy2
    
        distance = math.sqrt(dx ** 2 + dy ** 2) 
    
        return distance