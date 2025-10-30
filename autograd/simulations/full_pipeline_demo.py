"""
Complete Pipeline: Detection → Tracking → Pursuit → Interception

1. First: Get detection working
2. Then: Add tracking
3. Finally: Add pursuit
"""
import numpy as np
import matplotlib.pyplot as plt
from autograd.tensor import Tensor
from autograd.drone_problems.multi_target_tracker import MultiTargetDetector
from autograd.drone_problems.simple_tracker import SimpleTracker
from autograd.drone_problems.pursuit import pure_pursuit


# ============================================================================
# STEP 1: IMAGE GENERATION 
# ============================================================================

def create_threat_image(threat_x, threat_y, img_size=100, crosshair_size=5):
    """
    Create image with crosshair pattern at threat position.
    Args:
        threat_x, threat_y: Center position of threat
        img_size: Size of square image
        crosshair_size: Half-length of crosshair arms
    
    Returns:
        numpy array (img_size, img_size) with crosshair
    """
    img = np.zeros((img_size, img_size), dtype=np.float32)
    cx = int(threat_x)
    cy = int(threat_y)
    
    y_start = max(0, cy - crosshair_size)
    y_end = min(img_size, cy + crosshair_size + 1)
    
    x_start = max(0, cx - crosshair_size)
    x_end = min(img_size, cx + crosshair_size + 1)
    
    if 0 <= cx < img_size:
        img[y_start:y_end, cx] = 1.0
    
    if 0 <= cy < img_size:
        img[cy, x_start:x_end] = 1.0
    
    return img


# ============================================================================
# STEP 2: TEST DETECTION
# ============================================================================

def test_detection():
    """Test that detection works on our synthetic image"""
    print("STEP 1 TEST: Detection")
    detector = MultiTargetDetector()
    
    print("Creating image with threat at (50, 50)...")
    img = create_threat_image(50, 50)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', origin='upper')
    plt.title('Generated Threat Image (Crosshair)')
    plt.colorbar()
    plt.show()
    
    print("Running detector...")
    img_tensor = Tensor(img, requires_grad=False)
    detections = detector.detect_all_targets(img_tensor, use_nms=True)
    
    print(f"\nDetection results:")
    print(f"  Found {len(detections)} detections")
    for i, (box, score) in enumerate(detections):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        print(f"  Detection {i+1}: center=({cx:.1f}, {cy:.1f}), score={score:.3f}")
    
    print("\n✓ If you see detections near (50, 50), Step 1 works!")
    print("="*70)


# ============================================================================
# STEP 3: SIMPLE DRONES
# ============================================================================

class ThreatDrone:
    """Threat that moves and appears in images"""
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.path_x = [x]
        self.path_y = [y]
    
    def update(self, dt=0.1):
        """Move threat"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.path_x.append(self.x)
        self.path_y.append(self.y)
    
    def get_image(self):
        """Generate image showing this threat"""
        return create_threat_image(self.x, self.y)


class DefenderDrone:
    """Defender that pursues"""
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.vx = 0.0
        self.vy = 0.0
        self.path_x = [x]
        self.path_y = [y]
    
    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy
    
    def update(self, dt=0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.path_x.append(self.x)
        self.path_y.append(self.y)
    
    @property
    def pos(self):
        return (self.x, self.y)


# ============================================================================
# STEP 4: FULL PIPELINE
# ============================================================================

def run_full_pipeline():
    """
    Complete pipeline simulation WITH UNTRAINED DETECTOR
    """
    print("FULL PIPELINE: Detection → Tracking → Pursuit")
    
    detector = MultiTargetDetector()
    tracker = SimpleTracker()
    threat = ThreatDrone(x=60, y=60, vx=-1.5, vy=-1.5)
    defender = DefenderDrone(x=10, y=10, speed=6.0)
    
    dt = 0.1
    max_time = 20.0
    t = 0
    
    print(f"Initial state:")
    print(f"  Threat:   ({threat.x:.1f}, {threat.y:.1f})")
    print(f"  Defender: {defender.pos}")
    
    while t < max_time:
        # PHASE 1 - DETECTION
        img = threat.get_image()
        img_tensor = Tensor(img, requires_grad=False)
        detections = detector.detect_all_targets(img_tensor, use_nms=True, iou_threshold=0.3)
        if detections:
            # PHASE 2 - TRACKING
            tracked_box = tracker.update(detections, current_time=t)
            if tracked_box:
                tracked_cx = (tracked_box[0] + tracked_box[2]) / 2
                tracked_cy = (tracked_box[1] + tracked_box[3]) / 2
                tracked_pos = (tracked_cx, tracked_cy)
                tracked_vel = tracker.velocity
        
                # PHASE 3 - PURSUIT
                pursuit_vel = pure_pursuit(
                    defender.pos,
                    defender.speed,
                    tracked_pos,
                    tracked_vel,
                    lookahead_time=1.0
                )
                defender.set_velocity(pursuit_vel[0], pursuit_vel[1])

        defender.update(dt)
        threat.update(dt)
        
        distance = np.sqrt((defender.x - threat.x)**2 + (defender.y - threat.y)**2)
        
        if int(t * 10) % 10 == 0:
            print(f"t={t:5.1f}s: distance={distance:6.2f}")
        
        if distance < 5.0:
            print("-" * 70)
            print(f"✅ INTERCEPTED at t={t:.2f}s")
            break
        
        t += dt
    
    plt.figure(figsize=(10, 10))
    plt.plot(threat.path_x, threat.path_y, 'r-', label='Threat', linewidth=2)
    plt.plot(defender.path_x, defender.path_y, 'b-', label='Defender', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.title('Full Pipeline: Detection → Tracking → Pursuit')
    plt.show()


if __name__ == "__main__":
    print("  FULL PIPELINE DEMO - INCREMENTAL BUILD")
    run_full_pipeline()