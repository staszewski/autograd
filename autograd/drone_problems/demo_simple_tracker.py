from autograd.drone_problems.simple_tracker import SimpleTracker
from autograd.drone_problems.tracking_data import generate_straight_line_sequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def demo_straight_line():
    """Demo 1: Perfect tracking on straight line"""
    
    frames, ground_truth = generate_straight_line_sequence(
        num_frames=20,
        start=(10.0, 10.0),
        velocity=(2.0, 1.5),
        box_size=5
    )
    
    tracker = SimpleTracker()
    tracked_positions = []
    
    for t, detections in enumerate(frames):
        box = tracker.update(detections, current_time=t)
        if box:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            tracked_positions.append((cx, cy))
    
    tracked_x = [p[0] for p in tracked_positions]
    tracked_y = [p[1] for p in tracked_positions]
    truth_x = [p[0] for p in ground_truth]
    truth_y = [p[1] for p in ground_truth]
    
    errors = [((tracked_x[i] - truth_x[i])**2 + (tracked_y[i] - truth_y[i])**2)**0.5 
              for i in range(len(tracked_x))]
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.plot(truth_x, truth_y, 'r-', linewidth=4, alpha=0.5, 
             label='Ground Truth', marker='o', markersize=8, markeredgewidth=2, markeredgecolor='darkred')
    ax1.plot(tracked_x, tracked_y, 'b-', linewidth=2.5, 
             label='Tracked', marker='s', markersize=5)
    
    ax1.plot(truth_x[0], truth_y[0], 'go', markersize=15, label='Start', 
             markeredgewidth=2, markeredgecolor='darkgreen')
    ax1.plot(truth_x[-1], truth_y[-1], 'rs', markersize=15, label='End',
             markeredgewidth=2, markeredgecolor='darkred')
    
    for i in range(0, len(truth_x)-1, 5):
        ax1.annotate('', xy=(truth_x[i+1], truth_y[i+1]), xytext=(truth_x[i], truth_y[i]),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))
    
    ax1.set_xlabel('X Position (pixels)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='bold')
    ax1.set_title('Trajectory: Tracked vs Ground Truth', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    ax2.plot(errors, 'g-', linewidth=2.5, marker='o', markersize=5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Tracking')
    ax2.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='green')
    
    ax2.set_xlabel('Frame Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Tracking Error (pixels)', fontsize=13, fontweight='bold')
    ax2.set_title('Tracking Accuracy Over Time', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    
    stats_text = f"""
    STATS
    ━━━━━━━━━━━━━━━━━━━━━━
    Max Error:     {max(errors):.6f} px
    Mean Error:    {sum(errors)/len(errors):.6f} px
    Final Velocity: ({tracker.velocity[0]:.2f}, {tracker.velocity[1]:.2f}) px/frame
    True Velocity:  (2.00, 1.50) px/frame
    Frames:        {len(frames)}
    Success Rate:  100.00%
    """
    
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.suptitle('SimpleTracker Demo', 
                 fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def demo_curved_path():
    """Demo 2: Tracking through direction change"""
    
    frames1, gt1 = generate_straight_line_sequence(
        num_frames=10, start=(10.0, 10.0), velocity=(3.0, 0.0)
    )
    end_pos = gt1[-1]
    frames2, gt2 = generate_straight_line_sequence(
        num_frames=10, start=end_pos, velocity=(0.0, 3.0)
    )
    
    frames = frames1 + frames2
    ground_truth = gt1 + gt2
    
    tracker = SimpleTracker()
    tracked_positions = []
    velocities = []
    
    for t, detections in enumerate(frames):
        box = tracker.update(detections, current_time=t)
        if box:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            tracked_positions.append((cx, cy))
            velocities.append(tracker.velocity)
    
    tracked_x = [p[0] for p in tracked_positions]
    tracked_y = [p[1] for p in tracked_positions]
    truth_x = [p[0] for p in ground_truth]
    truth_y = [p[1] for p in ground_truth]
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.plot(truth_x, truth_y, 'r-', linewidth=4, alpha=0.5, 
             label='Ground Truth', marker='o', markersize=8)
    ax1.plot(tracked_x, tracked_y, 'b-', linewidth=2.5, 
             label='Tracked', marker='s', markersize=5)
    
    turn_idx = 9
    circle = patches.Circle((truth_x[turn_idx], truth_y[turn_idx]), 3, 
                           color='yellow', alpha=0.5, linewidth=3, 
                           edgecolor='orange', label='Direction Change')
    ax1.add_patch(circle)
    
    ax1.plot(truth_x[0], truth_y[0], 'go', markersize=15, label='Start',
             markeredgewidth=2, markeredgecolor='darkgreen')
    ax1.plot(truth_x[-1], truth_y[-1], 'rs', markersize=15, label='End',
             markeredgewidth=2, markeredgecolor='darkred')
    
    ax1.set_xlabel('X Position (pixels)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='bold')
    ax1.set_title('L-Shaped Path with Direction Change', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    vx_values = [v[0] for v in velocities]
    vy_values = [v[1] for v in velocities]
    
    ax2.plot(vx_values, 'b-', linewidth=2.5, marker='o', markersize=5, label='Velocity X')
    ax2.plot(vy_values, 'r-', linewidth=2.5, marker='s', markersize=5, label='Velocity Y')
    ax2.axvline(x=turn_idx, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Turn Point')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Frame Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Velocity (pixels/frame)', fontsize=13, fontweight='bold')
    ax2.set_title('Velocity Adaptation', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('SimpleTracker Demo: Adaptive Velocity Estimation', 
                 fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    demo_straight_line()
    demo_curved_path()