import numpy as np
import matplotlib.pyplot as plt
from autograd.drone_problems.pursuit import pure_pursuit

class SimpleDrone:
    """Simplified 2D drone (easier than full Drone class)"""
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
        """Move drone based on current velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.path_x.append(self.x)
        self.path_y.append(self.y)
    
    @property
    def pos(self):
        return (self.x, self.y)
    
    @property
    def vel(self):
        return (self.vx, self.vy)


def run_interception_sim(defender_speed=5.0, threat_speed=2.0, max_time=30.0,
                        threat_start=(50, 50), threat_direction=(-1, -1)):
    """
    Run interception simulation.
    
    Args:
        defender_speed: Speed of defender drone
        threat_speed: Speed of threat drone
        max_time: Maximum simulation time (seconds)
        threat_start: Starting position of threat (x, y)
        threat_direction: Direction threat flies (dx, dy) - will be normalized
    
    Returns:
        defender: Defender drone object with trajectory
        threat: Threat drone object with trajectory
        intercepted: Whether interception succeeded
        time_to_intercept: Time taken (or None)
    """
    defender = SimpleDrone(x=0, y=0, speed=defender_speed)
    threat = SimpleDrone(x=threat_start[0], y=threat_start[1], speed=threat_speed)
    
    direction_array = np.array(threat_direction)
    direction_norm = direction_array / np.linalg.norm(direction_array)
    threat.set_velocity(direction_norm[0] * threat_speed,
                       direction_norm[1] * threat_speed)
    
    dt = 0.1 
    intercept_distance = 2.0
    intercepted = False
    time_to_intercept = None
    t = 0
    
    print("Starting simulation...")
    print(f"Defender: speed={defender_speed}, start={defender.pos}")
    print(f"Threat:   speed={threat_speed}, start={threat.pos}, vel={threat.vel}")
    print(f"Threat direction: {threat_direction} (normalized: {direction_norm})")
    print("-" * 60)
    
    while t < max_time:
        pursuit_vel = pure_pursuit(
            defender.pos,
            defender.speed,
            threat.pos,
            threat.vel,
            lookahead_time=1.0
        )
        
        defender.set_velocity(pursuit_vel[0], pursuit_vel[1])
        
        defender.update(dt)
        threat.update(dt)
        
        distance = np.sqrt((defender.x - threat.x)**2 + (defender.y - threat.y)**2)
        
        if int(t * 10) % 10 == 0:
            print(f"t={t:5.1f}s: distance={distance:6.2f}, defender=({defender.x:.1f},{defender.y:.1f}), threat=({threat.x:.1f},{threat.y:.1f})")
        
        if distance < intercept_distance:
            intercepted = True
            time_to_intercept = t
            print("-" * 60)
            print(f"✅ INTERCEPTED at t={t:.2f}s, distance={distance:.2f}")
            break
        
        t += dt
    
    if not intercepted:
        print("-" * 60)
        print(f"❌ No interception within {max_time}s")
        final_distance = np.sqrt((defender.x - threat.x)**2 + (defender.y - threat.y)**2)
        print(f"Final distance: {final_distance:.2f}")
    
    return defender, threat, intercepted, time_to_intercept


def animate_interception(defender, threat, intercepted, time_to_intercept, save_path=None):
    """Create animated visualization of interception"""
    from matplotlib.animation import FuncAnimation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    total_frames = len(defender.path_x)
    if intercepted:
        intercept_frame = int(time_to_intercept * 10)
        end_frame = min(intercept_frame + 20, total_frames)
    else:
        end_frame = total_frames
    
    defender_line, = ax1.plot([], [], 'b-', linewidth=2, label='Defender', alpha=0.6)
    threat_line, = ax1.plot([], [], 'r-', linewidth=2, label='Threat', alpha=0.6)
    defender_dot, = ax1.plot([], [], 'bo', markersize=12, markeredgewidth=2, 
                            markeredgecolor='darkblue')
    threat_dot, = ax1.plot([], [], 'rs', markersize=12, markeredgewidth=2,
                          markeredgecolor='darkred')
    
    ax1.plot(defender.path_x[0], defender.path_y[0], 'go', markersize=10, 
            label='Start', alpha=0.5)
    ax1.plot(threat.path_x[0], threat.path_y[0], 'mo', markersize=10, alpha=0.5)
    
    all_x = defender.path_x[:end_frame] + threat.path_x[:end_frame]
    all_y = defender.path_y[:end_frame] + threat.path_y[:end_frame]
    margin = 5
    ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax1.set_xlabel('X Position', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position', fontsize=13, fontweight='bold')
    ax1.set_title('Live Pursuit', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        fontsize=12, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlim(0, end_frame * 0.1)
    distances = [np.sqrt((defender.path_x[i] - threat.path_x[i])**2 +
                        (defender.path_y[i] - threat.path_y[i])**2)
                for i in range(end_frame)]
    ax2.set_ylim(0, max(distances) * 1.1)
    
    distance_line, = ax2.plot([], [], 'g-', linewidth=2.5)
    distance_dot, = ax2.plot([], [], 'go', markersize=10)
    ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=2,
               alpha=0.7, label='Intercept Threshold')
    
    if intercepted:
        ax2.axvline(x=time_to_intercept, color='orange', linestyle='--',
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Distance', fontsize=13, fontweight='bold')
    ax2.set_title('Distance Over Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    status = "SUCCESS" if intercepted else "FAILED"
    fig.suptitle(f'Pure Pursuit - {status}',
                fontsize=17, fontweight='bold', y=0.98)
    
    def init():
        """Initialize animation"""
        defender_line.set_data([], [])
        threat_line.set_data([], [])
        defender_dot.set_data([], [])
        threat_dot.set_data([], [])
        distance_line.set_data([], [])
        distance_dot.set_data([], [])
        time_text.set_text('')
        return defender_line, threat_line, defender_dot, threat_dot, distance_line, distance_dot, time_text
    
    def update(frame):
        """Update animation frame"""
        defender_line.set_data(defender.path_x[:frame+1], defender.path_y[:frame+1])
        threat_line.set_data(threat.path_x[:frame+1], threat.path_y[:frame+1])
        
        defender_dot.set_data([defender.path_x[frame]], [defender.path_y[frame]])
        threat_dot.set_data([threat.path_x[frame]], [threat.path_y[frame]])
        
        times = np.arange(frame + 1) * 0.1
        dists = distances[:frame+1]
        distance_line.set_data(times, dists)
        distance_dot.set_data([times[-1]], [dists[-1]])
        
        current_time = frame * 0.1
        current_dist = distances[frame]
        time_text.set_text(f'Time: {current_time:.1f}s\nDistance: {current_dist:.2f}')
        
        if intercepted and frame >= int(time_to_intercept * 10):
            time_text.set_text(f'INTERCEPTED!\nTime: {time_to_intercept:.1f}s')
        
        return defender_line, threat_line, defender_dot, threat_dot, distance_line, distance_dot, time_text
    
    anim = FuncAnimation(fig, update, init_func=init, frames=end_frame,
                        interval=50, blit=True, repeat=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20, dpi=100)
        print(f"✅ Saved to {save_path}")
    plt.show()
    
    return anim 

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  PURE PURSUIT INTERCEPTION SIMULATION")
    print("="*70 + "\n")
    
    # Scenario 1: Threat approaching (easy)
    print("\n" + "="*70)
    print("  Scenario 1: Threat Approaching Defender (Easy)")
    print("="*70)
    defender, threat, intercepted, time_to_intercept = run_interception_sim(
        defender_speed=5.0,
        threat_speed=2.0,
        threat_start=(50, 50),
        threat_direction=(-1, -1),
        max_time=30.0
    )
    anim1 = animate_interception(defender, threat, intercepted, time_to_intercept, save_path="scenario1_approaching.gif")
    
    # Scenario 2: Threat fleeing (harder)
    print("\n" + "="*70)
    print("  Scenario 2: Threat Fleeing from Defender (Harder)")
    print("="*70)
    defender, threat, intercepted, time_to_intercept = run_interception_sim(
        defender_speed=5.0,
        threat_speed=2.0,
        threat_start=(20, 20),
        threat_direction=(1, 1),
        max_time=30.0
    )
    anim2 = animate_interception(defender, threat, intercepted, time_to_intercept, save_path="scenario2_fleeing.gif")
    
    # Scenario 3: Threat too fast (impossible)
    print("\n" + "="*70)
    print("  Scenario 3: Threat Faster Than Defender (Impossible)")
    print("="*70)
    defender, threat, intercepted, time_to_intercept = run_interception_sim(
        defender_speed=2.0,
        threat_speed=5.0,
        threat_start=(20, 20),
        threat_direction=(1, 1),
        max_time=30.0
    )
    anim3 = animate_interception(defender, threat, intercepted, time_to_intercept, save_path="scenario3_impossible.gif")
    
    print("\n" + "="*70)
    print("  All Scenarios Complete!")
    print("="*70 + "\n")