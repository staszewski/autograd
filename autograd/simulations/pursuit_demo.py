"""
Demo: Defender intercepts threat using pure pursuit
"""
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
    # Initialize drones
    defender = SimpleDrone(x=0, y=0, speed=defender_speed)
    threat = SimpleDrone(x=threat_start[0], y=threat_start[1], speed=threat_speed)
    
    # Normalize and set threat direction
    direction_array = np.array(threat_direction)
    direction_norm = direction_array / np.linalg.norm(direction_array)
    threat.set_velocity(direction_norm[0] * threat_speed,
                       direction_norm[1] * threat_speed)
    
    # Simulation parameters
    dt = 0.1  # Time step (seconds)
    intercept_distance = 2.0  # Consider intercepted if within 2 units
    
    intercepted = False
    time_to_intercept = None
    t = 0
    
    print("Starting simulation...")
    print(f"Defender: speed={defender_speed}, start={defender.pos}")
    print(f"Threat:   speed={threat_speed}, start={threat.pos}, vel={threat.vel}")
    print(f"Threat direction: {threat_direction} (normalized: {direction_norm})")
    print("-" * 60)
    
    # Simulation loop
    while t < max_time:
        # Compute pursuit direction
        pursuit_vel = pure_pursuit(
            defender.pos,
            defender.speed,
            threat.pos,
            threat.vel,
            lookahead_time=1.0
        )
        
        # Update defender velocity
        defender.set_velocity(pursuit_vel[0], pursuit_vel[1])
        
        # Update both drones
        defender.update(dt)
        threat.update(dt)
        
        # Check distance
        distance = np.sqrt((defender.x - threat.x)**2 + (defender.y - threat.y)**2)
        
        # Print progress every second
        if int(t * 10) % 10 == 0:
            print(f"t={t:5.1f}s: distance={distance:6.2f}, defender=({defender.x:.1f},{defender.y:.1f}), threat=({threat.x:.1f},{threat.y:.1f})")
        
        # Check interception
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


def visualize_interception(defender, threat, intercepted, time_to_intercept):
    """Create visualization of interception"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT: Trajectory plot
    ax1.plot(defender.path_x, defender.path_y, 'b-', linewidth=2,
             label='Defender Path', marker='o', markersize=3, markevery=10)
    ax1.plot(threat.path_x, threat.path_y, 'r-', linewidth=2,
             label='Threat Path', marker='s', markersize=3, markevery=10)
    
    # Mark start positions
    ax1.plot(defender.path_x[0], defender.path_y[0], 'go', markersize=15,
             label='Defender Start', markeredgewidth=2, markeredgecolor='darkgreen')
    ax1.plot(threat.path_x[0], threat.path_y[0], 'mo', markersize=15,
             label='Threat Start', markeredgewidth=2, markeredgecolor='darkred')
    
    # Mark interception point
    if intercepted:
        ax1.plot(defender.path_x[-1], defender.path_y[-1], 'y*', markersize=25,
                label=f'Intercept ({time_to_intercept:.1f}s)',
                markeredgewidth=2, markeredgecolor='orange')
    
    ax1.set_xlabel('X Position', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position', fontsize=13, fontweight='bold')
    ax1.set_title('Pursuit Trajectories', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    # RIGHT: Distance over time
    times = np.arange(len(defender.path_x)) * 0.1
    distances = [np.sqrt((defender.path_x[i] - threat.path_x[i])**2 +
                        (defender.path_y[i] - threat.path_y[i])**2)
                for i in range(len(times))]
    
    ax2.plot(times, distances, 'g-', linewidth=2.5, marker='o', markersize=4)
    ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=2,
               alpha=0.7, label='Intercept Threshold')
    
    if intercepted:
        ax2.axvline(x=time_to_intercept, color='orange', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Intercept at {time_to_intercept:.1f}s')
    
    ax2.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Distance Between Drones', fontsize=13, fontweight='bold')
    ax2.set_title('Closing Distance Over Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    status = "SUCCESS" if intercepted else "FAILED"
    plt.suptitle(f'Pure Pursuit Interception Demo - {status}',
                fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


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
        threat_direction=(-1, -1),  # Flying toward origin
        max_time=30.0
    )
    visualize_interception(defender, threat, intercepted, time_to_intercept)
    
    # Scenario 2: Threat fleeing (harder)
    print("\n" + "="*70)
    print("  Scenario 2: Threat Fleeing from Defender (Harder)")
    print("="*70)
    defender, threat, intercepted, time_to_intercept = run_interception_sim(
        defender_speed=5.0,
        threat_speed=2.0,
        threat_start=(20, 20),
        threat_direction=(1, 1),  # Flying away from origin
        max_time=30.0
    )
    visualize_interception(defender, threat, intercepted, time_to_intercept)
    
    # Scenario 3: Threat too fast (impossible)
    print("\n" + "="*70)
    print("  Scenario 3: Threat Faster Than Defender (Impossible)")
    print("="*70)
    defender, threat, intercepted, time_to_intercept = run_interception_sim(
        defender_speed=2.0,
        threat_speed=5.0,
        threat_start=(20, 20),
        threat_direction=(1, 1),  # Fast threat escaping
        max_time=30.0
    )
    visualize_interception(defender, threat, intercepted, time_to_intercept)
    
    print("\n" + "="*70)
    print("  All Scenarios Complete!")
    print("="*70 + "\n")