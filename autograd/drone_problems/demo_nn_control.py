import numpy as np
from autograd.drone_problems.drone_physics import Drone
from autograd.loss import MSELoss
from autograd.mlp import MLP
from autograd.tensor import Tensor

def calculate_simple_control(x, y, target_x, target_y):
    dx = target_x - x
    dy = target_y - y
    
    
    if dy > 0:  
        thrust = 0.8
    else:
        thrust = 0.4
    
    if dx > 0:
        torque = 0.3
    else:
        torque = -0.3
        
    return thrust, torque

def generate_training_data(num_samples=1000):
    states = []
    controls = []
    
    for i in range(num_samples):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5) 
        vx = np.random.uniform(-2, 2)
        vy = np.random.uniform(-2, 2)
        angle = np.random.uniform(0, 2*np.pi)
        
        target_x = np.random.uniform(-5, 5)
        target_y = np.random.uniform(-5, 5)
        
        state = [x, y, vx, vy, angle, target_x, target_y]
        states.append(state)
        
        thrust, torque = calculate_simple_control(x, y, target_x, target_y)
        controls.append([thrust, torque])
        
    
    states_array = np.array(states).T
    controls_array = np.array(controls).T

    X = Tensor(states_array, requires_grad=False)
    Y = Tensor(controls_array, requires_grad=False)

    return X, Y

def train_network(mlp, X, Y, learning_rate, epochs=1000, verbose=False):
    for epoch in range(epochs):
        predictions = mlp(X)
        loss = MSELoss.apply(predictions, Y)

        for param in mlp.parameters():
            param.zero_grad()
        
        loss.backward()
        
        for param in mlp.parameters():
            param._data -= learning_rate * param._grad
            
        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
    
    return mlp(X)

def test_ai_pilot(mlp):
    drone = Drone(x=0, y=0, vx=0, vy=0, angle=0.3)
    target_x, target_y = 2, 3 

    print(f"Mission: reach target ({target_x}, {target_y}) from (0, 0)")

    control_steps = 50
    for step in range(control_steps):
        x, y, vx, vy, angle = drone.get_state()
        state = [x, y, vx, vy, angle, target_x, target_y]

        state_tensor = Tensor(np.array(state).reshape(7, 1), requires_grad=False)
        controls = mlp(state_tensor)
        thrust, torque = controls.data[0, 0], controls.data[1, 0]

        drone.update(thrust, torque)

        distance = ((x - target_x)**2 + (y - target_y)**2)**0.5
        print(f"Step {step}: Pos=({x:.2f},{y:.2f}), Distance={distance:.2f}, Controls=({thrust:.2f},{torque:.2f})")
        if distance < 0.5: 
            print("Target reached")
            break



if __name__ == "__main__":
    states, controls = generate_training_data()
    mlp = MLP(input_size=7, hidden_size=10, output_size=2, activation="tanh")
    final_preds = train_network(mlp, states, controls, learning_rate=0.01, epochs=10000, verbose=True)
    print('AI Pilot:')
    test_ai_pilot(mlp)