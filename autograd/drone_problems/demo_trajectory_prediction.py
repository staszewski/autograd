from typing import List,Tuple
from autograd.loss import MSELoss
from autograd.tensor import Tensor
import numpy as np
from autograd.arithmetic import TanhOperation

class RNNWeights:
    def __init__(self, input_size=2, hidden_size=4, output_size=2):
        self.input = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.hidden = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.output = Tensor(np.random.randn(hidden_size, output_size) * 0.1, requires_grad=True)

def rnn_step(input_t, hidden_prev, weights):
    activation_input = input_t @ weights.input + hidden_prev @ weights.hidden + weights.bias
    hidden = TanhOperation.apply(activation_input)
    return hidden

positions = [(8, 13), (10, 14), (12, 15), (14, 16), (16, 17), (18, 18)]

def convert_pos_to_tensor(positions: List[Tuple]):
    tensors = []
    for pos in positions:
        x, y = pos
        tensors.append(Tensor([[x, y]]))
    return tensors

def convert_to_velocities(positions):
    velocities = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        velocities.append((dx, dy))
    return velocities

def create_training_data(tensor_positions):
    input_sequence = tensor_positions[:-1]
    target = tensor_positions[-1]

    return input_sequence, target

def create_velocity_training_data(positions):
    velocities = convert_to_velocities(positions)
    velocity_tensors = convert_pos_to_tensor(velocities) 
    
    input_sequence = velocity_tensors[:-1]
    target = velocity_tensors[-1]
    
    return input_sequence, target

def generate_multiple_sequences():
    sequences = []
    
    starting_points = [(0, 0), (10, 15), (5, 8), (20, 3), (15, 12)]
    
    for start_x, start_y in starting_points:
        sequence = []
        for i in range(6): 
            x = start_x + i * 2
            y = start_y + i * 1
            sequence.append((x, y))
        sequences.append(sequence)
    
    return sequences

def forward_sequence(input_sequence, weights):
    hidden = Tensor(np.zeros((1, 4)), requires_grad=True)

    for input_t in input_sequence:
        hidden = rnn_step(input_t, hidden, weights)

    return hidden

def predict_position(hidden_state, output_weights):
    result = hidden_state @ output_weights 
    return result

def train_velocity_network(epochs=100, learning_rate=0.01):
    sequences = generate_multiple_sequences()
    rnn_weights = RNNWeights()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for sequence in sequences:
            input_sequence, target = create_velocity_training_data(sequence)
            
            hidden = forward_sequence(input_sequence, rnn_weights)
            prediction = predict_position(hidden, rnn_weights.output)
            loss = MSELoss.apply(prediction, target)
            
            total_loss += loss.data
            
            rnn_weights.input.zero_grad()
            rnn_weights.hidden.zero_grad() 
            rnn_weights.bias.zero_grad()
            rnn_weights.output.zero_grad()
            
            loss.backward()
            
            rnn_weights.input._data -= learning_rate * rnn_weights.input._grad
            rnn_weights.hidden._data -= learning_rate * rnn_weights.hidden._grad
            rnn_weights.bias._data -= learning_rate * rnn_weights.bias._grad
            rnn_weights.output._data -= learning_rate * rnn_weights.output._grad
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss/len(sequences):.4f}")
    
    return rnn_weights

def test_velocity_rnn(trained_weights):
    test_positions = [(10, 15), (12, 16), (14, 17), (16, 18)]
    
    velocities = convert_to_velocities(test_positions)
    print(f"Position sequence: {test_positions}")
    print(f"Velocity sequence: {velocities}")
    
    input_sequence, target = create_velocity_training_data(test_positions)
    
    hidden = forward_sequence(input_sequence, trained_weights)
    velocity_prediction = predict_position(hidden, trained_weights.output)
    
    print(f"Target velocity: {target.data}")
    print(f"Predicted velocity: {velocity_prediction.data}")
    print(f"Error: {abs(velocity_prediction.data - target.data)}")
    
    last_position = test_positions[-1]
    predicted_next_pos = (
        last_position[0] + velocity_prediction.data[0][0],
        last_position[1] + velocity_prediction.data[0][1]
    )
    expected_next_pos = (18, 19) 
    
    print(f"Last known position: {last_position}")
    print(f"Expected next position: {expected_next_pos}")
    print(f"Predicted next position: ({predicted_next_pos[0]:.2f}, {predicted_next_pos[1]:.2f})")
    
    print(f"\nPattern recognition")
    new_test_positions = [(0, 0), (2, 1), (4, 2), (6, 3)]
    new_input_sequence, new_target = create_velocity_training_data(new_test_positions)
    
    new_hidden = forward_sequence(new_input_sequence, trained_weights)
    new_velocity_pred = predict_position(new_hidden, trained_weights.output)
    
    print(f"New sequence: {new_test_positions}")
    print(f"Should predict velocity: (2, 1)")
    print(f"Actually predicted: ({new_velocity_pred.data[0][0]:.2f}, {new_velocity_pred.data[0][1]:.2f})")
    

def demo_trajectory_prediction():
    weights = train_velocity_network()
    test_velocity_rnn(weights)

if __name__ == "__main__":
    demo_trajectory_prediction()