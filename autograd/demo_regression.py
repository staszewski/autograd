from autograd import Tensor
from autograd.mlp import MLP
import numpy as np
from autograd.loss import MSELoss
import matplotlib.pyplot as plt

def generate_sin_data(n_samples=100):
    x = np.linspace(-2*np.pi, 2*np.pi, n_samples)
    y = np.sin(x)  
    
    X = Tensor(x.reshape(1, -1), requires_grad=False)
    Y = Tensor(y.reshape(1, -1), requires_grad=False)
    return X, Y

def train_network(mlp, X, Y, learning_rate, epochs=1000, verbose=False):
    loss_history = []
    for epoch in range(epochs):
        predictions = mlp(X)
        loss = MSELoss.apply(predictions, Y)
        loss_history.append(loss.data)
        
        for param in mlp.parameters():
            param.zero_grad()
        
        loss.backward()
        
        for param in mlp.parameters():
            param._data -= learning_rate * param._grad

        loss_history.append(loss.data)

        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    return loss_history

def compare_mlps():
    X, Y = generate_sin_data()
    print("Training MLP with tanh activation...")
    mlp_tanh = MLP(input_size=1, hidden_size=4, output_size=1, activation="tanh")
    loss_history_tanh = train_network(mlp_tanh, X, Y, learning_rate=0.2, epochs=1000, verbose=True)

    print("Training MLP with sigmoid activation...")
    mlp_sigmoid = MLP(input_size=1, hidden_size=4, output_size=1, activation="sigmoid")
    loss_history_sigmoid = train_network(mlp_sigmoid, X, Y, learning_rate=0.2, epochs=1000, verbose=True)
    
    print("Training MLP with relu activation...")
    mlp_relu = MLP(input_size=1, hidden_size=4, output_size=1, activation="relu")
    loss_history_relu = train_network(mlp_relu, X, Y, learning_rate=0.2, epochs=1000, verbose=True)
    
    plt.plot(loss_history_tanh, label="tanh")
    plt.plot(loss_history_sigmoid, label="sigmoid")
    plt.plot(loss_history_relu, label="relu")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_mlps()