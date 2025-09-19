from autograd import Tensor
from autograd.mlp import MLP
from autograd.loss import MSELoss
import numpy as np

def generate_data(n_samples=100):
    x = np.linspace(-np.pi, np.pi, n_samples)
    y = np.sin(x)

    y1 = np.sin(x)
    y2 = np.cos(x)

    X = Tensor(x.reshape(1, -1), requires_grad=False)
    Y = Tensor(np.vstack([y1, y2]), requires_grad=False)

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

def test_multi_output_nn():
    X, Y = generate_data()
    print(f"X shape: {X.data.shape}")
    print(f"Y shape: {Y.data.shape}")  
    mlp = MLP(input_size=1, hidden_size=4, output_size=2, activation="tanh")
    final_preds = train_network(mlp, X, Y, learning_rate=0.01, epochs=1000, verbose=True)
    print(f"Predictions shape: {final_preds.data.shape}")
    true_sin = Y.data[0]
    true_cos = Y.data[1]
    pred_sin = final_preds.data[0]
    pred_cos = final_preds.data[1]
    sin_error = np.mean((pred_sin - true_sin)**2)
    cos_error = np.mean((pred_cos - true_cos)**2)
    print(f"Sin loss: {sin_error:.4f}")
    print(f"Cos loss: {cos_error:.4f}")
    return final_preds


if __name__ == "__main__":
    test_multi_output_nn()