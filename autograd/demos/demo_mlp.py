from autograd import Tensor
from autograd.mlp import MLP
from autograd.loss import MSELoss
import numpy as np

def demo_mlp_xor():
    X = Tensor([[0, 0, 1, 1], [0, 1, 0, 1]], requires_grad=False)
    y = Tensor([[0, 1, 1, 0]], requires_grad=False)
    
    mlp = MLP(input_size=2, hidden_size=4, output_size=1)
    
    learning_rate = 0.2
    epochs = 500 
    
    for epoch in range(epochs):
        predictions = mlp(X)
        loss = MSELoss.apply(predictions, y)
        
        for param in mlp.parameters():
            param.zero_grad()
        
        loss.backward()
        
        if epoch % 20 == 0:
            total_grad = sum(np.abs(param._grad).sum() for param in mlp.parameters())
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}, Total |grad|: {total_grad:.6f}")
            
            params = mlp.parameters()
            print(f"  W1 grad max: {np.abs(params[0]._grad).max():.6f}")
            print(f"  b1 grad max: {np.abs(params[1]._grad).max():.6f}")
            print(f"  W2 grad max: {np.abs(params[2]._grad).max():.6f}")
            print(f"  b2 grad max: {np.abs(params[3]._grad).max():.6f}")
        
        for param in mlp.parameters():
            param._data -= learning_rate * param._grad
    
    print("\nFinal predictions:")
    final_preds = mlp(X)
    for i in range(4):
        print(f"Input: [{X.data[0, i]}, {X.data[1, i]}] -> Predicted: {final_preds.data[0, i]:.3f}, Target: {y.data[0, i]}")

if __name__ == "__main__":
    demo_mlp_xor()