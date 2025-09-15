from autograd.mlp import MLP
from autograd import Tensor
import numpy as np

def test_mlp_creation():
    mlp = MLP(input_size=2, hidden_size=3, output_size=1)

    assert hasattr(mlp, 'W1')
    assert hasattr(mlp, 'b1') 
    assert hasattr(mlp, 'W2')
    assert hasattr(mlp, 'b2')

    assert mlp.W1.data.shape == (3, 2)
    assert mlp.b1.data.shape == (3, 1)
    assert mlp.W2.data.shape == (1, 3)
    assert mlp.b2.data.shape == (1, 1)

def test_mlp_gradients():
    mlp = MLP(input_size=2, hidden_size=3, output_size=1)
    x = Tensor([[1.0], [-2.0]], requires_grad=True) 

    output = mlp(x)

    assert output.data.shape == (1, 1)

    output.backward()

    for param in mlp.parameters():
        assert param._grad is not None
        assert param._grad.shape == param.data.shape
        assert param._requires_grad == True

def test_mlp_different_sizes():
    output_size = 5 
    mlp = MLP(input_size=3, hidden_size=4, output_size=output_size)
    x = Tensor([[1.0], [-2.0], [3.0]], requires_grad=True) 

    output = mlp(x)

    assert output.data.shape == (output_size, 1)