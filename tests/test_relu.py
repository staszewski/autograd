from autograd import Tensor
import numpy as np

def test_relu():
    x = Tensor([-1, 2, -3, 4], requires_grad=True)
    y = x.relu()

    y.backward()

    assert np.array_equal(x._grad, [0, 1, 0, 1]) 

def test_relu_zero_input():
    x = Tensor([0, 0, 0, 0], requires_grad=True)
    y = x.relu()

    y.backward()

    assert np.array_equal(y.data, [0, 0, 0, 0])
    assert np.array_equal(x._grad, [0, 0, 0, 0])

def test_relu_chain_rule():
    x = Tensor([-2, 1, -1, 3], requires_grad=True)
    y = x.relu()  # [0, 1, 0, 3]
    z = y * 2     # [0, 2, 0, 6]
    z.backward()
    
    expected_grad = [0, 2, 0, 2]
    assert np.array_equal(x._grad, expected_grad)