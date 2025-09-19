from autograd.arithmetic import TanhOperation
from autograd.tensor import Tensor
import numpy as np

def test_tanh_forward():
    x = Tensor([0, 1, -1], requires_grad=False)
    y = TanhOperation.apply(x)

    expected = [0, 0.7616, -0.7616]
    assert np.allclose(y.data, expected, atol=1e-4)

def test_tanh_backward():
    x = Tensor([0, 1, -1], requires_grad=True)
    y = TanhOperation.apply(x)

    y.backward()

    # tanh'(x) = 1 - tanh²(x)
    expected_grad = [1, 0.42, 0.42]
    assert np.allclose(x._grad, expected_grad, atol=1e-4)

def test_tanh_chain_rule():
    x = Tensor([1], requires_grad=True)
    y = x.tanh()
    z = y * 3 

    z.backward()

    # dz/dx = 3 * (1 - tanh²(1))
    expected_grad = 3 * (1 - np.tanh(1) ** 2)
    assert np.allclose(x._grad, [expected_grad], atol=1e-4)
