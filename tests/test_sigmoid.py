from autograd import Tensor
import numpy as np

def test_sigmoid():
    x = Tensor([0, 2], requires_grad=True)
    y = x.sigmoid()

    y.backward()
    
    expected_grad = [0.25, 0.10499359]
    assert np.allclose(x._grad, expected_grad, atol=1e-6)


def test_sigmoid_forward():
    x = Tensor([0.0, 1.0, -1.0], requires_grad=False)
    y = x.sigmoid()
    
    expected = [0.5, 0.7311, 0.2689]  # σ(0), σ(1), σ(-1)
    assert np.allclose(y.data, expected, atol=1e-4)

def test_sigmoid_chain_rule():
    x = Tensor([1.0], requires_grad=True)
    y = x.sigmoid()  # σ(1) ≈ 0.731
    z = y * 2        # 2 * σ(1)
    z.backward()
    
    expected_grad = 2 * y.data * (1 - y.data) 
    assert np.allclose(x._grad, [expected_grad], atol=1e-3)

def test_sigmoid_large_values():
    x = Tensor([100.0, -100.0], requires_grad=True)
    y = x.sigmoid()
    
    expected = [1.0, 0.0]
    assert np.allclose(y.data, expected, atol=1e-10)
