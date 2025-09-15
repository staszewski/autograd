from autograd import Tensor
import numpy as np

def test_sigmoid():
    x = Tensor([0, 2], requires_grad=True)
    y = x.sigmoid()

    y.backward()
    
    expected_grad = [0.25, 0.10499359]
    assert np.allclose(x._grad, expected_grad, atol=1e-6)
