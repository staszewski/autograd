from autograd.operations.avg_pool_2d_operation import AvgPool2dOperation
from autograd.tensor import Tensor
import numpy as np

def test_avg_pool_forward():
    input = Tensor([[1, 3], [2, 4]])
    
    result = AvgPool2dOperation.apply(input)
    
    expected = np.array([[2.5]])
    
    assert np.array_equal(result.data, expected)

def test_avg_pool_backward():
    input = Tensor([[1, 3], [2, 4]], requires_grad=True)

    result = AvgPool2dOperation.apply(input)

    result.backward()

    expected_grad = np.array([[0.25, 0.25], 
                              [0.25, 0.25]])

    assert np.array_equal(input._grad, expected_grad)