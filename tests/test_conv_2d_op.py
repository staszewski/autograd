from autograd.operations.conv_2d_operation import Conv2DOperation
from autograd.tensor import Tensor
import numpy as np

def test_conv_2d_forward():
    input_tensor = Tensor([[1,2,3], [4,5,6], [7,8,9]])
    kernel = Tensor([[1,0], [0,1]])
    
    result = Conv2DOperation.apply(input_tensor, kernel)
    
    expected = [[6, 8], [12, 14]]

    assert np.allclose(result.data, expected)

def test_conv_2d_backward():
    input_tensor = Tensor([[1,2,3], [4,5,6], [7,8,9]], requires_grad=True)
    kernel = Tensor([[1,0], [0,1]], requires_grad=True)
    
    result = Conv2DOperation.apply(input_tensor, kernel)

    result.backward()

    assert input_tensor._grad is not None
    assert kernel._grad is not None
    
    expected_grad_kernel = np.array([[12, 16], [24, 28]])
    assert np.array_equal(kernel._grad, expected_grad_kernel)

    expected_grad_input = np.array([[1, 1, 0], [1, 2, 1], [0, 1, 1]])
    assert np.array_equal(input_tensor._grad, expected_grad_input)
