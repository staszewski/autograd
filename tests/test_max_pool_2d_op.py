from autograd.operations.max_pool_2d_operation import MaxPool2dOperation
from autograd.tensor import Tensor
import numpy as np

def test_max_pool_forward():
    input = Tensor([[1,3], [2,4]])

    result = MaxPool2dOperation.apply(input)

    expected = np.array([[4]])

    assert np.array_equal(result.data, expected)

def test_max_pool_backward():
    input = Tensor([[1,3], [2,4]], requires_grad=True)

    result = MaxPool2dOperation.apply(input)

    result.backward()

    expected_grad = np.array([[0.0, 0.0], [0.0, 1.0]])

    assert np.array_equal(input._grad, expected_grad)
