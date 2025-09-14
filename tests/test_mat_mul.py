from autograd import Tensor
import numpy as np

def test_mat_mul():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[5, 6], [7, 8]], requires_grad=True)
    c = a @ b

    assert np.array_equal(c.data, [[19, 22], [43, 50]])

    c.backward()
    """
    a._grad = grad_output @ b.T
        = [[1, 1], [1, 1]] @ [[5, 6], [7, 8]].T
        = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]]

    a._grad[0,0] = 1*5 + 1*6 = 5 + 6 = 11
    a._grad[0,1] = 1*7 + 1*8 = 7 + 8 = 15
    a._grad[1,0] = 1*5 + 1*6 = 5 + 6 = 11  
    a._grad[1,1] = 1*7 + 1*8 = 7 + 8 = 15

    So a._grad = [[11, 15], [11, 15]]
    """
    assert np.array_equal(a._grad, [[11, 15], [11, 15]])
    """
    b._grad = a.T @ grad_output
        = [[1, 2], [3, 4]].T @ [[1, 1], [1, 1]]
        = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]]

    b._grad[0,0] = 1*1 + 3*1 = 1 + 3 = 4
    b._grad[0,1] = 1*1 + 3*1 = 1 + 3 = 4
    b._grad[1,0] = 2*1 + 4*1 = 2 + 4 = 6
    b._grad[1,1] = 2*1 + 4*1 = 2 + 4 = 6

    So b._grad = [[4, 4], [6, 6]]
    """
    assert np.array_equal(b._grad, [[4, 4], [6, 6]])

def test_mat_mul_neural_layer():
    W = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    x = Tensor([[1], [2], [3]], requires_grad=True)

    assert W.data.shape == (2, 3)
    assert x.data.shape == (3, 1)

    y = W @ x 

    assert y.data.shape == (2, 1)
    assert np.array_equal(y.data, [[14], [32]])

def test_mat_mul_with_relu():
    W = Tensor([[-1, 2]], requires_grad=True)  # 1x2
    x = Tensor([[1], [1]], requires_grad=True) # 2x1
    
    linear = W @ x 
    output = linear.relu() 
    output.backward()
    
    assert np.array_equal(output.data, [[1]])
    assert np.array_equal(W._grad, [[1, 1]])
    """
    W._grad = [[1, 1]]
    x._grad = [[-1], [2]]
    """
    assert np.array_equal(x._grad, [[-1], [2]])

