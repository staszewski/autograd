from autograd.tensor import Tensor

def test_negation_simple():
    a = Tensor(4.0, requires_grad=True)
    b = -a
    
    b.backward()
    
    assert a._grad == -1

def test_negation_in_expression():
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)

    c = b + (-a)

    c.backward()

    assert a._grad == -1
    assert b._grad == 1
    assert c._grad == 1