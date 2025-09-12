from autograd import Tensor

def test_division():
    a = Tensor(10.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)

    c = a / b
    assert c.data == 5.0

    c.backward()

    assert a._grad == 0.5
    assert b._grad == -2.5
