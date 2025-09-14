from autograd import Tensor

def test_division():
    a = Tensor(10.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)

    c = a / b
    assert c.data == 5.0

    c.backward()

    assert a._grad == 0.5
    assert b._grad == -2.5

def test_division_with_constant_numerator():
    b = Tensor(2.0, requires_grad=True)
    k = 8.0
    c = k / b

    assert c.data == 4.0

    c.backward()
    
    assert b._grad == -2.0

def test_division_with_squared_denominator():
    a = Tensor(4.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)

    c = a / b**2

    assert c.data == 1.0

    c.backward()
    
    assert a._grad == 0.25
    assert b._grad == -1.0