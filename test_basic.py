#!/usr/bin/env python3

from autograd import Tensor

def test_basic_creation():
    t = Tensor(5.0, requires_grad=True)
    assert t.data == 5.0
    assert t.requires_grad == True

def test_addition():
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    
    c = a + b
    assert c.data == 5.0
    
    c.backward()

    assert a._grad == 1.0
    assert b._grad == 1.0
    assert c._grad == 1.0

def test_addition_multiple_times():
    a = Tensor(53.0, requires_grad=True)
    b = a + 3
    c = a + 4
    d = a + 5
    e = a + 10 

    f = b + c + d + e
    f.backward()

    assert a._grad == 4.0

def test_subtraction():
    a = Tensor(5.0, requires_grad=True)
    three = Tensor(3.0, requires_grad=False)
    two = Tensor(2.0, requires_grad=False)
    
    b = a - three
    c = b - two
    
    c.backward()

    assert a._grad == 1.0
    assert b._grad == 1.0

if __name__ == "__main__":
    test_basic_creation()
    test_addition()
    test_addition_multiple_times()
    test_subtraction()
    print("All tests passed!")
    