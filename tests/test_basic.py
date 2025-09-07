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
    assert b._grad == 1.0
    assert c._grad == 1.0
    assert d._grad == 1.0
    assert e._grad == 1.0
    assert f._grad == 1.0

def test_subtraction():
    a = Tensor(5.0, requires_grad=True)

    b = a - 3 
    c = b - 2 

    c.backward()

    assert a._grad == 1.0
    assert b._grad == 1.0
    assert c._grad == 1.0

def test_subtration_simple():
    x = Tensor(5.0, requires_grad=True)
    y = Tensor(5.0, requires_grad=True)

    z = x - y 

    z.backward()

    assert x._grad == 1.0
    assert y._grad == -1.0
    assert z._grad == 1.0

def test_subtraction_multiple_times():
    a = Tensor(5.0, requires_grad=True)
    b = a - 3
    c = a - 2
    d = a - 1
    e = a - 10

    f = b + c + d + e

    f.backward()

    assert a._grad == 4.0
    assert b._grad == 1.0
    assert c._grad == 1.0
    assert d._grad == 1.0
    assert e._grad == 1.0
    assert f._grad == 1.0