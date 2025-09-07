from autograd.tensor import Tensor

import pytest
import math

def test_pow():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    c = a ** b
    assert c.data == 8.0

    c.backward()

    assert a._grad == 12.0
    assert b._grad == pytest.approx(8 * math.log(2))

def test_pow_square():
    a = Tensor(3.0, requires_grad=True)
    
    c = a ** 2
    assert c.data == 9.0
    
    c.backward()
    
    # ∂c/∂a = ∂/∂a(a^2) = 2a = 2 × 3.0 = 6.0
    assert a._grad == 6.0

def test_pow_cube():
    a = Tensor(3.0, requires_grad=True)
    
    c = a ** 3 
    assert c.data == 27.0
    
    c.backward()
    
    # ∂/∂a(a^2) = 3a^2 = 3 × 3^2 = 27.0
    assert a._grad == 27.0