#!/usr/bin/env python3
"""
Essential tests for our Tensor implementation.
"""

from autograd import Tensor

def test_basic_creation():
    """Test tensor creation."""
    t = Tensor(5.0, requires_grad=True)
    assert t.data == 5.0
    assert t.requires_grad == True

def test_addition():
    """Test addition operation."""
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    
    # Forward pass
    c = a + b
    assert c.data == 5.0
    
    # Backward pass
    c.backward()
    
    # Use .item() to get scalar from 0-dimensional array
    assert a._grad.item() == 1.0
    assert b._grad.item() == 1.0

if __name__ == "__main__":
    test_basic_creation()
    test_addition()
    print("All tests passed!")
    