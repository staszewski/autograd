#!/usr/bin/env python3

"""
This test file is very much LLM generated causse I'm afraid that I'm missing something.
"""

from autograd import Tensor


def test_complex_graph_gradients():
    """Test gradient computation in complex computation graphs"""
    # Create: f = (a + b) + (a + c) + (a + a)
    # This should give: df/da = 3, df/db = 1, df/dc = 1
    
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = Tensor(4.0, requires_grad=True)
    
    # Build computation graph: f = (a + b) + (a + c) + (a + a)
    temp1 = a + b      # 5
    temp2 = a + c      # 6
    temp3 = a + a      # 4
    f = temp1 + temp2 + temp3  # Should be: f = 15, df/da = 3
    
    f.backward()
    
    # For f = (a + b) + (a + c) + (a + a):
    # df/da = 1 + 1 + 2 = 4  (a appears 4 times in the computation)
    # df/db = 1 (b appears once)
    # df/dc = 1 (c appears once)
    
    print(f"Expected: a=4.0, b=1.0, c=1.0")
    print(f"Actual: a={a._grad}, b={b._grad}, c={c._grad}")
    
    assert abs(a._grad - 4.0) < 1e-6
    assert abs(b._grad - 1.0) < 1e-6
    assert abs(c._grad - 1.0) < 1e-6

def test_gradient_accumulation():
    """Test that gradients accumulate properly"""
    a = Tensor(2.0, requires_grad=True)
    
    # First computation: b = a + 1
    b = a + 1
    b.backward()
    first_grad = a._grad
    
    # Should be 1.0
    assert abs(first_grad - 1.0) < 1e-6

def test_scalar_operations():
    """Test that scalar operations work correctly"""
    a = Tensor(2.0, requires_grad=True)
    
    # This should work: Tensor + scalar
    b = a + 3.0
    assert b.data == 5.0
    
    # This should also work: scalar + Tensor (needs __radd__)
    c = 4.0 + a  # This will fail if __radd__ is not implemented!
    assert c.data == 6.0

def test_memory_management():
    """Test that the backward pass works for long chains"""
    a = Tensor(1.0, requires_grad=True)
    
    # Build a long chain: x = (((a + 1) + 1) + 1) + 1
    x = a
    for i in range(10):
        x = x + 1
    
    x.backward()
    
    # Gradient should be 1.0 (each addition contributes 1)
    assert abs(a._grad - 1.0) < 1e-6

def test_original_failing_case():
    """Test the original case that was failing"""
    a = Tensor(53.0, requires_grad=True)
    b = a + 3
    c = a + 4
    d = a + 5
    e = a + 10 

    f = b + c + d + e
    f.backward()

    print(f'a._grad = {a._grad} (expected 4.0)')
    # This should be 4.0 because a appears 4 times in the computation
    assert a._grad == 4.0

