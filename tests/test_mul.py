from autograd import Tensor

def test_multiplication():
    """Test basic multiplication operation"""
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)
    
    c = a * b
    assert c.data == 12.0
    
    c.backward()
    
    assert a._grad == 4.0
    assert b._grad == 3.0

def test_multiplication_chain_1():
    """Test multiplication in a chain: z = (a * b) + c"""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = Tensor(4.0, requires_grad=True)
    
    # z = (a * b) + c = (2 * 3) + 4 = 10
    temp = a * b
    z = temp + c
    
    assert z.data == 10.0
    
    z.backward()
    
    # Chain rule: ∂z/∂a = ∂z/∂temp * ∂temp/∂a = 1 * b = 3
    # ∂z/∂b = ∂z/∂temp * ∂temp/∂b = 1 * a = 2  
    # ∂z/∂c = 1
    assert a._grad == 3.0
    assert b._grad == 2.0
    assert c._grad == 1.0

def test_multiplication_chain_2():
    """Test multiplication in a chain: z = (a * b) + (a * c)"""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = Tensor(4.0, requires_grad=True)
    
    # z = (a * b) + (a * c) = (2 * 3) + (2 * 4) = 14
    temp1 = a * b
    temp2 = a * c
    z = temp1 + temp2

    assert z.data == 14.0

    z.backward()

    assert a._grad == 7.0
    assert b._grad == 2.0
    assert c._grad == 2.0