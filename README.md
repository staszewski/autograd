# Autograd

A minimal implementation of automatic differentiation from scratch. Learn how modern deep learning frameworks compute gradients automatically.

## 🚀 Features

- **Addition**: `a + b`
- **Subtraction**: `a - b` 
- **Multiplication**: `a * b`
- **Automatic Gradients**: Computes derivatives via backpropagation
- **Scalar & Tensor Operations**: Works with both `tensor + scalar` and `scalar + tensor`

## 📖 Quick Start

```python
from autograd import Tensor

# Create tensors with gradient tracking
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

# Build expression: z = x * y + x
temp = x * y  # 6.0
z = temp + x  # 8.0

# Compute gradients automatically
z.backward()

print(f"∂z/∂x = {x._grad}")  # 4.0 (y + 1)
print(f"∂z/∂y = {y._grad}")  # 2.0 (x)
```

## 🧮 Math Behind the Magic

### Basic Derivatives
```
∂(a + b)/∂a = 1,  ∂(a + b)/∂b = 1
∂(a - b)/∂a = 1,  ∂(a - b)/∂b = -1  
∂(a * b)/∂a = b,  ∂(a * b)/∂b = a
```

### Chain Rule
For nested functions `z = f(g(x))`:
```
∂z/∂x = ∂z/∂g × ∂g/∂x
```

### Gradient Accumulation
When a variable appears multiple times, gradients add:
```python
# z = x * y + x * 2
# ∂z/∂x = y + 2  (sum of all paths)
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/test_mul.py -v
```

## 🏗️ Architecture

```
autograd/
├── tensor.py      # Tensor class with gradient tracking
├── context.py     # Saves data for backward pass  
├── operation.py   # Abstract operation base class
└── arithmetic.py  # Add, Sub, Mul implementations
```

Built with clean OOP principles for educational purposes.
```