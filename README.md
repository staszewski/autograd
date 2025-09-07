# Autograd

Learning repo for creating auto grad. Work in progress.

# Operations added so far
- Addition
- Subtraction
- Multiplication

##  Math

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

## Testing

```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/test_mul.py -v
```