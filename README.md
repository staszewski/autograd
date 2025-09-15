# Autograd

A simple educational implementation of automatic differentiation built in Python.

# Operations added so far
- Addition (with broadcasting)
- Subtraction  
- Multiplication
- Division
- Power
- Negation
- Matrix Multiplication
- **ReLU Activation Function**

# Core Features
- **Automatic Differentiation Engine**
- **Computation Graph Tracking** 
- **Context System** for gradient computation
- **Broadcasting Support** for tensor operations
- **Gradient Accumulation** for multiple paths

# Neural Network Components
- Multi-Layer Perceptron (MLP) implementation
- MSE Loss function
- XOR learning demonstration

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

### Installation
```bash
# Clone
git clone <your-repo-url>
cd autograd

# Install dependencies (uv is the preferance in 2025)
uv sync
```

### Running the Demo
```bash
# Run the XOR learning demo
uv run python autograd/demo_mlp.py
```

## Testing
- Comprehensive test suite covering all operations
- Integration tests for neural networks
- Bug detection and edge case handling
```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/test_mul.py -v
```