## What is a Gradient?
A gradient tells you how much the output changes when you change an input by a tiny amount.
If you have output = f(input), then:
gradient = ∂output/∂input

### Simple Examples

**Example 1: Linear function**
```python
y = x
# If you increase x by 1, y increases by 1
# So ∂y/∂x = 1
```

**Example 2: Scaled function**
```python
y = 2x
# If you increase x by 1, y increases by 2
# So ∂y/∂x = 2
```

**Example 3: Negative function**
```python
y = -x
# If you increase x by 1, y decreases by 1
# So ∂y/∂x = -1
```

### Multiple Variables

**Example: Addition**
```python
z = x + y
# ∂z/∂x = 1 (increasing x by 1 increases z by 1)
# ∂z/∂y = 1 (increasing y by 1 increases z by 1)
```

**Example: Subtraction**
```python
z = x - y
# ∂z/∂x = 1 (increasing x by 1 increases z by 1)
# ∂z/∂y = -1 (increasing y by 1 decreases z by 1)
```

### Chain Rule

When you have nested functions: `z = f(g(x))`

```python
z = f(g(x))
∂z/∂x = ∂z/∂g × ∂g/∂x
```

**Example:**
```python
# Given: z = y + 1, y = 2x
# Find ∂z/∂x:
# ∂z/∂x = ∂z/∂y × ∂y/∂x = 1 × 2 = 2
```

### Multiple Paths

When a variable affects the output through multiple paths, add all contributions:

**Example:**
```python
# Given: z = x - y, y = x
# ∂z/∂x has two contributions:
# Direct: ∂z/∂x = 1 (from z = x - y, coefficient of x is +1)
# Through y: ∂z/∂y × ∂y/∂x = (-1) × 1 = -1
# Total: ∂z/∂x = 1 + (-1) = 0
```