## What is a Gradient?
A gradient tells you how much the output changes when you change an input by a tiny amount.
If you have output = f(input), then:
gradient = ∂output/∂input

## Math examples
Simple Examples:
Example 1: y = x
If you increase x by 1, y increases by 1
So ∂y/∂x = 1
Example 2: y = 2x
If you increase x by 1, y increases by 2
So ∂y/∂x = 2
Example 3: y = -x
If you increase x by 1, y decreases by 1
So ∂y/∂x = -1
Multiple Variables:
Example: z = x + y
∂z/∂x = 1 (increasing x by 1 increases z by 1)
∂z/∂y = 1 (increasing y by 1 increases z by 1)
Example: z = x - y
∂z/∂x = 1 (increasing x by 1 increases z by 1)
∂z/∂y = -1 (increasing y by 1 decreases z by 1)
Chain Rule:
When you have nested functions: z = f(g(x))
∂z/∂x = ∂z/∂g × ∂g/∂x
Example:
)
To find ∂z/∂x:
∂z/∂x = ∂z/∂y × ∂y/∂x = 1 × 2 = 2
Multiple Paths:
When a variable affects the output through multiple paths, add all contributions:
Example:
)
∂z/∂x has two contributions:
Direct: ∂z/∂x = 1 (from z = x - y, coefficient of x is +1)
Through y: ∂z/∂y × ∂y/∂x = (-1) × 1 = -1
Total: ∂z/∂x = 1 + (-1) = 0