"""
Educational Autograd Engine

A simple implementation of automatic differentiation for learning purposes.
This engine demonstrates the core concepts behind modern deep learning frameworks.
"""

from .tensor import Tensor
from .context import Context
from .operation import Operation
from .arithmetic import AddOperation, SubOperation

__version__ = "0.1.0"
__all__ = ["Tensor", "Context", "Operation", "AddOperation", "SubOperation"]
