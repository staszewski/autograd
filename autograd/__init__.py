"""
Educational Autograd Engine

A simple implementation of automatic differentiation for learning purposes.
"""

from .tensor import Tensor
from .context import Context
from .operation import Operation
from .arithmetic import AddOperation, SubOperation, MulOperation, PowOperation, DivOperation

__version__ = "0.1.0"
__all__ = ["Tensor", "Context", "Operation", "AddOperation", "SubOperation", "MulOperation", "PowOperation", "DivOperation"]
