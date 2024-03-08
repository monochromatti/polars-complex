import polars as pl
from .api_extensions import ComplexMethods

@pl.api.register_expr_namespace("c")
class ComplexArithmetic(ComplexMethods):
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._expr * other
        return self._expr.complex.multiply(other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return other * self._expr
        return other.complex.multiply(self._expr)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._expr / other
        return self._expr.complex.quotient(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return other / self._expr
        return other.complex.quotient(self._expr)