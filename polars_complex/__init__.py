from polars import col, struct

from .extensions import ComplexExpr

__all__ = ["ccol"]


class ComplexCol:
    def __call__(self, name, *more_names) -> ComplexExpr:
        if not more_names:
            expr = col(name).struct.rename_fields(["real", "imag"])
        elif len(more_names) == 1:
            real = col(name)
            imag = col(next(iter(more_names)))
            expr = struct(real.alias("real"), imag.alias("imag"))
        else:
            raise ValueError("Could not create `ComplexExpr` from arguments.")
        return ComplexExpr(expr)


ccol: ComplexCol = ComplexCol()
