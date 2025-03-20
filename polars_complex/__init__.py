import polars as pl

from .extensions import ComplexExpr

__all__ = ["ccol"]


def ccol(name, *more_names) -> ComplexExpr:
    if not more_names:
        if ".*" in name:
            expr = pl.struct(
                name.replace(".*", ".re"),
                name.replace(".*", ".im"),
            )
        else:
            expr = pl.struct(
                pl.col(name).struct[0],
                pl.col(name).struct[1],
            ).alias(name)
    elif len(more_names) == 1:
        expr = pl.struct(pl.col(name), pl.col(next(iter(more_names))))
    else:
        err_msg = "`ccol` accepts at most two arguments (real and imaginary columns)"
        raise ValueError(err_msg)
    return ComplexExpr(real=expr.struct[0], imag=expr.struct[1])
