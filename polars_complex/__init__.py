import polars as pl

from .extensions import ComplexExpr

__all__ = ["ccol"]


def ccol(name, *more_names) -> ComplexExpr:
    if not more_names:
        if ".*" in name:
            pyexpr = pl.struct(
                name.replace(".*", ".re"),
                name.replace(".*", ".im"),
            )._pyexpr
        else:
            pyexpr = pl.struct(
                pl.col(name).struct[0],
                pl.col(name).struct[1],
            )._pyexpr
    elif len(more_names) == 1:
        pyexpr = pl.struct(pl.col(name), pl.col(next(iter(more_names))))._pyexpr
    else:
        err_msg = "`ccol` accepts at most two arguments (real and imaginary columns)"
        raise ValueError(err_msg)
    return ComplexExpr._from_pyexpr(pyexpr)
