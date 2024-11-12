import polars as pl
from polars._typing import IntoExpr
from polars.expr.expr import Expr

from . import arithmetic as ar


@pl.api.register_expr_namespace("complex")
class ComplexExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def into(self):
        return ComplexExpr(
            pl.struct(self._expr.alias("real"), pl.lit(0.0).alias("imag"))
        )

    def explode(self, name: str | None = None):
        expr = self._expr
        if name:
            expr = expr.struct.rename_fields([f"{name}.real", f"{name}.imag"])
        return expr.struct.unnest()


class ComplexExpr(Expr):
    def __init__(self, expr: Expr) -> None:
        self.expr = expr
        self.real = self.expr.struct.field("real")
        self.imag = self.expr.struct.field("imag")

    @property
    def _pyexpr(self):
        return self.expr._pyexpr

    def alias(self, name: str):
        return ComplexExpr(self.expr.alias(name))

    def cast_to_self(self, other: IntoExpr | "ComplexExpr"):
        if isinstance(other, ComplexExpr):
            return other
        if isinstance(other, (int, float)):
            return ComplexExpr(pl.lit(other))
        if isinstance(other, Expr):
            return ComplexExpr(other)
        raise ValueError(f"Cannot cast {type(other)} to ComplexExpr.")

    def __add__(self, other):
        return ComplexExpr(ar.add(self, self.cast_to_self(other)))

    def __radd__(self, other):
        return self.__add__(self.cast_to_self(other))

    def __sub__(self, other):
        return ComplexExpr(ar.subtract(self, self.cast_to_self(other)))

    def __rsub__(self, other):
        return ComplexExpr((self.cast_to_self(other) - self.expr) * -1)

    def __mul__(self, other):
        return ComplexExpr(ar.multiply(self, self.cast_to_self(other)))

    def __rmul__(self, other):
        return ComplexExpr(self.__mul__(self.cast_to_self(other)))

    def __truediv__(self, other):
        return ComplexExpr(ar.divide(self, self.cast_to_self(other)))

    def __rtruediv__(self, other):
        return ComplexExpr(self.cast_to_self(other) / self.expr)

    def exp(self):
        return ComplexExpr(ar.exp(self))

    def sin(self):
        return ComplexExpr(ar.sin(self))

    def cos(self):
        return ComplexExpr(ar.cos(self))

    def pow(self, n: int | float):  # pyright: ignore [reportIncompatibleMethodOverride]
        return ComplexExpr(ar.pow(self, n))

    def squared_modulus(self):
        return ComplexExpr(ar.squared_modulus(self))

    def modulus(self):
        return ComplexExpr(ar.modulus(self))

    def phase(self):
        return ComplexExpr(ar.phase(self))

    def unwrap_phase(self):
        return ComplexExpr(ar.unwrap_phase(self))

    def conj(self):
        return ComplexExpr(ar.conj(self))


# @pl.api.register_dataframe_namespace("complex")
# class ComplexFrame:
#     def __init__(self, df: pl.DataFrame):
#         self._df = df

#     def unnest(self, names: str | list[str] | None = None):
#         df = self._df
#         if names is None:
#             names = [name for name in df.columns if name.endswith("[c]")]
#         elif not isinstance(names, list):
#             names = [names]
#         if len(names) < 1:
#             warnings.warn("No complex columns found.")
#             return df
#         else:
#             stems = [name.replace("[c]", "") for name in names]
#             return df.select(
#                 pl.all().exclude(names),
#                 *[
#                     pl.col(name).struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
#                     for (name, stem) in zip(names, stems)
#                 ],
#             ).unnest(*names)

#     def nest(self, varnames: str | list[str] = None):
#         if varnames is None:
#             varnames = [
#                 name.replace(".real", "")
#                 for name in self._df.columns
#                 if name.endswith(".real")
#                 and name.replace(".real", ".imag") in self._df.columns
#             ]
#             varnames = list(set(varnames))
#         elif isinstance(varnames, str):
#             varnames = [varnames]

#         complex_columns = self._df.select(
#             pl.col(f"{varname}.{comp}")
#             for varname in varnames
#             for comp in ("real", "imag")
#         ).columns

#         for varname in varnames:
#             if f"{varname}.real" not in complex_columns:
#                 raise ValueError(f"Column {varname}.real missing.")
#             if f"{varname}.imag" not in complex_columns:
#                 raise ValueError(f"Column {varname}.imag missing.")

#         return self._df.select(
#             pl.all().exclude(complex_columns),
#             *[
#                 pl.struct(
#                     pl.col(f"{var}.real").alias("real"),
#                     pl.col(f"{var}.imag").alias("imag"),
#                 ).alias(f"{var}[c]")
#                 for var in varnames
#             ],
#         )

#     def struct(self, *args, **kwargs):
#         return self.nest(*args, **kwargs)
