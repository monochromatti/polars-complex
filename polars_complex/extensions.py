import polars as pl
from polars._typing import IntoExpr
from polars.expr.expr import Expr
from polars.polars import PyExpr
from polars._typing import IntoExprColumn


@pl.api.register_expr_namespace("complex")
class ComplexExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def into(self, prefix: str = ""):
        name = prefix or self._expr.meta.output_name()
        pyexpr = pl.struct(
            self._expr.alias(f"{name}.re"),
            pl.lit(0.0).alias(f"{name}.im"),
        )._pyexpr
        return ComplexExpr._from_pyexpr(pyexpr)


def parse_into_complex_expression(input: IntoExpr) -> PyExpr:
    if isinstance(input, pl.Expr):
        expr = input
    elif isinstance(input, str):
        if ".*" in input:
            expr = pl.struct(
                input.replace(".*", ".re"),
                input.replace(".*", ".im"),
            )
        else:
            expr = pl.col(input)
    else:
        expr = pl.struct(pl.lit(input).alias("literal.re"), pl.lit(0.0).alias("literal.im"),)
    return expr._pyexpr


class ComplexExpr(Expr):
    real: Expr
    imag: Expr

    def __init__(self, real: Expr, imag: Expr) -> None:
        self.real = real
        self.imag = imag
        self._pyexpr = pl.struct(real, imag)._pyexpr

    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> "ComplexExpr":
        """Expects pyexpr to be a struct, with real and imaginary fields, respectively."""
        expr = cls.__new__(cls)
        expr._pyexpr = pyexpr
        expr.real = expr.struct[0]
        expr.imag = expr.struct[1]
        return expr

    def __add__(self, other: IntoExpr) -> Expr:
        expr = ComplexExpr._from_pyexpr(parse_into_complex_expression(other))
        return ComplexExpr(self.real + expr.real, self.imag + expr.imag)

    def __sub__(self, other: IntoExpr) -> "ComplexExpr":
        expr = ComplexExpr._from_pyexpr(parse_into_complex_expression(other))
        return ComplexExpr(self.real - expr.real, self.imag - expr.imag)

    def __mul__(self, other) -> "ComplexExpr":
        expr = ComplexExpr._from_pyexpr(parse_into_complex_expression(other))
        real = self.real * expr.real - self.imag * expr.imag
        imag = self.real * expr.imag + self.imag * expr.real
        return ComplexExpr(
            real.alias(self.real.meta.output_name()),
            imag.alias(expr.imag.meta.output_name()),
        )

    def __truediv__(self, other) -> "ComplexExpr":
        expr = ComplexExpr._from_pyexpr(parse_into_complex_expression(other))
        norm = expr.real.pow(2) + expr.imag.pow(2)
        real = (self.real * expr.real + self.imag * expr.imag) / norm
        imag = (self.imag * expr.real - self.real * expr.imag) / norm
        return ComplexExpr(real, imag)

    def conj(self) -> "ComplexExpr":
        return ComplexExpr(self.real, self.imag * -1.0)

    def arg(self, *, unwrap: bool = False) -> Expr:
        if unwrap:
            return self.arg_unwrap()
        return pl.arctan2(self.real, self.imag)

    def arg_unwrap(self) -> Expr:
        diff = self - self.shift()
        if_lower = pl.when(diff < -3.14159).then(diff + 2 * 3.14159)
        if_higher = pl.when(diff > 3.14159).then(diff - 2 * 3.14159)
        corrected_diff = (
            if_higher.otherwise(if_lower.otherwise(diff)).fill_null(0).cum_sum()
        )
        return self.head(1).cast(pl.Float64) + corrected_diff

    def modulus(self) -> Expr:
        return (self.real.pow(2) + self.imag.pow(2)).sqrt()

    def pow(
        self, exponent: IntoExprColumn | int | float, *, k: int = 0
    ) -> "ComplexExpr":
        magnitude = self.modulus().pow(exponent)
        arg = exponent * self.arg() + 2 * 3.14159265359 * k
        name = self.meta.output_name()
        return ComplexExpr(
            (magnitude * arg.cos()).alias(f"{name}.re"),
            (magnitude * arg.sin()).alias(f"{name}.im"),
        )

    def exp(self) -> "ComplexExpr":
        name = self.meta.output_name()
        return ComplexExpr(
            (self.real.exp() * self.imag.cos()).alias(f"{name}.re"),
            (self.real.exp() * self.imag.sin()).alias(f"{name}.im"),
        )

    def sin(self) -> "ComplexExpr":
        name = self.meta.output_name()
        return ComplexExpr(
            (self.real.sin() * self.imag.cosh()).alias(f"{name}.re"),
            (self.real.cos() * self.imag.sinh()).alias(f"{name}.im"),
        )

    def cos(self) -> "ComplexExpr":
        name = self.meta.output_name()
        return ComplexExpr(
            (self.real.cos() * self.imag.cosh()).alias(f"{name}.re"),
            (self.real.sin() * self.imag.sinh() * -1).alias(f"{name}.im"),
        )


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
