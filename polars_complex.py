import json
import re

import numpy as np
import polars as pl


@pl.api.register_expr_namespace("complex")
class ComplexArithmetic:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    @property
    def names(self):
        meta_dict = json.loads(self._expr.meta.serialize())
        if self._expr.meta.has_multiple_outputs():
            names = _finditem(meta_dict, "Columns")
            return names
        else:
            name = self._expr.meta.output_name()
            name = re.sub(r"\.real|\.imag", "", name)
            return [name]

    def cformat(self, name: str):
        return name if name.endswith("[c]") else name + "[c]"

    def struct(self, name: str = None):
        if self._expr.meta.has_multiple_outputs():
            name_real, name_imag = self.names
            stem = name or name_real.replace(".real", "")
            return pl.struct(pl.col(name_real), pl.col(name_imag)).alias(
                self.cformat(stem)
            )
        else:
            stem = self.names[0]
            if ".real" in stem:
                raise ValueError(
                    'Provide a column `name` with ".real"/".imag" omitted. '
                    "Corresponding `name`.real and `name`.imag columns are assumed."
                )
            alias = self.cformat(name or stem)
            return pl.struct(
                pl.col(f"{stem}.real").alias(alias.replace("[c]", ".real")),
                pl.col(f"{stem}.imag").alias(alias.replace("[c]", ".imag")),
            ).alias(alias)

    def unnest(self):
        stem = self.names[0].replace("[c]", "")
        self._expr = self._expr.struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
        return self._explode()

    def _explode(self):
        return self._expr.struct[0], self._expr.struct[1]

    def rename(self, name):
        if not name.endswith("[c]"):
            name = name + "[c]"
        return self._expr.alias(name)

    def real(self):
        return self._expr.struct[0]

    def imag(self):
        return self._expr.struct[1]

    def phase(self):
        return pl.arctan2(self._expr.struct[0], self._expr.struct[1])

    def unwrap_phase(self):
        diff = self._expr - self._expr.shift()
        if_lower = pl.when(diff < -3.14159).then(diff + 2 * 3.14159)
        if_higher = pl.when(diff > 3.14159).then(diff - 2 * 3.14159)
        corrected_diff = (
            if_higher.otherwise(if_lower.otherwise(diff)).fill_null(0).cumsum()
        )
        return self._expr.head(1).cast(pl.Float64) + corrected_diff

    def modulus(self, name: str = None):
        real, imag = self._explode()
        if name:
            return (real.pow(2) + imag.pow(2)).sqrt().alias(name)
        return (real.pow(2) + imag.pow(2)).sqrt()

    def conj(self, name: str = None):
        real, imag = self._explode()
        imag *= -1
        return self.set_alias(real, imag, name)

    def quotient(self, expr, name: str = None):
        self._expr = self._expr.name.suffix_fields(".1")
        real1, imag1 = self._explode()
        real2, imag2 = expr.name.suffix_fields(".2").complex._explode()
        real = (real1 * real2 + imag1 * imag2) / (real2.pow(2) + imag2.pow(2))
        imag = (real2 * imag1 - real1 * imag2) / (real2.pow(2) + imag2.pow(2))
        return self.set_alias(real, imag, name)

    def divide(self, expr, name: str = None):
        return self.quotient(expr, name=name)

    def ratio(self, expr, name: str = None):
        return self.quotient(expr, name=name)

    def inverse(self, name: str = None):
        real, imag = self._explode()
        real /= real.pow(2) + imag.pow(2)
        imag /= -1 * (real.pow(2) + imag.pow(2))
        return self.set_alias(real, imag, name)

    def multiply(self, expr, name: str = None):
        a, b = self._explode()
        c, d = expr.complex._explode()
        real = a * c - b * d
        imag = a * d + b * c
        return self.set_alias(real, imag, name)

    def subtract(self, expr, name: str = None):
        real, imag = (self._expr - expr).complex._explode()
        return self.set_alias(real, imag, name)

    def difference(self, expr, name: str = None):
        return self.subtract(expr, name=name)

    def relative_difference(self, expr, name: str = None):
        real, imag = self.quotient(expr).complex._explode()
        real -= 1
        if name:
            alias = self.cformat(name)
        else:
            alias = self.cformat(self.names[0])
        real = real.alias(alias.replace("[c]", ".real"))
        imag = imag.alias(alias.replace("[c]", ".imag"))
        return pl.struct(real, imag).alias(alias)

    def set_alias(self, real, imag, name: str = None):
        if name:
            alias = self.cformat(name)
        else:
            alias = self.cformat(self.names[0])
        real = real.alias(alias.replace("[c]", ".real"))
        imag = imag.alias(alias.replace("[c]", ".imag"))
        return pl.struct(real, imag).alias(alias)


@pl.api.register_dataframe_namespace("complex")
class ComplexFrame:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def unnest(self, names: str | list[str] = None):
        df = self._df
        if names is None:
            names = [name for name in df.columns if name.endswith("[c]")]
        elif not isinstance(names, list):
            names = [names]
        stems = [name.replace("[c]", "") for name in names]
        return df.select(
            pl.all().exclude(names),
            *[
                pl.col(name).struct.rename_fields([f"{stem}.real", f"{stem}.imag"])
                for (name, stem) in zip(names, stems)
            ],
        ).unnest(names)

    def nest(self, varnames: str | list[str] = None):
        if varnames is None:
            varnames = [
                name.replace(".real", "")
                for name in self._df.columns
                if name.endswith(".real")
                and name.replace(".real", ".imag") in self._df.columns
            ]
            varnames = list(set(varnames))
        elif isinstance(varnames, str):
            varnames = [varnames]

        complex_columns = self._df.select(
            pl.col(f"{varname}.{comp}")
            for varname in varnames
            for comp in ("real", "imag")
        ).columns

        for varname in varnames:
            if f"{varname}.real" not in complex_columns:
                raise ValueError(f"Column {varname}.real missing.")
            if f"{varname}.imag" not in complex_columns:
                raise ValueError(f"Column {varname}.imag missing.")

        return self._df.select(
            pl.all().exclude(complex_columns),
            *[
                pl.struct(
                    pl.col(f"{var}.real").alias("real"),
                    pl.col(f"{var}.imag").alias("imag"),
                ).alias(f"{var}[c]")
                for var in varnames
            ],
        )


def pl_fft(df, xname, id_vars=None, rfft=True):
    """
    Compute the Fast Fourier Transform (FFT) of the given DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        xname (str): The name of the column representing the x-axis values.
        id_vars (list, optional): List of column names to use as grouping variables. Defaults to None.
        real_valued (bool, optional): Whether the input data is real-valued. Defaults to True, and uses `rfft`.

    Returns:
        pl.DataFrame: The DataFrame containing the FFT results.
    """

    def fftfreq(df):
        if rfft:
            return np.fft.rfftfreq(
                len(df[xname]),
                abs(df[xname][1] - df[xname][0]),
            )
        else:
            return np.fft.fftfreq(
                len(df[xname]),
                abs(df[xname][1] - df[xname][0]),
            )

    def varname_iter(fft_dict, value_vars):
        for name in value_vars:
            for component, operator in zip(("real", "imag"), (np.real, np.imag)):
                yield pl.Series(
                    f"{name}.{component}",
                    operator(fft_dict[name]),
                    dtype=df.schema[name],
                )

    id_vars = id_vars or []
    value_vars = [var for var in df.columns if var not in id_vars and var != xname]

    frames = []
    fft_transform = np.fft.rfft if rfft else np.fft.fft
    if not id_vars:
        fft_dict = {name: fft_transform(df[name].to_numpy()) for name in value_vars}
        frames.append(
            pl.DataFrame(
                (
                    pl.Series("freq", fftfreq(df)),
                    *varname_iter(fft_dict, value_vars),
                )
            )
        )
    else:
        for id_vals, group in df.group_by(*id_vars, maintain_order=True):
            if isinstance(id_vals, (float, int, str)):
                id_vals = [id_vals]
            fft_dict = {
                name: fft_transform(group[name].to_numpy()) for name in value_vars
            }
            frames.append(
                pl.DataFrame(
                    (
                        pl.Series("freq", fftfreq(group)),
                        *varname_iter(fft_dict, value_vars),
                    )
                )
                .with_columns(
                    pl.lit(value).cast(df.schema[name]).alias(name)
                    for name, value in zip(id_vars, list(id_vals))
                )
                .select(
                    *(pl.col(name) for name in id_vars),
                    pl.col("freq"),
                    pl.all().exclude("freq", *id_vars),
                )
            )
    return pl.concat(frames)


def _finditem(d, key="Columns"):
    """
    Recursive function to search for the first occurrence of the key `key` in a
    nested dictionary of mixed dict and list values.

    Parameters:
        d (dict): The dictionary to search in.
        key (str): The key to search for, default is "Columns".

    Returns:
        list or None: The value associated with the key "Columns", or None if not found.
    """
    if isinstance(d, dict):
        if key in d:
            return d[key]
        else:
            for value in d.values():
                result = _finditem(value, key)
                if result is not None:
                    return result
    elif isinstance(d, list):
        for item in d:
            result = _finditem(item, key)
            if result is not None:
                return result
    return None


def zero_quadrature(s: pl.Series):
    real, imag = s.struct.unnest().get_columns()
    pha = np.arange(-1.571, 1.571, 1e-3)
    imag_outer = np.outer(real, np.sin(pha)) + np.outer(imag, np.cos(pha))
    pha_opt = pha[np.sum(np.abs(imag_outer) ** 2, axis=0).argmin()]
    return real * np.cos(pha_opt) - imag * np.sin(pha_opt)
