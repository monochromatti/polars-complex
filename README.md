# polars-complex


This is a [namespace
extension](https://docs.pola.rs/py-polars/html/reference/api/polars.api.register_expr_namespace.html)
for the [polars](https://github.com/pola-rs/polars) library, which
provides a complex number representation for the DataFrame type,
together with an implementation of complex number arithmetic.

## Creating the complex number representation

<div class="columns">

<div class="column" width="60%">

``` python
import polars as pl
import numpy as np

pl.Config.set_float_precision(3)

df = pl.DataFrame({
    "x": np.random.random(6),
    "y": np.random.random(6),
})
```

</div>

<div class="column" width="40%">

    shape: (6, 2)
    ┌───────┬───────┐
    │ x     ┆ y     │
    │ ---   ┆ ---   │
    │ f64   ┆ f64   │
    ╞═══════╪═══════╡
    │ 0.512 ┆ 0.960 │
    │ 0.341 ┆ 0.811 │
    │ 0.972 ┆ 0.951 │
    │ 0.513 ┆ 0.256 │
    │ 0.465 ┆ 0.137 │
    │ 0.635 ┆ 0.510 │
    └───────┴───────┘

</div>

</div>

<div class="columns">

<div class="column" width="60%">

``` python
import polars_complex

df = df.with_columns(
    pl.col("x", "y").complex.struct("z") # or .struct().alias("z[c]")
)
```

</div>

<div class="column" width="40%">

    shape: (6, 3)
    ┌───────┬───────┬───────────────┐
    │ x     ┆ y     ┆ z[c]          │
    │ ---   ┆ ---   ┆ ---           │
    │ f64   ┆ f64   ┆ struct[2]     │
    ╞═══════╪═══════╪═══════════════╡
    │ 0.512 ┆ 0.960 ┆ {0.512,0.960} │
    │ 0.341 ┆ 0.811 ┆ {0.341,0.811} │
    │ 0.972 ┆ 0.951 ┆ {0.972,0.951} │
    │ 0.513 ┆ 0.256 ┆ {0.513,0.256} │
    │ 0.465 ┆ 0.137 ┆ {0.465,0.137} │
    │ 0.635 ┆ 0.510 ┆ {0.635,0.510} │
    └───────┴───────┴───────────────┘

</div>

</div>

Note that the `[c]` suffix is the adopted convention to signal a complex
number representation. If the columns to be converted to the Struct
representation are named with the suffixes “.real” and “.imag”, with a
common prefix,

``` python
df = pl.DataFrame({
    "z.real": np.random.random(6),
    "z.imag": np.random.random(6),
})
print(df)
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ z.real ┆ z.imag │
    │ ---    ┆ ---    │
    │ f64    ┆ f64    │
    ╞════════╪════════╡
    │ 0.387  ┆ 0.657  │
    │ 0.086  ┆ 0.083  │
    │ 0.673  ┆ 0.196  │
    │ 0.530  ┆ 0.642  │
    │ 0.158  ┆ 0.050  │
    │ 0.670  ┆ 0.038  │
    └────────┴────────┘

then the construction is a little more succinct:

``` python
df = df.with_columns(
    pl.col("z").complex.struct()
)
print(df)
```

    shape: (6, 3)
    ┌────────┬────────┬───────────────┐
    │ z.real ┆ z.imag ┆ z[c]          │
    │ ---    ┆ ---    ┆ ---           │
    │ f64    ┆ f64    ┆ struct[2]     │
    ╞════════╪════════╪═══════════════╡
    │ 0.387  ┆ 0.657  ┆ {0.387,0.657} │
    │ 0.086  ┆ 0.083  ┆ {0.086,0.083} │
    │ 0.673  ┆ 0.196  ┆ {0.673,0.196} │
    │ 0.530  ┆ 0.642  ┆ {0.530,0.642} │
    │ 0.158  ┆ 0.050  ┆ {0.158,0.050} │
    │ 0.670  ┆ 0.038  ┆ {0.670,0.038} │
    └────────┴────────┴───────────────┘

## Arithmetic

Common operations, such as extracting phase, modulus, conjugation,
division, multiplication, and addition, are supported. They result in a
new complex number representation, or a numerical representation,
depending on the operation.

Using the dataframe above, here are some examples:

``` python
df = df.select(
    pl.col("z[c]"),
    (pl.col("z[c]").complex.conj()).alias("conj(z)[c]"),
    (pl.col("z[c]").complex.modulus()).alias("|z|"),
    (pl.col("z[c]").complex.phase()).alias("arg(z)"),
    (pl.col("z[c]") / 2.024).alias("w[c]"),
).with_columns(
    pl.col("z[c]").complex.divide(pl.col("w[c]")).alias("z/w[c]"),
    pl.col("z[c]").complex.multiply(pl.col("w[c]")).alias("z*w[c]"),
)
print(df)
```

    shape: (6, 7)
    ┌───────────────┬────────────────┬───────┬────────┬───────────────┬────────────────┬───────────────┐
    │ z[c]          ┆ conj(z)[c]     ┆ |z|   ┆ arg(z) ┆ w[c]          ┆ z/w[c]         ┆ z*w[c]        │
    │ ---           ┆ ---            ┆ ---   ┆ ---    ┆ ---           ┆ ---            ┆ ---           │
    │ struct[2]     ┆ struct[2]      ┆ f64   ┆ f64    ┆ struct[2]     ┆ struct[2]      ┆ struct[2]     │
    ╞═══════════════╪════════════════╪═══════╪════════╪═══════════════╪════════════════╪═══════════════╡
    │ {0.387,0.657} ┆ {0.387,-0.657} ┆ 0.762 ┆ 0.532  ┆ {0.191,0.325} ┆ {2.024,0.000}  ┆ {-0.139,0.251 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.086,0.083} ┆ {0.086,-0.083} ┆ 0.120 ┆ 0.803  ┆ {0.043,0.041} ┆ {2.024,0.000}  ┆ {0.000,0.007} │
    │ {0.673,0.196} ┆ {0.673,-0.196} ┆ 0.701 ┆ 1.287  ┆ {0.332,0.097} ┆ {2.024,-0.000} ┆ {0.205,0.131} │
    │ {0.530,0.642} ┆ {0.530,-0.642} ┆ 0.832 ┆ 0.691  ┆ {0.262,0.317} ┆ {2.024,0.000}  ┆ {-0.064,0.336 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.158,0.050} ┆ {0.158,-0.050} ┆ 0.166 ┆ 1.267  ┆ {0.078,0.024} ┆ {2.024,0.000}  ┆ {0.011,0.008} │
    │ {0.670,0.038} ┆ {0.670,-0.038} ┆ 0.671 ┆ 1.514  ┆ {0.331,0.019} ┆ {2.024,0.000}  ┆ {0.221,0.025} │
    └───────────────┴────────────────┴───────┴────────┴───────────────┴────────────────┴───────────────┘

## Converting back to numeric representation

Converting back (unnesting the structs) can be done either in the
dataframe namespace,

``` python
print(df.complex.unnest())
```

    shape: (6, 12)
    ┌───────┬────────┬────────┬────────┬───┬──────────┬──────────┬──────────┬──────────┐
    │ |z|   ┆ arg(z) ┆ z.real ┆ z.imag ┆ … ┆ z/w.real ┆ z/w.imag ┆ z*w.real ┆ z*w.imag │
    │ ---   ┆ ---    ┆ ---    ┆ ---    ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
    │ f64   ┆ f64    ┆ f64    ┆ f64    ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
    ╞═══════╪════════╪════════╪════════╪═══╪══════════╪══════════╪══════════╪══════════╡
    │ 0.762 ┆ 0.532  ┆ 0.387  ┆ 0.657  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.139   ┆ 0.251    │
    │ 0.120 ┆ 0.803  ┆ 0.086  ┆ 0.083  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.000    ┆ 0.007    │
    │ 0.701 ┆ 1.287  ┆ 0.673  ┆ 0.196  ┆ … ┆ 2.024    ┆ -0.000   ┆ 0.205    ┆ 0.131    │
    │ 0.832 ┆ 0.691  ┆ 0.530  ┆ 0.642  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.064   ┆ 0.336    │
    │ 0.166 ┆ 1.267  ┆ 0.158  ┆ 0.050  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.011    ┆ 0.008    │
    │ 0.671 ┆ 1.514  ┆ 0.670  ┆ 0.038  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.221    ┆ 0.025    │
    └───────┴────────┴────────┴────────┴───┴──────────┴──────────┴──────────┴──────────┘

or on individual columns,

``` python
print(df.select(
    pl.col("w[c]").complex.unnest()
))
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ w.real ┆ w.imag │
    │ ---    ┆ ---    │
    │ f64    ┆ f64    │
    ╞════════╪════════╡
    │ 0.191  ┆ 0.325  │
    │ 0.043  ┆ 0.041  │
    │ 0.332  ┆ 0.097  │
    │ 0.262  ┆ 0.317  │
    │ 0.078  ┆ 0.024  │
    │ 0.331  ┆ 0.019  │
    └────────┴────────┘
