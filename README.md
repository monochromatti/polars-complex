# polars-complex


This is a [namespace
extension](https://docs.pola.rs/py-polars/html/reference/api/polars.api.register_expr_namespace.html)
for the [polars](https://github.com/pola-rs/polars) library, which
provides a complex number representation for the DataFrame type,
together with an implementation of complex number arithmetic.

## Creating the complex number representation

``` python
import polars as pl
import numpy as np

pl.Config.set_float_precision(3)

df = pl.DataFrame({
    "x": np.random.random(6),
    "y": np.random.random(6),
})
print(df)
```

    shape: (6, 2)
    ┌───────┬───────┐
    │ x     ┆ y     │
    │ ---   ┆ ---   │
    │ f64   ┆ f64   │
    ╞═══════╪═══════╡
    │ 0.173 ┆ 0.326 │
    │ 0.800 ┆ 0.113 │
    │ 0.643 ┆ 0.990 │
    │ 0.477 ┆ 0.963 │
    │ 0.468 ┆ 0.338 │
    │ 0.058 ┆ 0.247 │
    └───────┴───────┘

``` python
import polars_complex

df = df.with_columns(
    pl.col("x", "y").complex.struct("z") # or .struct().alias("z[c]")
)
print(df)
```

    shape: (6, 3)
    ┌───────┬───────┬───────────────┐
    │ x     ┆ y     ┆ z[c]          │
    │ ---   ┆ ---   ┆ ---           │
    │ f64   ┆ f64   ┆ struct[2]     │
    ╞═══════╪═══════╪═══════════════╡
    │ 0.173 ┆ 0.326 ┆ {0.173,0.326} │
    │ 0.800 ┆ 0.113 ┆ {0.800,0.113} │
    │ 0.643 ┆ 0.990 ┆ {0.643,0.990} │
    │ 0.477 ┆ 0.963 ┆ {0.477,0.963} │
    │ 0.468 ┆ 0.338 ┆ {0.468,0.338} │
    │ 0.058 ┆ 0.247 ┆ {0.058,0.247} │
    └───────┴───────┴───────────────┘

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
    │ 0.752  ┆ 0.196  │
    │ 0.875  ┆ 0.828  │
    │ 0.178  ┆ 0.261  │
    │ 0.687  ┆ 0.356  │
    │ 0.496  ┆ 0.432  │
    │ 0.906  ┆ 0.798  │
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
    │ 0.752  ┆ 0.196  ┆ {0.752,0.196} │
    │ 0.875  ┆ 0.828  ┆ {0.875,0.828} │
    │ 0.178  ┆ 0.261  ┆ {0.178,0.261} │
    │ 0.687  ┆ 0.356  ┆ {0.687,0.356} │
    │ 0.496  ┆ 0.432  ┆ {0.496,0.432} │
    │ 0.906  ┆ 0.798  ┆ {0.906,0.798} │
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
    │ {0.752,0.196} ┆ {0.752,-0.196} ┆ 0.777 ┆ 1.315  ┆ {0.372,0.097} ┆ {2.024,0.000}  ┆ {0.260,0.146} │
    │ {0.875,0.828} ┆ {0.875,-0.828} ┆ 1.205 ┆ 0.813  ┆ {0.432,0.409} ┆ {2.024,-0.000} ┆ {0.040,0.716} │
    │ {0.178,0.261} ┆ {0.178,-0.261} ┆ 0.316 ┆ 0.600  ┆ {0.088,0.129} ┆ {2.024,0.000}  ┆ {-0.018,0.046 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.687,0.356} ┆ {0.687,-0.356} ┆ 0.773 ┆ 1.093  ┆ {0.339,0.176} ┆ {2.024,0.000}  ┆ {0.170,0.241} │
    │ {0.496,0.432} ┆ {0.496,-0.432} ┆ 0.658 ┆ 0.855  ┆ {0.245,0.213} ┆ {2.024,0.000}  ┆ {0.029,0.212} │
    │ {0.906,0.798} ┆ {0.906,-0.798} ┆ 1.208 ┆ 0.849  ┆ {0.448,0.394} ┆ {2.024,0.000}  ┆ {0.091,0.715} │
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
    │ 0.777 ┆ 1.315  ┆ 0.752  ┆ 0.196  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.260    ┆ 0.146    │
    │ 1.205 ┆ 0.813  ┆ 0.875  ┆ 0.828  ┆ … ┆ 2.024    ┆ -0.000   ┆ 0.040    ┆ 0.716    │
    │ 0.316 ┆ 0.600  ┆ 0.178  ┆ 0.261  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.018   ┆ 0.046    │
    │ 0.773 ┆ 1.093  ┆ 0.687  ┆ 0.356  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.170    ┆ 0.241    │
    │ 0.658 ┆ 0.855  ┆ 0.496  ┆ 0.432  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.029    ┆ 0.212    │
    │ 1.208 ┆ 0.849  ┆ 0.906  ┆ 0.798  ┆ … ┆ 2.024    ┆ 0.000    ┆ 0.091    ┆ 0.715    │
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
    │ 0.372  ┆ 0.097  │
    │ 0.432  ┆ 0.409  │
    │ 0.088  ┆ 0.129  │
    │ 0.339  ┆ 0.176  │
    │ 0.245  ┆ 0.213  │
    │ 0.448  ┆ 0.394  │
    └────────┴────────┘
