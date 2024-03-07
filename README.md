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
    │ 0.229 ┆ 0.858 │
    │ 0.856 ┆ 0.541 │
    │ 0.782 ┆ 0.061 │
    │ 0.081 ┆ 0.242 │
    │ 0.925 ┆ 0.152 │
    │ 0.708 ┆ 0.512 │
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
    │ 0.229 ┆ 0.858 ┆ {0.229,0.858} │
    │ 0.856 ┆ 0.541 ┆ {0.856,0.541} │
    │ 0.782 ┆ 0.061 ┆ {0.782,0.061} │
    │ 0.081 ┆ 0.242 ┆ {0.081,0.242} │
    │ 0.925 ┆ 0.152 ┆ {0.925,0.152} │
    │ 0.708 ┆ 0.512 ┆ {0.708,0.512} │
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
    │ 0.462  ┆ 0.966  │
    │ 0.077  ┆ 0.616  │
    │ 0.142  ┆ 0.827  │
    │ 0.588  ┆ 0.663  │
    │ 0.490  ┆ 0.515  │
    │ 0.535  ┆ 0.811  │
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
    │ 0.462  ┆ 0.966  ┆ {0.462,0.966} │
    │ 0.077  ┆ 0.616  ┆ {0.077,0.616} │
    │ 0.142  ┆ 0.827  ┆ {0.142,0.827} │
    │ 0.588  ┆ 0.663  ┆ {0.588,0.663} │
    │ 0.490  ┆ 0.515  ┆ {0.490,0.515} │
    │ 0.535  ┆ 0.811  ┆ {0.535,0.811} │
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
    │ {0.462,0.966} ┆ {0.462,-0.966} ┆ 1.071 ┆ 0.446  ┆ {0.228,0.477} ┆ {2.024,0.000}  ┆ {-0.355,0.441 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.077,0.616} ┆ {0.077,-0.616} ┆ 0.621 ┆ 0.124  ┆ {0.038,0.304} ┆ {2.024,0.000}  ┆ {-0.185,0.047 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.142,0.827} ┆ {0.142,-0.827} ┆ 0.839 ┆ 0.170  ┆ {0.070,0.409} ┆ {2.024,-0.000} ┆ {-0.328,0.116 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.588,0.663} ┆ {0.588,-0.663} ┆ 0.886 ┆ 0.726  ┆ {0.291,0.328} ┆ {2.024,0.000}  ┆ {-0.046,0.386 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.490,0.515} ┆ {0.490,-0.515} ┆ 0.710 ┆ 0.760  ┆ {0.242,0.254} ┆ {2.024,0.000}  ┆ {-0.013,0.249 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
    │ {0.535,0.811} ┆ {0.535,-0.811} ┆ 0.971 ┆ 0.583  ┆ {0.264,0.401} ┆ {2.024,0.000}  ┆ {-0.184,0.429 │
    │               ┆                ┆       ┆        ┆               ┆                ┆ }             │
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
    │ 1.071 ┆ 0.446  ┆ 0.462  ┆ 0.966  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.355   ┆ 0.441    │
    │ 0.621 ┆ 0.124  ┆ 0.077  ┆ 0.616  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.185   ┆ 0.047    │
    │ 0.839 ┆ 0.170  ┆ 0.142  ┆ 0.827  ┆ … ┆ 2.024    ┆ -0.000   ┆ -0.328   ┆ 0.116    │
    │ 0.886 ┆ 0.726  ┆ 0.588  ┆ 0.663  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.046   ┆ 0.386    │
    │ 0.710 ┆ 0.760  ┆ 0.490  ┆ 0.515  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.013   ┆ 0.249    │
    │ 0.971 ┆ 0.583  ┆ 0.535  ┆ 0.811  ┆ … ┆ 2.024    ┆ 0.000    ┆ -0.184   ┆ 0.429    │
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
    │ 0.228  ┆ 0.477  │
    │ 0.038  ┆ 0.304  │
    │ 0.070  ┆ 0.409  │
    │ 0.291  ┆ 0.328  │
    │ 0.242  ┆ 0.254  │
    │ 0.264  ┆ 0.401  │
    └────────┴────────┘
