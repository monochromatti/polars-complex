import numpy as np
import polars as pl
from polars import col

from polars_complex import ccol

test_data = pl.DataFrame(
    {
        "a": np.random.rand(10),
        "b": np.random.rand(10),
        "c": np.random.rand(10),
        "d": np.random.rand(10),
    }
)


class TestAddition:
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("a", "b") + ccol("c", "d")).alias("z1+z2"))

    def test_ccol_lit(self):
        test_data.with_columns(
            (ccol("a", "b") + pl.lit(1.0).complex.into()).alias("z1+1")
        )

    def test_lit_ccol(self):
        test_data.with_columns(
            (pl.lit(1.0).complex.into() + ccol("a", "b")).alias("1+z1")
        )


class TestSubtraction:
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("a", "b") - ccol("c", "d")).alias("z1-z2"))

    def test_ccol_lit(self):
        test_data.with_columns(
            (ccol("a", "b") - pl.lit(1.0).complex.into()).alias("z1+1")
        )

    def test_lit_ccol(self):
        test_data.with_columns(
            (pl.lit(1.0).complex.into() - ccol("a", "b")).alias("1+z1")
        )


class TestMultiplication:
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("a", "b") * ccol("c", "d")).alias("z1 * z2"))

    def test_ccol_lit(self):
        test_data.with_columns(
            (ccol("a", "b") * pl.lit(1.0).complex.into()).alias("z1 * 1")
        )

    def test_lit_ccol(self):
        test_data.with_columns(
            (pl.lit(1.0).complex.into() * ccol("a", "b")).alias("1 * z1")
        )


class TestDivision:
    def test_ccol_ccol(self):
        test_data.with_columns((ccol("a", "b") / ccol("c", "d")).alias("z1/z2"))

    def test_ccol_lit(self):
        test_data.with_columns(
            (ccol("a", "b") / pl.lit(1.0).complex.into()).alias("z1 / 1")
        )

    def test_lit_ccol(self):
        test_data.with_columns(
            (pl.lit(1.0).complex.into() / ccol("a", "b")).alias("1 / z1")
        )



# class TestTranscendental:
    # def test_exp(self):
    #     test_data.with_columns((ccol("a", "b").exp()).alias("exp(z1)"))

    # def test_sin(self):
    #     test_data.with_columns((ccol("a", "b").sin()).alias("sin(z1)"))

    # def test_cos(self):
    #     test_data.with_columns((ccol("a", "b").cos()).alias("cos(z1)"))

    # def test_pow(self):
    #     test_data.with_columns((ccol("a", "b").pow(2)).alias("z1^2"))