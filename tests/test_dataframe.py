import numpy as np
import polars as pl

from polars_complex import ccol

test_data = pl.DataFrame(
    {
        "a": [1] * 5 + [2] * 5,
        "b": np.random.rand(10),
        "c": np.random.rand(10),
        "d": np.random.rand(10),
    }
)

def test_groupby():
    result = test_data.group_by("a").agg(
        ccol("b", "c").first()
    )
    assert result.height == 2, f"Expected 2 rows, got {result.height} rows."
