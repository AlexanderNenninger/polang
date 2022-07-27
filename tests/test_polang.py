from unittest import TestCase

import numpy as np
from polang import polang
from polars import DataFrame


class TestPolang(TestCase):
    def setUp(self) -> None:
        self.eps = 1e-6
        self.df = DataFrame(
            {
                "a": np.linspace(0, 1),
                "b": np.sin(np.linspace(0, 1)),
                "c": np.sin(np.linspace(0, 1)),
                "d": np.sin(np.linspace(0, 1)),
                "e": np.sin(np.linspace(0, 1)),
            }
        )

    def test_parser(self):
        exprs = polang("sum(a - b)")
        df = self.df.select(exprs)
        pass
