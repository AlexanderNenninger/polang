from unittest import TestCase

import numpy as np
from polang import polang
from polars import DataFrame, col


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
        assert (df - (self.df.a - self.df.b).sum())[0, 0] < self.eps

        exprs = polang("a - b")
        df = self.df.select(exprs)
        assert (df - (self.df.a - self.df.b)).sum()[0, 0] < self.eps

        exprs = polang("(a - b) * c")
        df = self.df.select(exprs)
        assert (df - ((self.df.a - self.df.b) * self.df.c)).sum()[0, 0] < self.eps

    def test_number(self):
        exprs = polang("2.1 * a - 3")
        df = self.df.select(exprs)
        assert df.a[0] == 2.1 * self.df.a[0] - 3
        pass
