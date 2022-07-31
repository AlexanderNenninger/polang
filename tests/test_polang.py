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
                "c": np.cos(np.linspace(0, 1)),
                "d": np.tan(np.linspace(0, 1)),
                "e": np.arctan(np.linspace(0, 1)),
            }
        )

    def test_parser(self):
        exprs = polang("sum(a - b)")
        df = self.df.select(exprs)
        self.assertAlmostEqual(df[0, 0], (self.df.a - self.df.b).sum())

        exprs = polang("a - b")
        df = self.df.select(exprs)
        self.assertTrue((df - (self.df.a - self.df.b)).sum()[0, 0] < self.eps)

        exprs = polang("(a - b) * c")
        df = self.df.select(exprs)
        self.assertTrue(
            (df - ((self.df.a - self.df.b) * self.df.c)).sum()[0, 0] < self.eps
        )

    def test_number(self):
        exprs = polang("2.1 * a - 3")
        df = self.df.select(exprs)
        self.assertEqual(df.a[0], 2.1 * self.df.a[0] - 3)

    def test_alias(self):
        expr = polang("alias(a, 'test')")
        assert str(expr) == 'col("a").alias("test")'
        df = self.df.select(expr)
        self.assertEqual(df.columns[0], "test")

    def test_shift_by_1(self):
        # Custom functions and function calls inside parentheses
        expr = polang("(2 * shift(b, 1))")
        df = self.df.select(expr.alias("db/da"))

    def test_shift_by_neg_1(self):
        # Custom functions and function calls inside parentheses
        expr = polang("shift(b, -1)")
        df = self.df.select(expr.alias("db/da"))

    def test_derivative(self):
        expr = polang("(shift(b, 1) - shift(b, -1)) / (shift(a,1) - shift(a,-1))")
        df = self.df.select(expr.alias("db/da"))
        self.assertTrue(df[1, 0] - self.df.c[1] < 1e-4)
