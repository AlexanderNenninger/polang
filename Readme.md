# Polang

*Do not use this with untrusted input!*

*Polang* is a simple arithmetic parser that compiles to [Polars Expressions](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Expr.html#polars.Expr).

## But Why?

My usecase is tabular data with a lot of columns that need transformation. 

## Usage

```python
df = DataFrame({"a": np.linspace(0, 1), "b": np.sin(np.linspace(0, 1))})
df.select(polang("(sin(a) + b) * 3.2"))
````



## Currently Supported Features

 + `+ - * /`
 + Braces `( )`
 + Any method of polars expressions.

## Roadmap

 + Macros

