# Polang

*Polang* is a super simple arithmetic parser that compiles to (Polars Expressions)[https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Expr.html#polars.Expr]. 

## Usage

```python
df = DataFrame({"a": np.linspace(0, 1), "b": np.sin(np.linspace(0, 1))})
df.select(polang("sin(a) + b"))
````



## Currently Supported Features

 + `+ - * /`
 + Braces `( )`
 + Any method of polars expressions.

## Roadmap

 + Numbers
 + Macros

