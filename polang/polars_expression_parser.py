from abc import ABC, abstractclassmethod, abstractmethod
from operator import add, mul, neg, sub, truediv
from typing import List
from urllib.parse import ParseResult

from polars import DataFrame, Expr, col
from pyparsing import (
    FollowedBy,
    Forward,
    ParserElement,
    QuotedString,
    Suppress,
    Word,
    delimited_list,
    identbodychars,
    identchars,
    infix_notation,
    one_of,
    opAssoc,
)
from pyparsing import pyparsing_common as ppc


class Operand:
    def __init__(self, tokens: ParseResult) -> None:
        self.value = tokens[0]

    @abstractmethod
    def eval(self) -> Expr | str | float | int:
        pass

    @abstractclassmethod
    def parser(cls) -> ParserElement:
        pass

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"


class Column(Operand):
    def eval(self) -> Expr:
        return col(self.value)

    @classmethod
    def parser(cls):
        return Word(identchars, identbodychars).set_parse_action(cls)


class Integer(Operand):
    def eval(self) -> int:
        return int(self.value)

    @classmethod
    def parser(cls):
        return (ppc.integer + ~FollowedBy(".")).set_parse_action(cls)


class Floatingpoint(Operand):
    def eval(self) -> float:
        return float(self.value)

    @classmethod
    def parser(cls):
        return ppc.fnumber.set_parse_action(cls)


class String(Operand):
    def eval(self):
        return self.value

    @classmethod
    def parser(cls):
        return QuotedString(quoteChar="'").set_parse_action(cls)


def raises_not_implemented(*args, **kwargs):
    """Raises NotImplemented error when called.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError()


class Operator(ABC):
    def __init__(self, tokens):
        self.tokens = tokens
        self.func = raises_not_implemented
        self.children = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.func}({self.tokens})"

    def __call__(self, *args):
        return self.func(*args)

    def eval(self):
        return self.func(*[c.eval() for c in self.children])


class InfixOperator(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.children = [tokens[0][0], tokens[0][2]]
        match tokens[0][1]:
            case "*":
                self.func = mul
            case "/":
                self.func = truediv
            case "+":
                self.func = add
            case "-":
                self.func = sub
            case _:
                raise ValueError(f"Unknown infix operator {tokens[0][1]}")


class PrefixOperator(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.children = [tokens[0][1]]
        match tokens[0][0]:
            case "-":
                self.func = neg
            case "+":
                self.func = lambda x: x
            case _:
                raise ValueError(f"Unknown prefix operator {tokens[0][0]}")


def method2fun(fname):
    def call(*args):
        return getattr(args[0], fname)(*args[1:])

    return call


class Function(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        match tokens:
            case [fname]:
                raise ValueError("Zero argument functions are not supported.")
            case [fname, *args]:
                self.func = method2fun(fname)
                self.children = args


def make_polang() -> ParserElement:

    parse_tree = Forward()
    # Function calls

    function_name = Word(identchars, identbodychars)
    function_body = Forward()
    function_body <<= (
        function_name
        + Suppress("(")
        + delimited_list(parse_tree, min=1)
        + Suppress(")")
    )

    # Calculations
    operand = (
        Integer.parser()
        | Floatingpoint.parser()
        | String.parser()
        | function_body.set_parse_action(Function)
        | Column.parser().set_parse_action(Column)
    )

    parse_tree <<= infix_notation(
        operand,
        [
            (one_of("+ -"), 1, opAssoc.RIGHT, PrefixOperator),
            (one_of("* /"), 2, opAssoc.LEFT, InfixOperator),
            (one_of("+ -"), 2, opAssoc.LEFT, InfixOperator),
        ],
    )

    return parse_tree


def find_columns(node: Operator | Operand) -> List[str]:
    """Find all column names used in an expression.

    Args:
        node (Operator | Operand): The root of the parse tree.

    Returns:
        List: A list of column names.
    """

    def dfs(node, columns):
        if isinstance(node, Column):
            columns.append(node.value)
        if hasattr(node, "children"):
            for child in node.children:
                dfs(child, columns)

    columns: List[str] = []
    dfs(node, columns)
    return columns


def polang(s: str) -> Expr:
    return make_polang().parseString(s)[0].eval()


def can_select_polang(df: DataFrame, s: str) -> bool:
    ast = make_polang().parseString(s)[0]
    columns = find_columns(ast)
    return set(columns).issubset(df.columns)


if __name__ == "__main__":
    print("Parsed Expression: ", polang("a - b"))
    print("Parsed Expression: ", polang("a - b * c"))
    print("Parsed Expression: ", polang("a - b + c*-d*e"))
    print("Parsed Expression: ", polang("a--b + c*-d"))
