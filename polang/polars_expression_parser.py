from abc import ABC, abstractclassmethod, abstractmethod
from operator import add, methodcaller, mul, neg, sub, truediv

from polars import Expr, col
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
    def __init__(self, tokens) -> None:
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
    def eval(self) -> Expr:
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


class Operator(ABC):
    def __init__(self, tokens):
        self.tokens = tokens
        self.func = lambda x: x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.func}({self.tokens})"

    def __call__(self, *args):
        return self.func(*args)

    @abstractmethod
    def eval(self):
        pass


class InfixOperator(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.fst = tokens[0][0]
        self.snd = tokens[0][2]
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

    def eval(self):
        return self.func(self.fst.eval(), self.snd.eval())


class PrefixOperator(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.fst = tokens[0][1]
        match tokens[0][0]:
            case "-":
                self.func = neg
            case "+":
                self.func = lambda x: x
            case _:
                raise ValueError(f"Unknown prefix operator {tokens[0][0]}")

    def eval(self):
        return self.func(self.fst.eval())


class Function(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        match tokens:
            case [fname]:
                raise ValueError("Zero argument functions are not supported.")
            case [fname, arg]:
                self.func = methodcaller(fname)
                self.arg = arg
            case [fname, arg, *args]:
                self.func = methodcaller(fname, *[arg.eval() for arg in args])
                self.arg = arg

    def eval(self):
        if self.arg is None:
            return self.func()
        return self.func(self.arg.eval())


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


def polang(s: str) -> Expr:
    parsed = make_polang().parseString(s)[0]
    return parsed.eval()


if __name__ == "__main__":
    print("Parsed Expression: ", polang("a - b"))
    print("Parsed Expression: ", polang("a - b * c"))
    print("Parsed Expression: ", polang("a - b + c*-d*e"))
    print("Parsed Expression: ", polang("a--b + c*-d"))
