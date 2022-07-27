from abc import ABC, abstractmethod
from operator import add, methodcaller, mul, neg, sub, truediv

from polars import Expr, col
from pyparsing import (ParserElement, Word, alphas, infix_notation, one_of,
                       opAssoc)


class Operand:
    def __init__(self, tokens):
        self.name = tokens[0]

    def __str__(self):
        return str(self.name)

    def __repr__(self) -> str:
        return f"Operand({self.name})"

    def eval(self):
        return col(self.name)


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
                raise ValueError(f"Unknown prefix operator {tokens[0][1]}")

    def eval(self):
        return self.func(self.fst.eval())


class Function(Operator):
    def __init__(self, tokens):
        super().__init__(tokens)
        self.args = tokens[0][1:]
        match tokens[0][0]:
            case _:
                self.func = methodcaller(tokens[0][0])

    def eval(self):
        return self.func(*[arg.eval() for arg in self.args])


def make_polang() -> ParserElement:
    operand = Word(alphas).setParseAction(Operand)

    return infix_notation(
        operand,
        [
            (one_of(["sin", "cos", "sum", "mean"]), 1, opAssoc.RIGHT, Function),
            (one_of("+ -"), 1, opAssoc.RIGHT, PrefixOperator),
            (one_of("* /"), 2, opAssoc.LEFT, InfixOperator),
            (one_of("+ -"), 2, opAssoc.LEFT, InfixOperator),
        ],
    )


def polang(s: str) -> Expr:
    return make_polang().parseString(s)[0].eval()


if __name__ == "__main__":
    print("Parsed Expression: ", polang("a - b"))
    print("Parsed Expression: ", polang("a - b * c"))
    print("Parsed Expression: ", polang("a - b + c*-d*e"))
    print("Parsed Expression: ", polang("a--b + c*-d"))
