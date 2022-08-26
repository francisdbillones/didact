from __future__ import annotations

from enum import Enum
import math

import numpy as np
from graphviz import Digraph

from didact.autograd.backwards import add_backward, mul_backward, pow_backward


class Op(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 5


op_to_str = {
    Op.ADD: "+",
    Op.SUB: "-",
    Op.MUL: "*",
    Op.DIV: "/",
    Op.POW: "**",
}


class Value:
    def __init__(
        self, n: float, parent_op: str = None, parent_values: list[Value] = None
    ):
        self.n = n
        self.grad = 0.0
        self.parent_op = parent_op
        self.parent_values = parent_values

    def add(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            n=self.n + other.n,
            parent_op="add",
            parent_values=[self, other],
        )

    def sub(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self + (-1 * other)

    def mul(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            n=self.n * other.n,
            parent_op="mul",
            parent_values=[self, other],
        )

    def div(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self * (other ** -1)

    def pow(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return Value(
            n=self.n ** other.n,
            parent_op="pow",
            parent_values=[self, other],
        )

    def backward(self, grad=1.0):
        self.grad += grad
        if not self.parent_values:
            return

        a, b = self.parent_values

        backwards_fn = {
            "add": add_backward,
            "mul": mul_backward,
            "pow": pow_backward,
        }[self.parent_op]

        a_grad, b_grad = backwards_fn(a.n, b.n, grad)
        a.backward(a_grad)
        b.backward(b_grad)

    def zero_grad(self):
        self.grad = 0

        if self.parent_values is not None:
            for parent in self.parent_values:
                parent.zero_grad()

    def __repr__(self):
        return self.repr()

    def repr(self, depth=1):
        if depth != 0:
            return f"Value({self.n: 0.4g}, parent_op={repr(self.parent_op)}, parent_values=[{', '.join(value.repr(depth=depth-1) for value in self.parent_values)}])"

        else:
            return f"Value({self.n: 0.4g}, parent_op={repr(self.parent_op)}, parent_values={'[...]' if self.parent_values else '[]'})"


def register_op(op_name):
    func = getattr(Value, op_name)

    setattr(Value, f"__{op_name}__", lambda self, other: func(self, other))
    setattr(Value, f"__r{op_name}__", lambda self, other: func(self, other))


for op in ["add", "sub", "mul", "div", "pow"]:
    register_op(op)
