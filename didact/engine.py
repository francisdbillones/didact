from __future__ import annotations

from enum import Enum
from math import prod
import math

import numpy as np
from graphviz import Digraph


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
        if self.parent_op == "add":
            a.backward(grad)
            b.backward(grad)

        elif self.parent_op == "mul":
            a.backward(grad * b.n)
            b.backward(grad * a.n)

        elif self.parent_op == "pow":
            a, b = self.parent_values
            a.backward(grad * b.n * a.n ** (b.n - 1))
            b.backward(grad * (a.n ** b.n) * math.log(a.n))

    def zero_grad(self):
        self.grad = 0

        if self.parent_values is not None:
            for parent in self.parent_values:
                parent.zero_grad()

    def __repr__(self):
        return f"Value({self.n}, parent_op={self.parent_op}, parent_values={self.parent_values})"


def register_op(op_name):
    func = getattr(Value, op_name)

    setattr(Value, f"__{op_name}__", lambda self, other: func(self, other))
    setattr(Value, f"__r{op_name}__", lambda self, other: func(other, self))


for op in ["add", "sub", "mul", "div", "pow"]:
    register_op(op)
