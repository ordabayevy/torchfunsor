# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import operator

import pytest
import torch

from funsor import Variable


def test_variable():
    x = Variable("x", dtype=torch.float32)
    assert x("x") is x
    assert x(x) is x
    y = Variable("y", dtype=torch.float32)
    assert x("y") is y
    assert x(x="y") is y
    assert x(x=y) is y
    assert y is not x
    assert y(x) is x

    xp1 = x + 1.0
    assert xp1(x=2.0) == 3.0


def test_substitute():
    x = Variable("x", dtype=torch.float32)
    y = Variable("y", dtype=torch.float32)
    z = Variable("z", dtype=torch.float32)

    f = x * y + x * z

    assert f(y=2) == x * 2 + x * z
    assert f(z=2) == x * y + x * 2
    assert f(y=x) == x * x + x * z
    assert f(x=y) == y * y + y * z
    assert f(y=z, z=y) == x * z + x * y
    assert f(x=y, y=z, z=x) == y * z + y * x


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("op", ["abs", "neg"])
def test_operator_unary(op, value):
    expected = getattr(operator, op)(value)

    x = Variable("x", dtype=torch.float32)
    lazy_unary = getattr(operator, op)(x)
    actual = lazy_unary(value)

    assert actual == expected


@pytest.mark.parametrize("value", [torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)])
@pytest.mark.parametrize(
    "op",
    [
        "abs",
        "ceil",
        "floor",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "acos",
        "cos",
    ],
)
def test_torch_unary(op, value):
    expected = getattr(torch, op)(value)

    x = Variable("x", dtype=torch.float32)
    lazy_unary = getattr(torch, op)(x)
    actual = lazy_unary(value)

    assert actual == expected


BINARY_OPS = [
    # max,
    # min,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.lt,
    operator.le,
    operator.ge,
    operator.gt,
]
BOOLEAN_OPS = [operator.and_, operator.or_, operator.xor]


@pytest.mark.parametrize("value1", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("value2", [0.0, 0.8, 1.0])
@pytest.mark.parametrize("binary_op", BINARY_OPS + BOOLEAN_OPS)
def test_binary(binary_op, value1, value2):
    if binary_op in BOOLEAN_OPS:
        value1 = bool(value1)
        value2 = bool(value2)
    try:
        expected = binary_op(value1, value2)
    except ZeroDivisionError:
        return

    x1 = Variable("x1", dtype=torch.float32)
    x2 = Variable("x2", dtype=torch.float32)
    actual = binary_op(x1, x2)(value1, value2)

    assert actual == expected


REDUCE_OPS = [
    torch.sum,
    # torch.prod,
    torch.logsumexp,
    torch.mean,
    torch.std,
    torch.var,
    torch.amax,
    torch.amin,
    torch.all,
    torch.any,
]


@pytest.mark.parametrize("reduce_op", REDUCE_OPS)
def test_reduce_all(reduce_op):
    x = Variable("x", dtype=torch.float32)
    y = Variable("y", dtype=torch.float32)
    z = Variable("z", dtype=torch.float32)
    x_value = torch.arange(2)
    y_value = torch.randn(3)
    z_value = torch.rand(4)

    f = x * y + z
    actual = f.reduce(reduce_op, {"y": y_value, "x": x_value, "z": z_value})

    expected = reduce_op(x_value[:, None, None] * y_value[:, None] + z_value, dim=(0, 1, 2))

    assert actual == expected
