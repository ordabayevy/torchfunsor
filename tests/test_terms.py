# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from typing import Annotated

import pytest
import torch

from funsor import Variable
from funsor.utilities.testing import check_funsor

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


@pytest.mark.parametrize("domain", [Annotated[torch.Tensor, int, ()], Annotated[torch.Tensor, torch.float32, ()]])
def test_variable(domain):
    dtype, shape = domain.__metadata__
    x = Variable("x", dtype, shape)
    check_funsor(x, {"x": domain}, domain)
    assert x("x") is x
    assert x(x) is x
    y = Variable("y", dtype, shape)
    assert x("y") is y
    assert x(x="y") is y
    assert x(x=y) is y
    assert y is not x
    assert y(x) is x

    xp1 = x + 1.0
    assert xp1(x=2.0) == 3.0


def test_substitute():
    x = Variable("x", torch.float32)
    y = Variable("y", torch.float32)
    z = Variable("z", int)

    f = x * y + x * z
    check_funsor(f, {"x": torch.float32, "y": torch.float32, "z": int}, torch.float32)

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

    x = Variable("x", torch.float32)
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

    x = Variable("x", torch.float32)
    lazy_unary = getattr(torch, op)(x)
    actual = lazy_unary(value)

    assert actual == expected


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

    x1 = Variable("x1", torch.float32)
    x2 = Variable("x2", torch.float32)
    actual = binary_op(x1, x2)(value1, value2)

    assert actual == expected


@pytest.mark.parametrize("reduce_op", REDUCE_OPS)
def test_reduce_all(reduce_op):
    x = Variable("x", torch.float32)
    y = Variable("y", torch.float32)
    z = Variable("z", torch.float32)
    x_value = torch.arange(2)
    y_value = torch.randn(3)
    z_value = torch.rand(4)

    f = x * y + z
    actual = f.reduce(reduce_op, {"y": y_value, "x": x_value, "z": z_value})

    expected = reduce_op(x_value[:, None, None] * y_value[:, None] + z_value, dim=(0, 1, 2))

    assert actual == expected


@pytest.mark.parametrize("event_shape", [(3, 2)], ids=str)
@pytest.mark.parametrize("dim", [None, 0, (1,), (0, 1)], ids=str)
@pytest.mark.parametrize("keepdims", [False, True], ids=str)
@pytest.mark.parametrize("reduce_op", REDUCE_OPS)
def test_reduce_event(reduce_op, event_shape, dim, keepdims):
    data = torch.randn((5,) + event_shape)
    i = Variable("i", int)
    x = data[i]
    if reduce_op is torch.logsumexp and dim is None:
        dim = (0, 1)
    reduced_funsor = reduce_op(x, dim=dim, keepdims=keepdims)

    # compute expected shape
    dim = (0, 1) if dim is None else dim
    dim = (dim,) if isinstance(dim, int) else dim
    if keepdims:
        shape = tuple(1 if i in dim else event_shape[i] for i in range(len(event_shape)))
    else:
        shape = tuple(event_shape[i] for i in range(len(event_shape)) if i not in dim)
    dtype = torch.bool if reduce_op in (torch.all, torch.any) else x.dtype
    check_funsor(reduced_funsor, {"i": int}, Annotated[torch.Tensor, dtype, shape])

    for i in range(5):
        actual = reduced_funsor(i=i)
        expected = reduce_op(data[i], dim=dim, keepdim=keepdims)
        torch.testing.assert_close(actual, expected)
