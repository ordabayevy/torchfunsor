# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import inspect
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch
import torch.fx as fx
from torch.fx.graph_module import _forward_from_src
from torch.fx.node import Node, Target
from torch.fx.proxy import Proxy, TracerBase


def is_impure_node(node: Node) -> bool:
    if node.op == "placeholder":
        return False
    return node.is_impure()


class FunsorTracer(fx.Tracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.root = torch.nn.Module()
        self.graph = fx.Graph(tracer_cls=type(self))
        self.tensor_attrs = {}
        self.proxies = {}

    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[Node], "fx.Proxy"] = None,
    ):
        if proxy_factory_fn is None:

            def proxy_factory_fn(node):
                return Funsor(node, self)

        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        self.proxies[target] = proxy
        return proxy


class Funsor(Proxy):
    def __init__(self, node: Node, tracer: TracerBase | None = None):
        super().__init__(node, tracer)
        output = self.tracer.create_node(
            "output",
            "output",
            (self.tracer.create_arg(self),),
            {},
            # type_expr=type(self),
        )
        self.funsor_graph = copy.deepcopy(tracer.graph)
        # self.funsor_graph.output(self.tracer.create_arg(self))
        self.funsor_graph.eliminate_dead_code(is_impure_node=is_impure_node)
        self.tracer.graph.erase_node(output)

        python_code = self.funsor_graph.python_code(root_module="self")
        self._code = python_code.src
        self._lineno_map = python_code._lineno_map

        # cls = type(self.tracer.root)
        # co_fields = self._graph._co_fields if hasattr(self._graph, "_co_fields") else {}
        co_fields = {}
        self.forward = _forward_from_src(self._code, python_code.globals, co_fields)
        sig = inspect.signature(self.forward)
        self.inputs = list(sig.parameters)[1:]

    def __call__(self, *args, **kwargs) -> "Funsor":
        # out = self.forward(self.tracer.root, 0)

        # Parse args
        if len(args) > len(self.inputs):
            raise ValueError(f"Too many arguments. Expected no more than {len(self.inputs)} but got {len(args)}")
        subs = dict(zip(self.inputs, args))

        # Parse kwargs
        for k in kwargs:
            if k in subs:
                raise ValueError(f"Duplicate argument {k}")
            elif k in self.inputs:
                subs[k] = kwargs[k]
            else:
                raise ValueError(f"Invalid argument {k}")

        for key in self.inputs:
            if key not in subs:
                subs[key] = self.tracer.proxies[key]

        subs["self"] = self.tracer.root

        return self.forward(**subs)


tracer = FunsorTracer()

x = tracer.create_proxy("placeholder", "x", (), {}, type_expr=None)
y = tracer.create_proxy("placeholder", "y", (), {}, type_expr=None)
u = (x * y).logsumexp(1)
w = u(x=torch.ones(10, 30, 1).cuda(), y=torch.ones(30, 30).cuda())
compiled_u = torch.compile(u.__call__)
compiled_w = compiled_u(x=torch.ones(10000, 2900, 1).cuda(), y=torch.ones(2900, 2900).cuda())
