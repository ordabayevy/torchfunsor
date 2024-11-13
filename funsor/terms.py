# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.graph_module import _forward_from_src
from torch.fx.node import Argument, Target


class FunsorTracer(fx.Tracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.root = torch.nn.Module()
        self.graph = fx.Graph(tracer_cls=type(self))
        self.tensor_attrs: dict[torch.Tensor | ScriptObject | FakeScriptObject, str] = {}
        self.variable_cache: dict[str, Variable] = {}
        self.node_cache: dict[Any, fx.Node] = {}

    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[fx.Node], fx.Proxy] = None,
    ) -> "Funsor":
        if proxy_factory_fn is None:

            def proxy_factory_fn(node):
                return Funsor(node, self)

        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        return proxy

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> fx.Node:
        # Create a cache key based on the arguments
        cache_key = (kind, target, args, tuple(kwargs.items()))

        # Check if the node already exists in the cache
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        # Otherwise, create the node and cache it
        if type_expr is None:
            type_expr = Any
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_cache[cache_key] = node
        return node


tracer_stack: list[FunsorTracer] = []
tracer_stack.append(FunsorTracer())


def is_impure_node(node: fx.Node) -> bool:
    if node.op == "placeholder":
        return False
    return node.is_impure()


class Funsor(fx.Proxy):
    def __init__(
        self,
        node: fx.Node,
        tracer: FunsorTracer | None = None,
    ) -> None:
        if tracer is None:
            tracer = tracer_stack[-1]
        super().__init__(node, tracer)

        memo = {}
        self.funsor_graph = fx.Graph(tracer_cls=type(tracer))
        self.funsor_graph.graph_copy(tracer.graph, val_map=memo, return_output_node=False)
        self.funsor_graph.output(memo[node], type_expr=node.type)
        self.funsor_graph.eliminate_dead_code(is_impure_node=is_impure_node)

        python_code = self.funsor_graph.python_code(root_module="self")
        self._code = python_code.src
        self._lineno_map = python_code._lineno_map
        co_fields = {}
        self.forward = _forward_from_src(self._code, python_code.globals, co_fields)
        sig = inspect.signature(self.forward)

        self.inputs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            elif param.annotation is inspect.Parameter.empty:
                self.inputs[name] = Any
            else:
                self.inputs[name] = param.annotation
        self.output = node.type

    @property
    def __annotations__(self) -> dict[str, Any]:
        type_hints = dict(self.inputs)
        type_hints["return"] = self.output
        return type_hints

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Funsor):
            return False
        else:
            return self.node == other.node

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Parse args
        if len(args) > len(self.inputs):
            raise ValueError(f"Too many arguments. Expected no more than {len(self.inputs)} but got {len(args)}")
        subs = dict(zip(self.inputs, args))

        # Parse kwargs
        for key in kwargs:
            if key in subs:
                raise ValueError(f"Duplicate argument {key}")
            elif key in self.inputs:
                subs[key] = kwargs[key]
            else:
                raise ValueError(f"Invalid argument {key}")

        for key in self.inputs:
            if key not in subs:
                subs[key] = self.tracer.variable_cache[key]
            elif isinstance(subs[key], str):
                # TODO: Handle creating new variables
                subs[key] = self.tracer.variable_cache[subs[key]]

        subs["self"] = self.tracer.root

        return self.forward(**subs)

    def __repr__(self) -> str:
        return f"Funsor({', '.join([f'{key}: {value}' for key, value in self.inputs.items()])}) -> {self.output}"

    def reduce(self, reduce_op: Callable, reduced_vars: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, value in reduced_vars.items():
            if key not in self.inputs:
                raise ValueError(f"Invalid argument {key}")
            if value.ndim != 1:
                raise ValueError(f"Expected 1-dimensional tensor for {key} but got {value.ndim}-dimensional tensor")

        in_dims_list = []
        for name in self.inputs:
            in_dims = tuple(0 if name == key else None for key in self.inputs)
            in_dims_list.append(in_dims)

        func = self.__call__
        for in_dims in reversed(in_dims_list):
            func = torch.vmap(func, in_dims, 0)

        args = tuple(reduced_vars[key] for key in self.inputs)

        return reduce_op(func(*args), dim=tuple(range(len(reduced_vars))))


class Variable(Funsor):
    def __init__(self, name: str, output: Any = None, tracer: FunsorTracer | None = None):
        if tracer is None:
            tracer = tracer_stack[-1]
        placeholder_node = tracer.create_node("placeholder", name, (), {}, type_expr=output)
        super().__init__(placeholder_node, tracer)
        tracer.variable_cache[name] = self

    @property
    def name(self) -> str:
        return self.node.name

    def __repr__(self):
        return f"Variable({self.node.name}: {self.output})"
