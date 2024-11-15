# Copyright Contributors to the TorchFunsor project.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable
from typing import Annotated, Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch._library.fake_class_registry import FakeScriptObject
from torch.fx.experimental.meta_tracer import (
    MetaProxy,
    MetaTracer,
    gen_constructor_wrapper,
    manual_meta_overrides,
)
from torch.fx.node import Argument, Target


def node_to_meta(v):
    if isinstance(v, fx.Node):
        meta_val = v.meta.get("val")
        assert meta_val is not None, f"Node {v} does not have a meta value"
        return meta_val
    return v


class FunsorTracer(MetaTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.root = torch.nn.Module()
        self.graph = fx.Graph(tracer_cls=type(self))
        self.tensor_attrs: dict[torch.Tensor | ScriptObject | FakeScriptObject, str] = {}
        self.variable_cache: dict[str, Variable] = {}
        self.node_cache: dict[Any, fx.Node] = {}

        self.meta_args = {}

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, torch.Tensor):
            meta_val = a.to(device="meta")
            a = super().create_arg(a)
            a.meta["val"] = meta_val
            return a
        return super().create_arg(a)

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

        rv = fx.Tracer.create_proxy(self, kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

        if kind == "placeholder" and target in self.meta_args:
            raise ValueError("Should not be here")

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = pytree.tree_map(node_to_meta, rv.node.args)
            kwargs_metas = pytree.tree_map(node_to_meta, rv.node.kwargs)

            if kind == "call_function":
                meta_target = manual_meta_overrides.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                if target == "__getitem__":
                    # Scalar tensors lead to the following error:
                    # RuntimeError: Tensor.item() cannot be called on meta tensors
                    # This is a workaround to convert scalar tensors to Python scalars
                    indices = args_metas[1]
                    new_indices = ()
                    for index in indices:
                        if isinstance(index, torch.Tensor) and index.ndim == 0:
                            new_indices += (0,)
                        else:
                            new_indices += (index,)
                    args_metas = (args_metas[0], new_indices)
                meta_out = getattr(args_metas[0], target)(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
            elif kind == "call_module":
                assert hasattr(self, "orig_forward")
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)  # type: ignore[misc, arg-type]
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    assert isinstance(attr_itr, torch.Tensor)
                    meta_out = attr_itr.to(device="meta")
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            # TODO
            assert isinstance(rv, torch.fx.Proxy), "Dont support composite output yet"
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv

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
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_cache[cache_key] = node
        return node


tracer_stack: list[FunsorTracer] = []
tracer_stack.append(FunsorTracer())


def is_impure_node(node: fx.Node) -> bool:
    if node.op == "placeholder":
        return False
    return node.is_impure()


class Funsor(MetaProxy):
    """
    Abstract base class for immutable functional tensors.

    Args:
        node:
            The underlying FX node.
        tracer:
            The tracer to use for creating the Funsor.
    """

    def __init__(
        self,
        node: fx.Node,
        tracer: FunsorTracer | None = None,
    ) -> None:
        if tracer is None:
            tracer = tracer_stack[-1]
        super().__init__(node, tracer)

        self._inputs: dict[str, Any] | None = None
        self._graph: fx.Graph | None = None

    def install_tensor_meta(self, tensor_meta):
        self.node.meta["val"] = tensor_meta
        self.node.type = type(tensor_meta)
        super().install_tensor_meta(tensor_meta)

    @property
    def graph(self):
        if self._graph is None:
            memo = {}
            self._graph = fx.Graph(tracer_cls=type(self.tracer))
            self._graph.graph_copy(self.tracer.graph, val_map=memo, return_output_node=False)
            self._graph.output(memo[self.node], type_expr=self.node.type)
            self._graph.eliminate_dead_code(is_impure_node=is_impure_node)
        return self._graph

    @property
    def inputs(self) -> dict[str, Any]:
        if self._inputs is None:
            self._inputs = {}
            for graph_node in self.graph.nodes:
                if graph_node.op == "placeholder":
                    meta_val = graph_node.meta["val"]
                    self._inputs[graph_node.name] = Annotated[type(meta_val), meta_val.dtype, tuple(meta_val.shape)]
        return self._inputs

    @property
    def output(self) -> Any:
        meta_val = self.node.meta["val"]
        return Annotated[type(meta_val), meta_val.dtype, tuple(meta_val.shape)]

    @property
    def __annotations__(self) -> dict[str, Any]:
        type_hints = dict(self.inputs)
        type_hints["return"] = self.output
        return type_hints

    # TODO: revisit this
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

        interpreter = fx.Interpreter(self.tracer.root, graph=self.graph)
        return interpreter.run(*tuple(subs[key] for key in self.inputs))

    def __repr__(self) -> str:
        return f"Funsor({', '.join([f'{name}: {output}' for name, output in self.inputs.items()])}) -> {self.output}"

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
    """
    Funsor representing a single free variable.

    Example:

    >>> x = Variable("x")
    >>> y = Variable("y")
    >>> f = x * y + x
    >>> f(x=1, y=2)

    Args:
        name:
            The name of the variable.
        dtype:
            The data type of the variable.
        shape:
            The shape of the variable.
        tracer:
            The tracer to use for creating the variable.
    """

    def __init__(
        self, name: str, dtype: torch.dtype, shape: tuple[int, ...] = (), tracer: FunsorTracer | None = None
    ) -> None:
        if tracer is None:
            tracer = tracer_stack[-1]
        tracer.meta_args[name] = torch.empty(shape, dtype=dtype, device="meta")

        placeholder_node = tracer.create_node("placeholder", name, (), {})
        super().__init__(placeholder_node, tracer)

        self.install_tensor_meta(tracer.meta_args[name])
        tracer.variable_cache[name] = self

    @property
    def name(self) -> str:
        return self.node.name

    def __repr__(self) -> str:
        return f"Variable({self.node.name}: {self.output})"
