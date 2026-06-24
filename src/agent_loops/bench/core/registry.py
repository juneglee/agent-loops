from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_loops.loops import (
    plan_and_solve,
    react,
    single_call,
)

LOOPS = {
    m.NAME: m
    for m in (
        single_call,
        react,
        plan_and_solve,
    )
}

LOOP_KWARGS = {
    "react": {"max_steps": 10},
}


def kwargs_for(name: str) -> dict:
    return LOOP_KWARGS.get(name) or LOOP_KWARGS.get(name.split("+")[0], {})


LAYERS: dict[str, Any] = {}


def with_layers(loops: dict, layer_names: list) -> tuple[dict, list]:
    if not layer_names:
        return loops, []
    raise KeyError(layer_names[0])


@dataclass(frozen=True)
class Stack:
    name: str
    run: Callable[..., Any]
    kwargs: dict[str, Any]
    layers: tuple[str, ...] = ()


class Registry:
    def __init__(
        self, loops: dict[str, Any] | None = None, layers: dict[str, Any] | None = None
    ) -> None:
        self._loops = dict(loops if loops is not None else LOOPS)
        self._layers = dict(layers if layers is not None else LAYERS)

    def names(self) -> list[str]:
        return list(self._loops)

    def resolve(self, name: str, layers: list[str] | tuple[str, ...] = ()) -> Stack:
        module = self._loops[name]
        if not layers:
            return Stack(name=name, run=module.run, kwargs=dict(kwargs_for(name)))
        wrapped, _ = with_layers({name: module}, list(layers))
        stacked = next(iter(wrapped.values()))
        return Stack(
            name=stacked.NAME,
            run=stacked.run,
            kwargs=dict(kwargs_for(stacked.NAME)),
            layers=tuple(layers),
        )

    def extra_tools(self, layers: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        return with_layers({}, list(layers))[1]

    @classmethod
    def default(cls) -> Registry:
        return cls()


__all__ = [
    "LAYERS",
    "LOOPS",
    "LOOP_KWARGS",
    "Registry",
    "Stack",
    "kwargs_for",
    "with_layers",
]
