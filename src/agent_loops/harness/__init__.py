from __future__ import annotations

from typing import Any


def apply(loop: Any, layers: list) -> Any:

    def run(task: str, env: Any, llm: Any, history: list | None = None, **kwargs: Any):
        built = []
        for factory in layers:
            layer = factory()
            built.append(layer)
            if hasattr(layer, "wrap_env"):
                env = layer.wrap_env(env)
            if hasattr(layer, "wrap_llm"):
                llm = layer.wrap_llm(llm)
        trace = loop.run(task=task, env=env, llm=llm, history=history, **kwargs)
        for layer in built:
            if hasattr(layer, "finish"):
                layer.finish(trace)
        return trace

    run.NAME = "+".join([loop.NAME, *[f.NAME for f in layers]])
    return run
