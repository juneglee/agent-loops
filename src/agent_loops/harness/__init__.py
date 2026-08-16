from __future__ import annotations

from typing import Any


def apply(loop: Any, layers: list) -> Any:

    def run(task: str, env: Any, llm: Any, history: list | None = None, **kwargs: Any):
        for factory in layers:
            layer = factory()
            if hasattr(layer, "wrap_env"):
                env = layer.wrap_env(env)
            if hasattr(layer, "wrap_llm"):
                llm = layer.wrap_llm(llm)
        return loop.run(task=task, env=env, llm=llm, history=history, **kwargs)

    run.NAME = "+".join([loop.NAME, *[f.NAME for f in layers]])
    return run
