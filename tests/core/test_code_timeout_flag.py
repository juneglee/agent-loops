from types import SimpleNamespace

import agent_loops.bench.bfcl.track as track_mod
import agent_loops.bench.core.codeact_setup as codeact_mod
from agent_loops.bench.bfcl.runner import run_case


def test_run_case_forwards_code_timeout_to_the_env(monkeypatch):
    captured: dict = {}

    class _CapturingEnv:
        def __init__(self, case, long_context=False, **kwargs):
            captured.update(kwargs)
            self.calls: list = []

    monkeypatch.setattr(track_mod, "BFCLEnv", _CapturingEnv)
    monkeypatch.setattr(track_mod, "tools_for_case", lambda case: [])
    monkeypatch.setattr(track_mod, "turns_of", lambda case: [])
    monkeypatch.setattr(codeact_mod, "prepare", lambda env, tools, name: ([], None))

    loop = SimpleNamespace(NAME="react", run=None)
    run_case(
        {"id": "c0"}, loop, lambda tools: None, "multi_turn_base", code_timeout=9.5
    )

    assert captured["code_timeout"] == 9.5
