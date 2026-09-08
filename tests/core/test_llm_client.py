from agent_loops.bench.core.llm import build_payload, call, parse_response


def test_parses_tool_call_from_openai_response():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "ls", "arguments": '{"path": "docs"}'}}
                    ],
                }
            }
        ]
    }

    out = parse_response(raw)

    assert out["tool_calls"] == [{"name": "ls", "arguments": {"path": "docs"}}]
    assert out["parse_ok"] is True


def test_parses_plain_text_response_as_no_tool_call():
    raw = {"choices": [{"message": {"content": "the file is a.txt"}}]}

    out = parse_response(raw)

    assert out["tool_calls"] is None
    assert out["text"] == "the file is a.txt"
    assert out["parse_ok"] is True


def test_malformed_arguments_json_is_parse_failure_not_exception():
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "ls", "arguments": '{"path": "docs"'}}
                    ],
                }
            }
        ]
    }

    out = parse_response(raw)

    assert out["parse_ok"] is False
    assert out["tool_calls"] is None


def test_detects_quiet_failure_when_text_mentions_tool_but_no_call():
    raw = {
        "choices": [
            {"message": {"content": '{"name": "ls", "arguments": {"path": "."}}'}}
        ]
    }

    out = parse_response(raw)

    assert out["tool_calls"] is None
    assert out["quiet_failure"] is True


def test_payload_forces_temperature_zero_by_default():
    payload = build_payload(model="m", messages=[], tools=[])

    assert payload["temperature"] == 0


def test_non_dict_arguments_are_a_parse_failure():
    from agent_loops.bench.core.llm import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "ls", "arguments": "[1, 2]"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    out = parse_response(raw)
    assert out["parse_ok"] is False and out["tool_calls"] is None


def test_missing_tool_name_is_a_parse_failure():
    from agent_loops.bench.core.llm import parse_response

    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    out = parse_response(raw)
    assert out["parse_ok"] is False and out["tool_calls"] is None


def test_build_payload_carries_seed_through_extra():
    from agent_loops.bench.core.llm import build_payload

    payload = build_payload(model="m", messages=[], tools=None, seed=42)
    assert payload["seed"] == 42


def test_transport_error_returns_the_same_keys_as_a_parsed_response(monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    out = call(messages=[{"role": "user", "content": "hi"}])
    assert out["parse_ok"] is False and out["error"].startswith("ConnectionError")
    assert set(out) >= set(parse_response({}))


def test_local_llm_forwards_temperature_and_seed_to_the_runtime(monkeypatch):
    import agent_loops.bench.core.local_llm as mod

    sent = {}

    def spy(**kwargs):
        sent.update(kwargs)
        return {
            "tool_calls": None,
            "text": "",
            "parse_ok": True,
            "quiet_failure": False,
            "truncated": False,
            "raw": {},
        }

    monkeypatch.setattr(mod, "call", spy)
    llm = mod.LocalLLM([], seed=7)
    llm(messages=[{"role": "user", "content": "x"}])
    assert sent.get("seed") == 7
    assert sent.get("temperature") == 0.0


def test_transport_errors_are_retried_before_giving_up(monkeypatch):
    import requests

    from agent_loops.bench.core import llm as mod

    attempts = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def flaky(*_a, **_k):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("down")
        return _Resp()

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    out = mod.call(base_url="http://x/v1", model="m", messages=[], tools=[])

    assert len(attempts) == 3
    assert out["text"] == "hello" and out.get("error") is None


def test_transport_error_after_all_attempts_is_reported(monkeypatch):
    import requests

    from agent_loops.bench.core import llm as mod

    monkeypatch.setattr(
        requests,
        "post",
        lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    out = mod.call(base_url="http://x/v1", model="m", messages=[], tools=[])

    assert out["parse_ok"] is False and "ConnectionError" in out["error"]


def test_local_llm_forwards_cache_prompt_only_when_set(monkeypatch):
    from agent_loops.bench.core import local_llm as mod

    seen = []

    def spy(**kwargs):
        seen.append(kwargs)
        return {"tool_calls": None, "text": "", "parse_ok": True}

    monkeypatch.setattr(mod, "call", spy)
    mod.LocalLLM([], base_url="http://x/v1")([{"role": "user", "content": "hi"}])
    mod.LocalLLM([], base_url="http://x/v1", cache_prompt=False)(
        [{"role": "user", "content": "hi"}]
    )

    assert "cache_prompt" not in seen[0]
    assert seen[1]["cache_prompt"] is False


def test_local_llm_keeps_the_last_transport_error_text(monkeypatch):
    from agent_loops.bench.core import local_llm as mod

    monkeypatch.setattr(
        mod,
        "call",
        lambda **_k: {
            "tool_calls": None,
            "text": "",
            "parse_ok": False,
            "error": "ReadTimeout: slow",
        },
    )
    llm = mod.LocalLLM([], base_url="http://x/v1")
    llm([{"role": "user", "content": "hi"}])

    assert llm.errors == 1 and llm.last_error == "ReadTimeout: slow"


def test_timeouts_are_not_retried(monkeypatch):
    import requests

    from agent_loops.bench.core import llm as mod

    attempts = []

    def slow(*_a, **_k):
        attempts.append(1)
        raise requests.ReadTimeout("slow")

    monkeypatch.setattr(requests, "post", slow)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    out = mod.call(base_url="http://x/v1", model="m", messages=[], tools=[])

    assert len(attempts) == 1 and "ReadTimeout" in out["error"]


def test_local_llm_forwards_max_tokens_and_timeout(monkeypatch):
    from agent_loops.bench.core import local_llm as mod

    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return {"tool_calls": None, "text": "", "parse_ok": True}

    monkeypatch.setattr(mod, "call", spy)
    mod.LocalLLM([], base_url="http://x/v1", timeout=300, max_tokens=4096)(
        [{"role": "user", "content": "hi"}]
    )

    assert seen["timeout"] == 300 and seen["max_tokens"] == 4096
