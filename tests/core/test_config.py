from agent_loops.bench.core.config import RunConfig, dumps


def test_grammar_off_is_recorded_as_unconstrained():
    cfg = RunConfig(loop="react", grammar=None)

    assert cfg.constrained is False
    assert cfg.grammar_hash is None


def test_grammar_on_records_hash_not_raw_text():
    cfg = RunConfig(loop="react", grammar='root ::= "x"')

    assert cfg.constrained is True
    assert len(cfg.grammar_hash) == 12
    assert "grammar" not in cfg.to_dict()
    assert cfg.to_dict()["grammar_hash"]


def test_same_grammar_gives_same_hash():
    a = RunConfig(loop="a", grammar='root ::= "x"')
    b = RunConfig(loop="b", grammar='root ::= "x"')

    assert a.grammar_hash == b.grammar_hash


def test_key_distinguishes_grid_cells():
    free = RunConfig(loop="react", model_id="e2b", runtime="llamacpp")
    grammar = RunConfig(
        loop="react", model_id="e2b", runtime="llamacpp", grammar='root ::= "x"'
    )
    other_model = RunConfig(loop="react", model_id="e4b", runtime="llamacpp")

    assert free.key() != grammar.key()
    assert free.key() != other_model.key()
    assert free.key() == "react/e2b/llamacpp/free"


def test_dumps_is_json_and_keeps_extra_fields():
    import json

    cfg = RunConfig(loop="react", model_id="e2b")

    out = json.loads(dumps(cfg, accuracy=0.42))

    assert out["loop"] == "react"
    assert out["accuracy"] == 0.42
    assert out["temperature"] == 0.0
