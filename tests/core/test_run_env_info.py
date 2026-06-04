from agent_loops.bench.core.config import run_env_info


def test_env_info_has_the_protocol_fields_and_survives_offline():
    info = run_env_info(base_url="http://127.0.0.1:9/v1")
    for key in ("runtime_version", "scorer", "code_commit", "machine", "python"):
        assert key in info, key
    assert info["scorer"].startswith("bfcl-eval") or info["scorer"] == ""
    assert isinstance(info["runtime_version"], str)


def test_full_config_records_every_runconfig_field_even_offline_and_without_git():
    from dataclasses import fields

    from agent_loops.bench.core.config import RunConfig, full_config

    cfg = full_config(
        model="m", base_url="http://127.0.0.1:9/v1", started_at="2026-08-31T00:00:00Z"
    )

    expected = ({f.name for f in fields(RunConfig)} - {"grammar"}) | {
        "grammar_hash",
        "constrained",
    }
    missing = expected - set(cfg)
    assert not missing, f"missing fields: {sorted(missing)}"
    assert cfg["code_commit"], "a source hash must exist even without git"
    assert cfg["dataset_revision"].startswith("bfcl-eval")
    assert cfg["model_id"] == "m" and cfg["started_at"] == "2026-08-31T00:00:00Z"
