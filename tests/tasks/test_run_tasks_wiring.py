import json
from pathlib import Path

from scripts.run_tasks import run_dataset
from tests.replay.tasks import UniversalReplay, replay_mode

SAMPLES = Path(__file__).resolve().parents[1] / "fixtures" / "samples"


def test_run_dataset_writes_results_and_traces(tmp_path):
    written = run_dataset(
        tasks_path=SAMPLES / "tasks.json",
        cells=["single_turn_single_step", "multi_turn_multi_step"],
        loops=["react", "single_call"],
        layers=[],
        limit=30,
        repeats=1,
        temperature=0.0,
        seed=0,
        instruction_variant=None,
        code_timeout=5.0,
        bash_timeout=10.0,
        model="fake",
        base_url="http://x",
        out_dir=tmp_path / "results",
        llm_factory=lambda case, name: (
            lambda tools: UniversalReplay(case, replay_mode(name))
        ),
    )
    assert len(written) == 2
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    for key in (
        "when",
        "cell",
        "model",
        "prompt_version",
        "tools_version",
        "dataset",
        "layers",
        "summaries",
        "cases",
        "config",
    ):
        assert key in payload, key
    assert payload["dataset"].startswith("tasks:samples@")
    assert set(payload["summaries"]) == {"react", "single_call"}
    assert (
        "mean_tps" in payload["summaries"]["react"]
        and "mean_duplicate_tool_calls" in payload["summaries"]["react"]
    )
    assert payload["summaries"]["react"]["accuracy"] == 1.0
    traces = list((tmp_path / "results" / "traces").rglob("*.jsonl"))
    assert traces, (
        "traces must be saved so failure patterns can be classified afterwards"
    )
    line = json.loads(traces[0].read_text(encoding="utf-8").splitlines()[0])
    for key in ("turn", "step", "text", "tool_calls", "tool_name", "observation"):
        assert key in line, key


def test_layers_rename_loops_and_are_recorded(tmp_path):
    written = run_dataset(
        tasks_path=SAMPLES / "tasks.json",
        cells=["single_turn_single_step"],
        loops=["react"],
        layers=["todo"],
        limit=30,
        repeats=1,
        temperature=0.0,
        seed=0,
        instruction_variant=None,
        code_timeout=5.0,
        bash_timeout=10.0,
        model="fake",
        base_url="http://x",
        out_dir=tmp_path / "results",
        llm_factory=lambda case, name: (
            lambda tools: UniversalReplay(case, replay_mode(name))
        ),
    )
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert set(payload["summaries"]) == {"react+todo"} and payload["layers"] == ["todo"]
