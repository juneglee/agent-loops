from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_loops.bench.tasks.env import WorkspaceEnv
from agent_loops.bench.tasks.format import dataset_revision, fixture_dir, load_tasks
from agent_loops.bench.tasks.score import final_answer, score
from agent_loops.tools import schemas


def truncated(case: dict[str, Any], n_turns: int | None) -> dict[str, Any]:
    if not n_turns or n_turns >= len(case["turns"]):
        return case
    out = dict(case)
    out["turns"] = case["turns"][:n_turns]
    out["gt_calls"] = case["gt_calls"][:n_turns]
    return out


class TaskTrack:
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.tasks_path = path / "tasks.json" if path.is_dir() else path
        self.base = self.tasks_path.parent
        self.name = f"tasks:{self.base.name}"
        self._cases: list[dict[str, Any]] | None = None

    def revision(self) -> str:
        return dataset_revision(self.tasks_path)

    def all_cases(self) -> list[dict[str, Any]]:
        if self._cases is None:
            self._cases = load_tasks(self.tasks_path)
        return self._cases

    def cases(self, cell: str) -> list[dict[str, Any]]:
        return [c for c in self.all_cases() if c["cell"] == cell]

    def turns_of(self, case: dict[str, Any]) -> list[str]:
        return list(case["turns"])

    def tools_for(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        return schemas()

    def make_env(self, case: dict[str, Any], budgets: Any) -> WorkspaceEnv:
        return WorkspaceEnv(
            fixture_dir(case, self.base),
            code_timeout=budgets.code_timeout,
            bash_timeout=budgets.bash_timeout,
        )

    def score(
        self, case: dict[str, Any], env: Any, turn_traces: list[Any]
    ) -> tuple[bool, str | None]:
        scored = truncated(case, len(turn_traces))
        answer = final_answer(turn_traces[-1]) if turn_traces else ""
        return score(scored, env, self.base, answer)
