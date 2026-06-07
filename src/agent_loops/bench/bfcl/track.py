from __future__ import annotations

from typing import Any

from agent_loops.bench.bfcl.adapter import (
    dataset_revision,
    load_ground_truth,
    tools_for_case,
    turns_of,
)
from agent_loops.bench.bfcl.env import BFCLEnv


class BfclTrack:
    def __init__(self, category: str, only_classes: set[str] | None = None) -> None:
        self.category = category
        self.only_classes = only_classes
        self.name = f"bfcl:{category}"

    def revision(self) -> str:
        return dataset_revision()

    def cases(self, cell: str) -> list[dict[str, Any]]:
        from agent_loops.bench.bfcl.partition import partition

        return partition(self.category, only_classes=self.only_classes).get(cell, [])

    def turns_of(self, case: dict[str, Any]) -> list[str]:
        return turns_of(case)

    def tools_for(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        return tools_for_case(case)

    def make_env(self, case: dict[str, Any], budgets: Any) -> BFCLEnv:
        long_context = "long_context" in self.category or "composite" in self.category
        return BFCLEnv(
            case, long_context=long_context, code_timeout=budgets.code_timeout
        )

    def score(
        self, case: dict[str, Any], env: Any, turn_traces: list[Any]
    ) -> tuple[bool, str | None]:
        try:
            from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
                multi_turn_checker,
            )

            from agent_loops.bench.bfcl.runner import _decoded_from_turn_traces

            n_turns = len(turn_traces)
            scored = multi_turn_checker(
                multi_turn_model_result_list_decoded=_decoded_from_turn_traces(
                    turn_traces
                ),
                multi_turn_ground_truth_list=load_ground_truth(self.category)[
                    case["id"]
                ][:n_turns],
                test_entry=case,
                test_category=self.category,
                model_name=f"agentloops_{turn_traces[0].loop if turn_traces else 'unknown'}",
            )
            if bool(scored.get("valid")):
                return True, None
            return False, f"{scored.get('error_type')}: {scored.get('error_message')}"[
                :200
            ]
        finally:
            _release_scorer_instances(case["id"])


def _release_scorer_instances(case_id: str) -> None:
    try:
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    except ImportError:
        return
    g = vars(multi_turn_utils)
    for key in [k for k in g if k.endswith("_instance") and f"_{case_id}_" in k]:
        del g[key]
