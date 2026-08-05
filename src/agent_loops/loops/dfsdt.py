"""DFSDT (ToolLLM, arXiv:2307.16789, https://github.com/OpenBMB/ToolBench).

Source: arXiv:2307.16789 — "ToolLLM: Facilitating Large Language Models to Master
16000+ Real-world APIs". Reference implementation: https://github.com/OpenBMB/ToolBench
(``toolbench/inference/Algorithms/DFS.py`` ``DFS_tree_search.DFS`` with ``with_filter=False``,
``Tree/Tree.py``, ``Prompts/Tree_search_prompts.py`` ``DIVERSITY_PROMPT``).

The authors' algorithm (``DFS``, pre-order, ``with_filter=False`` = DFSDT):
  * A node is a point where the model is asked. Each node may receive up to
    ``tree_beam_size`` children (here ``breadth``). Children are generated one at a time
    and each child subtree is searched to completion before the next sibling is
    generated — the next sibling is generated whenever the child subtree returned without
    a final answer, not only after a give-up.
  * From the second child on, the prompt is extended with ``DIVERSITY_PROMPT``: the
    previous children's first actions as a JSON list of ``{name, arguments,
    function_output}`` (plus ``again, your former observation: …`` when the node itself
    has an observation) and the fixed sentence that all previous trails failed and the
    model must act differently. The message is not inherited by the child nodes.
  * The model ends a trail with ``Finish with Final Answer`` or ``Finish by Giving Up``.
    The first final answer terminates the whole search. A give-up marks the leaf pruned
    and returns ``prune_back_length = 2``: the node that emitted the give-up is closed and
    the search resumes at that node's parent with the next sibling.
  * Reaching ``single_chain_max_step`` (node depth) or an LLM parse error marks the node
    pruned and returns 1: the parent simply continues with its next sibling. Tool errors
    are ordinary observations and never prune.
  * A thought-only response creates a node and the search recurses into it, so a thought
    consumes one unit of width and one unit of depth.
  * ``max_query_count`` bounds the number of LLM calls over the whole tree.
  * When the search ends without a final answer, one give-up node's text is adopted as the
    answer with ``finish_type = "give_up"``.

Adaptations for this repository:
  * Native tool calling instead of the authors' ``Finish`` function: a response without
    tool calls is a Final Answer when ``response_is_complete`` sees the ``Final:`` marker, a
    Giving Up when ``response_gives_up`` sees the ``Give up:`` marker, a parse error when the
    client set ``parse_ok = False``, and otherwise a thought.
  * Depth counts model responses along the chain (thought or action node = 1). The
    authors count Thought/Action/Action-Input nodes separately (``12`` ≈ 4 actions);
    ``max_depth = 6`` is the scaled default. ``breadth = 2`` (authors' default
    ``tree_beam_size``) and ``max_calls = 30`` (authors use 200 with a hosted API; the
    on-device budget is scaled to our other loops).
  * A multi-call response is one action node: every call is executed in order and
    recorded as a ``Step`` (``llm_response`` on the first only). Its diversity entry uses
    the first call's name/arguments and a list of observations as ``function_output``.
  * The budget is checked before a call, so the last permitted response is still used
    (the authors discard the response that crosses ``max_query_count``).
  * On exhaustion the last give-up text is adopted (the authors pick one at random);
    it is written into the last ``Step`` that carries an ``llm_response`` under ``text``
    with ``finish_type = "give_up"`` because ``Trace`` has no final-text field, and the
    trace is labelled ``give_up``. Exhaustion without any give-up is labelled
    ``parse_fail`` when a parse error occurred, otherwise ``max_depth``.
  * ``DIVERSITY_PROMPT`` lives here verbatim because ``bench.prompts`` holds only the
    per-loop instruction; it is appended as a ``user`` message.
  * Environment side effects are not rolled back on backtracking. The authors'
    ``io_state`` is deep-copied per node because their APIs are read-only; a file-system
    environment keeps whatever a pruned branch did. Snapshot/restore is a harness concern.

Not included (deliberately): the API retriever, ToolBench training data, the
``with_filter=True`` LLM ranking of siblings (plain DFS, not DFSDT), and any automatic
branching on tool errors.
"""

from __future__ import annotations

import json
from typing import Any

from agent_loops.bench.prompts import build_messages
from agent_loops.loops.base import (
    Step,
    Trace,
    calls_of,
    response_gives_up,
    response_is_complete,
)

NAME = "dfsdt"

DIVERSITY_PROMPT = (
    "This is not the first time you try this task, all previous trails failed.\n"
    "Before you generate my thought for this state, I will first show you your previous "
    "actions for this state, and then you must generate actions that is different from all "
    "of them. Here are some previous actions candidates:\n"
    "{previous_candidate}\n"
    "Remember you are now in the intermediate state of a trail, you will first analyze the "
    "now state and previous action candidates, then make actions that is different from all "
    "the previous."
)

PRUNE_BACK_LENGTH = 2
FINAL = "final"
BUDGET = "budget"


def _node(
    history: list[dict[str, Any]],
    action: dict[str, Any] | None = None,
    gave_up: dict[str, Any] | None = None,
    pruned: bool = False,
) -> dict[str, Any]:
    return {
        "history": history,
        "children": [],
        "action": action,
        "gave_up": gave_up,
        "pruned": pruned,
    }


def _first_action(node: dict[str, Any]) -> dict[str, Any] | None:
    while node["action"] is None and node["children"]:
        node = node["children"][0]
    return node["action"]


def _diversity_message(node: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        a for a in (_first_action(c) for c in node["children"]) if a is not None
    ]
    if not candidates:
        return None
    shown = json.dumps(candidates, indent=2, ensure_ascii=False, default=str) + "\n"
    if node["action"] is not None:
        shown += (
            f"again, your former observation: {node['action']['function_output']}\n"
        )
    return {
        "role": "user",
        "content": DIVERSITY_PROMPT.replace("{previous_candidate}", shown),
    }


def run(
    task: str,
    env: Any,
    llm: Any,
    breadth: int = 2,
    max_calls: int = 30,
    max_depth: int = 6,
    history: list | None = None,
) -> Trace:
    trace = Trace(task=task, loop=NAME)
    prior = list(history or [])
    state: dict[str, Any] = {"calls": 0, "give_ups": [], "parse_failed": False}

    def expand(node: dict[str, Any], depth: int) -> str | int:
        if node["gave_up"] is not None:
            state["give_ups"].append(node["gave_up"])
            return PRUNE_BACK_LENGTH
        if node["pruned"] or depth >= max_depth:
            return 1
        for _ in range(breadth):
            if state["calls"] >= max_calls:
                return BUDGET
            context = list(node["history"])
            diversity = _diversity_message(node) if node["children"] else None
            if diversity is not None:
                context.append(diversity)
            response = llm(messages=build_messages(NAME, task, context, prior=prior))
            state["calls"] += 1
            calls = calls_of(response)
            child_history = [
                *node["history"],
                {"role": "assistant", "content": response},
            ]
            if not calls:
                trace.steps.append(Step(llm_response=response))
                if response_is_complete(response):
                    return FINAL
                if response_gives_up(response):
                    text = str(response.get("text") or "")
                    child = _node(
                        child_history,
                        gave_up=response,
                        action={
                            "name": "Finish by Giving Up",
                            "arguments": {"reason": text},
                            "function_output": "",
                        },
                    )
                elif response.get("parse_ok") is False:
                    state["parse_failed"] = True
                    child = _node(child_history, pruned=True)
                else:
                    child = _node(child_history)
            else:
                observations = []
                for i, (name, arguments) in enumerate(calls):
                    observation = env.execute(name, arguments)
                    observations.append(observation)
                    trace.steps.append(
                        Step(
                            llm_response=response if i == 0 else None,
                            tool_name=name,
                            tool_arguments=arguments,
                            observation=observation,
                        )
                    )
                    child_history.append({"role": "tool", "content": observation})
                first_name, first_args = calls[0]
                child = _node(
                    child_history,
                    action={
                        "name": first_name,
                        "arguments": first_args,
                        "function_output": observations[0]
                        if len(observations) == 1
                        else observations,
                    },
                )
            node["children"].append(child)
            result = expand(child, depth + 1)
            if result in (FINAL, BUDGET):
                return result
            if result > 1:
                return result - 1
        return 1

    result = expand(_node([]), 0)
    if result == FINAL:
        return trace.stop("success")
    if result == BUDGET:
        return trace.stop("max_steps")
    if state["give_ups"]:
        adopted = state["give_ups"][-1]
        for step in reversed(trace.steps):
            if step.llm_response is not None:
                step.llm_response = {
                    **step.llm_response,
                    "text": adopted.get("text", ""),
                    "finish_type": "give_up",
                }
                break
        return trace.stop("give_up")
    return trace.stop("parse_fail" if state["parse_failed"] else "max_depth")
