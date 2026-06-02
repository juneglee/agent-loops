import importlib
import inspect
import json

import pytest

pytestmark = pytest.mark.integration


def test_data_comes_from_the_installed_scorer_package():
    import bfcl_eval

    from agent_loops.bench.bfcl.adapter import (
        DATASET_VERSION,
        dataset_revision,
        package_data_dir,
    )

    assert (
        package_data_dir()
        .resolve()
        .is_relative_to(__import__("pathlib").Path(bfcl_eval.__file__).parent.resolve())
    )
    assert (package_data_dir() / f"{DATASET_VERSION}_multi_turn_base.json").exists()
    assert DATASET_VERSION in dataset_revision() and "bfcl-eval" in dataset_revision()


def test_function_docs_match_backend_methods_for_every_class():
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING

    from agent_loops.bench.bfcl.adapter import _FUNC_DOC, package_data_dir

    for cls, filename in _FUNC_DOC.items():
        documented = {
            json.loads(l)["name"]
            for l in (package_data_dir() / "multi_turn_func_doc" / filename).open()
            if l.strip()
        }
        inst = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[cls]), cls)()
        methods = {
            n
            for n, _ in inspect.getmembers(inst, inspect.ismethod)
            if not n.startswith("_")
        }
        assert documented == methods, (
            f"{cls}: only in docs {documented - methods} / only in backend {methods - documented}"
        )


def test_ground_truth_calls_only_documented_functions():
    from agent_loops.bench.bfcl.adapter import (
        _FUNC_DOC,
        load_cases,
        load_ground_truth,
        package_data_dir,
    )

    docs = {
        cls: {
            json.loads(l)["name"]
            for l in (package_data_dir() / "multi_turn_func_doc" / fn).open()
            if l.strip()
        }
        for cls, fn in _FUNC_DOC.items()
    }
    gt = load_ground_truth("multi_turn_base")
    for case in load_cases("multi_turn_base"):
        allowed = set().union(*(docs[k] for k in case["involved_classes"]))
        for turn in gt[case["id"]]:
            for call in turn:
                assert call.split("(", 1)[0] in allowed, f"{case['id']}: {call}"
