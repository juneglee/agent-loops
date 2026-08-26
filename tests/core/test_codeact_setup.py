from agent_loops.bench.bfcl.partition import partition
from agent_loops.bench.core.codeact_setup import code_signatures, code_tools


def _case_tools():
    from agent_loops.bench.bfcl.adapter import tools_for_case

    return tools_for_case(partition()["single_turn_multi_step"][0])


def test_only_execute_code_is_exposed_natively():
    tools = code_tools(_case_tools())

    assert [t["function"]["name"] for t in tools] == ["execute_code"]


def test_execute_code_takes_a_code_string():
    fn = code_tools([])[0]["function"]

    assert fn["parameters"]["required"] == ["code"]
    assert fn["parameters"]["properties"]["code"]["type"] == "string"


def test_signatures_list_every_file_tool():
    text = code_signatures(_case_tools())

    for name in ("ls", "pwd", "cat", "mkdir", "mv", "touch"):
        assert f"{name}(" in text, f"{name} signature missing"


def test_signatures_mark_optional_parameters():
    text = code_signatures(
        [
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "boolean"},
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
    )

    assert "path: string" in text
    assert "a: boolean = None" in text


def test_signatures_do_not_prescribe_an_output_format():
    text = code_signatures(_case_tools())

    assert "JSON" not in text
