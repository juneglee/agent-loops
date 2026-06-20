from agent_loops.loops.base import response_is_complete


def test_completion_marker_in_text_counts_as_complete():
    assert response_is_complete({"tool_calls": None, "text": "Final: moved two files"})
    assert response_is_complete({"tool_calls": None, "text": "Response: done"})
    assert response_is_complete({"tool_calls": None, "text": "Task completed"})


def test_plain_text_without_marker_is_not_complete():
    assert not response_is_complete(
        {"tool_calls": None, "text": "Let me list the directory first"}
    )
    assert not response_is_complete({"tool_calls": None, "text": "I cannot do this"})


def test_structured_completion_fields_count_as_complete():
    assert response_is_complete({"tool_calls": None, "text": "", "completed": True})
    assert response_is_complete({"tool_calls": None, "text": "", "final": "done"})


def test_marker_is_read_only_from_first_or_last_line():
    assert response_is_complete({"text": "All done.\nFinal: moved two files"})
    assert not response_is_complete(
        {"text": "Progress so far:\nFinal: step one\nnext I will move the file"}
    )
