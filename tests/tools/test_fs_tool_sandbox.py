import subprocess

import pytest

from agent_loops.tools import implementations
from agent_loops.tools.fs import ToolError
from agent_loops.tools.fs_tool import sandbox_available

pytestmark = pytest.mark.skipif(
    not sandbox_available(), reason="the OS sandbox is not available here"
)


def test_commands_are_never_refused_by_the_harness(tmp_path):
    tools = implementations(tmp_path, version="fs_tool")
    for command in (
        "ls /",
        "find / -maxdepth 1 -type d",
        "python3 -c 'print(1)'",
        "for i in 1 2; do echo $(( i * 2 )); done",
    ):
        try:
            tools["Bash"](command=command)
        except ToolError as exc:
            assert "blocked" not in str(exc).lower(), command


def test_writes_inside_the_workspace_work(tmp_path):
    tools = implementations(tmp_path, version="fs_tool")
    tools["Bash"](command="mkdir -p d && echo hi > d/a.txt && cp d/a.txt d/b.txt")
    assert (tmp_path / "d" / "b.txt").read_text() == "hi\n"


def test_writes_outside_the_workspace_are_refused_by_the_operating_system(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("keep")
    tools = implementations(tmp_path / "ws", version="fs_tool")
    (tmp_path / "ws").mkdir(exist_ok=True)

    with pytest.raises(ToolError) as exc:
        tools["Bash"](command=f"rm -f {outside}")

    assert "blocked" not in str(exc.value).lower()
    assert outside.read_text() == "keep"


def test_a_delete_sweep_of_the_whole_disk_changes_nothing(tmp_path):
    outside = tmp_path.parent / "survivor.txt"
    outside.write_text("keep")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "inside.txt").write_text("gone")
    tools = implementations(ws, version="fs_tool")

    tools["Bash"](command=f"find {tmp_path.parent} -name '*.txt' -delete", timeout=5)

    assert outside.read_text() == "keep"
    assert not (ws / "inside.txt").exists()


def test_the_sandbox_profile_names_the_workspace_only(tmp_path):
    from agent_loops.tools.fs_tool import _sandbox_profile

    profile = _sandbox_profile(tmp_path.resolve())
    assert f'(subpath "{tmp_path.resolve()}")' in profile
    assert "(deny file-write*)" in profile
    assert subprocess.run(["/usr/bin/true"], check=False).returncode == 0


def test_the_shell_is_wrapped_by_whichever_isolation_the_platform_has(tmp_path):
    from agent_loops.tools.fs_tool import BWRAP, SANDBOX, _shell_argv

    argv = _shell_argv("echo hi", tmp_path)
    assert argv[-3:] == ["/bin/bash", "-c", "echo hi"]
    if SANDBOX.exists():
        assert argv[0] == str(SANDBOX) and argv[1] == "-p"
    else:
        assert argv[0] == str(BWRAP)
        assert "--ro-bind" in argv and str(tmp_path.resolve()) in argv
