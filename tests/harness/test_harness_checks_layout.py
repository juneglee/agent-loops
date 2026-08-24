from agent_loops.harness.checks.gfs import GFS_CHECKS
from agent_loops.harness.checks.workspace import WORKSPACE_CHECKS
from agent_loops.harness.verifier import verifier


def test_tables_live_in_checks_package_and_verifier_defaults_to_gfs():
    assert {"mkdir", "touch", "echo", "rm", "rmdir", "mv", "cp", "cd"} <= set(
        GFS_CHECKS
    )
    assert {"write_file", "edit_file", "bash"} <= set(WORKSPACE_CHECKS)
    layer = verifier()()
    assert layer._checks is GFS_CHECKS
