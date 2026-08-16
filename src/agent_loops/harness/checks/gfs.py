from __future__ import annotations

from collections.abc import Callable
from typing import Any

Check = Callable[[Any, dict, dict], "str | None"]


def _gfs(env: Any) -> Any:
    return getattr(env, "_instances", {}).get("GorillaFileSystem")


def _contents(env: Any) -> dict | None:
    fs = _gfs(env)
    return None if fs is None else fs._current_dir.contents


def _check_mkdir(env: Any, args: dict, obs: dict) -> str | None:
    contents = _contents(env)
    if contents is None:
        return None
    name = args.get("dir_name")
    item = contents.get(name)
    if item is None or not hasattr(item, "contents"):
        return f"directory '{name}' missing in the current directory after mkdir"
    return None


def _check_touch(env: Any, args: dict, obs: dict) -> str | None:
    contents = _contents(env)
    if contents is None:
        return None
    name = args.get("file_name")
    item = contents.get(name)
    if item is None or not hasattr(item, "content"):
        return f"file '{name}' missing in the current directory after touch"
    return None


def _check_echo(env: Any, args: dict, obs: dict) -> str | None:
    if args.get("file_name") is None:
        return None
    contents = _contents(env)
    if contents is None:
        return None
    item = contents.get(args["file_name"])
    if item is None or not hasattr(item, "content"):
        return f"file '{args['file_name']}' missing after echo"
    if item.content != args.get("content"):
        return f"content of '{args['file_name']}' differs from what was written"
    return None


def _check_removed(key: str, verb: str) -> Check:
    def check(env: Any, args: dict, obs: dict) -> str | None:
        contents = _contents(env)
        if contents is None:
            return None
        name = args.get(key)
        if name in contents:
            return f"'{name}' still exists after {verb}"
        return None

    return check


def _check_mv(env: Any, args: dict, obs: dict) -> str | None:
    contents = _contents(env)
    if contents is None:
        return None
    src, dst = args.get("source"), args.get("destination")
    if src in contents:
        return f"source '{src}' still exists after mv"
    if dst not in contents:
        return f"destination '{dst}' missing in the current directory after mv"
    return None


def _check_cp(env: Any, args: dict, obs: dict) -> str | None:
    contents = _contents(env)
    if contents is None:
        return None
    dst = args.get("destination")
    if dst not in contents:
        return f"destination '{dst}' missing in the current directory after cp"
    return None


def _check_cd(env: Any, args: dict, obs: dict) -> str | None:
    folder = str(args.get("folder", "")).rstrip("/")
    if folder in ("", ".", "..", "/") or "/" in folder:
        return None
    fs = _gfs(env)
    if fs is None:
        return None
    if fs._current_dir.name != folder:
        return f"current directory after cd is '{fs._current_dir.name}', not '{folder}'"
    return None


GFS_CHECKS: dict[str, Check] = {
    "mkdir": _check_mkdir,
    "touch": _check_touch,
    "echo": _check_echo,
    "rm": _check_removed("file_name", "rm"),
    "rmdir": _check_removed("dir_name", "rmdir"),
    "mv": _check_mv,
    "cp": _check_cp,
    "cd": _check_cd,
}
