from __future__ import annotations

import argparse
from pathlib import Path

from agent_loops.bench.tasks.fetch import fetch_mcpmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks/mcpmark_fs")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the sha256 check against the archives the measurements used",
    )
    a = ap.parse_args()
    counts = fetch_mcpmark(Path(a.tasks), a.categories or None, verify=not a.no_verify)
    for category, n in counts.items():
        print(f"{category}: {n} files")


if __name__ == "__main__":
    main()
