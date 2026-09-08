import os
import sys
from pathlib import Path

root = Path(os.environ["FILESYSTEM_TEST_DIR"])
note = root / "note.txt"
if not note.is_file():
    print("note.txt missing")
    sys.exit(1)
if note.read_text(encoding="utf-8").strip() != "done":
    print("unexpected content")
    sys.exit(1)
print("ok")
sys.exit(0)
