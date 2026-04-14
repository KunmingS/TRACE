import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_ANNOTATOR_ROOT = ROOT / "trace-annotator"

for path in (ROOT, TRACE_ANNOTATOR_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Keep job persistence isolated from the user's real ~/.trace directory.
os.environ["HOME"] = tempfile.mkdtemp(prefix="trace-test-home-")

