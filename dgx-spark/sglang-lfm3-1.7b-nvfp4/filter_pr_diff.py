#!/usr/bin/env python3
"""Keep only the python/ file diffs of a GitHub PR diff (drops tests/docs)."""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
blocks = re.split(r"(?m)^(?=diff --git )", open(src).read())
keep = [b for b in blocks if b.startswith("diff --git a/python/")]
open(dst, "w").write("".join(keep))
print(f"kept {len(keep)} of {len(blocks) - 1} file diffs (python/ only)")
