"""Compare moo_interp's registered builtin names with ToastStunt source.

Run from the repository root with:

    uv run --with ../moo-conformance-tests scripts/toast_builtin_inventory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from moo_conformance import extract_builtin_specs

from moo_interp.builtin_functions import BuiltinFunctions


TOAST_ROOT = Path.home() / "src" / "toaststunt"


def main() -> int:
    toast_names = {spec.name for spec in extract_builtin_specs(TOAST_ROOT)}
    local_names = set(BuiltinFunctions().functions)
    result = {
        "toast_count": len(toast_names),
        "local_count": len(local_names),
        "shared_count": len(toast_names & local_names),
        "missing": sorted(toast_names - local_names),
        "extra": sorted(local_names - toast_names),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["missing"] or result["extra"] else 0


if __name__ == "__main__":
    sys.exit(main())
