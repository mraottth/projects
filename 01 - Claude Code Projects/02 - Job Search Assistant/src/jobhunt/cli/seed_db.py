"""Copy one SQLite database to another path, consistently (SQLite's online backup API, which also
picks up anything still in the source's -wal file). Used once by scripts/start.sh, to seed the
Litestream-managed database from the file the app used before (see DECISIONS.md D-020).

    python -m jobhunt.cli.seed_db SOURCE TARGET
"""

import sys
from pathlib import Path

from jobhunt.core.backup import copy_sqlite


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    source, target = Path(argv[0]), Path(argv[1])
    if not source.exists():
        print(f"seed_db: {source} doesn't exist", file=sys.stderr)
        return 1
    if target.exists():
        print(f"seed_db: {target} already exists; not overwriting it", file=sys.stderr)
        return 1
    copy_sqlite(source, target)
    print(f"seed_db: copied {source} to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
