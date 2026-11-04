"""Run the frozen task-profile rules over complete source Parquet shards."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.profile_bulk import main


if __name__ == "__main__":
    raise SystemExit(main())
