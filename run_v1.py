from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from verity_ranker.cli import main


if __name__ == "__main__":
    main()

