"""Generate the bundled, reproducible synthetic train and test registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_generator import DATASET_SIZE, SEED, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DATASET_SIZE, help="Approximate combined record count")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = generate_dataset(Path(__file__).parent / "data", args.size, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()