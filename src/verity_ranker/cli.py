import argparse
from pathlib import Path

from .pipeline import run_pipeline
from .validate import validate_ranked_output


def main() -> None:
    parser = argparse.ArgumentParser(prog="verity-ranker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--jd", required=True)
    run_parser.add_argument("--candidates", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--report", required=True)
    run_parser.add_argument("--weights", default="configs/scoring_weights.json")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--file", required=True)

    args = parser.parse_args()

    if args.command == "run":
        result = run_pipeline(
            jd_path=Path(args.jd),
            candidates_dir=Path(args.candidates),
            output_path=Path(args.output),
            report_path=Path(args.report),
            weights_path=Path(args.weights),
        )
        print(f"ranked_output={result.output_path}")
        print(f"audit_report={result.report_path}")
        print(f"candidates_ranked={result.candidate_count}")

    if args.command == "validate":
        validate_ranked_output(Path(args.file))
        print(f"valid_output={args.file}")

