from dataclasses import dataclass
from pathlib import Path

from .candidates import load_candidates
from .io import load_weights, write_audit_report, write_ranked_output
from .jd import parse_jd
from .scoring import rank_candidates
from .validate import validate_ranked_output


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    report_path: Path
    candidate_count: int


def run_pipeline(
    jd_path: Path,
    candidates_dir: Path,
    output_path: Path,
    report_path: Path,
    weights_path: Path,
) -> PipelineResult:
    role = parse_jd(jd_path.read_text(encoding="utf-8"))
    candidates = load_candidates(candidates_dir)
    weights = load_weights(weights_path)
    scores = rank_candidates(candidates, role, weights)
    write_ranked_output(output_path, scores)
    validate_ranked_output(output_path)
    write_audit_report(report_path, role, scores)
    return PipelineResult(output_path=output_path, report_path=report_path, candidate_count=len(scores))

