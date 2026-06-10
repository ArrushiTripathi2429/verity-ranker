import json
import tempfile
import unittest
from pathlib import Path

from verity_ranker.pipeline import run_pipeline


class V1PipelineTest(unittest.TestCase):
    def test_sample_pipeline_runs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ranked_output.csv"
            report = Path(tmp) / "audit_report.json"
            result = run_pipeline(
                jd_path=root / "data" / "sample" / "jd.txt",
                candidates_dir=root / "data" / "sample" / "candidates",
                output_path=output,
                report_path=report,
                weights_path=root / "configs" / "scoring_weights.json",
            )
            self.assertEqual(result.candidate_count, 4)
            self.assertTrue(output.exists())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], "v1_basic_ranker")
            self.assertEqual(payload["ranked_candidates"][0]["candidate_id"], "C001")


if __name__ == "__main__":
    unittest.main()

