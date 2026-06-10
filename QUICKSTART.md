# Quickstart

## 1. Run The Sample Pipeline

```bash
python run_v1.py run --jd data/sample/jd.txt --candidates data/sample/candidates --output outputs/final/ranked_output.csv --report outputs/final/audit_report.json
```

## 2. Validate The Output

```bash
python run_v1.py validate --file outputs/final/ranked_output.csv
```

## 3. Inspect Outputs

- `outputs/final/ranked_output.csv`
- `outputs/final/audit_report.json`

## 4. Replace Sample Data

Put the official challenge JD and candidate files under `data/raw/`, then point the command at those paths.

Do not assume the sample output schema is the official schema. Once the official format is known, update `src/verity_ranker/validate.py`.

