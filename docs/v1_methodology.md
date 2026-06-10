# V1 Methodology

V1 uses a deterministic scoring pipeline:

1. Parse the JD.
2. Extract required and preferred skills from a controlled vocabulary.
3. Parse each candidate profile.
4. Extract candidate skills and simple evidence snippets.
5. Score candidates with configurable weights.
6. Sort by final score.
7. Write ranked CSV and JSON audit report.

## Known Limitations

- It can be fooled by inflated resume claims.
- It does not verify GitHub or portfolio evidence.
- It does not run graph expansion.
- It does not run multi-agent review.
- It does not evaluate fairness or rank stability.

These limitations are intentional for V1.

