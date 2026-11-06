# Local validation — 2026-11-03

- Prompt-building and retrieval functions were compared structurally with the submitted-study source; retained functions are unchanged.
- Both CSVs contain 200 synthetic records: 100 fictional travellers, 180 training trips and 20 evaluation trips.
- All four bundled prompt examples were generated successfully.
- All 20 target prompts for each dataset/configuration were checked: changing every target choice to a sentinel value left all messages unchanged; synthetic record IDs do not appear in chat content.
- Metric arithmetic was checked on hand-calculable cases, including invalid predictions. Duplicate and incomplete prediction cohorts are rejected.
- The three CPU unit tests passed. No real model, GPU, API call, empirical dataset or manuscript result was used.
- No local machine paths or credential patterns were found in the release files.

Test environment: Python environment with NumPy 2.2.5, pandas 2.3.3 and scikit-learn 1.7.2.

Run `python -m unittest discover -s tests -v` to repeat the checks.
