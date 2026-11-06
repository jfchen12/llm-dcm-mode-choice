# LLM–DCM travel mode choice: synthetic examples

A demonstration of **zero-shot (ZS)** and **retrieved few-shot (FS)** prompt construction for travel mode choice, using entirely synthetic data. It includes prompt generation and evaluation tools, but no model inference or full experimental results.

## Contents

- 200 synthetic records for each of the HRTS and LPMC input schemas.
- ZS/FS prompt-building and retrieval functions, with 11 training examples per FS prompt.
- Readable prompt examples and machine-readable chat messages in `prompts/`.
- Prediction templates and explicitly labelled mock outputs in `examples/`.
- Evaluation of accuracy, macro-F1, mode-specific precision/recall/F1, and modal-share MAE.

No real participant records or model responses based on them are included. Synthetic data are generated from hand-written rules and pseudorandom draws, without reading survey records, fitted parameters or empirical summary statistics.

## Quick start

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Generate a single prompt.
python scripts/generate_prompts.py --dataset hrts --config zs
python scripts/generate_prompts.py --dataset hrts --config fs

# Generate FS prompts for all 20 LPMC evaluation trips.
python scripts/generate_prompts.py --dataset lpmc --config fs --all
```

Use `hrts` or `lpmc` for the dataset and `zs` or `fs` for the configuration. On Windows, activate the environment with `.venv\Scripts\activate`.

Generated files are saved to `outputs/prompts/`. Only the `messages` list in each JSONL record is intended as model input; other fields are metadata. These scripts do not call an API or load a model.

## Data and retrieval

Each dataset contains 100 synthetic travellers with two trips each. A fixed, mode-stratified 90/10 **trip-level split** assigns 180 records to training and 20 to evaluation. Travellers may appear in both groups.

FS standardises numerical retrieval features using the training records and selects the 11 most similar examples by cosine similarity. Same-traveller records are neither excluded nor prioritised. Examples are described as trips by other residents, and traveller IDs are omitted from prompts. Target choices are hidden, and evaluation records are excluded from retrieval. ZS uses no examples.

See the [data description](data/synthetic/README.md) for fields, units and generation rules. To regenerate the synthetic CSVs and evaluation fixtures:

```bash
python scripts/generate_synthetic_data.py
```

Use the bundled CSVs for consistent examples; numerical-library changes may affect regenerated samples.

## Evaluate predictions

Prediction files require unique `trip_id` values matching the label file and a `predicted_mode` column. The mode codes differ by dataset:

| Code | HRTS | LPMC |
|---|---|---|
| 1 | Driving | Walking |
| 2 | Public transport | Cycling |
| 3 | Cycling | Public transport |
| 4 | Walking | Driving |
| -1 | Invalid/unparsed | Invalid/unparsed |

Fill in a prediction template from `examples/`, or try the supplied mock predictions:

```bash
python scripts/evaluate_predictions.py \
  --dataset hrts \
  --labels examples/hrts_evaluation_labels.csv \
  --predictions examples/hrts_mock_predictions.csv \
  --output-dir outputs/hrts_mock_evaluation
```

For LPMC, replace `hrts` with `lpmc` in the command. **Mock predictions select the fastest available mode; they are not LLM outputs or DCM estimates.**

Accuracy and macro-F1 use a 0–1 scale; macro-F1 averages all four modes. Modal-share MAE is calculated from discrete predicted labels:

$$
\mathrm{MAE}_{\mathrm{share}} = \frac{100}{4}\sum_{i=1}^{4}|\hat{s}_i-s_i|.
$$

Here, shares are fractions and MAE is in percentage points. Invalid predictions remain in the evaluation denominator, so predicted shares may sum to less than one. Choice probabilities are not averaged by this evaluator.

## Scope and provenance

The ZS/FS prompt and retrieval functions retain their original text construction and feature definitions; [source hashes](prompts/provenance.json) document their provenance. This demonstration excludes other prompting configurations, DCM estimation, model execution and cluster job scripts.
Additional prompting configurations and supporting code are planned for future updates.
