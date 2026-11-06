"""Evaluate discrete mode labels. Does not compute probability-averaged DCM shares."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODES = {'hrts': {1: 'drive', 2: 'public_transport', 3: 'cycle', 4: 'walk'},
         'lpmc': {1: 'walk', 2: 'cycle', 3: 'public_transport', 4: 'drive'}}


def evaluate(dataset, labels_path, predictions_path, output_dir):
    labels = pd.read_csv(labels_path, dtype={'trip_id': str})
    predictions = pd.read_csv(predictions_path, dtype={'trip_id': str})
    label = 'choice_mode' if dataset == 'hrts' else 'travel_mode'
    for frame in (labels, predictions):
        if frame.empty or frame.trip_id.isna().any() or not frame.trip_id.is_unique:
            raise ValueError('Expected nonempty, unique, nonmissing trip IDs')
    if set(labels.trip_id) != set(predictions.trip_id):
        raise ValueError('Labels and predictions must cover exactly the same trips; no silent row dropping')
    truth_values = pd.to_numeric(labels[label], errors='raise')
    pred_values = pd.to_numeric(predictions.predicted_mode, errors='raise')
    if not truth_values.isin([1, 2, 3, 4]).all(): raise ValueError('True modes must be codes 1--4')
    if not pred_values.isin([-1, 1, 2, 3, 4]).all():
        raise ValueError('Predicted modes must be codes 1--4 or -1 for an invalid/unparsed answer')
    labels = labels[['trip_id', label]].assign(**{label: truth_values.astype(int)})
    predictions = predictions[['trip_id', 'predicted_mode']].assign(predicted_mode=pred_values.astype(int))
    frame = labels.merge(predictions, on='trip_id', validate='one_to_one')
    y, pred = frame[label], frame.predicted_mode
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[1, 2, 3, 4], zero_division=0)
    rows = []
    for i, (code, name) in enumerate(MODES[dataset].items()):
        rows.append({'mode': name, 'precision': float(precision[i]), 'recall': float(recall[i]),
                     'f1': float(f1[i]), 'support': int(support[i]),
                     'tp': int(((y == code) & (pred == code)).sum()),
                     'fp': int(((y != code) & (pred == code)).sum()),
                     'fn': int(((y == code) & (pred != code)).sum()),
                     'observed_share': float((y == code).mean()), 'predicted_share': float((pred == code).mean()),
                     'share_bias_pp': float(100 * ((pred == code).mean() - (y == code).mean()))})
    metrics = {'n': len(frame), 'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1.mean()),
               'share_mae_pp': float(np.mean([abs(r['share_bias_pp']) for r in rows])),
               'invalid_count': int((pred == -1).sum()), 'invalid_share': float((pred == -1).mean()),
               'aggregation': 'discrete predicted labels; all observations retained in denominators'}
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / 'per_mode.csv', index=False)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=MODES, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/evaluation'))
    args = parser.parse_args()
    evaluate(args.dataset, args.labels, args.predictions, args.output_dir)
