"""Run with python -m unittest discover -s tests -v. No model or network needed."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_prompts import generate
from evaluate_predictions import evaluate
from generate_synthetic_data import make_dataset


class DemoChecks(unittest.TestCase):
    def test_synthetic_inputs(self):
        for dataset, seed in [('hrts', 20261103), ('lpmc', 20261104)]:
            frame = make_dataset(dataset, seed)
            self.assertEqual(len(frame), 200)
            self.assertEqual(frame.traveller_id.nunique(), 100)
            self.assertTrue(frame.trip_id.is_unique)
            self.assertTrue(frame.is_synthetic.all())
            self.assertEqual(frame.sample_role.value_counts().to_dict(), {'train': 180, 'evaluation': 20})
            if dataset == 'hrts':
                for _, row in frame.iterrows():
                    self.assertEqual(row[f'av_{int(row.choice_mode)}'], 1)

    def test_target_labels_and_identifiers_do_not_reach_messages(self):
        for dataset in ('hrts', 'lpmc'):
            original = ROOT / f'data/synthetic/{dataset}_synthetic.csv'
            frame = pd.read_csv(original)
            label = 'choice_mode' if dataset == 'hrts' else 'travel_mode'
            frame.loc[frame.sample_role == 'evaluation', label] = 999
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                tmp = Path(tmp)
                changed = tmp / 'changed.csv'
                frame.to_csv(changed, index=False)
                for config in ('zs', 'fs'):
                    before = generate(dataset, config, original, tmp / 'before', limit=None)
                    after = generate(dataset, config, changed, tmp / 'after', limit=None)
                    self.assertEqual(before, after)
                    self.assertEqual(len(before), 20)
                    for record in before:
                        self.assertNotIn('syn_', json.dumps(record['messages']))
                        self.assertEqual(len(record['example_trip_ids']), 11 if config == 'fs' else 0)

    def test_metric_arithmetic_and_invalid_denominators(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            labels = tmp / 'labels.csv'; predictions = tmp / 'predictions.csv'
            pd.DataFrame({'trip_id': ['a','b','c','d'], 'travel_mode': [1,2,3,4]}).to_csv(labels,index=False)
            pd.DataFrame({'trip_id': ['d','c','b','a'], 'predicted_mode': [1,1,1,1]}).to_csv(predictions,index=False)
            result = evaluate('lpmc',labels,predictions,tmp/'out')
            self.assertAlmostEqual(result['accuracy'], .25)
            self.assertAlmostEqual(result['macro_f1'], .1)
            self.assertAlmostEqual(result['share_mae_pp'], 37.5)
            pd.DataFrame({'trip_id': ['a','b','c','d'], 'predicted_mode': [-1,-1,-1,-1]}).to_csv(predictions,index=False)
            result = evaluate('lpmc',labels,predictions,tmp/'out')
            self.assertEqual(result['invalid_count'],4)
            self.assertAlmostEqual(result['share_mae_pp'],25.)
            self.assertEqual(result['accuracy'],0.)
            pd.DataFrame({'trip_id': ['a','a','c','d'], 'predicted_mode': [1,2,3,4]}).to_csv(predictions,index=False)
            with self.assertRaises(ValueError): evaluate('lpmc',labels,predictions,tmp/'out')
            pd.DataFrame({'trip_id': ['a','b','c'], 'predicted_mode': [1,2,3]}).to_csv(predictions,index=False)
            with self.assertRaises(ValueError): evaluate('lpmc',labels,predictions,tmp/'out')


if __name__ == '__main__': unittest.main()
