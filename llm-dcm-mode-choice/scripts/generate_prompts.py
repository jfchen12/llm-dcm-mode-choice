"""Generate ZS/FS chat messages from the bundled synthetic data; no model call."""
import argparse
import importlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate(dataset, config, data_path, output_dir, limit=1):
    frame = pd.read_csv(data_path, dtype={'trip_id': str, 'traveller_id': str})
    if frame.empty or not frame.trip_id.is_unique:
        raise ValueError('Expected nonempty data with unique trip IDs')
    if not frame.is_synthetic.astype(str).str.lower().eq('true').all():
        raise ValueError('This public demo accepts synthetic records only')
    if set(frame.sample_role) != {'train', 'evaluation'}:
        raise ValueError('Expected train and evaluation sample_role values')
    label = 'choice_mode' if dataset == 'hrts' else 'travel_mode'
    train = frame[frame.sample_role == 'train'].copy()
    targets = frame[frame.sample_role == 'evaluation'].drop(columns=[label]).copy()
    if limit is not None:
        if limit < 1: raise ValueError('limit must be positive')
        targets = targets.iloc[:limit]
    template = importlib.import_module(f'prompt_templates.{dataset}_{config}')
    retriever = None
    if config == 'fs':
        if len(train) < 11: raise ValueError('FS requires at least 11 training rows')
        retriever = importlib.import_module(f'prompt_templates.{dataset}_retrieval').FlatRetriever(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, query in targets.iterrows():
        if config == 'fs':
            examples = retriever.get_examples(query, k=11)
            messages = template.build_fewshot_prompt(query, examples)
            ids = examples.trip_id.tolist()
            assert query.trip_id not in ids and set(ids) <= set(train.trip_id)
        else:
            ids = []
            messages = (template.build_zeroshot_prompt(query) if dataset == 'hrts' else
                        [{'role': 'system', 'content': 'You are an expert in travel behavior modelling.'},
                         {'role': 'user', 'content': template.build_prompt(query)}])
        records.append({'dataset': dataset, 'configuration': config, 'synthetic': True,
                        'target_trip_id': query.trip_id, 'example_trip_ids': ids, 'messages': messages})
    path = output_dir / f'{dataset}_{config}_messages.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    # Metadata stays outside the actual chat messages.
    first = records[0]
    text = ('SYNTHETIC DEMONSTRATION — not a real participant or manuscript result.\n'
            f"Dataset schema: {dataset}; configuration: {config}; target: {first['target_trip_id']}\n"
            f"Training examples: {', '.join(first['example_trip_ids']) or 'none'}\n\n")
    text += '\n\n'.join(f"[{m['role'].upper()}]\n{m['content']}" for m in first['messages'])
    (output_dir / f'{dataset}_{config}_example.txt').write_text(text + '\n', encoding='utf-8')
    print(f'{len(records)} prompt(s): {path}')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hrts', 'lpmc'], required=True)
    parser.add_argument('--config', choices=['zs', 'fs'], required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/prompts')
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--all', action='store_true', help='Generate all 20 evaluation prompts')
    args = parser.parse_args()
    generate(args.dataset, args.config, args.data or ROOT / f'data/synthetic/{args.dataset}_synthetic.csv',
             args.output_dir, None if args.all else args.limit)
