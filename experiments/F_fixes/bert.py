"""
The 'BERT (fine-tuned)' comparison model of the final comparison (修正の実験方針.md, 6): E4's FinBERT fine-tuning, re-run
for each split on the releases whose reaction window closed before the split's final period, then used to predict
every release filed in the final period. Settings are E4's; the reading follows what the development kept for
zero-shot FinBERT in that split (the cleaned text if fix 1 was kept; head 128 + tail 382 tokens if fix 2 'headtail'
was kept; otherwise the first 512 tokens). Run after develop.py, on a free GPU:

    python experiments/F_fixes/bert.py

Predictions: data/cache/fixes/bert_ft_{split}.csv (outside Git).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from events import texts
from experiments.F_fixes import common as C

e4 = C._load('e4_run', 'experiments/E4_finbert_finetune/run.py')
SELECTION = Path(__file__).resolve().parent / 'results' / 'dev_selection.json'


def make_encode(cleaned, headtail):
    import torch

    def encode(tokenizer, paths):
        txts = [C.clean_text(p, None if headtail else 6000) if cleaned else texts.read(p, 6000) for p in paths]
        if not headtail:
            return tokenizer(txts, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        ids, mask = [], []
        for t in txts:
            tk = tokenizer.encode(t, add_special_tokens=False)
            if len(tk) > 510:
                tk = tk[:128] + tk[-382:]
            seq = [tokenizer.cls_token_id] + tk + [tokenizer.sep_token_id]
            ids.append(seq + [tokenizer.pad_token_id] * (512 - len(seq)))
            mask.append([1] * len(seq) + [0] * (512 - len(seq)))
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}
    return encode


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    p = C.base_panel(D.connect())
    for split in ['B', 'A', 'C']:
        out = C.CACHE / f'bert_ft_{split}.csv'
        if out.exists():
            print(split, 'done', flush=True)
            continue
        reading = selection[split]['finbert']['reading']
        cleaned, headtail = reading != 'current', reading == 'fix2_headtail'
        e4.encode = make_encode(cleaned, headtail)
        start = C.final_start(split)
        train = p[(p.filing_date >= '2015-01-01') & (p.window_end < start)][['accession', 'abn_r_0_1', 'text_path']]
        pred_df = p[p.filing_date >= start][['accession', 'text_path']].reset_index(drop=True)
        print(f'{split}: train {len(train)} (cleaned={cleaned}, headtail={headtail}), predict {len(pred_df)}', flush=True)
        model, tok, info = e4.train(train.reset_index(drop=True))
        preds = e4.predict(model, tok, pred_df)
        preds.rename('bert_ft').rename_axis('accession').reset_index().to_csv(out, index=False)
        (C.CACHE / f'bert_ft_{split}.json').write_text(json.dumps({**info, 'cleaned': cleaned, 'headtail': headtail},
                                                                  indent=1), encoding='utf-8')
        del model


if __name__ == '__main__':
    main()
