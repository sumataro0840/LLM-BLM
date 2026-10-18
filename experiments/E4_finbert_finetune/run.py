"""
E4: does the zero-shot gpt-4o-mini score carry information beyond FinBERT fine-tuned on earlier reactions?
Everything follows preregistration.md. Run once, after the pre-registration commit:

    python experiments/E4_finbert_finetune/run.py            # train, predict, store, evaluate
    python experiments/E4_finbert_finetune/run.py --smoke    # a few steps on 64 releases; no evaluation, nothing stored

Training uses releases filed 2015-2022 only; the evaluation sample (E2's sample A, from 2023-11) does not overlap.
"""
import argparse
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from events.evaluate import holm, permutation_test, regress
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = Path(__file__).resolve().parent / 'results'
MODEL_DIR = REPO / 'data' / 'cache' / 'models' / 'finbert_ft_e4'
BASE_MODEL = 'yiyanghkust/finbert-tone'
REVISION = '4921590d3c0c3832c0efea24c8381ce0bda7844b'          # the same version as E2's FinBERT scores
TRAIN = ('2015-01-01', '2022-12-31')
PREDICT_FROM = '2023-01-01'
HP = {'lr': 2e-5, 'weight_decay': 0.01, 'epochs': 2, 'batch': 8, 'accumulate': 2, 'warmup': 0.10, 'max_len': 512,
      'seed': 0, 'precision': 'bf16 autocast', 'target': 'abn_r_0_1, winsorized at the training 1%/99% points, standardized'}
N_PERMUTATIONS = 1000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load('e2_run', 'experiments/E2_event_reaction/run.py')
e3 = _load('e3_run', 'experiments/E3_analyst_surprise/run.py')


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def releases_with_text(conn):
    panel = pd.read_csv(e3.PANEL, usecols=['accession', 'filing_date', 'abn_r_0_1'])
    rel = db.load_releases(conn, with_text=True)[['accession', 'text_path']]
    return panel.merge(rel, on='accession')


def encode(tokenizer, paths):
    # the first ~6,000 characters always cover 512 tokens
    return tokenizer([texts.read(p, 6000) for p in paths], truncation=True, max_length=HP['max_len'],
                     padding='max_length', return_tensors='pt')


def train(train_df, smoke=False):
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
    seed_everything(HP['seed'])
    device = 'cuda'
    tok = BertTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, revision=REVISION, num_labels=1,
                                                          problem_type='regression', ignore_mismatched_sizes=True).to(device)
    y = train_df.abn_r_0_1
    lo, hi = y.quantile(0.01), y.quantile(0.99)
    yw = y.clip(lo, hi)
    mean, sd = yw.mean(), yw.std()
    target = torch.tensor(((yw - mean) / sd).to_numpy(), dtype=torch.float32)
    enc = encode(tok, train_df.text_path)
    n = len(train_df)
    steps_per_epoch = math.ceil(n / HP['batch'] / HP['accumulate'])
    total = steps_per_epoch * HP['epochs'] if not smoke else 3
    opt = torch.optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=HP['weight_decay'])
    sched = get_linear_schedule_with_warmup(opt, int(HP['warmup'] * total), total)
    model.train()
    step, t0 = 0, time.time()
    g = torch.Generator().manual_seed(HP['seed'])
    for epoch in range(HP['epochs'] if not smoke else 1):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, HP['batch']):
            idx = order[i:i + HP['batch']]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=enc['input_ids'][idx].to(device), attention_mask=enc['attention_mask'][idx].to(device))
            loss = torch.nn.functional.mse_loss(out.logits.float().squeeze(-1), target[idx].to(device)) / HP['accumulate']
            loss.backward()
            if (i // HP['batch'] + 1) % HP['accumulate'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 100 == 0 or smoke:
                    print(f'  epoch {epoch + 1} step {step}/{total} loss {loss.item() * HP["accumulate"]:.3f} '
                          f'({time.time() - t0:.0f}s)', flush=True)
                if smoke and step >= total:
                    break
    stats = {'n_train': n, 'winsor': [float(lo), float(hi)], 'mean': float(mean), 'sd': float(sd), 'steps': step,
             'seconds': round(time.time() - t0, 1)}
    return model, tok, stats


def predict(model, tok, df):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            enc = encode(tok, df.text_path.iloc[i:i + 32])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(input_ids=enc['input_ids'].to('cuda'), attention_mask=enc['attention_mask'].to('cuda'))
            preds.extend(out.logits.float().squeeze(-1).cpu().tolist())
    return pd.Series(preds, index=df.accession.to_numpy(), name='ft_pred')


def fit(d, focal, controls, permute=False):
    d = d.dropna(subset=['y', focal])
    controls = [c for c in controls if d[c].std() > 0]
    full = regress(d, 'y', [focal, *controls], e2.FIXED)
    row = full.loc[focal]
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 'se_bp': float(row.se * 1e4), 't': float(row.t),
           'p_one_sided': float(e2.one_sided(row)), 'n': int(row.n), 'event_dates': int(row.clusters),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)],
           'others': {c: {'coef_bp': float(full.loc[c].coef * 1e4), 't': float(full.loc[c].t)}
                      for c in ('ft_pred', 'llm_composite', 'analyst_surprise') if c in full.index and c != focal}}
    if permute:
        out['permutation_p_one_sided'] = float(permutation_test(d, 'y', focal, controls, e2.FIXED,
                                                                n=N_PERMUTATIONS, seed=0)['p_one_sided'])
    return out


def evaluate(conn, preds):
    A = e3.sample_a(conn).merge(preds.rename_axis('accession').reset_index(), on='accession', how='left')
    d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
    d['ft_pred'] = e2.standardize(A['ft_pred'])
    m1 = e3.M1                                      # E2 controls + analyst surprise
    m2 = fit(d, 'llm_composite', m1 + ['ft_pred'], permute=True)
    m2['verdict'] = '支持' if m2['p_one_sided'] < 0.05 and m2['permutation_p_one_sided'] < 0.05 else '支持されない'
    s1 = fit(d, 'ft_pred', m1)
    s2 = {k: m2['others']['ft_pred'][k] for k in ('coef_bp', 't')}
    from scipy import stats
    s2['p_one_sided'] = float(stats.t.sf(s2['t'], df=m2['event_dates'] - 1))
    secondary = {'S1_ft_alone': s1, 'S2_ft_with_llm': s2}
    for (k, v), p in zip(secondary.items(), holm([v['p_one_sided'] for v in secondary.values()])):
        v['p_holm'] = float(p)
    dd = d.dropna(subset=['y', 'llm_composite', 'ft_pred'])
    base = [c for c in m1 if dd[c].std() > 0]
    r2 = {name: float(e2.r_squared(dd, base + xs) - e2.r_squared(dd, base))
          for name, xs in [('llm_only', ['llm_composite']), ('ft_only', ['ft_pred']), ('both', ['llm_composite', 'ft_pred'])]}
    r1 = fit(d, 'llm_composite', list(e2.CONTROLS) + ['ft_pred'])
    dr = e3.prepare(A, 'llm_composite', 'abn_r_0_1', rank=True)
    dr['ft_pred'] = A['ft_pred'].groupby(A.year_quarter).rank(pct=True)
    r2_rank = fit(dr, 'llm_composite', m1 + ['ft_pred'])
    corr = float(pd.Series(A.ft_pred).corr(A.llm_composite, method='spearman'))
    return {'M2_main': m2, 'secondary': secondary, 'r2_gain_over_controls': r2,
            'robustness': {'R1_without_analyst': r1, 'R2_ranks': r2_rank},
            'spearman_ft_llm': corr, 'n_with_prediction': int(A.ft_pred.notna().sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    conn = connect(config.DB_PATH)
    conn.execute('PRAGMA busy_timeout = 60000')
    rel = releases_with_text(conn)
    train_df = rel[rel.filing_date.between(*TRAIN) & rel.abn_r_0_1.notna()].reset_index(drop=True)
    pred_df = rel[rel.filing_date >= PREDICT_FROM].reset_index(drop=True)
    print(f'train {len(train_df)} releases, predict {len(pred_df)}', flush=True)
    if args.smoke:
        model, tok, stats = train(train_df.sample(64, random_state=0).reset_index(drop=True), smoke=True)
        print('smoke predictions:', predict(model, tok, pred_df.head(32)).describe().round(3).to_dict(), stats)
        return

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    model, tok, stats = train(train_df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    preds = predict(model, tok, pred_df)
    params = {'base_model': BASE_MODEL, 'revision': REVISION, 'train_window': TRAIN, 'predict_from': PREDICT_FROM,
              'hyperparameters': HP, 'code_commit': commit}
    run_id = db.get_or_create_score_run(conn, 'finbert_ft', f'{BASE_MODEL} fine-tuned on reactions', params, True,
                                        'E4, trained on 2015-2022 reactions')
    for acc, v in preds.items():
        db.save_scores(conn, run_id, acc, {'predicted_reaction': float(v)})
    print(f'score run {run_id}: {len(preds)} predictions ({stats["seconds"]}s training)', flush=True)

    result = evaluate(conn, preds)
    summary = {'code_commit': commit, 'score_run': run_id, 'training': stats, **result}
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
