"""
E10: ChronoGPT fine-tuned with LoRA on the reactions of 2015-2022 releases, against E4's fine-tuned FinBERT and
gpt-4o-mini. Everything follows preregistration.md. Run once, after the pre-registration commit:

    python experiments/E10_chrono_lora/run.py            # train, predict, store, evaluate (about 7-8 hours)
    python experiments/E10_chrono_lora/run.py --smoke    # a few steps on 64 releases: speed and GPU memory only;
                                                         # no evaluation, nothing stored

The training set, the prediction set and the evaluation sample are E4's: releases filed 2015-2022 to train, from
2023-01-01 to predict, E3's sample A (2023-11 .. 2026-08) to evaluate. The vintage is 2022 (text up to 2022-12-31),
so the model has read the training years but not the evaluation years.
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
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from events.evaluate import holm
from events.scorers.chrono import _chrono
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = Path(__file__).resolve().parent / 'results'
ADAPTER = REPO / 'data' / 'cache' / 'models' / 'chrono_lora_e10.pt'
VINTAGE = 2022
TRAIN = ('2015-01-01', '2022-12-31')
PREDICT_FROM = '2023-01-01'
HP = {'rank': 8, 'alpha': 16, 'targets': ['c_q', 'c_v'], 'lr_lora': 2e-4, 'lr_head': 1e-3, 'weight_decay': 0.0,
      'epochs': 1, 'accumulate': 16, 'warmup': 0.10, 'max_tokens': 1792, 'max_chars': 9000, 'seed': 0,
      'clip': 1.0, 'pooling': 'mean over tokens of the normalized last layer (as E5)',
      'target': 'abn_r_0_1, winsorized at the training 1%/99% points, standardized'}
N_PERMUTATIONS = 1000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e4 = _load('e4_run', 'experiments/E4_finbert_finetune/run.py')
e2, e3 = e4.e2, e4.e3


# ---------- LoRA ----------

class LoRA(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen; A small random, B zero (the start equals the original model)."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).T @ self.B.to(x.dtype).T) * self.scale


def add_lora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for block in model.blocks:
        if block.attn is None:
            continue
        for name in HP['targets']:
            layer = LoRA(getattr(block.attn, name), HP['rank'], HP['alpha'])
            setattr(block.attn, name, layer)
            params += [layer.A, layer.B]
    return params


def pooled(model, ids):
    """chrono.final_hidden with gradients and one checkpoint per block, then the mean over tokens (float32)."""
    norm = lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),))
    x0 = norm(model.embed(ids).bfloat16())
    x = x0
    ve = model.value_embeds(ids[0].view(-1))
    ve = [v.unsqueeze(0) if v is not None else None for v in ve]
    ve_enc, ve_dec = ve[:model.num_encoder_layers], ve[model.num_encoder_layers:]
    skips = []
    for i in range(model.num_encoder_layers):
        x = checkpoint(model.blocks[i], x, ve_enc[i], x0, use_reentrant=False)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        x = x + model.skip_weights[i] * skips.pop()
        x = checkpoint(model.blocks[model.num_encoder_layers + i], x, ve_dec[i], x0, use_reentrant=False)
    return norm(x)[0].float().mean(0)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ids_of(tok, path):
    return torch.tensor([tok.encode(texts.read(path, HP['max_chars']))[:HP['max_tokens']]], device='cuda')


def train(model, tok, train_df, smoke=False):
    seed_everything(HP['seed'])
    lora = add_lora(model)
    head = nn.Linear(1536, 1).cuda()
    y = train_df.abn_r_0_1
    lo, hi = y.quantile(0.01), y.quantile(0.99)
    yw = y.clip(lo, hi)
    mean, sd = yw.mean(), yw.std()
    target = torch.tensor(((yw - mean) / sd).to_numpy(), dtype=torch.float32)
    n = len(train_df)
    total = math.ceil(n / HP['accumulate']) * HP['epochs'] if not smoke else 4
    opt = torch.optim.AdamW([{'params': lora, 'lr': HP['lr_lora']}, {'params': head.parameters(), 'lr': HP['lr_head']}],
                            weight_decay=HP['weight_decay'])
    warm = max(1, int(HP['warmup'] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    model.train()
    g = torch.Generator().manual_seed(HP['seed'])
    step, t0, running = 0, time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(HP['epochs']):
        order = torch.randperm(n, generator=g).tolist()
        for k, i in enumerate(order):
            pred = head(pooled(model, ids_of(tok, train_df.text_path.iloc[i])))[0]
            loss = (pred - target[i].cuda()) ** 2 / HP['accumulate']
            loss.backward()
            running.append(loss.item() * HP['accumulate'])
            if (k + 1) % HP['accumulate'] == 0:
                torch.nn.utils.clip_grad_norm_(lora + list(head.parameters()), HP['clip'])
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 25 == 0 or smoke:
                    el = time.time() - t0
                    print(f'  step {step}/{total} loss {np.mean(running[-400:]):.3f} ({el:.0f}s, '
                          f'{el / (k + 1 + epoch * n):.2f}s per release, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)',
                          flush=True)
                if smoke and step >= total:
                    break
        if smoke:
            break
    stats = {'n_train': n, 'winsor': [float(lo), float(hi)], 'mean': float(mean), 'sd': float(sd), 'steps': step,
             'seconds': round(time.time() - t0, 1), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2),
             'trainable': int(sum(p.numel() for p in lora) + sum(p.numel() for p in head.parameters()))}
    return head, lora, stats


@torch.no_grad()
def predict(model, tok, head, df):
    model.eval()
    out = []
    for p in df.text_path:
        out.append(float(head(pooled(model, ids_of(tok, p)))[0]))
    return pd.Series(out, index=df.accession.to_numpy(), name='lora_pred')


def evaluate(conn, preds, ft_run):
    """P1: the LoRA prediction with E3's controls; P2: the same with E4's fine-tuned FinBERT added."""
    A = e3.sample_a(conn).merge(preds.rename('lora_pred').rename_axis('accession').reset_index(), on='accession', how='left')
    ft = db.load_scores(conn, ft_run, ['predicted_reaction'])['predicted_reaction'].rename('ft_pred')
    A = A.join(ft, on='accession')
    e5p = pd.read_csv(REPO / 'experiments' / 'E5_chrono_embeddings' / 'results' / 'predictions.csv')
    A = A.merge(e5p[e5p.block == 'A'][['accession', 'pred']].rename(columns={'pred': 'ridge_pred'}), on='accession', how='left')
    d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
    for c in ('lora_pred', 'ft_pred', 'ridge_pred'):
        d[c] = e2.standardize(A[c])
    m1 = e3.M1
    p1 = e4.fit(d, 'lora_pred', [c for c in m1 if c != 'llm_composite'], permute=True)
    p2 = e4.fit(d, 'lora_pred', [c for c in m1 if c != 'llm_composite'] + ['ft_pred'], permute=True)
    for r, ph in zip((p1, p2), holm([p1['p_one_sided'], p2['p_one_sided']])):
        r['p_holm'] = float(ph)
        r['verdict'] = '支持' if ph < 0.05 and r['permutation_p_one_sided'] < 0.05 else '支持されない'
    s1 = e4.fit(d, 'llm_composite', m1 + ['lora_pred'])
    s2 = e4.fit(d, 'lora_pred', [c for c in m1 if c != 'llm_composite'] + ['ridge_pred'])
    dd = d.dropna(subset=['y', 'llm_composite', 'lora_pred', 'ft_pred', 'ridge_pred'])
    base = [c for c in m1 if c != 'llm_composite' and dd[c].std() > 0]
    r2 = {name: float(e2.r_squared(dd, base + [name]) - e2.r_squared(dd, base))
          for name in ('llm_composite', 'ft_pred', 'ridge_pred', 'lora_pred')}
    r2['lora_and_ft'] = float(e2.r_squared(dd, base + ['lora_pred', 'ft_pred']) - e2.r_squared(dd, base))
    corr = {c: float(A.lora_pred.corr(A[c], method='spearman')) for c in ('ft_pred', 'ridge_pred', 'llm_composite')}
    return {'P1_lora_alone': p1, 'P2_lora_beyond_finetuned_finbert': p2,
            'S1_llm_with_lora': s1, 'S2_lora_beyond_ridge': s2,
            'r2_gain_over_controls': r2, 'spearman': corr, 'n_with_prediction': int(A.lora_pred.notna().sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    conn = connect(config.DB_PATH)
    conn.execute('PRAGMA busy_timeout = 60000')
    rel = e4.releases_with_text(conn)
    train_df = rel[rel.filing_date.between(*TRAIN) & rel.abn_r_0_1.notna()].reset_index(drop=True)
    pred_df = rel[rel.filing_date >= PREDICT_FROM].reset_index(drop=True)
    print(f'train {len(train_df)} releases, predict {len(pred_df)}', flush=True)
    chrono = _chrono()
    model, tok = chrono.load(VINTAGE)
    if args.smoke:
        head, lora, stats = train(model, tok, train_df.sample(64, random_state=0).reset_index(drop=True), smoke=True)
        t0 = time.time()
        p = predict(model, tok, head, pred_df.head(8))
        print('smoke:', stats, f'predict {(time.time() - t0) / 8:.2f}s per release', flush=True)
        return

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    head, lora, stats = train(model, tok, train_df)
    ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'lora': [p.detach().cpu() for p in lora], 'head': head.state_dict(), 'hp': HP, 'stats': stats}, ADAPTER)
    t0 = time.time()
    preds = predict(model, tok, head, pred_df)
    stats['predict_seconds'] = round(time.time() - t0, 1)
    params = {'model': f'manelalab/chrono-gpt-instruct-v1-{VINTAGE}1231 + LoRA', 'train_window': TRAIN,
              'predict_from': PREDICT_FROM, 'hyperparameters': HP, 'code_commit': commit}
    run_id = db.get_or_create_score_run(conn, 'chrono_lora', f'ChronoGPT {VINTAGE} LoRA on reactions', params, True,
                                        'E10, trained on 2015-2022 reactions')
    for acc, v in preds.items():
        db.save_scores(conn, run_id, acc, {'predicted_reaction': float(v)})
    print(f'score run {run_id}: {len(preds)} predictions', flush=True)
    ft_run = json.loads((REPO / 'experiments' / 'E4_finbert_finetune' / 'results' / 'summary.json')
                        .read_text(encoding='utf-8'))['score_run']
    result = evaluate(conn, preds, ft_run)
    summary = {'code_commit': commit, 'score_run': run_id, 'training': stats, **result}
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
