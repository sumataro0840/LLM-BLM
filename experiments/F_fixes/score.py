"""
Scores of the fix experiments that use no stock reaction (修正の実験方針.md, 4). Saved under data/cache/fixes/ (outside Git);
each step skips what is already saved, so an interrupted run continues.

    python experiments/F_fixes/score.py stats       # fix 1: removed share, fix 5 check (outlook words beyond the window)
    python experiments/F_fixes/score.py dictionary  # fix 1: Loughran-McDonald net tone of the cleaned text
    python experiments/F_fixes/score.py finbert     # fixes 1-2: sentence probabilities of the cleaned text (3 readings)
    python experiments/F_fixes/score.py chrono_raw  # baseline: vintage 2015, raw text, releases filed after 2023-10-31
    python experiments/F_fixes/score.py chrono      # fixes 1 and 4: vintage 2015, cleaned text, 3 vectors per release
    python experiments/F_fixes/score.py compress    # fix 5: LLMLingua-2 compression of the cleaned text (if stats says so)
    python experiments/F_fixes/score.py finbert5    # fix 5: FinBERT (first 200 sentences) of the compressed text
    python experiments/F_fixes/score.py chrono5     # fix 5: ChronoGPT vector (mean of the last layer) of the compressed text
    python experiments/F_fixes/score.py all

Releases: E5's set (E2's base sample with a main-window reaction and a text; only whether the reaction exists is used).
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from experiments.F_fixes.clean import clean

OUT = REPO / 'data' / 'cache' / 'fixes'
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'
VINTAGE = 2015
RAW_CHARS = 9000                 # as E5: more than fits, the model keeps the first 1,792 tokens
CLEAN_CHARS = 12000
PROMPT = '\nIn one word, this earnings release is'
MID_LAYER = 26                   # output of the 26th of the 52 blocks (the last encoder block)
OUTLOOK = re.compile(r'\b(outlook|guidance|expects?|expected|forecasts?)\b', re.I)
FINBERT_MAX_SENTENCES = 1000     # cap for the whole-text reading
COMPRESSOR = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
COMPRESSOR_REVISION = 'ebaba9b0e874dadd3003ffcff828e4397e568089'      # MIT licence
COMPRESS_TARGET = 1700           # GPT-2 tokens: below ChronoGPT's 1,780-token text window
COMPRESS_INPUT_CHARS = 60000     # the compressor reads at most this much of the cleaned text
CHUNK = 1000


def releases():
    """The releases to score. Without the database (e.g. on a rented GPU), from the CSV that export_releases() wrote."""
    import os
    csv = os.environ.get('FIXES_RELEASES_CSV')
    if csv:
        return pd.read_csv(csv).sort_values(['filing_date', 'accession']).reset_index(drop=True)
    from marketdb.db import events as db
    from marketdb.db.connection import connect
    p = pd.read_csv(PANEL, usecols=['accession', 'filing_date', 'abn_r_0_1'])
    p = p[p.abn_r_0_1.notna()]
    rel = db.load_releases(connect(config.DB_PATH), with_text=True)[['accession', 'text_path', 'accepted_et']]
    return p.drop(columns='abn_r_0_1').merge(rel, on='accession').sort_values(['filing_date', 'accession']).reset_index(drop=True)


def export_releases(path):
    """Write the release list (accession, filing date, text path, acceptance time) for scoring without the database."""
    releases().to_csv(path, index=False)


def cleaned(text_path):
    return clean(texts.read(text_path))


# ---------- stats and the fix 5 check ----------

def step_stats(rel):
    import tiktoken
    tok = tiktoken.get_encoding('gpt2')
    rows = []
    for r in rel.itertuples():
        raw = texts.read(r.text_path)
        c = clean(raw)
        m = OUTLOOK.search(c)
        first_tok = len(tok.encode(c[:m.start()])) if m else None
        m_raw = OUTLOOK.search(raw)
        rows.append({'accession': r.accession, 'filing_date': r.filing_date, 'raw_chars': len(raw), 'clean_chars': len(c),
                     'outlook_found': m is not None, 'outlook_token_clean': first_tok,
                     'outlook_char_raw': m_raw.start() if m_raw else None})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'clean_stats.csv', index=False)
    dev = df[df.filing_date.between('2016-01-01', '2022-12-31')]
    window = 1792 - 12        # the ChronoGPT window minus the appended prompt
    beyond = dev.outlook_found & (dev.outlook_token_clean > window)
    check = {'n_dev_releases': int(len(dev)), 'removed_share_mean': float(1 - (dev.clean_chars / dev.raw_chars).mean()),
             'outlook_found_share': float(dev.outlook_found.mean()),
             'outlook_beyond_chrono_window_share': float(beyond.mean()),
             'raw_outlook_beyond_15000_chars_share': float((dev.outlook_char_raw > 15000).mean()),
             'rule': 'fix 5 (LLMLingua) is run only if outlook_beyond_chrono_window_share >= 0.10',
             'run_fix5': bool(beyond.mean() >= 0.10)}
    (OUT / 'fix5_check.json').write_text(json.dumps(check, indent=1), encoding='utf-8')
    print(json.dumps(check, indent=1), flush=True)


# ---------- dictionary ----------

def step_dictionary(rel):
    from events.scorers.dictionary import load_lists, tone
    lists = load_lists(config.LM_DICTIONARY)
    rows = [{'accession': r.accession, **tone(cleaned(r.text_path), lists, int(r.accepted_et[:4]))} for r in rel.itertuples()]
    pd.DataFrame(rows)[['accession', 'net_tone', 'n_words']].to_csv(OUT / 'dictionary_clean.csv', index=False)
    print(f'dictionary: {len(rows)} releases', flush=True)


# ---------- FinBERT (zero-shot, sentence level) ----------

def step_finbert(rel):
    """Per release, the mean positive - negative probability over three readings of the cleaned text's sentences:
    head200 (first 200 sentences, as the current scorer), headtail (first 50 + last 150), all (every sentence, cap 1000)."""
    import torch
    from huggingface_hub import model_info
    from transformers import BertForSequenceClassification, BertTokenizer
    from events.scorers.finbert import sentences
    path = OUT / 'finbert_clean.csv'
    done = set(pd.read_csv(path).accession) if path.exists() else set()
    todo = rel[~rel.accession.isin(done)]
    print(f'finbert: {len(todo)} of {len(rel)} to score', flush=True)
    revision = model_info(config.FINBERT_MODEL).sha
    tok = BertTokenizer.from_pretrained(config.FINBERT_MODEL, revision=revision)
    model = BertForSequenceClassification.from_pretrained(config.FINBERT_MODEL, revision=revision).to('cuda').eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    split_sentences = lambda text: [s for s in re.split(r'(?<=[.!?])\s+|\n+', text)]
    rows, t0 = [], time.time()
    for k, r in enumerate(todo.itertuples(), 1):
        # the current scorer's sentence rule, without its cap of 200
        sents = [s.strip() for s in split_sentences(cleaned(r.text_path))
                 if 20 <= len(s.strip()) <= 1000 and re.search(r'[a-z]', s)][:FINBERT_MAX_SENTENCES]
        row = {'accession': r.accession, 'n_sentences': len(sents)}
        if sents:
            probs = []
            with torch.no_grad():
                for i in range(0, len(sents), 64):
                    b = tok(sents[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        probs.append(torch.softmax(model(**b).logits.float(), -1).cpu())
            p = torch.cat(probs).numpy()
            net = p[:, labels['positive']] - p[:, labels['negative']]
            row['head200'] = float(net[:200].mean())
            row['headtail'] = float(np.concatenate([net[:50], net[50:][-150:]]).mean())
            row['all'] = float(net.mean())
        rows.append(row)
        if k % 500 == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)
            rows = []
            print(f'  finbert {k}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    del model
    torch.cuda.empty_cache()


# ---------- ChronoGPT ----------

def _forward(chrono, model, ids, n_text):
    """One pass: (mean of the normalized last layer over the text tokens, last token of the last layer,
    mean of the normalized MID_LAYER output over the text tokens)."""
    import torch
    norm = lambda x: torch.nn.functional.rms_norm(x, (x.size(-1),))
    x0 = norm(model.embed(ids).bfloat16())
    x = x0
    ve = model.value_embeds(ids[0].view(-1))
    ve = [v.unsqueeze(0) if v is not None else None for v in ve]
    ve_enc, ve_dec = ve[:model.num_encoder_layers], ve[model.num_encoder_layers:]
    skips, mid = [], None
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, ve_enc[i], x0)
        skips.append(x)
        if i + 1 == MID_LAYER:
            mid = norm(x)
    for i in range(model.num_decoder_layers):
        x = x + model.skip_weights[i] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, ve_dec[i], x0)
    last = norm(x)[0]
    return (last[:n_text].float().mean(0).cpu().numpy(), last[-1].float().cpu().numpy(),
            mid[0, :n_text].float().mean(0).cpu().numpy())


def _run_chrono(rel, d, make_vectors, label):
    from events.scorers.chrono import _chrono
    d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in range(0, len(rel), CHUNK) if not (d / f'part{s // CHUNK:03d}.npz').exists()]
    print(f'{label}: {len(rel)} releases, {len(todo)} chunks to do', flush=True)
    if not todo:
        return
    chrono = _chrono()
    model, tok = chrono.load(VINTAGE)
    t0 = time.time()
    for s in todo:
        part = rel.iloc[s:s + CHUNK]
        arrays = [make_vectors(chrono, model, tok, r) for r in part.itertuples()]
        np.savez_compressed(d / f'part{s // CHUNK:03d}.npz', accession=part.accession.to_numpy(),
                            **{k: np.stack([a[k] for a in arrays]).astype(np.float16) for k in arrays[0]})
        print(f'  {label}: {s + len(part)}/{len(rel)} ({time.time() - t0:.0f}s)', flush=True)
    del model


def step_chrono_raw(rel):
    """The baseline (E5's vector) for the releases E5's vintage-2015 block did not cover."""
    ext = rel[rel.filing_date > '2023-10-31'].reset_index(drop=True)

    def vec(chrono, model, tok, r):
        return {'mean_last': chrono.embed(model, tok, texts.read(r.text_path, RAW_CHARS)).float().numpy()}
    _run_chrono(ext, OUT / 'chrono_raw_v2015_ext', vec, 'chrono_raw')


def step_chrono(rel):
    import torch

    def vec(chrono, model, tok, r):
        prompt = tok.encode(PROMPT)
        text = tok.encode(cleaned(r.text_path)[:CLEAN_CHARS])[:chrono.CONTEXT - len(prompt)]
        with torch.no_grad():
            mean_last, last_token, mean_mid = _forward(chrono, model, torch.tensor([text + prompt], device='cuda'), len(text))
        return {'mean_last': mean_last, 'last_token': last_token, 'mean_mid': mean_mid}
    _run_chrono(rel, OUT / 'chrono_clean_v2015', vec, 'chrono_clean')


def load_chrono(name, key):
    """(accessions, float32 matrix) of a finished ChronoGPT step."""
    parts = [np.load(f, allow_pickle=True) for f in sorted((OUT / name).glob('part*.npz'))]
    return np.concatenate([p['accession'] for p in parts]), np.concatenate([p[key] for p in parts]).astype(np.float32)


# ---------- fix 5 ----------

def compressed_texts():
    df = pd.read_csv(OUT / 'compressed.csv', keep_default_na=False)
    return dict(zip(df.accession, df.text))


def step_compress(rel):
    """LLMLingua-2 keeps the tokens it predicts to be informative, at the rate that brings the cleaned text to about
    COMPRESS_TARGET GPT-2 tokens (sentence punctuation and line breaks are always kept). A cleaned text that already
    fits is left as it is."""
    check = json.loads((OUT / 'fix5_check.json').read_text(encoding='utf-8'))
    if not check['run_fix5']:
        print('fix 5 is not required by the check')
        return
    import tiktoken
    from llmlingua import PromptCompressor
    path = OUT / 'compressed.csv'
    done = set(pd.read_csv(path, usecols=['accession']).accession) if path.exists() else set()
    todo = rel[~rel.accession.isin(done)]
    print(f'compress: {len(todo)} of {len(rel)}', flush=True)
    enc = tiktoken.get_encoding('gpt2')
    comp = PromptCompressor(model_name=COMPRESSOR, use_llmlingua2=True, device_map='cuda',
                            model_config={'revision': COMPRESSOR_REVISION})
    rows, t0 = [], time.time()
    for k, r in enumerate(todo.itertuples(), 1):
        c = cleaned(r.text_path)[:COMPRESS_INPUT_CHARS]
        n = len(enc.encode(c))
        if n > COMPRESS_TARGET:
            c = comp.compress_prompt(c, rate=COMPRESS_TARGET / n, force_tokens=['\n', '.', '!', '?'],
                                     drop_consecutive=True)['compressed_prompt']
        rows.append({'accession': r.accession, 'n_tokens_in': n, 'text': c})
        if k % 500 == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)
            rows = []
            print(f'  compress {k}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    del comp


def step_finbert5(rel):
    import torch
    from huggingface_hub import model_info
    from transformers import BertForSequenceClassification, BertTokenizer
    from events.scorers.finbert import sentences
    if not (OUT / 'compressed.csv').exists():
        return
    text_of = compressed_texts()
    revision = model_info(config.FINBERT_MODEL).sha
    tok = BertTokenizer.from_pretrained(config.FINBERT_MODEL, revision=revision)
    model = BertForSequenceClassification.from_pretrained(config.FINBERT_MODEL, revision=revision).to('cuda').eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    rows = []
    for r in rel.itertuples():
        sents = sentences(text_of[r.accession])          # the current scorer's rule (first 200 sentences)
        v = None
        if sents:
            with torch.no_grad():
                b = [tok(sents[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
                     for i in range(0, len(sents), 64)]
                pr = torch.cat([torch.softmax(model(**x).logits.float(), -1).cpu() for x in b]).numpy()
            v = float((pr[:, labels['positive']] - pr[:, labels['negative']]).mean())
        rows.append({'accession': r.accession, 'net_tone': v})
    pd.DataFrame(rows).to_csv(OUT / 'finbert_compressed.csv', index=False)
    print(f'finbert5: {len(rows)}', flush=True)


def step_chrono5(rel):
    import torch
    if not (OUT / 'compressed.csv').exists():
        return
    text_of = compressed_texts()

    def vec(chrono, model, tok, r):
        prompt = tok.encode(PROMPT)
        text = tok.encode(text_of[r.accession])[:chrono.CONTEXT - len(prompt)]
        with torch.no_grad():
            mean_last, _, _ = _forward(chrono, model, torch.tensor([text + prompt], device='cuda'), len(text))
        return {'mean_last': mean_last}
    _run_chrono(rel, OUT / 'chrono_comp_v2015', vec, 'chrono_comp')


STEPS = {'stats': step_stats, 'dictionary': step_dictionary, 'finbert': step_finbert, 'chrono_raw': step_chrono_raw,
         'chrono': step_chrono, 'compress': step_compress, 'finbert5': step_finbert5, 'chrono5': step_chrono5}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    names = sys.argv[1:]
    if not names or any(n not in STEPS and n != 'all' for n in names):
        print(__doc__)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rel = releases()
    for name in (list(STEPS) if 'all' in names else names):
        STEPS[name](rel)


if __name__ == '__main__':
    main()
