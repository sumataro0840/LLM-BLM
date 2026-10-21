"""
Exploratory (not pre-registered): the keep rule only compared fixes 2, 4 and 5 with fix 1 when fix 1 was kept, which
did not happen for FinBERT and ChronoGPT. Here each variant is compared with the current reading directly, on each
split's development period, with the same paired t (common.firm_compare) and the same halves.

    python experiments/F_fixes/explore_fixes245.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.F_fixes import common as C
from experiments.F_fixes import develop as DV

OUT = Path(__file__).resolve().parent / 'results'
VARIANTS = {'finbert': ('finbert_net_tone', ['fb_head200', 'fb_headtail', 'fb_all', 'fb_comp']),
            'chrono': ('chrono_base', ['chrono_mean', 'chrono_mid', 'chrono_last', 'chrono_comp'])}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    p = DV.panel_with_scores(D.connect())
    out = {}
    for split in ['B', 'A', 'C']:
        A = DV.dev_firm(p, split)
        h1, h2 = DV.halves_firm(A, split)
        for model, (base, variants) in VARIANTS.items():
            for v in variants:
                full = C.firm_compare(A, base, v)
                r = {'full': full, 'half1': C.firm_compare(h1, base, v)['delta'], 'half2': C.firm_compare(h2, base, v)['delta']}
                r['would_pass'] = DV.passes(full, {'delta': r['half1']}, {'delta': r['half2']})
                out.setdefault(split, {}).setdefault(model, {})[v] = r
                print(split, model, v, 'r2 %.4f -> %.4f delta %+.5f t %+.2f pass %s' % (
                    full['r2_gain_base'] * 100, full['r2_gain_fix'] * 100, full['delta'], full['t'], r['would_pass']), flush=True)
    (OUT / 'explore_fixes245.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
