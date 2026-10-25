"""
Pod side: one of smallcap/score.py's two ChronoGPT steps, with the same functions, on the bundle's release list.

    python score_chrono.py chrono_raw|chrono_clean OUTDIR      # run in bundle/LLM-BLM (see run.sh)

Writes OUTDIR/chrono_raw_v2015/part*.npz or OUTDIR/chrono_clean_v2015/part*.npz, the same files (same chunks of the
same list) as smallcap/score.py writes on the PC, so the PC's scoring finds them done.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from events import texts  # noqa: E402
from experiments.F_fixes import score as F  # noqa: E402


def main():
    step, out = sys.argv[1], Path(sys.argv[2])
    rel = pd.read_csv(REPO / 'data' / 'cache' / 'smallcap_releases.csv', dtype=str, keep_default_na=False)
    out.mkdir(parents=True, exist_ok=True)
    if step == 'chrono_raw':             # smallcap/score.py chrono_raw
        def vec(chrono, model, tok, r):
            return {'mean_last': chrono.embed(model, tok, texts.read(r.text_path, F.RAW_CHARS)).float().numpy()}
        F._run_chrono(rel, out / 'chrono_raw_v2015', vec, 'chrono_raw')
    elif step == 'chrono_clean':         # smallcap/score.py chrono_clean
        F.OUT = out
        F.step_chrono(rel)
    else:
        raise SystemExit(f'unknown step {step}')


if __name__ == '__main__':
    main()
