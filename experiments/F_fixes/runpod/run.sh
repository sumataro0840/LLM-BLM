#!/usr/bin/env bash
# Scoring of the fix experiments on a RunPod GPU (preregistration.md, 追記 2). Run inside the unpacked bundle:
#     cd /workspace/bundle && bash run.sh
# Takes the release list, the texts and the code from the bundle; downloads ChronoGPT vintage 2015, FinBERT and
# LLMLingua-2 from Hugging Face; writes /workspace/fixes_out.tar.gz (the scores, the log and the environment).
set -euo pipefail
cd "$(dirname "$0")"
BUNDLE=$(pwd)
LOG=$BUNDLE/LLM-BLM/data/cache/fixes/runpod.log
mkdir -p "$(dirname "$LOG")"

# the library versions of the scoring on the PC (torch is the one of the pod image; it is recorded)
pip install --quiet "transformers==5.19.0" "tokenizers==0.23.3" "tiktoken==0.14.0" "safetensors==0.8.0" \
    "huggingface_hub==2.2.0" "llmlingua==0.2.2" "nltk==3.10.3" "pandas==3.0.6" "numpy==2.5.3" \
    "python-dotenv==1.2.4" "accelerate==1.15.0" 2>&1 | tail -3

# ChronoGPT vintage 2015: the same commit and the same original file as on the PC
cd "$BUNDLE/LLM/chrono"
python chrono.py download 2015
python - <<'EOF'
import json
got = json.load(open('models/2015/meta.json'))
want = json.load(open('../../chrono_2015_meta_expected.json'))
for k in ('repo', 'revision', 'original_sha256'):
    assert got[k] == want[k], f'ChronoGPT 2015 differs from the PC: {k} {got[k]} != {want[k]}'
print('ChronoGPT 2015 matches the PC copy:', got['revision'][:10], got['original_sha256'][:10])
EOF

cd "$BUNDLE/LLM-BLM"
export FIXES_RELEASES_CSV=data/cache/fixes_releases.csv
export EVENTS_CHRONO_DIR=../LLM/chrono
python - <<'EOF' > data/cache/fixes/runpod_env.json
import json, platform, torch, importlib.metadata as m
pk = ['torch', 'transformers', 'tokenizers', 'tiktoken', 'safetensors', 'huggingface_hub', 'llmlingua', 'pandas', 'numpy']
print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                  'packages': {p: m.version(p) for p in pk}}, indent=1))
EOF
cat data/cache/fixes/runpod_env.json
python experiments/F_fixes/score.py stats dictionary finbert chrono compress finbert5 chrono5 2>&1 | tee "$LOG"

tar czf /workspace/fixes_out.tar.gz -C data/cache fixes
echo "done: /workspace/fixes_out.tar.gz"
