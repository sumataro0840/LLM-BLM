#!/usr/bin/env bash
# ChronoGPT scoring of a small-cap sample on a RunPod GPU (R1). Run inside the unpacked bundle:
#     cd /workspace/bundle && bash run.sh setup                         # once: Python, libraries, ChronoGPT 2015
#     CUDA_VISIBLE_DEVICES=0 bash run.sh chrono_raw                     # then each step (one GPU each)
#     CUDA_VISIBLE_DEVICES=1 bash run.sh chrono_clean
# setup installs the PC's Python 3.12 and library versions (torch included) and downloads ChronoGPT vintage 2015,
# checked against the PC's copy. A step writes /workspace/out_<step>.tar.gz (its vectors, log and environment).
set -euo pipefail
MODE=$1
cd "$(dirname "$0")"
BUNDLE=$(pwd)

if [ "$MODE" = setup ]; then
    if [ ! -x /workspace/venv/bin/python ]; then
        pip install --quiet uv 2>&1 | tail -1
        uv venv /workspace/venv --python 3.12 2>&1 | tail -1
        uv pip install --python /workspace/venv/bin/python --quiet "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
        uv pip install --python /workspace/venv/bin/python --quiet "transformers==5.19.0" "tokenizers==0.23.3" \
            "tiktoken==0.14.0" "safetensors==0.8.0" "huggingface_hub==2.2.0" "nltk==3.10.3" "pandas==3.0.6" \
            "numpy==2.5.3" "python-dotenv==1.2.4"
    fi
    export PATH=/workspace/venv/bin:$PATH
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
    echo "setup done"
    exit 0
fi

STEP=$MODE
export PATH=/workspace/venv/bin:$PATH
cd "$BUNDLE/LLM-BLM"
export EVENTS_CHRONO_DIR=../LLM/chrono
OUT=$BUNDLE/out
mkdir -p "$OUT"
python - <<'EOF' > "$OUT/env_$STEP.json"
import json, platform, torch, importlib.metadata as m
pk = ['torch', 'transformers', 'tiktoken', 'safetensors', 'huggingface_hub', 'pandas', 'numpy']
print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                  'packages': {p: m.version(p) for p in pk}}, indent=1))
EOF
python "$BUNDLE/score_chrono.py" "$STEP" "$OUT" > "$OUT/log_$STEP.txt" 2>&1
tar czf "/workspace/out_$STEP.tar.gz" -C "$OUT" "${STEP}_v2015" "env_$STEP.json" "log_$STEP.txt"
echo "done: /workspace/out_$STEP.tar.gz"
