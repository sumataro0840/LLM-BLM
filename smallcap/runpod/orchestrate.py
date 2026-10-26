"""
R1: the two ChronoGPT steps of the replication sample on one RunPod pod with two GPUs, from start to end, in a process
that survives this session (smallcap/runpod/start.ps1). Waits until the release fetch has finished, builds the bundle,
creates the pod, runs chrono_raw and chrono_clean side by side, brings the vectors back, checks them against the
release list and puts them where smallcap/score.py looks (data/cache/smallcap_rep/), so the PC's scoring finds them
done. The pod is deleted at the end, also when something fails. Log: data/cache/logs/runpod_rep.log.
"""
import math
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'smallcap' / 'runpod'))
import pod  # noqa: E402

LOGS = REPO / 'data' / 'cache' / 'logs'
TAG = os.environ.get('SMALLCAP_TAG', 'rep')   # rep: R1's sample; g3: the G3 sample (data/smallcap_g3.sqlite)
LOG = LOGS / f'runpod_{TAG}.log'
BUNDLE = REPO / 'data' / 'cache' / 'runpod' / f'smallcap_{TAG}.tar.gz'
STAGE = REPO / 'data' / 'cache' / 'runpod' / f'bundle_{TAG}'
BACK = REPO / 'data' / 'cache' / 'runpod' / f'{TAG}_out'
DEST = REPO / 'data' / 'cache' / f'smallcap_{TAG}'
STEPS = ['chrono_raw', 'chrono_clean']
SSH = 'C:/Program Files/Git/usr/bin/ssh.exe'
SCP = 'C:/Program Files/Git/usr/bin/scp.exe'
KEY = str(Path.home() / '.ssh' / 'id_ed25519_runpod')
OPTS = ['-i', KEY, '-o', 'StrictHostKeyChecking=accept-new', '-o', f'UserKnownHostsFile={Path.home() / ".ssh" / "known_hosts_runpod"}',
        '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30']
MAX_HOURS = 5
CHUNK = 1000


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')


def chain_text():
    p = LOGS / f'{TAG}_chain.log'
    if not p.exists():
        return ''
    raw = p.read_bytes()
    return raw.decode('utf-16', errors='ignore') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', 'ignore')


def ssh(addr, cmd, timeout=600, check=True):
    user_host, _, port = addr.partition(' -p ')
    return subprocess.run([SSH, *OPTS, '-p', port, user_host, cmd], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                          check=check)


def scp(addr, src, dst, timeout=7200):
    user_host, _, port = addr.partition(' -p ')
    subprocess.run([SCP, *OPTS, '-P', port, src.replace('POD:', f'{user_host}:'), dst.replace('POD:', f'{user_host}:')],
                   check=True, timeout=timeout, capture_output=True)


def check_and_place(step, n, accessions):
    sub = f'{step}_v2015'
    with tarfile.open(BACK / f'out_{step}.tar.gz') as tar:
        tar.extractall(BACK / step)
    parts = sorted((BACK / step / sub).glob('part*.npz'))
    got = np.concatenate([np.load(f, allow_pickle=True)['accession'] for f in parts])
    if len(parts) != math.ceil(n / CHUNK) or list(got) != list(accessions):
        raise RuntimeError(f'{step}: {len(parts)} parts / {len(got)} rows do not match the release list ({n})')
    (DEST / sub).mkdir(parents=True, exist_ok=True)
    placed = 0
    for f in parts:
        target = DEST / sub / f.name
        if not target.exists():           # a part the PC already made is the same chunk; keep it
            target.write_bytes(f.read_bytes())
            placed += 1
    log(f'{step}: {len(parts)} parts checked, {placed} placed in {DEST / sub}')


def main():
    global LOG, BUNDLE, STAGE, BACK, DEST
    test = '--test' in sys.argv        # 30 releases of the original sample, results kept apart
    name, db, extra = ('test2', 'data/smallcap.sqlite', ['--limit', '30']) if test else (TAG, f'data/smallcap_{TAG}.sqlite', [])
    if test:
        R = REPO / 'data' / 'cache' / 'runpod'
        LOG, BUNDLE, STAGE, BACK, DEST = (LOGS / 'runpod_test.log', R / 'smallcap_test2.tar.gz', R / 'bundle_test2',
                                          R / 'test2_out', R / 'test2_dest')
    LOGS.mkdir(parents=True, exist_ok=True)
    log('waiting for the release fetch')
    while not test and f'done: {TAG}_build_releases' not in chain_text():
        if 'failed' in chain_text():
            log('the data chain failed; stop')
            return
        time.sleep(300)
    log('building the bundle')
    subprocess.run([sys.executable, str(REPO / 'smallcap' / 'runpod' / 'build_bundle.py'), '--db', db, '--name', name, *extra],
                   check=True, cwd=REPO, capture_output=True)
    rel = pd.read_csv(STAGE / 'LLM-BLM' / 'data' / 'cache' / 'smallcap_releases.csv', dtype=str, keep_default_na=False)
    log(f'bundle: {len(rel)} releases, {BUNDLE.stat().st_size / 1e6:.0f} MB')
    pod_id, two = None, True
    try:
        for cloud in ('SECURE', 'COMMUNITY'):
            try:
                b = pod.create('llm-blm-r1', 'NVIDIA GeForce RTX 4090', 2, cloud)
                pod_id = b['id']
                break
            except SystemExit as e:
                log(f'2-GPU pod on {cloud} not created: {e}')
        if pod_id is None:
            b = pod.create('llm-blm-r1', 'NVIDIA GeForce RTX 4090', 1, 'SECURE')
            pod_id, two = b['id'], False
        log(f'pod {pod_id}: {b["costPerHr"]} USD/h, {"two GPUs" if two else "one GPU (steps one after the other)"}')
        addr = pod.wait_ssh(pod_id)
        for _ in range(20):
            if ssh(addr, 'nvidia-smi -L', check=False).returncode == 0:
                break
            time.sleep(15)
        log(f'ssh ready; uploading')
        scp(addr, str(BUNDLE), 'POD:/workspace/')
        ssh(addr, f'cd /workspace && tar xzf {BUNDLE.name}', timeout=1800)
        log('uploaded; setup')
        r = ssh(addr, 'cd /workspace/bundle && bash run.sh setup', timeout=3600)
        log(r.stdout.strip().splitlines()[-2] if r.stdout.strip() else 'setup output empty')
        if two:
            ssh(addr, 'cd /workspace/bundle && (CUDA_VISIBLE_DEVICES=0 nohup bash run.sh chrono_raw > /workspace/raw.out 2>&1 &) '
                      '&& (CUDA_VISIBLE_DEVICES=1 nohup bash run.sh chrono_clean > /workspace/clean.out 2>&1 &)')
        else:
            ssh(addr, "cd /workspace/bundle && (nohup bash -c 'bash run.sh chrono_raw; bash run.sh chrono_clean' "
                      "> /workspace/both.out 2>&1 &)")
        t0 = time.time()
        while True:
            done = ssh(addr, 'ls /workspace/out_chrono_raw.tar.gz /workspace/out_chrono_clean.tar.gz 2>/dev/null | wc -l',
                       check=False).stdout.strip()
            prog = ssh(addr, 'tail -qn1 /workspace/bundle/out/log_chrono_raw.txt /workspace/bundle/out/log_chrono_clean.txt '
                             '2>/dev/null', check=False).stdout.strip().replace('\n', ' | ')
            log(f'progress: {prog}')
            if done == '2':
                break
            if time.time() - t0 > MAX_HOURS * 3600:
                raise RuntimeError(f'not finished in {MAX_HOURS} hours')
            time.sleep(60 if test else 600)
        BACK.mkdir(parents=True, exist_ok=True)
        for step in STEPS:
            scp(addr, f'POD:/workspace/out_{step}.tar.gz', str(BACK / f'out_{step}.tar.gz'))
            check_and_place(step, len(rel), rel.accession)
        log('all done')
    except Exception:
        log('failed: ' + traceback.format_exc().replace('\n', ' / '))
    finally:
        if pod_id:
            try:
                pod.delete(pod_id)
                log(f'pod {pod_id} deleted')
            except BaseException as e:
                log(f'could not delete pod {pod_id}: {e}  -- delete it on the RunPod console')


if __name__ == '__main__':
    main()
