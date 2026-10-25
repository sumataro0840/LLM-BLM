"""
R1: FinBERT for the replication sample on a RunPod pod (the PC's GPU ran at 0.5 s a release, about 5 hours for the
rest), then the rest of the PC's scoring chain. Run detached (smallcap/runpod/start_finbert.ps1) after the PC's
`smallcap.score all` has been stopped:

1. the releases the PC already scored (FinBERT run of data/smallcap_rep.sqlite) are skipped;
2. a one-GPU pod runs the official scorer on the others with the PC's library versions; the pod is deleted at the end,
   also on failure;
3. the scores are checked (same method, model and params hash as the database's run) and saved with db.save_scores,
   only for the releases still without a score;
4. after the ChronoGPT pod (orchestrate.py) has finished, `python -m smallcap.score finbert chrono_raw chrono_clean`
   runs on the PC (both find their work done, or finish it), and "done: rep_score_all" / "all done" are appended to
   rep_chain.log, which starts the waiting R1 analysis (experiments/R1_replication/start_analysis.ps1).
Log: data/cache/logs/runpod_finbert.log.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'smallcap' / 'runpod'))
import orchestrate as O  # noqa: E402
import pod  # noqa: E402
from marketdb.db import events as db  # noqa: E402

LOG = O.LOGS / 'runpod_finbert.log'
DB = REPO / 'data' / 'smallcap_rep.sqlite'
WORK = REPO / 'data' / 'cache' / 'runpod' / 'finbert_rep'
O.LOG = LOG


def finbert_run(conn):
    row = conn.execute("SELECT score_run_id, model, params_sha256 FROM release_score_runs WHERE method = 'finbert'").fetchone()
    if row is None:
        raise RuntimeError('no FinBERT run in the replication database')
    return row


def pod_part():
    conn = sqlite3.connect(DB)
    run_id, model, digest = finbert_run(conn)
    done = db.scored_accessions(conn, run_id)
    conn.close()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'skip.txt').write_text('\n'.join(sorted(done)), encoding='utf-8')
    O.log(f'FinBERT run {run_id}: {len(done)} releases already scored on the PC')
    pod_id = None
    try:
        b = pod.create('llm-blm-r1-finbert', 'NVIDIA GeForce RTX 4090', 1, 'SECURE')
        pod_id = b['id']
        O.log(f'pod {pod_id}: {b["costPerHr"]} USD/h')
        addr = pod.wait_ssh(pod_id)
        for _ in range(20):
            if O.ssh(addr, 'nvidia-smi -L', check=False).returncode == 0:
                break
            time.sleep(15)
        O.scp(addr, str(REPO / 'data' / 'cache' / 'runpod' / 'smallcap_rep.tar.gz'), 'POD:/workspace/')
        O.ssh(addr, 'cd /workspace && tar xzf smallcap_rep.tar.gz', timeout=1800)
        O.scp(addr, str(REPO / 'events' / 'scorers' / 'finbert.py'), 'POD:/workspace/bundle/LLM-BLM/events/scorers/')
        O.scp(addr, str(REPO / 'smallcap' / 'runpod' / 'score_finbert.py'), 'POD:/workspace/bundle/')
        O.scp(addr, str(WORK / 'skip.txt'), 'POD:/workspace/bundle/')
        O.log('uploaded; setup')
        r = O.ssh(addr, 'cd /workspace/bundle && bash run.sh setup', timeout=3600)
        O.log(r.stdout.strip().splitlines()[-2] if r.stdout.strip() else 'setup output empty')
        O.ssh(addr, 'cd /workspace/bundle/LLM-BLM && (PATH=/workspace/venv/bin:$PATH nohup python ../score_finbert.py '
                    '/workspace/finbert.jsonl ../skip.txt > /workspace/finbert.out 2>&1 &)')
        t0 = time.time()
        while True:
            tail = O.ssh(addr, 'tail -n1 /workspace/finbert.out', check=False).stdout.strip()
            O.log(f'progress: {tail}')
            if tail == 'finbert done':
                break
            if 'Error' in tail or 'Traceback' in tail or time.time() - t0 > 3 * 3600:
                raise RuntimeError(f'FinBERT on the pod stopped: {tail}')
            time.sleep(300)
        O.scp(addr, 'POD:/workspace/finbert.jsonl', str(WORK / 'finbert.jsonl'))
    finally:
        if pod_id:
            try:
                pod.delete(pod_id)
                O.log(f'pod {pod_id} deleted')
            except BaseException as e:
                O.log(f'could not delete pod {pod_id}: {e}  -- delete it on the RunPod console')
    # check and save
    conn = sqlite3.connect(DB)
    conn.execute('PRAGMA busy_timeout = 60000')
    run_id, model, digest = finbert_run(conn)
    lines = (WORK / 'finbert.jsonl').read_text(encoding='utf-8').splitlines()
    head = json.loads(lines[0])
    if head['method'] != 'finbert' or head['model'] != model or db.params_hash(head['params']) != digest:
        raise RuntimeError(f'the pod scorer differs from run {run_id}: {head}')
    done = db.scored_accessions(conn, run_id)
    saved = 0
    for line in lines[1:]:
        r = json.loads(line)
        if r['accession'] not in done:
            db.save_scores(conn, run_id, r['accession'], r['values'], r['call'])
            saved += 1
    O.log(f'checked: same method, model and params hash as run {run_id}; saved {saved} releases '
          f'(the pod scored {len(lines) - 1}, header n_todo {head["n_todo"]})')
    conn.close()


def append_chain(text):
    with open(O.LOGS / 'rep_chain.log', 'ab') as f:            # PowerShell's Out-File wrote it in UTF-16 LE
        f.write((text + '\r\n').encode('utf-16-le'))


def pc_part():
    O.log('waiting for the ChronoGPT pod (runpod_rep.log)')
    rp = O.LOGS / 'runpod_rep.log'
    while True:
        t = rp.read_text(encoding='utf-8') if rp.exists() else ''
        if 'all done' in t or 'failed' in t:
            break
        time.sleep(120)
    O.log('ChronoGPT pod: ' + ('all done' if 'all done' in t else 'failed (the PC computes ChronoGPT itself)'))
    env = {**os.environ, 'SMALLCAP_DB': str(DB), 'SMALLCAP_CACHE': str(REPO / 'data' / 'cache' / 'smallcap_rep'),
           'PYTHONIOENCODING': 'utf-8'}
    with open(O.LOGS / 'rep_score_rest.log', 'w', encoding='utf-8') as out, \
            open(O.LOGS / 'rep_score_rest.err', 'w', encoding='utf-8') as err:
        r = subprocess.run([sys.executable, '-W', 'ignore', '-m', 'smallcap.score', 'finbert', 'chrono_raw', 'chrono_clean'],
                           cwd=REPO, env=env, stdout=out, stderr=err)
    if r.returncode != 0:
        raise RuntimeError('smallcap.score finbert chrono_raw chrono_clean failed (rep_score_rest.err)')
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    append_chain(f'done: rep_score_all {stamp} (FinBERT: PC and RunPod; ChronoGPT: RunPod)')
    append_chain(f'all done {stamp}')
    O.log('rep_chain.log: all done (the R1 analysis starts)')


def main():
    try:
        pod_part()
        pc_part()
        O.log('finished')
    except Exception:
        O.log('failed: ' + traceback.format_exc().replace('\n', ' / '))


if __name__ == '__main__':
    main()
