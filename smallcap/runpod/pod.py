"""
RunPod pods for the long GPU scoring of the small caps (ChronoGPT), through the REST API. The key is read from
LLM-BLM/.env (RUNPOD_API_KEY) and never printed. Standard library only.

    python smallcap/runpod/pod.py create NAME [GPU]    # one GPU pod (default NVIDIA GeForce RTX 4090), SSH key added
    python smallcap/runpod/pod.py get ID               # status, cost per hour, ssh address when ready
    python smallcap/runpod/pod.py wait ID              # until SSH is reachable; prints "user@host -p port"
    python smallcap/runpod/pod.py list
    python smallcap/runpod/pod.py delete ID            # terminate (billing stops)
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
API = 'https://rest.runpod.io/v1'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
PUBKEY = Path.home() / '.ssh' / 'id_ed25519_runpod.pub'


def key():
    for line in (REPO / '.env').read_text(encoding='utf-8').splitlines():
        name, _, value = line.partition('=')
        if name.strip() == 'RUNPOD_API_KEY':
            return value.strip().strip('"').strip("'")
    raise SystemExit('RUNPOD_API_KEY が .env にありません')


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Authorization': f'Bearer {key()}', 'Content-Type': 'application/json',
                                          'User-Agent': 'llm-blm-runpod/1.0'})   # Python's default is blocked (1010)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{method} {path}: HTTP {e.code} {e.read().decode(errors="replace")[:500]}')


def brief(p):
    ports = p.get('portMappings') or {}
    ssh = f"root@{p.get('publicIp')} -p {ports.get('22')}" if p.get('publicIp') and ports.get('22') else None
    return {'id': p.get('id'), 'name': p.get('name'), 'status': p.get('desiredStatus'), 'costPerHr': p.get('costPerHr'),
            'gpu': (p.get('machine') or {}).get('gpuTypeId') or p.get('gpuTypeIds'), 'ssh': ssh}


def create(name, gpu='NVIDIA GeForce RTX 4090', count=1, cloud='SECURE'):
    body = {'name': name, 'imageName': IMAGE, 'gpuTypeIds': [gpu], 'gpuCount': count, 'cloudType': cloud,
            'containerDiskInGb': 80, 'volumeInGb': 0, 'ports': ['22/tcp'],
            'env': {'PUBLIC_KEY': PUBKEY.read_text().strip()}}
    return brief(call('POST', '/pods', body))


def wait_ssh(pod_id, tries=60):
    for _ in range(tries):
        b = brief(call('GET', f'/pods/{pod_id}'))
        if b['ssh']:
            return b['ssh']
        time.sleep(10)
    raise RuntimeError('SSH の番号が 10 分たっても出ません')


def delete(pod_id):
    call('DELETE', f'/pods/{pod_id}')


def main():
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ('', [])
    if cmd == 'create':
        name = args[0]
        gpu = args[1] if len(args) > 1 else 'NVIDIA GeForce RTX 4090'
        print(json.dumps(create(name, gpu, int(args[2]) if len(args) > 2 else 1,
                                args[3] if len(args) > 3 else 'SECURE'), ensure_ascii=False))
    elif cmd == 'get':
        print(json.dumps(brief(call('GET', f'/pods/{args[0]}')), ensure_ascii=False))
    elif cmd == 'wait':
        print(wait_ssh(args[0]))
    elif cmd == 'list':
        pods = call('GET', '/pods')
        for p in pods if isinstance(pods, list) else pods.get('data', []):
            print(json.dumps(brief(p), ensure_ascii=False))
    elif cmd == 'delete':
        delete(args[0])
        print(f'deleted {args[0]}')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
