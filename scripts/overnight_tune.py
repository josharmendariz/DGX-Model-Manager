#!/usr/bin/env python3
"""Bounded, synthetic GB10 sweep. Execute only through the reviewed systemd units.

No API credentials, prompt history, model downloads, persistent profile edits, or
fleet concurrency changes. Host-maintenance launches use maintained DMM profiles.
"""
import argparse
import concurrent.futures
from contextlib import closing
import datetime as dt
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FLASH = 'hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid'
QWEN = 'hf_qwen_qwen3.8-27b-fp8'
PROFILES = {FLASH: 'qwen38-flash-dgx', QWEN: 'eugr/spark-vllm:latest'}
API = 'http://127.0.0.1:8000'
DMM = 'http://100.115.54.83:8090'
REMOTE_SCRIPT = '/home/josh/DGX-Model-Manager/scripts/overnight_tune.py'
RUN_ID = os.environ.get('DMM_RUN_ID', 'manual')
if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', RUN_ID):
    raise ValueError('Invalid run ID')
STATE = Path.home() / '.local/state/dmm-overnight' / RUN_ID


def candidates():
    # Eight cold starts, not a Cartesian search. All values remain hypotheses.
    return [dict(profile=p, util=u, context=c, seqs=s) for p, u, c, s in [
        (QWEN, .49, 262144, 2), (QWEN, .46, 262144, 2), (QWEN, .43, 262144, 2),
        (QWEN, .49, 196608, 4), (FLASH, .76, 262144, 8),
        (FLASH, .74, 262144, 4), (FLASH, .72, 262144, 4), (FLASH, .76, 196608, 4)]]


def command(argv, timeout=30, check=True):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=check).stdout


def request(path, data=None, timeout=15):
    req = urllib.request.Request(API + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def memory():
    info = {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    psi = Path('/proc/pressure/memory').read_text()
    return {'total_gib': info['MemTotal'] / 1024**2,
            'unified_unavailable_percent': 100 * (1 - info['MemAvailable'] / info['MemTotal']),
            'available_gib': info['MemAvailable'] / 1024**2,
            'cached_gib': info['Cached'] / 1024**2,
            'swap_used_gib': (info['SwapTotal'] - info['SwapFree']) / 1024**2,
            'swap_out_pages': int(vm['pswpout']),
            'psi_full_avg10': float(re.search(r'full avg10=(\S+)', psi)[1])}


def container():
    # Docker inspect contains environment secrets; keep the full value in memory only.
    data = json.loads(command(['docker', 'inspect', 'vllm_node']))[0]
    return {'id': data['Id'], 'image': data['Image'],
            'profile': data['Config'].get('Labels', {}).get('dgx.profile'),
            'args': data['Args'], 'status': data['State']['Status']}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_manifests():
    text = (ROOT / 'profiles/vLLM' / f'start_{QWEN}.sh').read_text()
    qwen = Path(re.search(r'--model\s+(\S+)', text)[1].replace('/root/', str(Path.home()) + '/', 1))
    flashes = sorted((Path.home() / '.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots').glob('*-fp8hybrid'))
    if not flashes:
        raise RuntimeError('Flash hybrid snapshot missing')
    result = {}
    for profile, directory in ((QWEN, qwen), (FLASH, flashes[0])):
        index = directory / 'model.safetensors.index.json'
        files = set(json.loads(index.read_text())['weight_map'].values())
        if not files or any(not (directory / f).is_file() or (directory / f).stat().st_size == 0
                            or not os.access(directory / f, os.R_OK) for f in files):
            raise RuntimeError('Indexed weights missing or unreadable')
        result[profile] = {'index_sha256': digest(index), 'indexed_files': len(files)}
    return result


def pins():
    return {'commit': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip(),
            'worker_sha256': digest(Path(__file__)), 'weights': weight_manifests(),
            'profiles': {p: digest(ROOT / 'profiles/vLLM' / f'start_{p}.sh') for p in PROFILES},
            'serve_sha256': digest(Path.home() / 'qwen3.8-Flash-DGX/scripts/serve.sh'),
            'images': {p: command(['docker', 'image', 'inspect', '--format', '{{.Id}}', i]).strip()
                       for p, i in PROFILES.items()}}


def ensure_idle():
    with urllib.request.urlopen(DMM + '/api/vllm/status', timeout=10) as r:
        status = json.load(r)
    if status.get('launch', {}).get('in_flight'):
        raise RuntimeError('DMM has a pending launch')
    text = urllib.request.urlopen(API + '/metrics', timeout=10).read().decode()
    for name in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+(\S+)', text, re.M)
        if not values:
            raise RuntimeError('Required activity metrics missing')
        if sum(float(v) for v in values):
            raise RuntimeError('Inference consumers have not drained')


def baseline_settings(args):
    def flag(name):
        return args[args.index(name) + 1]
    result = dict(profile=FLASH, util=float(flag('--gpu-memory-utilization')),
                  context=int(flag('--max-model-len')), seqs=int(flag('--max-num-seqs')))
    # This qualification covers the observed native-context, MTP=2 baseline only.
    speculative = json.loads(flag('--speculative-config'))
    if (result != dict(profile=FLASH, util=.76, context=262144, seqs=8)
            or speculative.get('method') != 'mtp' or speculative.get('num_speculative_tokens') != 2
            or '--rope-scaling' in args):
        raise RuntimeError('Baseline settings changed; review restoration plan')
    return result


def launch(settings, log):
    env = os.environ.copy()
    env.update(VLLM_GPU_MEMORY_UTILIZATION=str(settings['util']),
               VLLM_MAX_MODEL_LEN=str(settings['context']), VLLM_MAX_NUM_SEQS=str(settings['seqs']))
    path = ROOT / 'profiles/vLLM' / ('start_' + settings['profile'] + '.sh')
    # Already in a dedicated systemd unit, outside DMM's service cgroup.
    with log.open('wb') as out:
        return subprocess.Popen(['bash', str(path)], env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)


def kv_summary(text):
    def last(pattern):
        hits = re.findall(pattern, text)
        return float(hits[-1].replace(',', '')) if hits else None
    return {'kv_tokens': last(r'GPU KV cache size:\s*([\d,]+) tokens'),
            'kv_gib': last(r'Available KV cache memory:\s*([\d.]+) GiB'),
            'model_gib': last(r'(?:Model loading took|model weights take)\s*([\d.]+) GiB')}


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = [memory()]
        self.fault = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        first_swap = None
        low = 0
        while not self.stop.is_set():
            sample = memory()
            self.samples.append(sample)
            first_swap = sample['swap_out_pages'] if first_swap is None else first_swap
            low = low + 1 if sample['available_gib'] < 8 else 0
            if low >= 3 or sample['psi_full_avg10'] > 5 or (sample['swap_out_pages'] - first_swap) * os.sysconf('SC_PAGE_SIZE') > 256 * 1024**2:
                self.fault = 'Memory headroom, pressure, or swap-out guard tripped'
            self.stop.wait(2)

    def check(self):
        if self.fault:
            raise RuntimeError(self.fault)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=3)
        return {'min_available_gib': min(s['available_gib'] for s in self.samples),
                'max_psi_full_avg10': max(s['psi_full_avg10'] for s in self.samples),
                'max_swap_used_gib': max(s['swap_used_gib'] for s in self.samples),
                'swap_out_pages_delta': self.samples[-1]['swap_out_pages'] - self.samples[0]['swap_out_pages']}


def wait_ready(proc, timeout, prior_id, expected_profile, monitor=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if monitor:
            monitor.check()
        if proc.poll() not in (None, 0):
            raise RuntimeError('Profile launcher failed; inspect private host log')
        try:
            current = container()
            models = request('/v1/models')['data']
            if (models and current['id'] != prior_id and current['status'] == 'running'
                    and current['profile'] == expected_profile):
                return models[0]['id']
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            pass
        time.sleep(5)
    raise RuntimeError('Startup deadline exceeded')


def synthetic_tokens(model, target, seed, decode=False):
    # Fresh prefix per request prevents identical-prefix cache speedups.
    filler = ('The ordinary archive item contains no secret value. ' * 200)
    prefix = f'Benchmark {seed}. The secret marker is AMBER731. Remember it.\n'
    suffix = ('\nBegin with AMBER731, then write at least 200 words explaining queues in plain language.'
              if decode else '\nReturn only the secret marker, without explanation.')
    count = max(1, target // 10)
    for _ in range(5):
        text = prefix + (filler * (count // 200 + 1))[:count * len(filler) // 200] + suffix
        tokens = request('/tokenize', {'model': model, 'messages': [{'role': 'user', 'content': text}],
                                      'add_generation_prompt': True,
                                      'chat_template_kwargs': {'enable_thinking': False}}, timeout=60)['tokens']
        if abs(len(tokens) - target) < target * .02:
            return tokens
        count = max(1, int(count * target / len(tokens)))
    raise RuntimeError('Could not calibrate synthetic token length')


def infer(model, tokens, deadline, monitor, decode=False):
    start = time.monotonic()
    payload = json.dumps({
        'model': model, 'prompt': tokens, 'max_tokens': 128, 'temperature': 0,
        'stream': True, 'add_special_tokens': False, 'stream_options': {'include_usage': True}}).encode()
    first, usage, output, finish = None, {}, '', None
    endpoint = urllib.parse.urlsplit(API)
    with closing(http.client.HTTPConnection(endpoint.hostname, endpoint.port,
                 timeout=max(.01, deadline - time.monotonic()))) as connection:
        connection.request('POST', '/v1/completions', body=payload,
                           headers={'Content-Type': 'application/json'})
        r = connection.getresponse()
        if r.status != 200:
            raise RuntimeError('Inference HTTP status ' + str(r.status))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Inference batch deadline exceeded')
            if connection.sock is None:
                raise RuntimeError('Streaming deadline socket unavailable')
            connection.sock.settimeout(remaining)
            monitor.check()
            line = r.readline().decode()
            if not line:
                break
            if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
                continue
            value = json.loads(line[6:])
            usage = value.get('usage') or usage
            for choice in value.get('choices', []):
                chunk = choice.get('text', '')
                if chunk:
                    first = first or time.monotonic()
                    output += chunk
                finish = choice.get('finish_reason') or finish
    duration = time.monotonic() - start
    return {'input_tokens': len(tokens), 'seconds': duration,
            'ttft_s': None if first is None else first - start,
            'output_tokens': usage.get('completion_tokens'), 'finish_reason': finish,
            'decode_tps': (usage.get('completion_tokens', 0) - 1) / max(.001, time.monotonic() - first) if first and usage.get('completion_tokens', 0) > 1 else None,
            'needle_pass': 'AMBER731' in output and (finish == 'stop' or
                            (decode and finish == 'length' and usage.get('completion_tokens') == 128))}


def workloads(context):
    return [[17000], [176000], *([[229000]] if context >= 237192 else []),
            [17000, 176000], [17000] * 4]


def completed_requests():
    text = urllib.request.urlopen(API + '/metrics', timeout=10).read().decode()
    values = re.findall(r'^vllm:request_success_total(?:\{[^\n]*\})?\s+(\S+)', text, re.M)
    if not values:
        raise RuntimeError('Required completed-request metric missing')
    return sum(float(v) for v in values)


def kernel_faults(since):
    # Inspect in memory; publish counts only. Non-readable journals are a coverage gap.
    r = subprocess.run(['journalctl', '-k', '--since', '@' + str(int(since)), '--no-pager'],
                       capture_output=True, text=True, timeout=15)
    if r.returncode or 'permission' in r.stderr.lower():
        return None
    if 'No entries' in r.stdout:
        access = subprocess.run(['journalctl', '-k', '-n', '1', '--no-pager'],
                                capture_output=True, text=True, timeout=15)
        if access.returncode or 'No entries' in access.stdout or 'permission' in access.stderr.lower():
            return None
        return 0
    return len(re.findall(r'NV_ERR_NO_MEMORY|Out of memory:|oom-kill:', r.stdout))


def qualify(settings, directory, cutoff):
    result = {'settings': settings, 'batches': [], 'status': 'failed'}
    monitor = Monitor()
    monitor.thread.start()
    proc = None
    started_at = time.time()
    try:
        ensure_idle()
        begin = time.monotonic()
        prior_id = container()['id']
        proc = launch(settings, directory / 'launch.log')
        model = wait_ready(proc, min(1200, cutoff - time.monotonic()), prior_id, settings['profile'], monitor)
        result['startup_s'] = time.monotonic() - begin
        result['ready_memory'] = memory()
        result['reservation_gib'] = settings['util'] * result['ready_memory']['total_gib']
        result['kernel_fault_count'] = kernel_faults(started_at)
        if result['kernel_fault_count']:
            raise RuntimeError('Kernel allocation/OOM errors observed')
        # Read logs into memory; publish only aggregate numbers, never raw logs.
        # Docker sends most vLLM logs to stderr too.
        logged = subprocess.run(['docker', 'logs', 'vllm_node'], capture_output=True, text=True, timeout=30)
        result.update(kv_summary(logged.stdout + logged.stderr))
        if not result['kv_tokens']:
            raise RuntimeError('Measured KV token capacity missing; refusing long requests')
        result['full_context_kv_capacity'] = result['kv_tokens'] / settings['context']
        result['tool_smoke'] = tool_smoke(model)
        for index, sizes in enumerate(workloads(settings['context']) + [[17000]]):
            if time.monotonic() + 600 > cutoff:
                result['batches'].append({'sizes': sizes, 'skipped': 'time reserve'})
                continue
            if len(sizes) > settings['seqs'] or sum(sizes) + len(sizes) * 8192 > result['kv_tokens'] * .9:
                result['batches'].append({'sizes': sizes, 'skipped': 'sequence or measured KV budget'})
                continue
            ensure_idle()
            decode = index == len(workloads(settings['context']))
            tokens = [synthetic_tokens(model, n, f'{directory.name}-{index}-{i}', decode) for i, n in enumerate(sizes)]
            if any(len(t) + 8192 > settings['context'] for t in tokens):
                raise RuntimeError('Calibrated prompt exceeds context safety budget')
            # Parallelize requests only after exact token counts satisfy the KV bound.
            before = completed_requests()
            deadline = min(cutoff, time.monotonic() + 570)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tokens)) as pool:
                futures = [pool.submit(infer, model, t, deadline, monitor, decode) for t in tokens]
                records = [f.result() for f in futures]
            result['batches'].append({'sizes': sizes, 'decode': decode, 'requests': records,
                                     'aggregate_output_tps': sum(r['output_tokens'] or 0 for r in records) / max(r['seconds'] for r in records)})
            result['kernel_fault_count'] = kernel_faults(started_at)
            if result['kernel_fault_count']:
                raise RuntimeError('Kernel allocation/OOM errors observed')
            if completed_requests() - before != len(tokens):
                result['contaminated'] = True
                raise RuntimeError('Uncontrolled traffic detected during batch')
            if not all(r['needle_pass'] for r in records):
                raise RuntimeError('Synthetic retrieval or bounded completion failed')
        result['kernel_fault_count'] = kernel_faults(started_at)
        if result['kernel_fault_count']:
            raise RuntimeError('Kernel allocation/OOM errors observed')
        result['status'] = 'measured' if any('requests' in b for b in result['batches']) else 'incomplete'
    except Exception as exc:
        # Avoid serializing exceptions that may contain request URLs/configuration.
        result['failure_type'] = type(exc).__name__
        result['failure_reason'] = str(exc) if type(exc) is RuntimeError else type(exc).__name__
        result['failure_stage'] = 'startup' if 'startup_s' not in result else 'inference'
        result['abort_sweep'] = bool(monitor.fault or result['failure_stage'] == 'startup'
                                     or result.get('contaminated') or result.get('kernel_fault_count'))
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        result['memory'] = monitor.finish()
        write(directory / 'result.json', result)
    return result


def tool_smoke(model):
    value = request('/v1/chat/completions', {'model': model, 'messages': [
        {'role': 'user', 'content': 'Use get_status to inspect the inference service. Do not answer in prose.'}],
        'tools': [{'type': 'function', 'function': {'name': 'get_status',
            'description': 'Get service status; this benchmark does not execute the call.',
            'parameters': {'type': 'object', 'properties': {'service': {'type': 'string'}},
                           'required': ['service'], 'additionalProperties': False}}}],
        'tool_choice': 'auto', 'max_tokens': 128, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False}}, timeout=90)
    calls = value['choices'][0]['message'].get('tool_calls') or []
    if not any(c['function']['name'] == 'get_status' and
               json.loads(c['function']['arguments']) == {'service': 'inference'} for c in calls):
        raise RuntimeError('Tool-call schema smoke failed')
    return {'passed': True, 'executed': False}


def smoke():
    models = request('/v1/models')['data']
    answer = request('/v1/chat/completions', {'model': models[0]['id'],
        'messages': [{'role': 'user', 'content': 'Reply exactly READY'}], 'max_tokens': 32,
        'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=90)
    text = answer['choices'][0]['message'].get('content') or ''
    if 'READY' not in text:
        raise RuntimeError('Restore smoke failed')


def restore():
    if not (STATE / 'snapshot.json').exists():
        return  # Pre-switch validation failed; nothing was replaced.
    with (STATE / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot = json.loads((STATE / 'snapshot.json').read_text())
        if snapshot.get('restored'):
            if container()['profile'] != FLASH:
                raise RuntimeError('Restored engine changed')
            smoke()
            return
        if pins() != snapshot['pins']:
            raise RuntimeError('Pinned source/image drift; manual restore required')
        try:
            current = container()
        except subprocess.CalledProcessError:
            current = None
        if current and current['profile'] not in PROFILES:
            raise RuntimeError('Unexpected engine; manual restore required')
        # A pre-switch failure should not replace a still-healthy original engine.
        healthy_original = False
        if current and current['id'] == snapshot['container']['id']:
            try:
                smoke()
                healthy_original = True
            except Exception:
                pass
        if not healthy_original:
            proc = launch(snapshot['baseline'], STATE / 'restore.log')
            wait_ready(proc, 1500, current['id'] if current else None, FLASH)
            smoke()
        snapshot['restored'] = True
        snapshot['restored_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(STATE / 'snapshot.json', snapshot)


def rehearse():
    """Tiny live rehearsal on the existing engine; never launches/replaces it."""
    precheck()
    original = container()
    model = request('/v1/models')['data'][0]['id']
    result = {'mode': 'rehearsal', 'engine_replaced': False, 'batches': [], 'passed': False}
    monitor = Monitor()
    monitor.thread.start()
    try:
        result['tool_smoke'] = tool_smoke(model)
        for parallel in (1, 2):
            ensure_idle()
            tokens = [synthetic_tokens(model, 512, f'{RUN_ID}-{parallel}-{i}')
                      for i in range(parallel)]
            before = completed_requests()
            deadline = time.monotonic() + 60
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
                records = list(pool.map(lambda t: infer(model, t, deadline, monitor), tokens))
            if completed_requests() - before != parallel:
                raise RuntimeError('Rehearsal contaminated by other requests')
            result['batches'].append({'parallel': parallel, 'requests': records})
            if not all(r['needle_pass'] for r in records):
                raise RuntimeError('Rehearsal inference failed')
        if container()['id'] != original['id']:
            raise RuntimeError('Engine identity changed during rehearsal')
        result['passed'] = True
    finally:
        result['memory'] = monitor.finish()
        write(STATE / 'rehearsal.json', result)


def worker():
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STATE / 'snapshot.json').exists():
            raise RuntimeError('Use a fresh reviewed state directory for each run')
        # This script must not be invoked without systemd's independent restore hook.
        if not os.environ.get('DMM_OVERNIGHT_SYSTEMD'):
            raise RuntimeError('Run through the prepared systemd unit')
        start = float(os.environ['DMM_WINDOW_START'])
        if not start <= time.time() <= start + 900:
            raise RuntimeError('Outside the reviewed worker start window')
        ensure_idle()
        precheck()
        original = container()
        if original['profile'] != FLASH or original['status'] != 'running':
            raise RuntimeError('Expected serving Flash Next baseline')
        snapshot = {'pins': pins(), 'container': {k: v for k, v in original.items() if k != 'args'}, 'baseline': baseline_settings(original['args']),
                    'restored': False}
        write(STATE / 'snapshot.json', snapshot)
        remaining = float(os.environ['DMM_WINDOW_END']) - time.time() - 2700
        cutoff = time.monotonic() + min(4 * 3600 + 15 * 60, remaining)
        results = []
        owned_id = original['id']
        for index, settings in enumerate(candidates()):
            if time.monotonic() + 1800 > cutoff:
                break
            if pins() != snapshot['pins']:
                raise RuntimeError('Pinned source/image drift')
            current = container()
            if current['profile'] not in PROFILES or current['id'] != owned_id:
                raise RuntimeError('Unexpected engine identity; refusing replacement')
            directory = STATE / f'candidate-{index}'
            directory.mkdir(mode=0o700)
            results.append(qualify(settings, directory, cutoff))
            owned_id = container()['id']
            write(STATE / 'report.json', {'results': results, 'recommendation':
                'Review measured tradeoffs. No persistent defaults or fleet concurrency changed.',
                'provisional_selection': select_results(results),
                'limitations': ['Synthetic retrieval is not agent task-quality validation.',
                               'Uncontrolled application traffic can bias measurements.',
                               '196608 context does not cover the largest observed fleet requests.']})
            # A pressure failure ends the sweep rather than testing another risky launch.
            if results[-1].get('abort_sweep'):
                break
    # ExecStopPost runs restoration after normal exit, failure, timeout, or cancellation.


def precheck():
    if (STATE / 'snapshot.json').exists():
        raise RuntimeError('Previous snapshot exists; archive it after review')
    if command(['git', '-C', str(ROOT), 'diff', 'HEAD', '--', 'scripts', 'profiles/vLLM']).strip():
        raise RuntimeError('Tracked maintenance source is dirty')
    expected = os.environ.get('DMM_EXPECTED_COMMIT')
    if expected and pins()['commit'] != expected:
        raise RuntimeError('Deployed commit differs from reviewed schedule')
    ensure_idle()
    current = container()
    if current['profile'] != FLASH or current['status'] != 'running':
        raise RuntimeError('Flash baseline not serving')
    baseline_settings(current['args'])
    if current['image'] != pins()['images'][FLASH]:
        raise RuntimeError('Baseline image differs from restoration image tag')
    smoke()


def select_results(results):
    selected = {}
    for profile in PROFILES:
        eligible = [r for r in results if r['settings']['profile'] == profile
                    and r['settings']['context'] == 262144 and r['status'] == 'measured'
                    and r.get('kernel_fault_count') == 0 and not r.get('contaminated')
                    and all(any(b.get('sizes') == [size] and b.get('requests')
                                and all(q['needle_pass'] for q in b['requests'])
                                for b in r['batches']) for size in (17000, 176000, 229000))]
        if eligible:
            baseline = next((r for r in eligible if r['settings']['util'] ==
                             (.76 if profile == FLASH else .49)), None)
            if baseline is None:
                continue
            latency = lambda r: sum(q['seconds'] for b in r['batches']
                                   if len(b['sizes']) == 1 and not b.get('decode')
                                   for q in b.get('requests', []))
            acceptable = [r for r in eligible if latency(r) <= latency(baseline) * 1.10
                          and r['memory']['min_available_gib'] >= 10]
            if acceptable:
                chosen = min(acceptable, key=lambda r: r['settings']['util'])
                selected[profile] = {'settings': chosen['settings'],
                    'reason': 'Lowest tested reservation covering observed context, with >=10 GiB headroom and <=10% single-request elapsed regression.',
                    'max_tested_parallelism': max(len(b.get('requests', [])) for b in chosen['batches']),
                    'client_parallelism': 'Keep current admission pending a representative agent canary.',
                    'review_required': True}
    return selected


def freeze_after_drain(was_active, deadline, control_db=Path("/home/josh/agent-control/control.db")):
    """Hold a non-writing DB transaction across the final idle check and freeze.

    Dispatch records/claims work before executing it. The reserved writer lock
    prevents new admissions between this check and freezing the service cgroup.
    Rollback releases the lock without changing any controller data.
    """
    uri = control_db.resolve().as_uri() + '?mode=rw'
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(uri, uri=True, timeout=1) as db:
                db.execute('BEGIN IMMEDIATE')
                busy = (db.execute('SELECT count(*) FROM runs WHERE ended_at IS NULL').fetchone()[0] +
                        db.execute("SELECT count(*) FROM jobs WHERE state IN ('claimed','running')").fetchone()[0])
                if not busy:
                    if was_active:
                        command(['systemctl', '--user', 'freeze', 'agent-control.service'], timeout=15)
                    db.rollback()
                    return
                db.rollback()
        except sqlite3.OperationalError as exc:
            if 'locked' not in str(exc):
                raise
        time.sleep(5)
    raise RuntimeError('Controller did not drain within ten minutes')


def coordinate():
    start = float(os.environ['DMM_WINDOW_START'])
    if not start <= time.time() <= start + 60:
        raise RuntimeError('Outside the reviewed one-shot start window')
    expected = os.environ.get('DMM_EXPECTED_COMMIT')
    if expected != command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip():
        raise RuntimeError('Controller source differs from reviewed schedule')
    loaded = command(['ssh', 'gb10', 'systemctl', '--user', 'show', 'dmm-overnight-worker.service', '-p', 'LoadState', '--value']).strip()
    if loaded != 'loaded':
        raise RuntimeError('Remote recovery unit not installed')
    was_active = command(['systemctl', '--user', 'is-active', 'agent-control.service'], check=False).strip() == 'active'
    if was_active:
        watchdog = command(['systemctl', '--user', 'show', 'agent-control.service', '-p', 'WatchdogUSec', '--value']).strip()
        freezer = command(['systemctl', '--user', 'show', 'agent-control.service', '-p', 'FreezerState', '--value']).strip()
        if watchdog != '0' or freezer != 'running':
            raise RuntimeError('Controller watchdog/freezer state requires review')
    state = STATE / 'controller.json'
    if state.exists():
        raise RuntimeError('Controller state already exists; review previous run')
    write(state, {'was_active': was_active, 'paused': False})
    freeze_after_drain(was_active, time.monotonic() + 600)
    write(state, {'was_active': was_active, 'paused': was_active})
    command(['ssh', 'gb10', 'env', 'DMM_RUN_ID=' + RUN_ID, 'DMM_EXPECTED_COMMIT=' + expected, '/usr/bin/python3', REMOTE_SCRIPT, 'precheck'], timeout=150)
    command(['ssh', 'gb10', 'systemctl', '--user', 'start', 'dmm-overnight-worker.service'])
    until = time.monotonic() + 5.5 * 3600
    while time.monotonic() < until:
        status = command(['ssh', 'gb10', 'systemctl', '--user', 'is-active', 'dmm-overnight-worker.service'], check=False).strip()
        if status in ('inactive', 'failed'):
            return
        time.sleep(15)
    raise RuntimeError('Remote service deadline exceeded; check independent restore')


def resume():
    path = STATE / 'controller.json'
    if not path.exists():
        return
    prior = json.loads(path.read_text())
    # Remote unit executes restoration independently of this process/SSH connection.
    command(['ssh', 'gb10', 'systemctl', '--user', 'stop', 'dmm-overnight-worker.service'], timeout=1800)
    verified = command(['ssh', 'gb10', 'env', 'DMM_RUN_ID=' + RUN_ID, '/usr/bin/python3', REMOTE_SCRIPT, 'verify'], timeout=120)
    if verified.strip() != 'RESTORED':
        raise RuntimeError('Serving unverified; controller remains paused')
    if prior['was_active']:
        command(['systemctl', '--user', 'thaw', 'agent-control.service'], timeout=30)
        command(['systemctl', '--user', 'start', 'agent-control.service'], timeout=60)
    write(path, {**prior, 'resumed': True})


def schedule(date, output):
    day = dt.date.fromisoformat(date)
    when = dt.datetime.combine(day, dt.time(2), ZoneInfo('America/Chicago'))
    if when <= dt.datetime.now(dt.timezone.utc):
        raise ValueError('Choose a future 02:00 America/Chicago date')
    output.mkdir(parents=True, exist_ok=True)
    python = '/usr/bin/python3'
    # Remote deployed source location differs from the controlplane repository path.
    remote = '/home/josh/DGX-Model-Manager/scripts/overnight_tune.py'
    local = str(ROOT / 'scripts/overnight_tune.py')
    commit = command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip()
    end = (when + dt.timedelta(hours=5.5)).timestamp()
    run_id = when.astimezone(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    worker_unit = f'''[Unit]
Description=Bounded GB10 overnight comparison; restore on every exit
[Service]
Type=exec
Environment=DMM_OVERNIGHT_SYSTEMD=1
Environment=DMM_RUN_ID={run_id}
Environment=DMM_EXPECTED_COMMIT={commit}
Environment=DMM_WINDOW_END={end}
Environment=DMM_WINDOW_START={when.timestamp()}
UMask=0077
ExecStart={python} {remote} worker
ExecStopPost={python} {remote} restore
RuntimeMaxSec=16200
TimeoutStopSec=1800
KillMode=control-group
Restart=no
'''
    controller_unit = f'''[Unit]
Description=Drain fleet and coordinate bounded GB10 comparison
[Service]
Type=exec
Environment=DMM_WINDOW_START={when.timestamp()}
Environment=DMM_EXPECTED_COMMIT={commit}
Environment=DMM_RUN_ID={run_id}
UMask=0077
ExecStart={python} {local} coordinate
ExecStopPost={python} {local} resume
RuntimeMaxSec=19800
TimeoutStopSec=2100
Restart=no
'''
    timer = f'''[Unit]
Description=One-shot 02:00 Chicago comparison on {date}
[Timer]
OnCalendar={date} 02:00:00 America/Chicago
AccuracySec=1s
Persistent=false
Unit=dmm-overnight.service
[Install]
WantedBy=timers.target
'''
    for name, value in [('dmm-overnight-worker.service', worker_unit),
                        ('dmm-overnight.service', controller_unit), ('dmm-overnight.timer', timer)]:
        (output / name).write_text(value)
    print(json.dumps({'start': when.isoformat(), 'end': (when + dt.timedelta(hours=5.5)).isoformat(),
                      'directory': str(output), 'armed': False, 'candidates': candidates()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'schedule', 'worker', 'restore', 'coordinate', 'resume', 'verify', 'precheck', 'rehearse'])
    parser.add_argument('--date')
    parser.add_argument('--output', type=Path, default=ROOT / '.planning/overnight-units')
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps(candidates(), indent=2))
    elif args.mode == 'schedule':
        schedule(args.date, args.output)
    elif args.mode == 'verify':
        if not (STATE / 'snapshot.json').exists():
            current = container()
            if current['profile'] != FLASH or current['status'] != 'running':
                raise RuntimeError('Original Flash baseline not serving')
            baseline_settings(current['args'])
            smoke()
        else:
            snapshot = json.loads((STATE / 'snapshot.json').read_text())
            if not snapshot.get('restored') or container()['profile'] != FLASH:
                raise RuntimeError('Restoration not confirmed')
            baseline_settings(container()['args'])
            smoke()
        print('RESTORED')
    else:
        try:
            globals()[args.mode]()
        except Exception as exc:
            reason = str(exc) if type(exc) is RuntimeError else type(exc).__name__
            print(f'{args.mode} failed: {reason}; inspect private state on the owning host', file=sys.stderr)
            sys.exit(1)
