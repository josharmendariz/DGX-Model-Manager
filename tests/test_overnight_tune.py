"""No GPU, SSH, systemd changes, or real controller DB access in these tests."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune = load('overnight_tune')
fleet = load('analyze_fleet_workload')


def test_analysis_deduplicates_sessions_filters_old_and_vision(tmp_path):
    ctl, oc = tmp_path / 'ctl.db', tmp_path / 'oc.db'
    with sqlite3.connect(ctl) as db:
        db.execute('CREATE TABLE runs(agent,session_id,status,started_at)')
        db.executemany('INSERT INTO runs VALUES (?,?,?,?)', [
            ('sre', 's1', 'ok', 200000), ('sre', 's1', 'ok', 200001), ('hosted', 'missing', 'ok', 200000)])
    with sqlite3.connect(oc) as db:
        db.execute('CREATE TABLE message(session_id,time_created,data)')
        for model, timestamp in [('coding', 200000000), ('vision', 200000000), ('coding', 1)]:
            db.execute('INSERT INTO message VALUES (?,?,?)', ('s1', timestamp, json.dumps({
                'role': 'assistant', 'modelID': model, 'tokens': {'input': 100, 'output': 20,
                'cache': {'read': 200, 'write': 10}}, 'time': {'created': timestamp, 'completed': timestamp + 10},
                'content': 'PRIVATE CONTENT MUST NOT BE EXPORTED'})))
    report = fleet.analyze(ctl, oc, days=1, now=210000)
    assert report['agents']['sre']['requests'] == 1
    assert report['agents']['sre']['max_input_tokens'] == 310
    assert report['sessions'] == 2 and report['covered_sessions'] == 1
    assert report['peak_recorded_overlap'] == 1
    assert 'PRIVATE' not in json.dumps(report)


def test_baseline_refuses_unreviewed_runtime():
    args = ['--gpu-memory-utilization', '.76', '--max-model-len', '262144', '--max-num-seqs', '8', '--speculative-config', '{"method":"mtp","num_speculative_tokens":2}']
    assert tune.baseline_settings(args)['seqs'] == 8
    args[args.index('--max-num-seqs') + 1] = '4'
    with pytest.raises(RuntimeError, match='Baseline settings changed'):
        tune.baseline_settings(args)


def test_wait_ready_does_not_accept_old_serving_engine(monkeypatch):
    states = iter([{'id': 'old', 'status': 'running', 'profile': tune.FLASH},
                   {'id': 'new', 'status': 'running', 'profile': tune.QWEN}])
    monkeypatch.setattr(tune, 'container', lambda: next(states))
    monkeypatch.setattr(tune, 'request', lambda _: {'data': [{'id': 'new-model'}]})
    monkeypatch.setattr(tune.time, 'sleep', lambda _: None)
    class Process:
        def poll(self):
            return 0
    assert tune.wait_ready(Process(), 10, 'old', tune.QWEN) == 'new-model'


def fixture_snapshot(tmp_path, monkeypatch, restored=False):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    snapshot = {'pins': {'a': 1}, 'container': {'id': 'original'},
                'baseline': {'profile': tune.FLASH}, 'restored': restored}
    tune.write(tmp_path / 'snapshot.json', snapshot)
    monkeypatch.setattr(tune, 'pins', lambda: {'a': 1})
    return snapshot


def test_restore_relaunches_after_failed_candidate(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'candidate', 'profile': tune.QWEN})
    calls = []
    monkeypatch.setattr(tune, 'launch', lambda settings, log: calls.append(settings) or 'process')
    monkeypatch.setattr(tune, 'wait_ready', lambda *args: calls.append(args))
    monkeypatch.setattr(tune, 'smoke', lambda: calls.append('smoke'))
    tune.restore()
    assert calls[0]['profile'] == tune.FLASH
    assert calls[1] == ('process', 1500, 'candidate', tune.FLASH)
    assert json.loads((tmp_path / 'snapshot.json').read_text())['restored']


def test_restore_does_not_mark_success_if_smoke_fails(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'candidate', 'profile': tune.QWEN})
    monkeypatch.setattr(tune, 'launch', lambda *a: None)
    monkeypatch.setattr(tune, 'wait_ready', lambda *a: None)
    monkeypatch.setattr(tune, 'smoke', lambda: (_ for _ in ()).throw(RuntimeError('failed')))
    with pytest.raises(RuntimeError):
        tune.restore()
    assert not json.loads((tmp_path / 'snapshot.json').read_text())['restored']


def test_restore_refuses_unexpected_container_without_launch(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'operator', 'profile': 'other-profile'})
    monkeypatch.setattr(tune, 'launch', lambda *a: pytest.fail('must not replace another engine'))
    with pytest.raises(RuntimeError, match='Unexpected engine'):
        tune.restore()


def test_healthy_original_is_preserved_on_prelaunch_failure(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'original', 'profile': tune.FLASH})
    monkeypatch.setattr(tune, 'launch', lambda *a: pytest.fail('healthy original must remain'))
    monkeypatch.setattr(tune, 'smoke', lambda: None)
    tune.restore()


def test_controller_not_resumed_without_remote_health(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    tune.write(tmp_path / 'controller.json', {'was_active': True})
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        return 'NOT VERIFIED' if argv[-1] == 'verify' else ''
    monkeypatch.setattr(tune, 'command', command)
    with pytest.raises(RuntimeError, match='controller remains paused'):
        tune.resume()
    assert not any(argv[:3] == ['systemctl', '--user', 'start'] for argv in calls)


def test_schedule_local_zone_nonpersistent_restore_on_all_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'command', lambda *a: 'abc123')
    tune.schedule('2027-10-19', tmp_path)
    timer = (tmp_path / 'dmm-overnight.timer').read_text()
    worker = (tmp_path / 'dmm-overnight-worker.service').read_text()
    assert '02:00:00 America/Chicago' in timer and 'Persistent=false' in timer
    assert 'ExecStopPost=/usr/bin/python3 /home/josh/DGX-Model-Manager/scripts/overnight_tune.py restore' in worker
    assert 'RuntimeMaxSec=16200' in worker and 'TimeoutStopSec=1800' in worker
    assert 'DMM_EXPECTED_COMMIT=abc123' in worker


def test_past_schedule_rejected(tmp_path):
    with pytest.raises(ValueError, match='future'):
        tune.schedule('2020-01-01', tmp_path)


def test_selection_rejects_small_context_and_slow_or_unqualified_candidates():
    def result(util, context=262144, seconds=10, kernel=0):
        return {'settings': {'profile': tune.QWEN, 'util': util, 'context': context, 'seqs': 2},
                'status': 'measured', 'kernel_fault_count': kernel,
                'memory': {'min_available_gib': 20}, 'batches': [
                    {'sizes': [n], 'requests': [{'needle_pass': True, 'seconds': seconds}]}
                    for n in (17000, 176000, 229000)]}
    results = [result(.49), result(.46), result(.42, context=196608), result(.44, seconds=20), result(.43, kernel=None)]
    assert tune.select_results(results)[tune.QWEN]['settings']['util'] == .46


def test_kv_parser_comma_tokens():
    assert tune.kv_summary('Available KV cache memory: 25.06 GiB\nGPU KV cache size: 405,625 tokens')['kv_tokens'] == 405625


@pytest.mark.parametrize('name,value', [('VLLM_MAX_MODEL_LEN','0'), ('VLLM_MAX_MODEL_LEN','262145'),
    ('VLLM_MAX_NUM_SEQS','9'), ('VLLM_MAX_NUM_SEQS','-1'), ('VLLM_MAX_NUM_SEQS','08'),
    ('VLLM_GPU_MEMORY_UTILIZATION','1.0'), ('VLLM_GPU_MEMORY_UTILIZATION','0.0')])
def test_flash_invalid_overrides_exit_before_docker(name, value):
    import os
    script = Path(__file__).parents[1] / 'profiles/vLLM' / f'start_{tune.FLASH}.sh'
    result = subprocess.run(['bash', str(script)], env={**os.environ, name: value}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Invalid' in result.stderr or 'Overrides exceed' in result.stderr


def test_streaming_latency_usage_and_retrieval(monkeypatch):
    import io
    chunks = [
        {'choices': [{'text': 'AMBER', 'finish_reason': None}]},
        {'choices': [{'text': '731', 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'completion_tokens': 3}},
    ]
    response = io.BytesIO((''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n').encode())
    response.status = 200
    class Socket:
        def settimeout(self, seconds):
            assert seconds > 0
    class Connection:
        sock = Socket()
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return response
    monkeypatch.setattr(tune.http.client, 'HTTPConnection', Connection)
    class Monitor:
        def check(self):
            pass
    result = tune.infer('test', [1,2,3], tune.time.monotonic() + 5, Monitor())
    assert result['needle_pass'] and result['output_tokens'] == 3
    assert result['ttft_s'] is not None and result['decode_tps'] > 0


def test_no_selection_without_measured_baseline():
    r = {'settings': {'profile': tune.QWEN, 'util': .46, 'context': 262144, 'seqs': 2},
         'status': 'measured', 'kernel_fault_count': 0, 'memory': {'min_available_gib': 20},
         'batches': [{'sizes': [n], 'requests': [{'needle_pass': True, 'seconds': 1}]}
                     for n in (17000,176000,229000)]}
    assert tune.select_results([r]) == {}


def test_drain_holds_admission_reservation_across_freeze_and_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / 'ctl.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE runs(ended_at)')
        db.execute('CREATE TABLE jobs(state)')
    calls = []
    def command(argv, **kw):
        calls.append(argv)
        with sqlite3.connect(path, timeout=0) as competing:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                competing.execute("INSERT INTO jobs VALUES ('claimed')")
        return ''
    monkeypatch.setattr(tune, 'command', command)
    tune.freeze_after_drain(True, tune.time.monotonic() + 5, path)
    assert calls == [['systemctl', '--user', 'freeze', 'agent-control.service']]
    with sqlite3.connect(path, timeout=0) as db:
        db.execute("INSERT INTO jobs VALUES ('queued')")
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1


def test_active_run_prevents_freeze(tmp_path, monkeypatch):
    path = tmp_path / 'ctl.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE runs(ended_at)')
        db.execute('INSERT INTO runs VALUES (NULL)')
        db.execute('CREATE TABLE jobs(state)')
    monkeypatch.setattr(tune, 'command', lambda *a, **kw: pytest.fail('active work must not be frozen'))
    clock = iter([0, 0, 10])
    monkeypatch.setattr(tune.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(tune.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='did not drain'):
        tune.freeze_after_drain(True, 5, path)


def test_tool_smoke_validates_arguments_without_executing(monkeypatch):
    monkeypatch.setattr(tune, 'request', lambda *a, **kw: {'choices': [{'message': {
        'tool_calls': [{'function': {'name': 'get_status', 'arguments': '{"service":"inference"}'}}]}}]})
    assert tune.tool_smoke('test') == {'passed': True, 'executed': False}
    monkeypatch.setattr(tune, 'request', lambda *a, **kw: {'choices': [{'message': {'content': 'no call'}}]})
    with pytest.raises(RuntimeError, match='Tool-call schema'):
        tune.tool_smoke('test')
