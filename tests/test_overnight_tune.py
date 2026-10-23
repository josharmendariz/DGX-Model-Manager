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
    assert 'RuntimeMaxSec=15300' in worker and 'TimeoutStopSec=1800' in worker
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
        def close(self):
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


@pytest.mark.parametrize('valid', [True, False])
def test_rehearsal_never_launches_engine(tmp_path, monkeypatch, valid):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    monkeypatch.setattr(tune, 'precheck', lambda **kw: None)
    monkeypatch.setattr(tune, 'ensure_idle', lambda: None)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'original'})
    monkeypatch.setattr(tune, 'launch', lambda *a: pytest.fail('rehearsal must not replace engine'))
    monkeypatch.setattr(tune, 'request', lambda *a: {'data': [{'id': 'test'}]})
    monkeypatch.setattr(tune, 'tool_smoke', lambda *a: {'passed': True})
    monkeypatch.setattr(tune, 'synthetic_tokens', lambda *a: [1, 2])
    counts = iter([0, 1, 1, 3])
    monkeypatch.setattr(tune, 'completed_requests', lambda: next(counts))
    monkeypatch.setattr(tune, 'infer', lambda *a: {'needle_pass': valid})
    if valid:
        tune.rehearse()
    else:
        with pytest.raises(RuntimeError, match='Rehearsal inference failed'):
            tune.rehearse()
    report = json.loads((tmp_path / 'rehearsal.json').read_text())
    assert report['passed'] is valid
    assert not report['engine_replaced']


def test_infer_uses_real_http_connection_and_stream_parser(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            events = [{'choices': [{'text': 'AMBER731', 'finish_reason': 'stop'}]},
                      {'choices': [], 'usage': {'completion_tokens': 3}}]
            body = (''.join('data: ' + json.dumps(v) + '\n\n' for v in events) + 'data: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(tune, 'API', 'http://127.0.0.1:' + str(server.server_port))
    class Monitor:
        def check(self):
            pass
    try:
        result = tune.infer('test', [1], tune.time.monotonic() + 5, Monitor())
        assert result['needle_pass'] and result['output_tokens'] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_empty_recent_kernel_journal_is_clean_when_journal_readable(monkeypatch):
    replies = iter([subprocess.CompletedProcess([], 0, '-- No entries --', ''),
                    subprocess.CompletedProcess([], 0, 'Older kernel message', '')])
    monkeypatch.setattr(tune.subprocess, 'run', lambda *a, **kw: next(replies))
    assert tune.kernel_faults(123) == 0


def test_engine_drain_waits_until_idle_without_forcing(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    states = iter([{'running': 3, 'waiting': 1}, {'running': 1, 'waiting': 0}, {'running': 0, 'waiting': 0}])
    clock = [0]
    monkeypatch.setattr(tune, 'activity', lambda: next(states))
    monkeypatch.setattr(tune.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(tune.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    result = tune.wait_for_idle(900, allow_busy=True)
    assert result['idle'] and not result['forced_replacement'] and result['elapsed_s'] == 10


@pytest.mark.parametrize('allow_busy', [True, False])
def test_busy_deadline_is_bounded_and_requires_explicit_policy(tmp_path, monkeypatch, allow_busy):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    monkeypatch.setattr(tune, 'activity', lambda: {'running': 3, 'waiting': 0})
    clock = [0]
    monkeypatch.setattr(tune.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(tune.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    if allow_busy:
        result = tune.wait_for_idle(900, allow_busy=True, interval=300)
        assert result['forced_replacement'] and not result['idle']
    else:
        with pytest.raises(RuntimeError, match='Drain deadline: running=3'):
            tune.wait_for_idle(900, allow_busy=False, interval=300)
    result = json.loads((tmp_path/'admission.json').read_text())
    assert clock[0] == 900 and result['elapsed_s'] == 900
    assert result['forced_replacement'] is allow_busy


def test_missing_monitoring_never_authorizes_busy_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    monkeypatch.setattr(tune, 'activity', lambda: (_ for _ in ()).throw(RuntimeError('Required activity metrics missing')))
    with pytest.raises(RuntimeError, match='metrics missing'):
        tune.wait_for_idle(900, allow_busy=True)
    assert not (tmp_path/'admission.json').exists()


def test_remote_failure_preserves_safe_reason_and_does_not_copy_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'STATE', tmp_path)
    output='DMM_EVENT '+json.dumps({'event':'failure','mode':'admit','reason':'DMM has a pending launch'})+'\n'
    monkeypatch.setattr(tune.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([],1,output,'PRIVATE TOKEN DO NOT COPY'))
    with pytest.raises(RuntimeError, match='DMM has a pending launch'):
        tune.command(['ssh','test'])
    evidence=(tmp_path/'command-failure.json').read_text()
    assert 'pending launch' in evidence and 'PRIVATE TOKEN' not in evidence


@pytest.mark.parametrize('endpoint,binding', [('NORMAL_API','8000'),('BENCH_API','127.0.0.1:18000')])
def test_launch_isolates_benchmark_port_and_restores_normal_binding(tmp_path, monkeypatch, endpoint, binding):
    monkeypatch.setattr(tune, 'API', getattr(tune,endpoint))
    calls=[]
    monkeypatch.setattr(tune.subprocess,'Popen',lambda *a,**kw:calls.append(kw))
    tune.launch({'profile':tune.FLASH,'util':.76,'context':262144,'seqs':8},tmp_path/'launch.log')
    assert calls[0]['env']['VLLM_PUBLISH_BINDING']==binding


@pytest.mark.parametrize('profile', [tune.FLASH,tune.QWEN])
def test_profiles_reject_unreviewed_network_binding_before_docker(profile):
    import os
    script=Path(__file__).parents[1]/'profiles/vLLM'/f'start_{profile}.sh'
    result=subprocess.run(['bash',str(script)],env={**os.environ,'VLLM_PUBLISH_BINDING':'0.0.0.0:18000'},capture_output=True,text=True)
    assert result.returncode==2 and 'Invalid VLLM_PUBLISH_BINDING' in result.stderr


def test_first_candidate_uses_existing_admission_not_pre_snapshot_check(tmp_path, monkeypatch):
    monkeypatch.setattr(tune,'API',tune.BENCH_API)
    (tmp_path/'snapshot.json').write_text('{}')
    monkeypatch.setattr(tune,'precheck',lambda **kw:pytest.fail('snapshot already created; do not repeat precheck'))
    monkeypatch.setattr(tune,'check_manager_idle',lambda:None)
    monkeypatch.setattr(tune,'container',lambda:{'id':'original'})
    class Process:
        def poll(self):return 0
    calls=[]
    monkeypatch.setattr(tune,'launch',lambda *a:calls.append(tune.API) or Process())
    monkeypatch.setattr(tune,'wait_ready',lambda *a:'model')
    monkeypatch.setattr(tune,'kernel_faults',lambda *a:0)
    monkeypatch.setattr(tune,'tool_smoke',lambda *a:{'passed':True})
    monkeypatch.setattr(tune.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess([],0,'GPU KV cache size: 405,625 tokens',''))
    result=tune.qualify({'profile':tune.QWEN,'util':.49,'context':262144,'seqs':2},tmp_path,tune.time.monotonic()+100,first_replacement=True)
    assert 'failure_reason' not in result
    assert calls==[tune.BENCH_API]


def test_restore_forces_normal_serving_api(tmp_path, monkeypatch):
    fixture_snapshot(tmp_path,monkeypatch)
    monkeypatch.setattr(tune,'API',tune.BENCH_API)
    monkeypatch.setattr(tune,'container',lambda:{'id':'candidate','profile':tune.QWEN})
    calls=[]
    monkeypatch.setattr(tune,'launch',lambda *a:calls.append(tune.API))
    monkeypatch.setattr(tune,'wait_ready',lambda *a:None)
    monkeypatch.setattr(tune,'smoke',lambda:None)
    tune.restore()
    assert calls==[tune.NORMAL_API]


@pytest.mark.parametrize('binding',['8000','127.0.0.1:18000'])
@pytest.mark.parametrize('profile',[tune.FLASH,tune.QWEN])
def test_real_profile_shell_passes_exact_docker_binding(tmp_path,profile,binding):
    import os
    home=tmp_path/'home';home.mkdir()
    bin_path=tmp_path/'bin';bin_path.mkdir()
    output=tmp_path/'argv.jsonl'
    fake=bin_path/'docker'
    fake.write_text('#!/usr/bin/python3\nimport json,os,sys\nwith open(os.environ["TEST_DOCKER_ARGS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    if profile==tune.FLASH:
        serve=home/'qwen3.8-Flash-DGX/scripts/serve.sh';serve.parent.mkdir(parents=True)
        serve.write_text('#!/bin/bash\nexec docker run -p "${PORT}:8000" example-image\n');serve.chmod(0o755)
        snap=home/'.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fake-fp8hybrid';snap.mkdir(parents=True)
    script=Path(__file__).parents[1]/'profiles/vLLM'/f'start_{profile}.sh'
    result=subprocess.run(['bash',str(script)],env={**os.environ,'HOME':str(home),'PATH':str(bin_path)+':'+os.environ['PATH'],
        'TEST_DOCKER_ARGS':str(output),'SKIP_DROP_CACHES':'1','VLLM_PUBLISH_BINDING':binding},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    invocations=[json.loads(line) for line in output.read_text().splitlines()]
    run=next(args for args in invocations if args[0]=='run')
    assert run[run.index('-p')+1]==binding+':8000'


@pytest.mark.parametrize('allow',[True,False])
def test_schedule_requires_explicit_busy_replacement_policy(tmp_path,monkeypatch,allow):
    monkeypatch.setattr(tune,'command',lambda *a:'abc123')
    tune.schedule('2027-10-20',tmp_path,allow_busy=allow)
    for name in ('dmm-overnight.service','dmm-overnight-worker.service'):
        assert f'DMM_ALLOW_BUSY_REPLACEMENT={int(allow)}' in (tmp_path/name).read_text()


def test_restore_handles_missing_container_with_sanitized_command_failure(tmp_path,monkeypatch):
    fixture_snapshot(tmp_path,monkeypatch)
    monkeypatch.setattr(tune,'container',lambda:(_ for _ in ()).throw(tune.CommandFailure('docker exited 1')))
    calls=[]
    monkeypatch.setattr(tune,'launch',lambda settings,log:calls.append(settings['profile']))
    monkeypatch.setattr(tune,'wait_ready',lambda *args:None)
    monkeypatch.setattr(tune,'smoke',lambda:None)
    tune.restore()
    assert calls==[tune.FLASH]
    assert json.loads((tmp_path/'snapshot.json').read_text())['restored']


def test_wait_ready_tolerates_container_replacement_gap(monkeypatch):
    states=iter([None,{'id':'new','status':'running','profile':tune.QWEN}])
    def current():
        value=next(states)
        if value is None:raise tune.CommandFailure('docker exited 1')
        return value
    monkeypatch.setattr(tune,'container',current)
    monkeypatch.setattr(tune,'request',lambda _: {'data':[{'id':'new-model'}]})
    monkeypatch.setattr(tune.time,'sleep',lambda _:None)
    class Process:
        def poll(self):return 0
    assert tune.wait_ready(Process(),10,'original',tune.QWEN)=='new-model'


@pytest.mark.parametrize('stdout,code,expected', [('57.91\n', 0, 57.91), ('[N/A]\n', 0, None), ('', 9, None)])
def test_gpu_power_reads_draw_and_fails_soft(monkeypatch, stdout, code, expected):
    monkeypatch.setattr(tune.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], code, stdout, ''))
    assert tune.gpu_power() == expected


def test_monitor_peak_power_only_counts_window():
    monitor = tune.Monitor()
    monitor.power = [(1.0, 80.0), (5.0, 15.0), (6.0, None), (7.0, 17.0)]
    assert monitor.peak_power(5.0) == 17.0
    assert monitor.peak_power(8.0) is None


@pytest.mark.parametrize('peak,capped', [(57.9, False), (16.5, True), (None, True)])
def test_capped_or_unreadable_gpu_power_aborts_sweep(tmp_path, monkeypatch, peak, capped):
    class Monitor:
        fault = None
        class thread:
            start = staticmethod(lambda: None)
        def check(self): pass
        def peak_power(self, since): return peak
        def finish(self): return {'min_available_gib': 20}
    done = [0]
    monkeypatch.setattr(tune, 'Monitor', Monitor)
    monkeypatch.setattr(tune, 'check_manager_idle', lambda: None)
    monkeypatch.setattr(tune, 'ensure_idle', lambda: None)
    monkeypatch.setattr(tune, 'container', lambda: {'id': 'original'})
    monkeypatch.setattr(tune, 'launch', lambda *a: type('P', (), {'poll': lambda self: 0})())
    monkeypatch.setattr(tune, 'wait_ready', lambda *a: 'model')
    monkeypatch.setattr(tune, 'memory', lambda: {'total_gib': 121.6})
    monkeypatch.setattr(tune, 'kernel_faults', lambda *a: 0)
    monkeypatch.setattr(tune, 'tool_smoke', lambda *a: {'passed': True})
    monkeypatch.setattr(tune, 'workloads', lambda context: [])
    monkeypatch.setattr(tune, 'synthetic_tokens', lambda model, n, seed, decode: [1] * n)
    monkeypatch.setattr(tune, 'completed_requests', lambda: done[0])
    def infer(*a):
        done[0] += 1
        return {'output_tokens': 128, 'seconds': 10.0, 'needle_pass': True}
    monkeypatch.setattr(tune, 'infer', infer)
    monkeypatch.setattr(tune.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], 0, 'GPU KV cache size: 405,625 tokens', ''))
    result = tune.qualify({'profile': tune.FLASH, 'util': .76, 'context': 262144, 'seqs': 8},
                          tmp_path, tune.time.monotonic() + 7200, first_replacement=True)
    assert result['batches'][-1]['gpu_peak_w'] == peak
    assert result['status'] == ('failed' if capped else 'measured')
    assert bool(result.get('abort_sweep')) is capped
    assert bool(result.get('power_capped')) is capped
