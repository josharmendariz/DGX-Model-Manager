"""Replacement estimates and parser-only device defaults do not grant admission."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

import app as appmod

MEM = {'total_gib': 121.6, 'free_gib': 1.5, 'cuda_unavailable_gib': 120.1,
       'reclaimable_gib': 14.7, 'available_gib': 16.2}


@pytest.mark.parametrize('label', ['', 'unknown', 'flash'])
def test_replacement_estimate_uses_only_identified_reservation(tmp_path, monkeypatch, label):
    monkeypatch.setattr(appmod, '_mem_total_gib', lambda: MEM['total_gib'])
    script = tmp_path / 'start_flash.sh'
    script.write_text('--gpu-memory-utilization 0.76\n')
    monkeypatch.setattr(appmod, '_scan_profiles', lambda key: [
        {'id': 'start_flash', 'script': str(script), 'vram_gb': 92}])

    async def status(*args, **kwargs):
        return {'running': True, 'instances': [{'name': 'vllm_node', 'running': True}]}

    async def docker_label(*args):
        return label

    monkeypatch.setattr(appmod, '_engine_status', status)
    monkeypatch.setattr(appmod, '_docker_label', docker_label)
    budget = asyncio.run(appmod._preflight_budget(0.35, MEM))
    assert budget['usable_gib'] == pytest.approx(-77.5, abs=0.1)
    if label != 'flash':
        assert budget['replacement'] is None
    else:
        after = budget['replacement']
        assert after['profile'] == 'start_flash'
        assert after['reservation_credit_gib'] == 92.4
        assert after['usable_gib'] == pytest.approx(14.9, abs=0.1)
        assert after['usable_after_reclaim_gib'] == pytest.approx(29.6, abs=0.1)
        detail = appmod._preflight_memory(0.35, MEM, budget)[0]['detail']
        assert 'including resident engines' in detail
        assert 'not measured released memory' in detail
        assert 'do not measure KV capacity' in detail


def test_replacement_and_cache_credit_cannot_exceed_reservation(monkeypatch):
    async def credit(*args):
        return 92.4, 'start_flash'

    monkeypatch.setattr(appmod, '_running_profile_vram_credit', credit)
    monkeypatch.setattr(appmod, '_scan_profiles', lambda key: [])
    budget = asyncio.run(appmod._preflight_budget(0.35, {
        **MEM, 'cuda_unavailable_gib': 10.0, 'reclaimable_gib': 15.0}))
    assert budget['replacement']['reservation_credit_gib'] == 10.0
    assert budget['replacement']['usable_gib'] == budget['budget_gib']
    assert budget['replacement']['usable_after_reclaim_gib'] == budget['budget_gib']


def test_failed_credit_lookup_leaves_current_state_and_no_estimate(monkeypatch):
    async def credit(*args):
        raise RuntimeError('unavailable')

    monkeypatch.setattr(appmod, '_running_profile_vram_credit', credit)
    monkeypatch.setattr(appmod, '_scan_profiles', lambda key: [])
    budget = asyncio.run(appmod._preflight_budget(0.35, MEM))
    assert budget['replacement'] is None
    assert budget['usable_gib'] == pytest.approx(-77.5, abs=0.1)


def test_memory_and_budget_share_one_preflight_snapshot(tmp_path, monkeypatch):
    script = tmp_path / 'start_test.sh'
    script.write_text('vllm serve --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.35}\n')
    snapshots = []

    def memory():
        snapshots.append(True)
        return dict(MEM)

    async def empty(*args):
        return []

    async def budget(util, mem):
        return {**appmod._vllm_budget_gib(util, mem), 'replacement': None}

    monkeypatch.setattr(appmod, '_cuda_visible_memory', memory)
    monkeypatch.setattr(appmod, '_preflight_budget', budget)
    monkeypatch.setattr(appmod, '_preflight_footprint', lambda profile: [])
    for name in ('_preflight_reclaim_label', '_preflight_runtime', '_preflight_smoke'):
        monkeypatch.setattr(appmod, name, empty)
    result = asyncio.run(appmod._preflight_profile({'id': 'start_test', 'script': str(script)}))
    assert len(snapshots) == 1
    assert result['memory'] == MEM
    assert result['budget']['util'] == 0.35


@pytest.fixture
def fake_vllm(tmp_path):
    root = tmp_path / 'vllm'
    for folder in (root, root / 'utils', root / 'entrypoints', root / 'entrypoints/openai'):
        folder.mkdir(exist_ok=True)
        (folder / '__init__.py').write_text('')
    (root / 'config.py').write_text('''from dataclasses import dataclass
class DeviceConfig:
    def __init__(self, device="auto"):
        if device != "cpu":
            raise RuntimeError("GPU detection must not run")
class FieldInfo:
    default_factory = DeviceConfig
@dataclass
class VllmConfig:
    device_config: object = FieldInfo()
''')
    (root / 'utils/argparse_utils.py').write_text('''import argparse, dataclasses
from vllm.config import VllmConfig, DeviceConfig
class FlexibleArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        factory = dataclasses.fields(VllmConfig)[0].default.default_factory
        assert factory is DeviceConfig, "factory must be restored before argument parsing"
        return super().parse_args(*args, **kwargs)
''')
    (root / 'entrypoints/openai/cli_args.py').write_text('''import dataclasses
from vllm.config import VllmConfig
def make_arg_parser(parser):
    default = dataclasses.fields(VllmConfig)[0].default
    default.default_factory()
    parser.add_argument("--model")
    parser.add_argument("--gpu-memory-utilization", type=float)
    return parser
''')
    return tmp_path


@pytest.mark.parametrize('flags,code,marker', [
    (['--model', 'org/model', '--gpu-memory-utilization', '0.35'], 0, True),
    (['--not-a-real-vllm-flag'], 2, False),
    (['--gpu-memory-utilization', 'not-a-number'], 2, False),
])
def test_actual_probe_builds_cpu_default_and_still_rejects_bad_cli(fake_vllm, flags, code, marker):
    result = subprocess.run([sys.executable, '-c', appmod._PF_ARGPARSE_PROBE, *flags],
                            env={**os.environ, 'PYTHONPATH': str(fake_vllm)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == code, result.stderr
    assert ('ARGS_OK_SYNTAX_ONLY' in result.stdout.splitlines()) is marker
    assert 'GPU detection must not run' not in result.stderr


def test_unknown_device_factory_cannot_report_success(fake_vllm):
    config = fake_vllm / 'vllm/config.py'
    config.write_text(config.read_text().replace('default_factory = DeviceConfig',
                                                'default_factory = lambda: DeviceConfig()'))
    result = subprocess.run([sys.executable, '-c', appmod._PF_ARGPARSE_PROBE],
                            env={**os.environ, 'PYTHONPATH': str(fake_vllm)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout.strip() == 'ARGS_PROBE_UNSUPPORTED_DEVICE_DEFAULT'
    assert 'ARGS_OK_SYNTAX_ONLY' not in result.stdout


@pytest.mark.skipif(not shutil.which('node'), reason='node not installed')
@pytest.mark.parametrize('replacement', [None, {
    'profile': '<unsafe-profile>', 'usable_gib': 14.9, 'usable_after_reclaim_gib': 29.6,
    'reservation_credit_gib': 92.4}])
def test_preflight_ui_renders_current_and_estimated_balances_as_text(replacement):
    source = (Path(appmod.__file__).parent / 'dashboard.html').read_text()
    function = re.search(r'function renderPreflight\(out, r\) \{.*?\n\}', source, re.S).group(0)
    program = '''class Element {
      constructor() { this.children = []; this.textContent = ''; }
      appendChild(c) { this.children.push(c); }
      set innerHTML(v) { throw new Error('HTML sink used'); }
    }
    const document = {createElement: () => new Element()};
    ''' + function + '''
    const out = new Element();
    renderPreflight(out, JSON.parse(process.argv[1]));
    const text = el => el.textContent + el.children.map(text).join(' ');
    console.log(text(out));
    '''
    data = {'profile': 'test', 'verdict': 'warn', 'checks': [], 'budget': {
        'util': 0.35, 'budget_gib': 42.6, 'usable_gib': -77.5, 'replacement': replacement}}
    result = subprocess.run(['node', '-e', program, json.dumps(data)],
                            capture_output=True, text=True, check=True, timeout=10)
    assert 'Current state, before replacing any engine: -77.5' in result.stdout
    assert 'KV capacity and startup readiness are not measured' in result.stdout
    assert ('Estimated after replacing' in result.stdout) is bool(replacement)
