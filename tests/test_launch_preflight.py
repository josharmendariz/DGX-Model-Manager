"""Tests for launch preflight and load-progress parsing.

Regression cover for the 2026-08-04 outage, which had three independent causes
that all presented identically as "clicking Start did nothing":

  1. A profile generated before `vllm serve` became an image property, so the
     container exec'd `--model` and died in ~2s with exit 2.
  2. `--restart unless-stopped` on that broken container, which resurrected the
     crash loop across a reboot and made run-recipe.sh skip its own launch,
     taking the box's default model down with it.
  3. Page cache being charged against --gpu-memory-utilization, which left
     0.19 GiB of KV cache at a utilization that had been measured as correct.

Every check below is on a pure function, so none of this needs docker.
"""

import pytest

import app as appmod


BROKEN_SCRIPT = """#!/bin/bash
# Name: HF Qwen/Qwen3.6-35B-A3B-FP8
set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

exec docker run --name vllm_node --restart unless-stopped --gpus all -p 8000:8000 \\
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
  -e HF_HUB_OFFLINE=1 \\
  eugr/spark-vllm:latest \\
  --model "/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/abc" \\
  --gpu-memory-utilization 0.75 \\
  --max-model-len 32768
"""

FIXED_SCRIPT = BROKEN_SCRIPT.replace(
    "  eugr/spark-vllm:latest \\\n",
    "  eugr/spark-vllm:latest \\\n  vllm serve \\\n",
).replace("--restart unless-stopped ", "")

RECIPE_SCRIPT = """#!/bin/bash
# Name: HF Qwen/Qwen3.6-35B-A3B-FP8
set -euo pipefail
docker rm -f vllm_node 2>/dev/null || true
cd "$HOME/spark-vllm-docker"
exec ./run-recipe.sh qwen3.6-35b-a3b-fp8-solo -d
"""


def _levels(checks, name):
    return [c["level"] for c in checks if c["check"] == name]


# ── Script parsing ────────────────────────────────────────────────────────────

def test_parses_docker_run_script():
    facts = appmod._parse_launch_script(BROKEN_SCRIPT)
    assert facts["image"] == "eugr/spark-vllm:latest"
    assert facts["has_serve"] is False
    assert facts["restart"] == "unless-stopped"
    assert facts["util"] == 0.75
    assert facts["recipe_backed"] is False
    assert facts["clears_container"] is True
    assert facts["model"].endswith("/snapshots/abc")


def test_parses_recipe_backed_script():
    facts = appmod._parse_launch_script(RECIPE_SCRIPT)
    assert facts["recipe_backed"] is True
    assert facts["recipe"] == "qwen3.6-35b-a3b-fp8-solo"
    # A recipe-backed script declares no image or flags of its own; the recipe does.
    assert facts["util"] is None


# ── The entrypoint contract (cause #1) ────────────────────────────────────────

def test_missing_vllm_serve_is_a_blocking_failure():
    checks = appmod._preflight_static(BROKEN_SCRIPT, {})
    assert _levels(checks, "entrypoint") == ["fail"]
    fail = next(c for c in checks if c["check"] == "entrypoint")
    assert "exec `--model`" in fail["detail"]
    assert "vllm serve" in fail["fix"]


def test_fixed_script_passes_the_entrypoint_check():
    checks = appmod._preflight_static(FIXED_SCRIPT, {})
    assert _levels(checks, "entrypoint") == ["ok"]


def test_vllm_serve_against_a_self_serving_image_is_a_failure():
    script = FIXED_SCRIPT.replace("eugr/spark-vllm:latest", "vllm/vllm-openai:v0.20.0")
    checks = appmod._preflight_static(script, {})
    assert _levels(checks, "entrypoint") == ["fail"]


def test_recipe_backed_script_skips_the_entrypoint_check():
    checks = appmod._preflight_static(RECIPE_SCRIPT, {})
    assert _levels(checks, "entrypoint") == []
    assert _levels(checks, "shape") == ["ok"]


# ── The restart policy (cause #2) ─────────────────────────────────────────────

def test_restart_policy_is_flagged():
    checks = appmod._preflight_static(BROKEN_SCRIPT, {})
    assert _levels(checks, "restart_policy") == ["warn"]
    warn = next(c for c in checks if c["check"] == "restart_policy")
    assert "run-recipe.sh" in warn["detail"]


def test_generated_scripts_no_longer_emit_a_restart_policy():
    checks = appmod._preflight_static(FIXED_SCRIPT, {})
    assert _levels(checks, "restart_policy") == []


def test_missing_container_clear_is_flagged():
    script = FIXED_SCRIPT.replace("docker rm -f vllm_node 2>/dev/null || true\n", "")
    assert _levels(appmod._preflight_static(script, {}), "collision") == ["warn"]


# ── The memory budget (cause #3) ──────────────────────────────────────────────

def test_budget_charges_page_cache_against_utilization():
    # The exact shape of the failed launch: 121 GiB total, only 90 GiB free
    # because ~25 GiB sits in page cache.
    mem = {"total_gib": 121.0, "free_gib": 90.0, "reclaimable_gib": 25.0,
           "cuda_unavailable_gib": 31.0}
    b = appmod._vllm_budget_gib(0.55, mem)
    assert b["budget_gib"] == pytest.approx(66.55, abs=0.1)
    # Barely more than the 34.5 GiB the weights alone need — hence 0.19 GiB of KV.
    assert b["usable_gib"] == pytest.approx(35.55, abs=0.1)
    # What dropping the cache buys, at an identical utilization.
    assert b["usable_after_reclaim_gib"] == pytest.approx(60.55, abs=0.1)


def test_heavy_page_cache_fails_preflight(monkeypatch):
    monkeypatch.setattr(appmod, "_cuda_visible_memory", lambda: {
        "total_gib": 121.0, "free_gib": 90.0, "available_gib": 113.0,
        "reclaimable_gib": 25.0, "cuda_unavailable_gib": 31.0})
    checks = appmod._preflight_memory(0.55)
    assert checks[0]["level"] == "fail"
    assert "drop_caches" in checks[0]["fix"]
    assert "35.6 GiB" in checks[0]["detail"]


def test_moderate_page_cache_only_warns(monkeypatch):
    monkeypatch.setattr(appmod, "_cuda_visible_memory", lambda: {
        "total_gib": 121.0, "free_gib": 108.0, "available_gib": 113.0,
        "reclaimable_gib": 10.0, "cuda_unavailable_gib": 13.0})
    assert appmod._preflight_memory(0.55)[0]["level"] == "warn"


def test_cold_cache_passes(monkeypatch):
    monkeypatch.setattr(appmod, "_cuda_visible_memory", lambda: {
        "total_gib": 121.0, "free_gib": 118.0, "available_gib": 119.0,
        "reclaimable_gib": 2.0, "cuda_unavailable_gib": 3.0})
    assert appmod._preflight_memory(0.55)[0]["level"] == "ok"


def test_cuda_visible_memory_reports_free_not_available():
    mem = appmod._cuda_visible_memory()
    assert "error" not in mem
    # The whole point: these differ by the page cache, and CUDA sees the smaller.
    assert mem["free_gib"] <= mem["available_gib"]
    assert mem["cuda_unavailable_gib"] >= 0


# ── Load-progress parsing ─────────────────────────────────────────────────────

def test_log_lines_map_to_phases():
    cases = {
        "INFO [core.py:114] Initializing a V1 LLM engine (v0.23.1)": "engine_init",
        "INFO [default_loader.py:430] Loading weights took 14.32 seconds": "loading_weights",
        # Emitted when the weights are DONE — it must not advance past compiling.
        "INFO [gpu_model_runner.py:5279] Model loading took 34.5 GiB": "loading_weights",
        "INFO [backends.py:530] Using cache directory for torch.compile": "compiling",
        "INFO [gpu_model_runner.py:6507] Profiling CUDA graph memory: PIECEWISE=51": "profiling",
        "INFO [gpu_model_runner.py:6612] Estimated CUDA graph memory: 5.55 GiB": "kv_cache",
        "INFO [gpu_worker.py:569] Available KV cache memory: 25.97 GiB": "kv_cache",
        "INFO [backends.py:1148] Dynamo bytecode transform time: 0.84 s": "compiling",
        "INFO [monitor.py:53] torch.compile took 9.63 s in total": "compiling",
        "Capturing CUDA graphs (decode, FULL):   3%": "capturing",
        "INFO [gpu_model_runner.py:6680] Graph capturing finished in 9 secs": "capturing",
        "INFO:     Application startup complete.": "ready",
    }
    for line, expected in cases.items():
        assert appmod._classify_vllm_log_line(line) == expected, line


def test_unremarkable_lines_do_not_change_phase():
    assert appmod._classify_vllm_log_line("INFO: some unrelated chatter") is None


def test_phases_follow_the_measured_log_order():
    # From a real 178s cold start: torch.compile at ~63s, memory profiling at
    # ~118s, the KV verdict at ~122s, graph capture at ~143s.
    order = [p for p, _, _ in appmod._LOAD_PHASES]
    assert order.index("compiling") < order.index("profiling")
    assert order.index("profiling") < order.index("kv_cache")
    assert order.index("kv_cache") < order.index("capturing")


def test_phase_percentages_increase_monotonically():
    pcts = [pct for _, pct, _ in appmod._LOAD_PHASES]
    assert pcts == sorted(pcts)
    assert pcts[-1] == 100


def test_failure_reason_prefers_the_concrete_exception():
    # The real log: the ValueError appears twice, then an unhelpful RuntimeError
    # wrapper, all inside ~90 lines of traceback.
    tail = [
        "(EngineCore pid=217) ERROR [core.py:1231] Traceback (most recent call last):",
        "(EngineCore pid=217) ERROR [core.py:1231]     raise ValueError(",
        "(EngineCore pid=217) ERROR [core.py:1231] ValueError: To serve at least one "
        "request with the model's max seq len (262144), (2.64 GiB KV cache is needed, "
        "which is larger than the available KV cache memory (0.19 GiB).",
        "(APIServer pid=82) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}",
    ]
    reason = appmod._load_failure_reason(tail)
    assert reason.startswith("ValueError: To serve at least one request")
    assert "See root cause above" not in reason


def test_failure_reason_falls_back_to_the_wrapper_when_alone():
    tail = ["(APIServer pid=82) RuntimeError: Engine core initialization failed. "
            "See root cause above."]
    assert "Engine core initialization failed" in appmod._load_failure_reason(tail)


def test_failure_reason_never_returns_empty():
    assert appmod._load_failure_reason([])
    assert appmod._load_failure_reason(["", "   "])


def test_failure_reason_does_not_pass_off_ordinary_output_as_an_error():
    # A container that was replaced mid-stream has no exception in its log, only
    # request lines. Quoting one as "the error" is worse than saying nothing.
    tail = ['(APIServer pid=85) INFO: 127.0.0.1:50678 - "GET /v1/models HTTP/1.1" 200 OK']
    reason = appmod._load_failure_reason(tail)
    assert reason.startswith("No exception was logged")


def test_entrypoint_death_is_detected_as_a_failure():
    # What the broken script actually produced, before vLLM ever started.
    line = "/opt/nvidia/nvidia_entrypoint.sh: line 67: exec: --: invalid option"
    assert appmod._LOAD_FAILED_RE.search(line)


def test_kv_cache_exhaustion_is_detected_as_a_failure():
    assert appmod._LOAD_FAILED_RE.search("RuntimeError: Engine core initialization failed")


# ── Parsing the code, not the documentation ───────────────────────────────────
# Load scripts carry long rationale headers, and those headers quote the exact
# commands preflight looks for. Parsing raw text made the Qwen3.6 profile report
# its recipe as "(FP8," — lifted out of its own `# Description:` line — which
# then silently voided the memory-budget and dry-run checks.

DOCUMENTED_RECIPE_SCRIPT = '''#!/bin/bash
# Name: HF Qwen/Qwen3.6-35B-A3B-FP8
# Description: Tuned solo recipe via run-recipe.sh (FP8, util 0.55, 256K ctx)
#
# The `docker rm -f` below is load-bearing: run-recipe.sh reports "already
# running" for any existing vllm_node, including a crash-looping one.
set -euo pipefail

RECIPE="qwen3.6-35b-a3b-fp8-solo"

docker rm -f vllm_node 2>/dev/null || true

cd "$HOME/spark-vllm-docker"
exec ./run-recipe.sh "$RECIPE" -d
'''


def test_comment_prose_is_not_parsed_as_code():
    facts = appmod._parse_launch_script(DOCUMENTED_RECIPE_SCRIPT)
    assert facts["recipe"] == "qwen3.6-35b-a3b-fp8-solo"


def test_recipe_name_resolves_through_a_shell_variable():
    facts = appmod._parse_launch_script(DOCUMENTED_RECIPE_SCRIPT)
    # The use site reads `"$RECIPE"`; the value is assigned ten lines above it.
    assert "$" not in facts["recipe"]


def test_commented_out_command_does_not_count_as_present():
    script = FIXED_SCRIPT.replace(
        "docker rm -f vllm_node 2>/dev/null || true",
        "# docker rm -f vllm_node 2>/dev/null || true")
    assert appmod._parse_launch_script(script)["clears_container"] is False


def test_expand_script_vars_leaves_unknown_variables_alone():
    assert appmod._expand_script_vars("$NOPE/x", "A=1") == "$NOPE/x"
    assert appmod._expand_script_vars("${A}/x", 'A="val"') == "val/x"


def test_home_expands_in_mount_and_model_paths(monkeypatch):
    # An unexpanded $HOME reaches docker verbatim and is rejected as "invalid
    # characters for a local volume name", which reads as a broken model.
    monkeypatch.setenv("HOME", "/home/tester")
    facts = appmod._parse_launch_script(FIXED_SCRIPT)
    assert facts["mounts"] == ["/home/tester/.cache/huggingface:/root/.cache/huggingface"]
    assert "$" not in facts["model"]


def test_env_expansion_is_allowlisted(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    # Not on the allowlist, so it is never echoed into an HTTP response.
    assert appmod._expand_script_vars("$SECRET_TOKEN", "") == "$SECRET_TOKEN"


# ── Recipe smoke (subprocess branch) ──────────────────────────────────────────
# The recipe-backed smoke path spawns run-recipe.sh --dry-run. It deliberately
# uses a fake runner in a tmp checkout: a regression here (e.g. a renamed local
# reused in subprocess cwd) crashes only when the runner *exists* on the
# resolved dir — the case no pure-function test could ever hit, and the case
# production is always in.

def test_recipe_smoke_dry_runs_the_resolved_runner(tmp_path, monkeypatch):
    root = tmp_path / "spark-vllm-docker"
    (root / "recipes").mkdir(parents=True)
    runner = root / "run-recipe.sh"
    runner.write_text("#!/bin/bash\necho DRYRUN_OK $1\n")
    runner.chmod(0o755)
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "recipe_dir": str(root / "recipes")})
    import asyncio
    checks = asyncio.run(appmod._preflight_smoke(
        appmod._parse_launch_script(
            RECIPE_WRAPPER.replace("__XDIRE__", str(root / "recipes"))), "ignored"))
    assert checks and checks[0]["check"] == "smoke"
    assert checks[0]["level"] == "ok"
    assert "DRYRUN_OK" in checks[0]["detail"]
    # executed in the checkout root with the recipe as first arg
    assert "cwd" not in checks[0]  # no leak of internal state


RECIPE_WRAPPER = """#!/bin/bash
# Name: HF Qwen/Qwen3.6-35B-A3B-FP8
set -euo pipefail
docker rm -f vllm_node 2>/dev/null || true
RECIPE_DIR=__XDIRE__
RECIPE="qwen3.6-35b-a3b-fp8-solo"
cd "$RECIPE_DIR/.."
exec ./run-recipe.sh "$RECIPE" -d
"""


def test_recipe_smoke_missing_runner_is_skip_not_error(tmp_path, monkeypatch):
    (tmp_path / "recipes").mkdir()
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "recipe_dir": str(tmp_path / "recipes")})
    import asyncio
    script = RECIPE_WRAPPER.replace("__XDIRE__", str(tmp_path / "recipes"))
    checks = asyncio.run(appmod._preflight_smoke(
        appmod._parse_launch_script(script), "ignored"))
    assert checks[0]["check"] == "smoke"
    assert checks[0]["level"] == "skip"


