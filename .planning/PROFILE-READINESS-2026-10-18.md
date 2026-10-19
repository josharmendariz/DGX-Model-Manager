# Profile readiness audit — 2026-10-18 CDT

Observed live on GB10 at approximately 22:03 CDT, before any engine switch.
DMM remains at c005e60, active; its built-in jobs and launch queue are idle.
Completed authentication rollout documentation is published in PR #29 at a2ea8b2;
GitHub lint and pytest checks pass. Local validation: 692 passed, 4 skipped.
No merge or deployment performed.

## Recommended measurement candidate

`start_hf_qwen3-vl-4b-fp8`: both indexed safetensors shards exist, totaling
5.61 GiB; config.json is readable through the exact Docker bind mount. Image
`vllm/vllm-openai:v0.20.0` is already present (ID prefix bf05c8a0d56f).
Full preflight verdict is `warn`, with no failing checks. Argument smoke is
`skip` because the probe only extracts flags following `vllm serve`, while this
image starts its API server through its entrypoint. GPU/kernel compatibility and
successful inference remain unverified until an approved engine-switch window.

The existing utilization default is 0.75, context 32768, max sequences 2, FP8 KV.
The handoff's 0.20 suggestion is only an initial measurement hypothesis. Do not
write new defaults or claim KV capacity from disk size or admission estimates.
Of the five originally proposed tuning targets, this is the only one with files
verified present at the configured model location.

## Other candidates and failures

| Profile | Observed readiness |
| --- | --- |
| Qwen3-8B / Qwen3-14B | Exact HF repository directories referenced by scripts do not exist. Dynamic paths cause preflight to skip the mount probe. |
| Qwen2.5-14B GPTQ | `/mnt/models/qwen2.5-14b-instruct-gptq-int8` is absent; genuine mount_source failure. |
| DeepSeek distill 14B / 32B | Configured `/opt/models/models--deepseek-ai--...` directories exist but are empty; genuine mount_readable failures. Historical mount_source diagnosis has changed. |
| Nemotron Nano / Super | Cached snapshots have all indexed weight shards (18.01 / 74.80 GiB for inventory-selected snapshots). Preflight incorrectly passes Hub ID plus `/config.json` to `cat`. Reading resolved cache paths through the same HF mount succeeds. False preflight failure; does not establish inference compatibility. |
| Qwen3-Next-80B | `/mnt/models/qwen3-next-80b-a3b-nvfp4` absent; genuine mount_readable failure. |
| Qwen3-Coder-Next | 10/10 indexed shards, 42.68 GiB; mount probe succeeds. |
| GPT-OSS-120B | 15/15 indexed shards, 60.77 GiB; mount probe succeeds. |
| Qwen3.8-27B FP8 | Exact configured snapshot has 66/66 indexed shards, 28.75 GiB; mount probe succeeds. |
| Qwen3.8-27B BF16 | 18/18 indexed shards, 51.75 GiB; mount probe succeeds. |
| Unsloth Qwen3.8-27B NVFP4 | 2/2 indexed shards, 21.81 GiB; dynamic mount path skipped by preflight. |
| Qwen3.6-35B recipe | Inventory-selected snapshot has three missing indexed shards; recipe-specific target still needs verification before declaring recipe readiness. |

Inventory attribution is not authoritative: `_script_model_dir` selected the
Unsloth NVFP4 directory for the Qwen FP8 MTP script. Resolve exact script/recipe
paths before using inventory results as launch evidence.

## Preflight defects identified

- `_preflight_static` uses `m.split(':', 1)[-1]` for mount targets, retaining
  `:ro` and falsely warning that correctly mounted absolute model paths are outside
  the mount. VL-4B and Coder-Next actual container probes succeed.
- Hub IDs are treated as literal filesystem paths by both mount-scope and runtime
  mount-readability checks (Nemotron Nano/Super).
- `_parse_launch_script` does not extract environment-default utilization expressions
  for some diagnostics; the separate footprint parser correctly reports the VL-4B
  0.75 default and 91.2 GiB footprint. The full preflight budget field is null.
- `_preflight_smoke` skips entrypoint-style profiles with no `vllm serve` marker.
- Flash wrapper gets misleading collision and restart warnings despite delegating
  container removal and explicitly disabling restarts after launch.

No preflight code or production settings changed during this audit.

## Consumers and restoration baseline

Flash Next is serving; direct `/v1/models` succeeds, native context is 262144.
Container ID: `4c4f68ed11cd13a3e07c4a5ccded9f1f0dabb633db3820e8ae266a0ea3b0aa18`.
Started: `2026-10-16T03:18:53.41314325Z`.
Image ID: `sha256:826800d5af37e81df3af1dc77982376d657ab70ab2fe1fd2e0bf2378ae367076`.
Pinned profile: `start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid`.
Tracked wrapper defaults: GPU utilization 0.76, native context 262144, MTP 2.

One running inference request, zero waiting, KV usage 20.58%, and a loopback
connection were observed. The original client behind the connection is not known.
The autonomous fleet controller is active; this alone does not identify the caller.
No active Kubernetes jobs were listed; Spark scraper runs hourly, harvest every
four hours, GitHub job daily. RAG monitor service inactive, next timer October 24
at 22:17 CDT. Recheck consumers immediately before any approved switch.

## Initial decision boundary (superseded by execution update below)

Josh requested choosing the profile after the audit and confirming timing before
replacing Flash Next. Await that choice and timing; do not infer permission from
an idle snapshot. No engine stopped, restarted, switched, or cache reclaimed.

Proposed bounded window: at most 30 minutes including Flash restoration; candidate
startup deadline 10 minutes, abort on OOM or stalled progress, record load/KV-cache
logs, host/CUDA memory, startup duration, and a deterministic short text inference
(up to 64 output tokens, 60-second request deadline). Confirm usable context
concurrency from measured KV capacity. Start with utilization 0.20 only as a
hypothesis, adjust from measurements, restore Flash on success or failure, and
verify serving and a bounded inference response from the consumer network.
A vision-specific smoke would be separate from this short text smoke.

After measurements: update only the selected profile and its evidence, run required
tests, publish a tuning PR. Any merge or persistent deployment requires approval
naming that concrete change. Do not change unrelated preflight logic in the tuning
PR without extending scope explicitly. Move tracked code through Git/GitHub.

## Approved execution update

Josh selected `start_hf_qwen_qwen3.8-27b-fp8` after the audit and explicitly
approved proceeding immediately, including outage while a request was active.
The earlier VL-4B recommendation is retained as the initial audit result, not the
selected execution target. The chosen profile's existing utilization is 0.49,
context 262144, and max sequences 2. Measurement started at that existing baseline;
VL-4B's 0.20 hypothesis does not apply. Evidence and final settings will be recorded
in `docs/profile-measurements-2026-10-18.md`.

Execution completed: Qwen3.8-27B-FP8 became ready in 454.9 seconds at the
existing 0.49 / 262144 / 2 settings. Reported KV capacity was 25.06 GiB /
405625 tokens (1.55 full-context requests). Local and two concurrent
controlplane short text smokes passed. Retain those verified launch defaults;
qualify the profile metadata conservatively from measured capacity and remove
the stale eight-sequence note. The former 1190948 metadata figure came from the
calculator's aggressive utilization cap, not the measured 0.49 setting.

Flash Next was restored with its original image, pin, utilization 0.76, context
262144, and MTP 2. Consumer-network model-list and bounded inference passed at
22:31 CDT; the smoke took 1.114 seconds. The cold restoration exceeded the
harness's 720-second observation limit, so follow-up health checks continued
on the same initializing container; no extra restart was issued. Total window
was about 22 minutes 42 seconds. Both production profile files remain unchanged.
No persistent rollout or merge was performed. See the measurement document and
its JSON evidence for limits and exact identities.
