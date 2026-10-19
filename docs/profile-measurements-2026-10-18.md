# Qwen3.8-27B-FP8 measurement — 2026-10-18 CDT

Josh selected `start_hf_qwen_qwen3.8-27b-fp8` after the readiness audit and
approved an immediate temporary outage, including interruption of active work.
The audit's suggested VL-4B utilization of 0.20 does not apply to this profile.

## Measured configuration and result

The existing profile was measured without changing its launch arguments:
utilization **0.49**, context **262144**, max sequences **2**, default KV dtype.
No lower utilization was tested. Retain these measured defaults; this change
corrects the profile's capacity metadata and stale eight-sequence guidance.

| Observation | Result |
| --- | --- |
| Host pool | 121.627 GiB unified memory |
| Model files | 66/66 indexed shards, 28.75 GiB on disk |
| Loaded model memory | 28.51 GiB |
| Model loading duration | 208.85 seconds |
| Compiler duration | 53.71 seconds |
| Available KV-cache memory | 25.06 GiB |
| Allocated KV capacity | 405625 tokens total |
| Reported concurrency at 262144 tokens/request | 1.55x |
| Launch to successful `/v1/models` | 454.9 seconds |
| Host memory available at readiness | 52.572 GiB |
| Host memory available after initial smoke | 52.831 GiB |
| Engine GPU process memory after network smoke | 55912 MiB (54.60 GiB) |
| Recent memory pressure at readiness | some/full avg10, avg60, avg300 all 0.00 |

At launch, cache reclaim left 116.018 GiB MemAvailable and 116.169 GiB MemFree.
At readiness, MemFree was 21.871 GiB with 31.488 GiB cached. On unified memory,
MemAvailable and MemFree answer different questions; neither is the model's
measured allocation. GPU-wide total/used/free are reported as N/A by nvidia-smi;
its process memory is recorded separately above. Swap was already used before
this window, and SwapFree increased from 10.907 to 13.272 GiB. Nonzero swap alone
is not evidence of active pressure.

The generator computes `max_fitting_context` at its aggressive utilization cap,
not at the recommended utilization. Its old 1190948-token value therefore did
not describe the KV allocation at 0.49. This curated profile conservatively sets
that ceiling to 405625, the observed total KV-token allocation for this launch.
This does not change the generator or establish that its aggressive-cap estimate
is incorrect. Regenerating the profile would replace this manual qualification. This is
not a tested single-request context length or a promise across images/configs.
The declared model context and default request limit remain 262144. The two
sequence cap permits short-request batching; it does not imply capacity for two
simultaneous full-context requests, which would require at least 524288 tokens
before extra margin. A lower utilization would reduce this measured headroom,
so no unmeasured lower default is proposed.

## Bounded inference smoke

All requests used temperature 0, thinking disabled through chat_template_kwargs,
at most 64 output tokens, and a 60-second HTTP deadline:

- GB10-local request: `READY`, finish reason `stop`, 17 prompt + 2 output tokens,
  17.176 seconds including cold first-request overhead.
- Two requests submitted concurrently from controlplane to the direct GB10 API:
  each returned `BATCH_OK`, finish reason `stop`, 18 prompt + 4 output tokens.
  Durations were 31.364 and 30.578 seconds. Other consumers were allowed to run;
  these timings are bounded smoke observations, not isolated throughput results.

Full-context inference, vision, tool calls, sustained throughput, and a search
for the minimum utilization were not tested. The profile's model/API aliases
were verified at readiness. The profile's existing parser flags remain unchanged.

## Provenance and operating method

- DMM application: c005e60; no application deployment or DMM restart.
- Profile script SHA256 before measurement:
  `b4aaf468f18ef1c4169d104766d468fbca45e5d92124b6765aca771ea7f4f4c0`.
- Engine image ID (`eugr/spark-vllm:latest` at measurement):
  `sha256:6208462fec6ae77e87f3e8023f04946109561cc82a475d87d01eefcff8ae901c`.
- Engine version reported in startup log:
  `v0.23.1rc1.dev841+g39c1111fb.d20260706`.
- Exact model snapshot:
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Private GB10 evidence and unchanged-script backups:
  `~/.local/state/dgx-model-manager/measurements/20261019T030820Z`.
- Nonsecret measurement summary: `profile-measurements-2026-10-18.json`.

After verifying the existing container identity, idle DMM launch state, profile
files and local images, an operator harness used DMM's existing `_launch_argv`
scope helper and the maintained profile scripts over SSH. This was a host
maintenance launch, not an authenticated API launch; no bearer key was needed,
read, changed, or bypassed in the running service. DMM's HTTP launch lock and
admission gate were not exercised by this host launch. The harness checked idle
launch state first, serialized itself, bounded startup and inference, and restored
Flash in a finally block. No other engine was launched alongside it. The DMM
pinned default and tracked production profile files were preserved.

## Restoration

Flash Next was verified serving from controlplane at **22:31:02 CDT**, about
22 minutes 42 seconds after the original stop began. Its cold restoration took
about 775 seconds from the launch call to the observer's successful model list.
The harness's 720-second observation deadline expired during otherwise normal
initialization; it did not kill or restart the container. Follow-up health checks
completed on that same container, with no second launch.

- DMM reports `serving`, idle launch state, and the original pinned Flash profile.
- Container `a6997620cbcf8bfc5a695525a1c3a184fb156a27ddb86b159cf02d947f2e0306`
  uses the original image
  `sha256:826800d5af37e81df3af1dc77982376d657ab70ab2fe1fd2e0bf2378ae367076`;
  `OOMKilled` is false.
- Model aliases and 262144 context are restored. Settings remain utilization 0.76,
  native context, MTP 2. Flash logs report 14.6 GiB KV cache / 515598 tokens.
- Bounded direct inference from controlplane returned `READY`, finish reason
  `stop`, 17 prompt + 2 output tokens, in 1.114 seconds at 22:31:03 CDT.
- Both maintained production profile scripts match their pre-window backups
  byte for byte. DMM remains active on c005e60 with a clean tracked checkout;
  no DMM restart, application deployment, pin change, or authentication change.
- Protected unauthenticated config reads still return 401. The existing app
  module loads its configuration internally; no plaintext bearer key was used,
  and no credential or saved configuration was exported.
- No matching NVRM out-of-memory lines were found in the readable kernel journal
  during the window. Flash's cold load briefly increased memory PSI and swap use;
  recent post-restoration vmstat samples showed zero swap-out and 0–4 KiB/s
  swap-in, with 99% CPU idle. This was a short functional verification, not a
  sustained stability qualification.

Validation for the profile metadata change: shell syntax and full pytest suite
(692 passed, 4 skipped). No launch argv or defaults changed. The tuning PR is
separate from approval to merge or deploy its metadata change.
