# Fleet-informed overnight comparison

The comparison runs 02:00–07:30 **America/Chicago**, with agent-control pause
after active jobs drain and resume after verified Flash restoration. The next
approved window is October 20, 2026 (07:00–12:30 UTC). Deployment and activation
evidence stays in local `.planning/` artifacts rather than this generic runbook.
The schedule generator rejects past dates and missed runs do not catch up.

## Workload audit

Refresh the local metadata audit before choosing context limits. The analyzer
reads controller run/session identifiers and selected OpenCode JSON metadata
fields, without exporting prompt or tool bodies. It deduplicates sessions,
filters request timestamps, separates currently routed local aliases, and reports
per-agent input percentiles, output allowance and observed overlap. Missing
telemetry and historical routing changes remain coverage gaps.

Keep detailed fleet audit artifacts under local `.planning/` for operator review;
this public repository does not need private run histories. The prepared synthetic
workloads span 17k, 176k and 229k input tokens. With an 8192-token output allowance
and 10% margin, the largest test represents a 260,912-token context requirement.
The smaller context candidate therefore has limited workload coverage. Recorded
request overlap guides whether higher concurrency is necessary, while synthetic
parallel requests establish future batching potential.

Flash profile overrides preserve its existing default utilization .76, native
context 262144, sequence cap 8 and MTP=2. Qwen defaults remain .49 / 262144 / 2.
No persistent defaults or fleet concurrency settings are changed by the runner.

## Bounded experiment

`scripts/analyze_fleet_workload.py` refreshes the metadata audit without exporting
prompts, tool payloads, credentials, or raw message JSON. `scripts/overnight_tune.py`
prepares the plan and units and runs the host maintenance experiment.

| Model | Utilization hypotheses | Context | Sequence cap |
| --- | --- | ---: | ---: |
| Qwen FP8 | .49 baseline, .46, .43 | 262144 | 2 |
| Qwen FP8 | .49 | 196608 | 4 |
| Flash Next | .76 baseline | 262144 | 8 |
| Flash Next | .74, .72 | 262144 | 4 |
| Flash Next | .76 | 196608 | 4 |

This is eight bounded candidates, not an exhaustive optimum search.
The Qwen FP8 rows are currently suspended (`SUSPENDED_PROFILES`), so a run tests
only the four Flash Next rows. The 2026-10-20 run hung GB10 for hours during the
Qwen .49 cold load, and the cause is not yet known. Changing
Flash utilization and sequence cap together measures deployable combinations;
it does not isolate the causal effect of either setting. Candidates at 196608
are limited to smaller workflows and never selected for shared fleet use.

Each launch measures wall-clock startup, parsed weight/KV GiB and KV tokens,
MemAvailable, memory PSI, swap use and swap-out, and readable kernel allocation
errors. No memory-savings claim is based on reservation arithmetic alone.
Each inference batch uses newly prefixed synthetic text, calibrated through the
model tokenizer, with a retrieval marker and bounded output. Single inputs target
17k, 176k and 229k tokens, plus mixed 17k+176k and four 17k requests when sequence
and measured KV bounds permit. A separate 128-token decode case measures bounded
decode speed. A synthetic tool-call schema smoke validates function arguments
without executing any tool. Retrieval uses up to 128 output tokens, while admission reserves
8192 per request. Batches never exceed 90% of measured KV capacity. Prompt
calibration is within 2%; actual token lengths are reported.

Reports include TTFT, end-to-end elapsed time, decode tokens/s, aggregate
end-to-end output tokens/s, output length and finish reason. A provisional shared
fleet selection requires all three single-request context cases, no kernel
allocation errors, >=10 GiB available memory, and no more than 10% single-request
elapsed regression against the measured original-utilization baseline. It selects the lowest
**tested** reservation meeting those bounds. This is a starting operating point,
not proof of agent task quality or a globally ideal setting. A representative
agent canary remains necessary before changing production defaults/concurrency.

If a candidate cannot expose required activity/KV metrics, startup fails, an
unexpected engine appears, traffic contaminates a batch, or memory guards trip,
the script stops the sweep and attempts to restore Flash. Quality failures remain recorded and
may allow subsequent healthy candidates; incomplete runs are not passing results.
A detected extra completed request marks results contaminated. Other traffic
outside batches or concurrent failed requests may escape that detector.

Every measured batch must also reach **25 W** peak GPU draw, read with nvidia-smi
every two seconds. Healthy GB10 load peaks at about 30–90 W. After a hard fault the
platform can latch a power cap of about 15 W that slows decode about threefold.
The driver reports no limit or throttle reason, so the draw is the only signal.
Only a full AC power removal has cleared the cap; a reboot did not. A batch below
the floor, or with no readable draw, fails the candidate and stops the sweep.

## Ownership, deadlines and recovery

Controlplane drains agent-control for up to ten minutes. A short SQLite writer
reservation (rolled back without changing data) spans the final ledger check and
service-cgroup freeze, preventing new recorded admissions during that transition.
Freezing the idle controller preserves process state; queued jobs remain queued.
The runner refuses a controller with an enabled systemd watchdog or an existing
freeze. After serving restoration it thaws the controller and ensures it is active.
Known caller pause/resume hooks may be supplied as a local service drop-in, with
identity checks and an independent resume failsafe. They are not a substitute for
checking engine activity: pausing a caller does not cancel its pending inference.

Admission records running/waiting counts under the GB10 run directory and waits
up to **15 minutes** for the engine to drain. The explicitly approved deadline
policy may then replace the original engine even with requests outstanding;
those requests will be interrupted. Without `DMM_ALLOW_BUSY_REPLACEMENT=1`, a busy
engine still causes an abort. Missing metrics, a DMM launch conflict, source or
image drift, and active controller jobs are never overridden by this policy.
The schedule generator enables that policy only with the explicit
`--allow-busy-replacement` flag; its default remains abort-on-busy.

Benchmark engines publish **127.0.0.1:18000** on GB10 instead of normal serving
port 8000. All measurements use that private endpoint. Applications using 8000
cannot reach the benchmark models, so retries do not contaminate the sweep.
The profile override accepts only normal port 8000 or this exact loopback mapping.
Restoration explicitly returns Flash Next to port 8000 before callers resume.
No firewall, routing, credentials or persistent application endpoints change.

GB10 executes the maintained DMM profiles from its deployed Git revision in a
separate user systemd unit. This is the explicit host-maintenance path, not the
HTTP start route: it preserves DMM authentication and performs its own checks,
private file lock, source/image/weight-manifest pins and DMM launch-idle checks. It does not share
DMM's in-process launch lock/admission. **Do not operate DMM start/stop controls or
restart DMM during the experiment.** Existing image tags and maintained Flash
serve script are fingerprinted; no image pull/build or source synchronization is
performed. Normal profile page-cache reclamation remains part of each launch.

The worker has `RuntimeMaxSec=15300` (4h15) and an **independent systemd
ExecStopPost restore**, with `TimeoutStopSec=1800`. Its internal cutoff reserves
45 minutes before 07:30 for restoration and starts no new candidate with less
than 30 minutes remaining before that cutoff. The worker refuses starts more than 30 minutes after 02:00. That start bound,
plus the hard worker maximum and restoration allowance, fit within 5.5 hours.
Restoration allows 25 minutes startup plus a bounded smoke within the 30-minute
outer allowance. This covers ordinary process errors, timeout, cancellation and
SSH disconnect; it cannot guarantee recovery across host/power/systemd failure.
There is no boot-persistent experiment timer or recovery change.

Restoration verifies source/image pins, restores Flash's baseline settings, and
requires a direct chat smoke. Controller resume is gated on remote restoration
confirmation plus another serving smoke, and occurs only if it was originally
active. Failed restoration leaves the controller paused for operator recovery.
Private state is under `~/.local/state/dmm-overnight/<UTC-run-id>/` on each owning
host. `report.json`, candidate `result.json` and restoration `snapshot.json` are
metadata artifacts; raw launcher logs stay private on GB10. Admission failures
persist `failure.json`, safe remote diagnostics persist `command-failure.json`,
and structured journal events retain the original precheck reason without copying
raw stderr, environment, prompts or credentials.

## Small live rehearsal

The `rehearse` mode uses the current Flash engine, tokenizer, monitoring and
streaming inference path with 512-token synthetic prompts at parallelism 1 and 2,
plus a tool-call schema smoke. It never invokes a profile or replaces a container.
Run it through temporary versions of the same coordinator/worker units with a
separate run ID and a short deadline, before installing the overnight units.
That exercises the real controller drain/freeze, SSH coordination, exit hooks,
serving verification and controller thaw. It qualifies orchestration and the
current inference protocol; it does not measure either model's cold startup.
A busy-engine rehearsal uses a shortened drain deadline and validates the approved
fallback without actually replacing the engine; background completions are marked
as expected traffic, not treated as isolated performance measurements.

## Prepare, review, then deploy

No timer is installed by the schedule generator:

```bash
python3 scripts/analyze_fleet_workload.py > /tmp/fleet-workload.json
python3 scripts/overnight_tune.py plan
python3 scripts/overnight_tune.py schedule --date 2026-10-20 --allow-busy-replacement
systemd-analyze --user verify .planning/overnight-units/*.service .planning/overnight-units/*.timer
```

The generated files are review artifacts, not committed unit definitions. The
worker unit binds the current Git commit. **After final commit/merge, regenerate
them from that exact revision**, and review the resulting date, UTC run ID,
revision and all timeouts. Qualification includes mocked recovery/streaming and drain-lock
checks, a disposable live systemd timeout/exit-hook drill, and live
tokenizer/monitor-schema compatibility; neither model has been
switched to qualify this new runner. Complete review before arming it.

After approval of the concrete revision and schedule:

1. Transfer the reviewed commit through private Git transport. Deploy that exact revision through
   Git on GB10 at `/home/josh/DGX-Model-Manager`; preserve generated/untracked
   profiles and all unrelated state. Do not restart DMM for this script change.
2. Generate units locally from the same revision. Install only
   `dmm-overnight-worker.service` into GB10's user unit directory, delivering the
   reviewed generated unit as a small release configuration artifact. Install
   `dmm-overnight.service` and `.timer` on controlplane. This is not a source file
   sync; application code travels only through Git.
3. On each host run `systemctl --user daemon-reload`, verify the applicable units,
   and check effective `ExecStart`, `ExecStopPost`, `RuntimeMaxUSec`,
   `TimeoutStopUSec` and worker revision binding. Confirm user lingering is already
   enabled and both managers remain available overnight. Check current Flash
   baseline, model files, images, consumers and absence of old run state again.
4. Only on controlplane, arm with
   `systemctl --user enable --now dmm-overnight.timer`. Inspect the next trigger;
   it must be the approved 02:00 Chicago date. The worker has no timer.
5. Afterward verify Flash's settings/health and controller state, review the JSON
   results, then disable/remove the one-shot timer and archive the run evidence.
   Consider recommendations in a separate reviewed profile-settings PR.

To cancel an armed run before 02:00, stop/disable the controlplane timer. To stop
an active experiment, stop `dmm-overnight.service` on controlplane; its exit hook
stops the GB10 worker, whose exit hook restores serving. If controlplane/SSH is
unavailable, stop `dmm-overnight-worker.service` directly on GB10 and inspect its
journal and private snapshot. Never resume the fleet while serving is unverified.
