# Profile readiness checks

The 2026-10-18 GB10 audit found three preflight defects: read-only mount
options were included in destination paths, Hub IDs were treated as filesystem
paths, and argument checks skipped images whose entrypoint starts the API server.

The repair keeps local paths and Hub IDs distinct:

- Mount scope strips options such as `:ro` and checks a directory boundary.
  `/models-other` is outside a bind at `/models`.
- Absolute model paths are checked by reading `config.json` through the profile's
  bind mounts. Hub IDs use Hugging Face's offline cache lookup in the image,
  including a statically resolved `--revision` when present.
- Missing local configurations remain failures. A missing cached Hub configuration
  fails an explicitly offline profile; an online profile receives a warning that
  its launch may need a download. An unavailable probe returns `skip`.
- Argument checks handle both `vllm serve ...` and image-entrypoint `--model ...`
  profiles. A successful check is explicitly labelled CLI syntax only. Numeric
  defaults for the three documented `VLLM_*` overrides can be
  read without evaluating shell code. The parser stops at the launch command's
  end, excluding subsequent commands such as `docker update`.

Probes use already-present images (`--pull never`), disable networking, request
no GPU access, and replace bind mounts with read-only `--mount` arguments that
cannot create missing source directories. Neither weights nor model code are
loaded. Genuine argparse rejections remain failures; import/device limitations
remain skips. An argument probe cannot establish GPU compatibility or inference
readiness.

Dynamic paths/revisions, custom Hub cache environment/download directories, and
named-volume mounts remain unknown when their exact layout cannot be established.
The diagnostic explicitly reports a skipped check. Recipe dry-run behavior is
unchanged. Wrapper collision/restart warnings remain separate diagnostics.

Validation includes mocked Docker arguments and execution of the actual cache
probe against a temporary Hugging Face cache with intact and dangling snapshot
symlinks. These tests require no Docker daemon, model weights, or GPU.

## Current-state and replacement estimates

The memory summary separates the current-state budget balance (including resident
engines) from an estimated balance after replacing a known labelled engine. The
second estimate reuses admission's label-only reservation identification, caps
credit to currently unavailable memory, and never double-counts cache reclaim
past the reservation limit. Unknown labels or unavailable status receive no
replacement estimate. This is advisory display data; admission logic and page-cache
thresholds are unchanged.

Reservation credit is not measured released memory. Neither balance is KV capacity
or proof a model will start. A negative current-state balance while Flash is
serving does not describe Nano after replacing Flash. Memory and budget use one
snapshot; numeric `VLLM_*` defaults supply utilization for parameterized profiles.

## GPU-free CLI syntax qualification

Installed vLLM versions materialize `VllmConfig.device_config` while constructing
their CLI parser. A GPU-free process cannot infer its automatic device default.
The isolated probe recognizes that specific `FieldInfo.default_factory` identity
and temporarily supplies `DeviceConfig(device="cpu")` only during parser
construction. It restores the factory before parsing the profile arguments.
It does not change platform detection, option definitions, or the supplied flags.
Unknown layouts/imports remain `skip`; no GPU access is added.

Success is reported as `CLI flags accepted (syntax only)`. This checks flag names
and parser value types; GPU-specific plugins, model compatibility, engine config
validation, weights, kernels and inference remain unqualified. The replacement
estimate and the CPU device default never feed a model launch.

See `preflight-cli-qualification-2026-10-18.json` for isolated offline image results.
These are tests of the probe, not an application deployment. GB10's running DMM
and Flash Next remain unchanged while this follow-up is prepared.

Qualification on GB10's existing images accepted all four real profile argument
lists (Nemotron Nano, Nemotron Super, Qwen3-VL-4B-FP8, Qwen3.8-27B-FP8). For each
profile, an unknown-flag control and an invalid-utilization control returned
argparse exit 2 without a success marker: twelve expected outcomes across three
image IDs. The exact probe hash and image IDs are recorded in the JSON evidence.
The installed DMM was not modified. Flash's container identity and start time
were unchanged, and a bounded controlplane inference check returned `READY` in
0.333 seconds afterward.
