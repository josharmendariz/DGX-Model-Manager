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
  profiles. Numeric defaults for the three documented `VLLM_*` overrides can be
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
unchanged. Wrapper collision/restart warnings and utilization-budget parsing are
outside this repair.

Validation includes mocked Docker arguments and execution of the actual cache
probe against a temporary Hugging Face cache with intact and dangling snapshot
symlinks. These tests require no Docker daemon, model weights, or GPU. Image-specific
runtime qualification remains a follow-up after an approved deployment; the
running GB10 application and Flash Next are unchanged by preparing this PR.
