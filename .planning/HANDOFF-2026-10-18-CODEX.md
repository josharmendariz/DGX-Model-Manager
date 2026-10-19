# Session handoff — 2026-10-18 (America/Chicago)

Josh asked for autonomous preparation and verification, stopping at approval or scope
boundaries. No live configuration or service was changed by Codex this session.

## Observed live

- GB10 DMM active at clean commit `1c2d785`.
- Josh saved a key in Settings. The key is persisted as a hash; presence/shape only
  were checked. `/api/config` reports `api_key_set: true`; `/api/debug/config`
  returns 401 with missing and deliberately invalid tokens. No key/hash was printed.
- `MODEL_MANAGER_ALLOW_UNAUTH=1` remains in the running process. A configured key
  takes precedence, so this exception does not bypass authentication now.
- No DMM background jobs or vLLM launch in flight at the last read-only check.
- Direct inference responds with Flash Next and its aliases. No model switching,
  refusal smoke, cache reclaim, engine stop, or restart was performed.
- LiteLLM Kubernetes mode enabled; source router YAML and ConfigMap are equivalent.
  The comparison printed no contents. This is a point-in-time check, not a guard
  deployed into the running application.
- Installed DMM user unit differs from the tracked template and has no drop-ins.

## Prepared locally

Implementation commit `30a5c1d`, branch `fix/auth-default-and-litellm-drift`:

- Remove the open-access exception from the shipped service template.
- Add `deploy/require-auth.conf` using `UnsetEnvironment` for customized units.
- Correct server-hash versus browser-plaintext storage documentation.
- Add read-only LiteLLM startup comparison: parsed YAML, five-second Kubernetes
  lookup, fixed diagnostics for drift/unavailable results, no sync or rollout.
- Add 19 regression cases for equivalence, drift, unavailable inputs, secrecy,
  disabled mode, and startup integration.
- Rollout and rollback procedure: `docs/authentication-rollout.md` on that branch.

Verified: **692 passed, 4 skipped**, Ruff 0.14.0 passed, unit plus drop-in accepted
by `systemd-analyze --user verify`. Tests requiring systemd sockets passed outside
the restricted sandbox. No production unit or manager was restarted for validation.

PR #26 is open at `9d213c8`; GitHub main remains `2315f4f`. The documentation
follow-up builds on #26 rather than discarding its Phase 5 reconciliation. Preserve
the user's untracked `.planning/HANDOFF-2026-10-19.md` as historical material.

## Approval boundary and remaining checks

1. Josh confirms an authenticated protected read works in his browser. The saved
   token must stay out of chat, commands, logs, Git, and handoffs.
2. Obtain approval to install only the authentication drop-in, reload systemd, and
   restart only GB10's DMM user service. Refresh job/launch state, persisted-key
   presence, effective unit, and inference identity first. Back up an existing
   drop-in if one has appeared. Preserve the customized unit and saved key.
3. Verify authentication, absence of the exception in the process environment,
   unchanged inference container identity/start time, and serving health afterward.
4. Publication, merge, and deployment of the implementation candidate remain
   separate approval decisions. PR #26 should land before its documentation
   follow-up. Move application code through Git; do not copy it between hosts.

Profile tuning needs a model-switching window. Refusal smoke needs approval because
a broken admission guard could launch a second engine. Modularization and Phase 6
were intentionally deferred. No branches were deleted and no memory files changed.
