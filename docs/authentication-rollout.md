# Authentication rollout for an existing GB10 deployment

Prepared 2026-10-18. Deployment and service restart require Josh's approval.
Keep DMM bound to Tailscale. Do not stop or switch inference engines.

## Verified before preparation

- GB10 runs DMM commit `1c2d785`; the user unit is active.
- Josh saved an API key through Settings. `/api/config` reports `api_key_set: true`.
- `config.json` contains a 64-character hexadecimal key hash; no hash or token was
  printed. Missing and invalid tokens receive 401 from `/api/debug/config`.
- No DMM background jobs or engine launches were running at the read-only check.
- Direct `/v1/models` responds with the Flash Next model and its aliases.
- The installed unit differs from the tracked template and has no drop-ins.

These observations must be refreshed immediately before a deployment.

## Authentication-only live change

This can be approved independently of publishing or deploying the startup drift code.
The exact drop-in is `deploy/require-auth.conf`:

```ini
[Service]
UnsetEnvironment=MODEL_MANAGER_ALLOW_UNAUTH
```

1. Have Josh confirm a protected read works in the authenticated browser. Do not
   ask him to send the token. The negative authentication checks alone do not
   establish that a valid-token request succeeds.
2. Recheck job/launch inactivity and inference health. Confirm the persisted key
   hash is present without printing it. Record the current engine container ID
   and start time so the same engine can be verified after the restart.
3. Inspect the effective unit and existing drop-ins for changes since preparation.
   Back up any existing `require-auth.conf` before replacing it. Preserve the
   installed unit and `config.json`.
4. Install the reviewed drop-in at
   `~/.config/systemd/user/dgx-model-manager.service.d/require-auth.conf`.
   Validate the installed unit with `systemd-analyze --user verify`, then run
   `systemctl --user daemon-reload` and restart only `dgx-model-manager.service`.
5. Verify DMM is active, its process environment no longer contains
   `MODEL_MANAGER_ALLOW_UNAUTH`, and `/api/config` still reports a configured key.
   Verify missing and invalid tokens receive 401 on the protected read. Have Josh
   verify the authenticated browser read still works.
6. Recheck the direct inference endpoint and compare the engine container ID/start
   time. Do not issue a model start/stop or a force request as part of this change.

If DMM fails after the restart, diagnose before changing authentication. If rollback
is needed, restore the backed-up drop-in (or remove only the newly installed file),
reload systemd, and restart DMM. Do not clear or rotate the saved key. With the key
still configured, restoring the old exception does not bypass key verification.

## Startup drift code release

The code change is separate from the authentication-only drop-in. Publish a feature
PR, obtain merge/deployment approval, and deploy through Git. Do not copy application
files between hosts. Refresh job/launch state before the approved DMM restart.

After deployment, inspect only messages beginning `LiteLLM config drift` in the
application logs. The result should be `source file matches ConfigMap`, a drift
warning, or an unavailable warning. Never dump the router file, ConfigMap contents,
complete unit environment, or unfiltered logs. A drift warning does not authorize
syncing either side or rolling the LiteLLM deployment.
