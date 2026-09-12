from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy" / "dgx-model-manager.service"


def test_deployed_user_unit_has_no_local_systemd_diagnostics():
    proc = subprocess.run(
        ["/usr/bin/systemd-analyze", "--user", "verify", str(UNIT)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    local_diagnostics = [
        line for line in proc.stderr.splitlines() if line.startswith(f"{UNIT}:")
    ]
    assert proc.returncode == 0, proc.stderr
    assert local_diagnostics == []
