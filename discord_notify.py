#!/usr/bin/env python3
"""Discord notification helpers for Hermes SysEng."""

from __future__ import annotations

import os
import sys
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path


def send_discord_alert(webhook_url: str, title: str, message: str, color: int = 0xFF0000) -> bool:
    """Send a Discord alert with embed format."""
    if not webhook_url:
        print("No Discord webhook configured")
        return False
    
    embed = {
        "title": title,
        "description": message[:4096],  # Discord 4096 char limit
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    payload = {"content": "", "embeds": [embed]}
    
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status in (200, 204)
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")
        return False


def send_troubleshooter_summary(webhook_url: str, alert_output: str, remediation_status: str) -> bool:
    """Send troubleshooter summary to Discord."""
    if not webhook_url:
        print("No Discord webhook configured")
        return False
    
    # Parse alert output to extract severity
    severity = "critical" if "critical" in alert_output.lower() else "warning"
    color = 0xFF0000 if severity == "critical" else 0xFFA500
    
    # Truncate alert output if needed
    alert_summary = alert_output[:1000] + "..." if len(alert_output) > 1000 else alert_output
    
    embed = {
        "title": "🚀 Autonomous Remediation Triggered",
        "description": f"**Status**: {remediation_status}\n\n**Alerts Detected**:\n```{alert_summary}```",
        "color": color,
        "fields": [
            {
                "name": "System Response",
                "value": "The autonomous troubleshooter has been triggered to analyze and fix the detected issues.",
                "inline": False,
            },
            {
                "name": "Next Steps",
                "value": "Monitor the troubleshooting logs at `~/.hermes/logs/troubleshooter.log` and `~/.hermes/logs/endpoint_remediation.log`",
                "inline": False,
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "Hermes SysEng - Autonomous Troubleshooter"},
    }
    
    payload = {"content": "⚠️ Autonomous Remediation Triggered", "embeds": [embed]}
    
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status in (200, 204)
    except Exception as e:
        print(f"Failed to send Discord alert: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: discord_notify.py <alert_output> <remediation_status>")
        sys.exit(1)
    
    alert_output = sys.argv[1]
    remediation_status = sys.argv[2]
    
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    
    if not webhook_url:
        print("No Discord webhook configured")
        sys.exit(1)
    
    success = send_troubleshooter_summary(webhook_url, alert_output, remediation_status)
    if success:
        print("Discord notification sent successfully")
    else:
        print("Failed to send Discord notification")
        sys.exit(1)


if __name__ == "__main__":
    main()
