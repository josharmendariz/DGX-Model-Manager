#!/usr/bin/env python3
"""Aggregate request metadata only; never export message bodies or tool results."""
import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

ALIASES = {'coding', 'fast', 'vllm-active', 'vllm-active-thinking'}


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def analyze(control, opencode, days=14, now=None):
    cutoff = (now or time.time()) - days * 86400
    with readonly(control) as ctl, readonly(opencode) as oc:
        runs = ctl.execute('SELECT agent,session_id,status FROM runs WHERE started_at>=?', (cutoff,)).fetchall()
        sessions = defaultdict(set)
        for agent, session, _ in runs:
            if session:
                sessions[session].add(agent)
        groups, events = defaultdict(list), []
        covered = set()
        for session, agents in sessions.items():
            # Selecting JSON metadata expressions avoids loading prompt/tool payloads.
            rows = oc.execute('''SELECT json_extract(data,'$.modelID'),
              json_extract(data,'$.tokens'), json_extract(data,'$.time') FROM message
              WHERE session_id=? AND time_created>=? AND json_extract(data,'$.role')='assistant' ''',
                              (session, int(cutoff * 1000)))
            for model, tokens, timing in rows:
                if model not in ALIASES or not tokens:
                    continue
                tok, clock = json.loads(tokens), json.loads(timing or '{}')
                cache = tok.get('cache') or {}
                size = tok.get('input', 0) + cache.get('read', 0) + cache.get('write', 0)
                if size <= 0:
                    continue
                output = tok.get('output', 0) + tok.get('reasoning', 0)
                for agent in agents:
                    groups[agent].append((size, output))
                covered.add(session)
                start, end = clock.get('created'), clock.get('completed')
                if start and end and end > start:
                    events.extend([(start, 1), (end, -1)])
        active = peak = 0
        for _, delta in sorted(events):
            active += delta
            peak = max(peak, active)
        result = {'days': days, 'generated_at_epoch': now or time.time(),
                  'runs': len(runs), 'sessions': len(sessions), 'covered_sessions': len(covered),
                  'peak_recorded_overlap': peak, 'agents': {}}
        maximum = 0
        for agent, samples in sorted(groups.items()):
            sizes, outputs = zip(*samples)
            maximum = max(maximum, max(sizes))
            result['agents'][agent] = {'requests': len(sizes), 'input_tokens': {
                f'p{int(p*100)}': percentile(sizes, p) for p in (.5, .9, .95, .99)},
                'max_input_tokens': max(sizes), 'max_output_plus_reasoning': max(outputs)}
        result['suggested_context_floor'] = math.ceil((maximum + 8192) * 1.10) if maximum else None
        result['limitations'] = ['Metadata coverage only; missing sessions are not zero demand.',
          'Aliases are classified using current routing; historical routing can differ.',
          'Overlap excludes incomplete timestamps and unobserved consumers.',
          'Context floor uses maximum observed input plus 8192 output tokens and 10% margin.']
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-db', default='/home/josh/agent-control/control.db')
    p.add_argument('--opencode-db', default='/home/josh/.local/share/opencode/opencode.db')
    p.add_argument('--days', type=int, default=14)
    a = p.parse_args()
    print(json.dumps(analyze(a.control_db, a.opencode_db, a.days), indent=2))
