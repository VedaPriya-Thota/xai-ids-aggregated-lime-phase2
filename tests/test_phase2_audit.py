"""Regression: every number in the Phase 2 report must reproduce from the saved CSVs (no experiment is run)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_report_numbers_reproduce_from_saved_outputs():
    out = subprocess.run([sys.executable, "scripts/phase2_audit_checks.py"], cwd=ROOT, capture_output=True, text=True,
                         env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    summary = [l for l in out.stdout.splitlines() if l.startswith("SUMMARY:")][0]
    assert "FAIL" not in summary and eval(summary.split(":", 1)[1])["PASS"] >= 80
