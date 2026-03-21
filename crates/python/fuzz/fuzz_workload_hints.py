#!/usr/bin/env python3
"""
Fuzz tests for workload hints.

Uses atheris for coverage-guided fuzzing.
Install: pip install atheris
Run: python fuzz_workload_hints.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import atheris
import cv_native as cv


def TestOneInput(data):
    """Fuzz workload hint combinations"""
    if len(data) < 2:
        return

    try:
        mode_idx = data[0] % 4
        modes = [
            cv.Runtime.ExecutionMode.Strict,
            cv.Runtime.ExecutionMode.Normal,
            cv.Runtime.ExecutionMode.AdaptiveBasic,
            cv.Runtime.ExecutionMode.AdaptiveAggressive,
        ]
        cv.Runtime.set_execution_mode(modes[mode_idx])

        hint_idx = data[0] % 4
        hints = [
            cv.Runtime.WorkloadHint.Latency,
            cv.Runtime.WorkloadHint.Throughput,
            cv.Runtime.WorkloadHint.PowerSave,
            cv.Runtime.WorkloadHint.Default,
        ]

        _ = hints[hint_idx]

        memory_mb = (data[0] * 17 + data[1]) % 10000 + 1
        device_idx = data[0] % 8

        try:
            cv.Runtime.reserve_device(device_idx, memory_mb)
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.best_device(memory_mb % 1000)
        except (RuntimeError, TypeError):
            pass

    except Exception:
        pass


def main():
    """Main entry point"""
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
