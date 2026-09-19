#!/usr/bin/env python3
"""
Fuzz tests for cv-native runtime API.

Uses atheris for coverage-guided fuzzing.
Install: pip install atheris
Run: python fuzz_runtime.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import atheris
import cv_native as cv


def TestOneInput(data):
    """Fuzz a single input"""
    if len(data) < 2:
        return

    with atheris.instrument_imports():
        try:
            device_idx = data[0] % 8
            memory_mb = (data[0] * 256 + data[1]) % 1000 + 1

            try:
                cv.Runtime.reserve_device(device_idx, memory_mb)
                cv.Runtime.release_device(device_idx)
            except (RuntimeError, TypeError):
                pass

            try:
                cv.Runtime.best_device(memory_mb % 100)
            except (RuntimeError, TypeError):
                pass

            mode_idx = data[0] % 4
            modes = [
                cv.Runtime.ExecutionMode.Strict,
                cv.Runtime.ExecutionMode.Normal,
                cv.Runtime.ExecutionMode.AdaptiveBasic,
                cv.Runtime.ExecutionMode.AdaptiveAggressive,
            ]
            cv.Runtime.set_execution_mode(modes[mode_idx])

            try:
                cv.Runtime.get_gpu_wait_timeout()
            except (RuntimeError, TypeError):
                pass

            try:
                cv.Runtime.get_global_load()
            except (RuntimeError, TypeError):
                pass

            try:
                cv.Runtime.get_device_info()
            except (RuntimeError, TypeError):
                pass

            try:
                group_id = (data[0] % 1000) + 1
                cv.Runtime.join_group(group_id)
            except (RuntimeError, TypeError):
                pass

            try:
                cv.Runtime.get_num_devices()
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
