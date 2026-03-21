#!/usr/bin/env python3
"""
Fuzz tests for device operations.

Uses atheris for coverage-guided fuzzing.
Install: pip install atheris
Run: python fuzz_device_ops.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import atheris
import cv_native as cv


def TestOneInput(data):
    """Fuzz device operations"""
    if len(data) < 4:
        return

    try:
        device_idx = data[0] % 8
        memory_mb = ((data[0] << 8) | data[1]) % 10000 + 1
        timeout_ms = ((data[2] << 8) | data[3]) % 60000
        group_id = ((data[0] << 8) | data[1]) % 1000 + 1

        try:
            cv.Runtime.reserve_device(device_idx, memory_mb)
        except (RuntimeError, TypeError, OverflowError):
            pass

        try:
            cv.Runtime.release_device(device_idx)
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.best_device(memory_mb % 1000)
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.set_gpu_wait_timeout(timeout_ms)
        except (RuntimeError, TypeError, OverflowError):
            pass

        try:
            cv.Runtime.get_gpu_wait_timeout()
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.join_group(group_id)
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.mock_init_device(device_idx, memory_mb)
        except (RuntimeError, TypeError):
            pass

        try:
            cv.Runtime.wait_for_gpu(device_idx, memory_mb, timeout_ms)
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
