#!/bin/bash
source "$(dirname "$0")/../.venv/bin/activate"
cd "$(dirname "$0")"
python3 manual_proc_a.py &
PID_A=$!
sleep 2 # let A start and hold memory
python3 manual_proc_b.py &
PID_B=$!
wait $PID_A
wait $PID_B
