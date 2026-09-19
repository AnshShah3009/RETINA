#!/bin/bash
source .venv/bin/activate
cd python_examples
python3 manual_proc_a.py &
PID_A=$!
sleep 2 # let A start and hold memory
python3 manual_proc_b.py &
PID_B=$!
wait $PID_A
wait $PID_B
