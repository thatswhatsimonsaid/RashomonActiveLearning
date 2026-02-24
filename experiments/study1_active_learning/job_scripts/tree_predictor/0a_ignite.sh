#!/bin/bash
if pgrep -f "1_smart_run.sh" > /dev/null; then
    echo "Smart Run is ALREADY running!"
else
    echo "Igniting Smart Run in the background..."
    nohup ./1_smart_run.sh > smart_run_log.txt 2>&1 &
    echo "Success! Process ID: $!"
    echo "Logs: tail -f smart_run_log.txt"
fi
