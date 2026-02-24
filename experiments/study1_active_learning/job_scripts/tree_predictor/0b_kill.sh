#!/bin/bash
if pgrep -f "1_smart_run.sh" > /dev/null; then
    echo "Stopping Smart Run..."
    pkill -f "1_smart_run.sh"
    echo "Smart Run stopped."
else
    echo "Smart Run is not running."
fi
read -p "Also cancel all your Slurm jobs? (y/n): " s_kill
[[ "$s_kill" == "y" ]] && scancel -u $USER
