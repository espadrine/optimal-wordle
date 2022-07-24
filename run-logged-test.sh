#!/bin/bash
set -x
touch "$logfile"
julia play-optimally.jl >"$logfile" &
tail -f "$logfile" | grep -E '^Improvement found: ..... -|^Explored .....:'
