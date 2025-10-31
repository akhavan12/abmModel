#!/usr/bin/env bash
tmux new -s myjob -d 'bash -lc "source ./env/bin/activate; python -u covid_abm_gpu_claude.py sweep 50 10000 |& tee -a logs/run_$(date +%F_%H-%M-%S).log"'
