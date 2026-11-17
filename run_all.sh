#!/bin/sh
# 全検証を回して results/ に出す。published/ は動画を作ったときの結果。
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
mkdir -p results
set -e
python sim_predator.py check
python sim_predator.py 1a
sh run_1b_all.sh
CICADA_TIE=1 python sim_predator.py 1b 10 5 10000 div 1000 50
CICADA_TIE=2 python sim_predator.py 1b 10 5 10000 div 1000 50
CICADA_TIE=1 python sim_predator.py 1b 20 5 2500 div 1000 50
CICADA_TIE=2 python sim_predator.py 1b 20 5 2500 div 1000 50
python boot_1b.py 10 10000
python boot_1b.py 20 2500
python sim_hybrid.py
python sim_hybrid_plane.py 100
python sim_hybrid_eq4.py
python sim_yoshimura.py
python sweep_yoshimura.py
python constC_yoshimura.py
python sim_ga.py check
python sim_ga.py 4a
python sim_ga.py 4b
python sim_ga.py prey_only
python sim_stuck.py
