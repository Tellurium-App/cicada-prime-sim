export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
python sim_predator.py 1b 10 5 10000 div 1000 50
python sim_predator.py 1b 10 0 10000 div 1000 50
python sim_predator.py 1b 10 5 10000 nodiv 1000 50
python sim_predator.py 1b 5 5 10000 div 1000 50
python sim_predator.py 1b 20 5 2500 div 1000 50
python sim_predator.py 1b 10 5 2500 div 1000 100
