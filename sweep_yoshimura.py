"""検証3 の追加観察（判定には使わない）: Fig 1A の近くで S を少しずつ動かし、200年・700年の生き残りを見る。"""
import json
from multiprocessing import Pool

from sim_yoshimura import simulate

SS = [round(0.940 + 0.001 * i, 3) for i in range(17)]


def one(args):
    S, R, fm = args
    s = simulate(S, R, 0.2, 700, fm, record=(200, 700))
    return S, R, fm, list(s[200]), list(s[700])


if __name__ == "__main__":
    jobs = [(S, R, fm) for fm in ("pair", "sum") for R in (1.45, 1.63) for S in SS]
    with Pool(30) as p:
        res = p.map(one, jobs)
    for r in res:
        print(r)
    json.dump(res, open("results/yoshimura_sweep.json", "w"))
