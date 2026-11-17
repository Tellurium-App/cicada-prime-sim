"""検証3 の対照（判定には使わない）: 産卵数を周期によらず一定にすると、縁で素数だけが残る形は消えるか。
C(y)=R·y（R=1.45）と、一定 C=1.45×15=21.75 を、S を 0.900〜0.970 で動かして 700年目の生き残りを比べる。"""
import json
from multiprocessing import Pool

from sim_yoshimura import PRIMES, simulate

SS = [round(0.900 + 0.002 * i, 3) for i in range(36)]


def one(args):
    S, mode = args
    cc = None if mode == "R*y" else 21.75
    return S, mode, list(simulate(S, 1.45, 0.2, 700, "pair", record=(700,), const_C=cc)[700])


if __name__ == "__main__":
    with Pool(30) as p:
        res = p.map(one, [(S, m) for m in ("R*y", "const") for S in SS])
    for m in ("R*y", "const"):
        rows = [r for r in res if r[1] == m]
        nonempty = [r for r in rows if r[2]]
        first = nonempty[0] if nonempty else None
        edge = [r for r in rows if 1 <= len(r[2]) <= 2]
        print(m, "| 最初に生き残りが出る S:", first, "| 縁(1〜2本)の点:", [(r[0], r[2]) for r in edge],
              "| 縁で全部素数:", sum(all(p in PRIMES for p in r[2]) for r in edge), "/", len(edge))
    json.dump(res, open("results/yoshimura_constC.json", "w"))
