"""検証2 追記: S×E 平面で、絶滅の縁に残る周期が素数かを見る（CRITERIA.md 追記節）。"""
import json
import sys
from multiprocessing import Pool

import numpy as np

from sim_hybrid import PRIMES, simulate, survivors

SS = [round(0.910 + 0.002 * i, 3) for i in range(31)]
EE = [round(0.130 + 0.005 * j, 3) for j in range(45)]


def one(args):
    S, E, allee, Nc = args
    phen, _ = simulate(S, E, Nc=Nc, T=1000, allee=allee)
    return S, E, survivors(phen)


if __name__ == "__main__":
    Nc = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    out = {}
    with Pool(30) as pool:
        for allee in ("genotype", "phenotype", "adult"):
            res = pool.map(one, [(S, E, allee, Nc) for S in SS for E in EE])
            edge = [r for r in res if 1 <= len(r[2]) <= 2]
            allp = [r for r in edge if all(p in PRIMES for p in r[2])]
            from collections import Counter
            cnt = Counter(tuple(r[2]) for r in edge)
            single = Counter(r[2][0] for r in edge if len(r[2]) == 1)
            surv_area = Counter(p for r in res for p in r[2])
            out[allee] = {
                "edge_points": len(edge), "edge_all_prime": len(allp),
                "share": round(len(allp) / len(edge), 3) if edge else None,
                "edge_sets_top": [(list(k), v) for k, v in cnt.most_common(12)],
                "single_survivor_counts": dict(sorted(single.items())),
                "survival_area_by_cycle": dict(sorted(surv_area.items())),
                "grid": res,
            }
            print(allee, {k: out[allee][k] for k in ("edge_points", "edge_all_prime", "share", "edge_sets_top", "single_survivor_counts", "survival_area_by_cycle")}, flush=True)
    json.dump(out, open(f"results/hybrid_plane_Nc{int(Nc)}.json", "w"), ensure_ascii=False)
