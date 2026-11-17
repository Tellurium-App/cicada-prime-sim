"""掲載版の式4を字面どおりに読んだ版（eq4="literal"）で Fig 1・Fig 5・縁の平面を回す。判定は CRITERIA.md のまま。"""
import json
from collections import Counter
from multiprocessing import Pool

from sim_hybrid import PAPER_1B, PRIMES, simulate, spearman, survivors
from sim_hybrid_plane import EE, SS


def one(args):
    S, E, allee = args
    phen, _ = simulate(S, E, Nc=100, T=1000, allee=allee, eq4="literal")
    return S, E, survivors(phen)


if __name__ == "__main__":
    out = {}
    with Pool(30) as pool:
        for allee in ("genotype", "phenotype", "adult"):
            a, _ = simulate(0.95, 0.5, Nc=100, T=100, allee=allee, eq4="literal")
            b, _ = simulate(0.95, 0.5, Nc=0, T=100, allee=allee, eq4="literal")
            rho, ours = spearman(PAPER_1B, b)
            f5 = {}
            for E in (0.24, 0.26, 0.27, 0.30):
                rows = [(Nc, survivors(simulate(0.944, E, Nc=Nc, T=1000, allee=allee, eq4="literal")[0])) for Nc in range(0, 301, 10)]
                sp = sum(1 for Nc, sv in rows if 20 <= Nc <= 250 and len(sv) == 1 and sv[0] in PRIMES)
                f5[E] = {"single_prime_in_20_250": sp, "rows": rows}
            res = pool.map(one, [(S, E, allee) for S in SS for E in EE])
            edge = [r for r in res if 1 <= len(r[2]) <= 2]
            allp = [r for r in edge if all(p in PRIMES for p in r[2])]
            out[allee] = {
                "fig1a_survivors": survivors(a), "fig1a_order": sorted(survivors(a), key=lambda p: -a[p]),
                "fig1b_survivors": survivors(b), "fig1b_order": ours, "fig1b_spearman": round(rho, 3),
                "fig5_single_prime": {E: v["single_prime_in_20_250"] for E, v in f5.items()},
                "plane_edge": len(edge), "plane_edge_all_prime": len(allp),
                "plane_edge_sets": [(list(k), v) for k, v in Counter(tuple(r[2]) for r in edge).most_common(8)],
                "fig5_rows": {E: v["rows"] for E, v in f5.items()}, "grid": res,
            }
            print(allee, {k: out[allee][k] for k in ("fig1a_survivors", "fig1a_order", "fig1b_survivors", "fig1b_order", "fig1b_spearman", "fig5_single_prime", "plane_edge", "plane_edge_all_prime", "plane_edge_sets")}, flush=True)
    json.dump(out, open("results/hybrid_eq4_literal.json", "w"), ensure_ascii=False)
