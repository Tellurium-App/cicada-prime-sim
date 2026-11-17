"""検証5: 短い周期から始めると近くの素数で止まるか（CRITERIA.md 検証5）。
  python sim_stuck.py
"""
import json
import numpy as np
from numba import njit, prange
from sim_ga import tables, PRIME, YMAX, mut, tourn


@njit(parallel=True)
def run(FY, w, Xfixed, lo, hi, N, gens, mu, runs, seed, ck):
    cx = np.zeros(YMAX + 1)
    for v in Xfixed:
        cx[v] += 1.0
    M = Xfixed.shape[0]
    vy = np.zeros(YMAX + 1)
    for b in range(2, YMAX + 1):
        s = 0.0
        for a in range(2, YMAX + 1):
            s += cx[a] * FY[a, b]
        vy[b] = s / M
    modes = np.zeros((runs, ck.shape[0]), np.int64)
    for r in prange(runs):
        np.random.seed(seed + r)
        Y = np.random.randint(lo, hi + 1, N).astype(np.int64)
        Yn = np.empty(N, np.int64)
        fy = np.empty(N)
        k = 0
        for g in range(1, gens + 1):
            for i in range(N):
                fy[i] = vy[Y[i]]
            for i in range(N):
                Yn[i] = tourn(fy, Y, N, True)
                if np.random.random() < mu:
                    v = Yn[i] + mut(w)
                    if 2 <= v <= YMAX:
                        Yn[i] = v
            Y[:] = Yn
            if k < ck.shape[0] and g == ck[k]:
                c = np.bincount(Y, minlength=YMAX + 1)
                modes[r, k] = np.argmax(c)
                k += 1
    return modes


def main():
    FX, FY = tables()
    sets = {"2-10": np.arange(2, 11), "2-20": np.arange(2, 21)}
    inits = {"2-5": (2, 5), "2-20": (2, 20), "2-50": (2, 50)}
    ck = np.array([3000, 30000])
    out = {}
    for pn, xs in sets.items():
        for iname, (lo, hi) in inits.items():
            for w in (1, 3, 10):
                m = run(FY, w, xs.astype(np.int64), lo, hi, 200, 30000, 0.1, 100, 50000, ck)
                rec = {}
                for j, g in enumerate(ck):
                    col = m[:, j]
                    vals, cnt = np.unique(col, return_counts=True)
                    rec[str(int(g))] = {
                        "modes": {int(v): int(c) for v, c in zip(vals, cnt)},
                        "le20": int((col <= 20).sum()),
                        "in11_13": int(np.isin(col, [11, 13]).sum()),
                        "is13_17": int(np.isin(col, [13, 17]).sum()),
                        "prime": int(PRIME[col].sum()),
                    }
                key = f"P{pn}_init{iname}_w{w}"
                out[key] = rec
                r = rec["30000"]
                print(key, "3k le20", rec["3000"]["le20"], "| 30k le20", r["le20"], "11/13", r["in11_13"],
                      "13/17", r["is13_17"], "prime", r["prime"], "modes", r["modes"], flush=True)
    json.dump(out, open("results/5_stuck.json", "w"), indent=1)


if __name__ == "__main__":
    main()
