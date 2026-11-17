"""検証3: Yoshimura et al. 2009（Evolution）整数版モデル。CRITERIA.md 検証3 の設定。

floor_mode="pair": 交配1組×遺伝子型ごとに切り捨て（原稿の「式3が4未満なら0」に合わせた既定）
floor_mode="sum" : 遺伝子型ごとに合計してから切り捨て
"""
import json
import math
import sys

import numpy as np

CY = list(range(10, 21))
K = len(CY)
PRIMES = {11, 13, 17, 19}


def simulate(S, R, E, T, floor_mode="pair", N0=1000, record=(), const_C=None):
    N = np.zeros((K, K, 22), dtype=np.int64)
    for i in range(K):
        N[i, i, 0] = N0
    phen = [[min(CY[a], CY[b]) for b in range(K)] for a in range(K)]
    snaps = {}
    for t in range(1, T + 1):
        N[:, :, 1:] = np.floor(N[:, :, :-1] * S).astype(np.int64)
        N[:, :, 0] = 0
        adults = []
        for a in range(K):
            for b in range(a, K):
                ag = phen[a][b]
                if N[a, b, ag] > 0:
                    na = math.floor(E * N[a, b, ag])
                    N[a, b, ag] = 0
                    if na > 0:
                        adults.append((a, b, na))
        tot = sum(x[2] for x in adults)
        if tot > 0:
            acc = {}
            for (a, b, na) in adults:
                C = R * CY[a] if const_C is None else const_C  # 母の表現型 = 短い方（const_C: 周期によらず一定にする対照）
                for (m, n, nm) in adults:
                    eq3 = 0.5 * na * (nm / tot) * C
                    for u in (a, b):
                        for v in (m, n):
                            key = (min(u, v), max(u, v))
                            val = eq3 / 4
                            if floor_mode == "pair":
                                val = math.floor(val)
                            acc[key] = acc.get(key, 0) + val
            for (u, v), val in acc.items():
                N[u, v, 0] += math.floor(val)
        if t in record:
            snaps[t] = surv(N, phen)
    return snaps


def surv(N, phen):
    out = {}
    for a in range(K):
        for b in range(a, K):
            s = int(N[a, b].sum())
            if s > 0:
                p = phen[a][b]
                out[p] = out.get(p, 0) + s
    return dict(sorted(out.items()))


if __name__ == "__main__":
    res = {}
    for fm in ("pair", "sum"):
        a = simulate(0.948, 1.45, 0.2, 700, fm, record=(100, 200, 400, 700))
        c = simulate(0.943, 1.63, 0.2, 600, fm, record=(200, 600))
        r = {
            "1A_y200": a[200], "1A_ok": bool(a[200]) and all(p in PRIMES for p in a[200]),
            "1B_y700": a[700], "1B_ok": list(a[700]) == [17],
            "1C_y600": c[600], "1C_ok": list(c[600]) == [13, 17],
            "trace_1A": a, "trace_1C": c,
        }
        res[fm] = r
        print(fm, {k: r[k] for k in ("1A_y200", "1A_ok", "1B_y700", "1B_ok", "1C_y600", "1C_ok")})
        print("   1A の途中:", a[100], "| 400年:", a[400])
    json.dump(res, open("results/yoshimura.json", "w"), ensure_ascii=False, indent=1, default=str)
