"""1b の P(17)-P(13) と P(13)-P(11) を、初期配置（1本）単位のブートストラップで幅をつける。"""
import json
import sys

import numpy as np

from sim_predator import run_many

L, runs = int(sys.argv[1]), int(sys.argv[2])
_, fY, _, _, _ = run_many(L, 5.0, True, runs, 1000, 3000, 2, 50)
cnt = np.stack([np.bincount(r, minlength=51) for r in fY])  # 本ごとの Y の数
rng = np.random.default_rng(0)
B = 2000
out = {}
for a, b in ((17, 13), (13, 11), (17, 19)):
    d = []
    for _ in range(B):
        idx = rng.integers(0, runs, runs)
        c = cnt[idx].sum(0)
        d.append((c[a] - c[b]) / c.sum())
    d = np.array(d)
    full = (cnt[:, a].sum() - cnt[:, b].sum()) / cnt.sum()
    out[f"P{a}-P{b}"] = {"est": round(float(full), 4), "ci95": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]}
print(L, runs, out)
json.dump(out, open(f"results/boot_L{L}_n{runs}.json", "w"))
