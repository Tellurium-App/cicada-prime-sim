"""検証2: 交雑説＋アリー効果（Tanaka et al. 2009 PNAS 106:8975、投稿前原稿の Model Specifications）の自前実装。

状態: N[a, b, age] = 遺伝子型 [a,b]（a<=b、10〜20年）・齢 age の幼虫の数（実数）
毎年: 幼虫は S で生き残る（式1）→ 齢が表現型（短い方が優性 = a）に達した群れは割合 E で成虫（式2）、
      残りの幼虫は消える → 同じ年に出た成虫は相手の割合 F で交配、子は 1/2×自分×F×C（式3）をメンデル遺伝で配る（式4）
アリー効果: 群れが Nc 未満なら 0（「群れ」の単位は allee= で選ぶ）

  allee="genotype"  遺伝子型×生まれ年ごと（幼虫の群れ）
  allee="phenotype" 表現型×生まれ年ごと（遺伝子型を合わせた群れ）
  allee="adult"     その年に出た成虫を表現型ごとに合計して Nc 未満なら繁殖なし
"""
import json
import sys

import numpy as np

CY = list(range(10, 21))
K = len(CY)
PRIMES = {11, 13, 17, 19}


def simulate(S, E, C=25.0, Nc=0.0, T=100, N0=1000.0, allee="genotype", init=None, eq4="mendel"):
    A = 21  # 齢 0..20
    N = np.zeros((K, K, A + 1))
    for i in range(K):
        N[i, i, 0] = N0 if init is None else init[i]
    phen = np.array([[min(CY[a], CY[b]) for b in range(K)] for a in range(K)])
    for t in range(1, T + 1):
        # 式1: 1年ぶん年をとり、S で生き残る
        N[:, :, 1:] = N[:, :, :-1] * S
        N[:, :, 0] = 0.0
        # 式2: 表現型の年齢に達した群れが成虫になる
        NA = np.zeros((K, K))
        for a in range(K):
            for b in range(a, K):
                age = phen[a, b]
                if N[a, b, age] > 0:
                    NA[a, b] = E * N[a, b, age]
                    N[a, b, age] = 0.0
        if allee == "adult" and Nc > 0:
            for p in CY:
                m = phen == p
                if NA[m].sum() < Nc:
                    NA[m] = 0.0
        tot = NA.sum()
        if tot > 0 and eq4 == "literal":
            # 掲載版の式4を字面どおり: N(i,m) = Σ_x Σ_z 1/2·N_A(i,x)·F(N_A(m,z))·C
            # （遺伝子型 [u,v] は、u を持つ親と v を持つ親の組ごとに 1/2·N_A·F·C をまるごと受け取る。1/4 に割らない）
            has = np.zeros((K, K))  # has[allele, k] = 遺伝子型 k(=a*K+b) の成虫で allele を持つものの数
            Fh = np.zeros(K)  # allele を持つ相手の割合の合計
            Ng = np.zeros(K)  # allele を持つ親の数の合計
            for a in range(K):
                for b in range(a, K):
                    if NA[a, b] > 0:
                        for al in {a, b}:
                            Ng[al] += NA[a, b]
                            Fh[al] += NA[a, b] / tot
            for u in range(K):
                for v in range(u, K):
                    val = 0.5 * Ng[u] * Fh[v] * C
                    if u != v:
                        val += 0.5 * Ng[v] * Fh[u] * C
                    N[u, v, 0] = val
        elif tot > 0:
            # 式3・4: ランダム交配。相手は数の割合 F で選ばれ、子の総数は 1/2×成虫×C
            # 配偶子の頻度（遺伝子型 [a,b] は a と b を 1/2 ずつ）
            g = np.zeros(K)
            for a in range(K):
                for b in range(a, K):
                    g[a] += NA[a, b] / 2
                    g[b] += NA[a, b] / 2
            g /= tot
            kids = 0.5 * tot * C
            for a in range(K):
                for b in range(a, K):
                    f = g[a] * g[b] * (1 if a == b else 2)
                    N[a, b, 0] = kids * f
        # アリー効果
        if Nc > 0 and allee == "genotype":
            N[(N > 0) & (N < Nc)] = 0.0
        elif Nc > 0 and allee == "phenotype":
            for p in CY:
                m = phen == p
                sub = N[m]  # (遺伝子型数, 齢)
                tot_age = sub.sum(axis=0)
                kill = (tot_age > 0) & (tot_age < Nc)
                sub[:, kill] = 0.0
                N[m] = sub
    by_phen = {p: float(N[phen == p].sum()) for p in CY}
    pure = {CY[i]: float(N[i, i].sum()) for i in range(K)}
    return by_phen, pure


def survivors(d, eps=1e-9):
    return sorted(p for p, v in d.items() if v > eps)


def spearman(order_paper, values):
    """order_paper: 論文の大きさの順（大きい順）。values: こちらの値。"""
    r_paper = {p: i for i, p in enumerate(order_paper)}
    ours = sorted(values, key=lambda p: -values[p])
    r_ours = {p: i for i, p in enumerate(ours)}
    n = len(order_paper)
    d2 = sum((r_paper[p] - r_ours[p]) ** 2 for p in order_paper)
    return 1 - 6 * d2 / (n * (n * n - 1)), ours


PAPER_1B = [16, 15, 18, 17, 19, 14, 20, 13, 12, 11, 10]


def fig1(allee):
    a_phen, a_pure = simulate(0.95, 0.5, Nc=100, T=100, allee=allee)
    b_phen, b_pure = simulate(0.95, 0.5, Nc=0, T=100, allee=allee)
    rho, ours = spearman(PAPER_1B, b_phen)
    return {
        "allee": allee,
        "1a_survivors_phen": survivors(a_phen),
        "1a_survivors_pure": survivors(a_pure),
        "1a_order": sorted(survivors(a_phen), key=lambda p: -a_phen[p]),
        "1a_values": {p: round(v, 1) for p, v in a_phen.items()},
        "1b_survivors": survivors(b_phen),
        "1b_order_ours": ours,
        "1b_spearman": round(rho, 3),
        "1b_values": {p: float(f"{v:.3g}") for p, v in b_phen.items()},
    }


def fig5(allee, E, S=0.944, T=1000):
    rows = []
    for Nc in range(0, 301, 10):
        phen, pure = simulate(S, E, Nc=Nc, T=T, allee=allee)
        rows.append({"Nc": Nc, "surv": survivors(phen), "surv_pure": survivors(pure)})
    single_prime = sum(1 for r in rows if 20 <= r["Nc"] <= 250 and len(r["surv"]) == 1 and r["surv"][0] in PRIMES)
    return rows, single_prime


if __name__ == "__main__":
    out = {}
    for allee in ("genotype", "phenotype", "adult"):
        r = fig1(allee)
        print(json.dumps(r, ensure_ascii=False))
        out[f"fig1_{allee}"] = r
        for E in (0.24, 0.26, 0.27, 0.30):
            rows, sp = fig5(allee, E)
            out[f"fig5_{allee}_E{E}"] = {"rows": rows, "single_prime_in_20_250": sp}
            line = " ".join(f"{r['Nc']}:{','.join(map(str, r['surv'])) or '-'}" for r in rows[::3])
            print(f"  fig5 E={E} 素数1本だけ(Nc20-250・24点中)={sp}  | {line}")
    json.dump(out, open("results/hybrid.json", "w"), ensure_ascii=False, indent=1)
