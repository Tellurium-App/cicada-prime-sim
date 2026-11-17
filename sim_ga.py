"""検証4: 突然変異の幅・受け入れ方・集団で、素数は残るか（CRITERIA.md 検証4）。

  python sim_ga.py check   # 表と sim_predator.fit_formula の突き合わせ
  python sim_ga.py 4a      # 1組の山登り（幅 × 受け入れ方）
  python sim_ga.py 4b      # 集団の GA
"""
import json
import math
import sys
import time

import numpy as np
from numba import njit, prange

from sim_predator import fit_formula, is_prime

YMAX = 200
ACCEPT = {"strict": 0, "geq": 1, "metro0.02": 2, "metro0.1": 3, "null": 4}
TEMP = {2: 0.02, 3: 0.1}
WIDTHS = [1, 3, 10, 30]


def tables():
    FX = np.zeros((YMAX + 1, YMAX + 1))
    FY = np.zeros((YMAX + 1, YMAX + 1))
    for X in range(2, YMAX + 1):
        for Y in range(2, YMAX + 1):
            g = math.gcd(X, Y)
            FX[X, Y] = 2 * g / Y - 1
            FY[X, Y] = 1 - 2 * g / X
    return FX, FY


PRIME = np.array([is_prime(n) for n in range(YMAX + 1)])


@njit
def accept(rule, new, old, T):
    if rule == 4:
        return True
    if rule == 0:
        return new > old + 1e-12
    if rule == 1:
        return new >= old - 1e-12
    if new >= old - 1e-12:
        return True
    return np.random.random() < math.exp((new - old) / T)


@njit(parallel=True)
def run_4a(FX, FY, prime, w, rule, T, starts, steps, seed):
    n = starts.shape[0]
    finalY = np.zeros(n, np.int64)
    prime_frac = np.zeros(n)
    yhist = np.zeros((n, YMAX + 1), np.int64)
    for r in prange(n):
        np.random.seed(seed + r)
        X, Y = starts[r, 0], starts[r, 1]
        pc = 0
        for s in range(steps):
            d = np.random.randint(1, w + 1)
            if np.random.random() < 0.5:
                d = -d
            if s % 2 == 0:
                Xn = X + d
                if 2 <= Xn < Y:
                    if accept(rule, FX[Xn, Y], FX[X, Y], T):
                        X = Xn
            else:
                Yn = Y + d
                if X < Yn <= YMAX:
                    if accept(rule, FY[X, Yn], FY[X, Y], T):
                        Y = Yn
            if s >= steps // 2:
                yhist[r, Y] += 1
                if prime[Y]:
                    pc += 1
        finalY[r] = Y
        prime_frac[r] = pc / (steps - steps // 2)
    return finalY, prime_frac, yhist


@njit
def mut(w):
    d = np.random.randint(1, w + 1)
    return -d if np.random.random() < 0.5 else d


@njit
def tourn(fit, pop, N, selection):
    a = np.random.randint(N)
    b = np.random.randint(N)
    if not selection:
        return pop[a]
    if fit[a] > fit[b] + 1e-12:
        return pop[a]
    if fit[b] > fit[a] + 1e-12:
        return pop[b]
    return pop[a] if np.random.random() < 0.5 else pop[b]


@njit(parallel=True)
def run_4b(FX, FY, prime, w, selection, N, gens, mu, runs, seed):
    prime_frac = np.zeros(runs)
    yhist = np.zeros((runs, YMAX + 1), np.int64)
    for r in prange(runs):
        np.random.seed(seed + r)
        X = np.random.randint(2, 51, N).astype(np.int64)
        Y = np.random.randint(2, 51, N).astype(np.int64)
        cx = np.zeros(YMAX + 1)
        cy = np.zeros(YMAX + 1)
        fx = np.zeros(N)
        fy = np.zeros(N)
        Xn = np.empty(N, np.int64)
        Yn = np.empty(N, np.int64)
        acc = 0.0
        for g in range(gens):
            cx[:] = 0.0
            cy[:] = 0.0
            for i in range(N):
                cx[X[i]] += 1.0
                cy[Y[i]] += 1.0
            ux = np.nonzero(cx)[0]
            uy = np.nonzero(cy)[0]
            vx = np.zeros(YMAX + 1)  # 値ごとの適応度（出てくる値だけ計算）
            vy = np.zeros(YMAX + 1)
            for a in ux:
                s = 0.0
                for b in uy:
                    s += cy[b] * FX[a, b]
                vx[a] = s / N
            for b in uy:
                s = 0.0
                for a in ux:
                    s += cx[a] * FY[a, b]
                vy[b] = s / N
            for i in range(N):
                fx[i] = vx[X[i]]
                fy[i] = vy[Y[i]]
            if g >= gens // 2:
                pc = 0
                for i in range(N):
                    yhist[r, Y[i]] += 1
                    if prime[Y[i]]:
                        pc += 1
                acc += pc / N
            for i in range(N):
                Xn[i] = tourn(fx, X, N, selection)
                Yn[i] = tourn(fy, Y, N, selection)
            for i in range(N):
                if np.random.random() < mu:
                    v = Xn[i] + mut(w)
                    if 2 <= v <= YMAX:
                        Xn[i] = v
                if np.random.random() < mu:
                    v = Yn[i] + mut(w)
                    if 2 <= v <= YMAX:
                        Yn[i] = v
            X[:] = Xn
            Y[:] = Yn
        prime_frac[r] = acc / (gens - gens // 2)
    return prime_frac, yhist


def rank_of(hist, y):
    order = np.argsort(-hist, kind="stable")
    return int(np.where(order == y)[0][0]) + 1


def summarize(hist, prime_frac):
    h = hist.sum(axis=0)
    top = [(int(y), round(float(h[y] / h.sum()), 4)) for y in np.argsort(-h, kind="stable")[:8]]
    return {
        "prime_frac": round(float(prime_frac.mean()), 4),
        "mode_Y": top[0][0],
        "top_Y": top,
        "rank13": rank_of(h, 13),
        "rank17": rank_of(h, 17),
        "mean_Y": round(float((h * np.arange(YMAX + 1)).sum() / h.sum()), 1),
    }


def check():
    FX, FY = tables()
    bad = 0
    for X in range(2, 80):
        for Y in range(2, 80):
            a = fit_formula(X, Y, True)
            if abs(a[0] - FX[X, Y]) > 1e-12 or abs(a[1] - FY[X, Y]) > 1e-12:
                bad += 1
    print("表と fit_formula の不一致:", bad)


def main_4a():
    FX, FY = tables()
    rng = np.random.default_rng(20261117)
    starts = []
    while len(starts) < 1000:
        X, Y = int(rng.integers(2, 51)), int(rng.integers(2, 51))
        if X < Y:
            starts.append((X, Y))
    starts = np.array(starts, np.int64)
    out = {}
    for w in WIDTHS:
        for name, rule in ACCEPT.items():
            t0 = time.time()
            fY, pf, hist = run_4a(FX, FY, PRIME, w, rule, TEMP.get(rule, 1.0), starts, 20000, 7000)
            res = summarize(hist, pf)
            res["final_prime"] = int(PRIME[fY].sum())
            fh = np.bincount(fY, minlength=YMAX + 1)
            res["final_top_Y"] = [(int(y), int(fh[y])) for y in np.argsort(-fh, kind="stable")[:8]]
            res["sec"] = round(time.time() - t0, 1)
            out[f"w{w}_{name}"] = res
            print(f"w={w:2d} {name:9s} prime_frac={res['prime_frac']:.3f} final_prime={res['final_prime']:4d} "
                  f"mode={res['mode_Y']:3d} r13={res['rank13']:3d} r17={res['rank17']:3d} top={res['top_Y'][:4]}")
    json.dump(out, open("results/4a.json", "w"), ensure_ascii=False, indent=1)


def main_4b():
    FX, FY = tables()
    out = {}
    for w in WIDTHS:
        for name, sel in (("ga", True), ("null", False)):
            t0 = time.time()
            pf, hist = run_4b(FX, FY, PRIME, w, sel, 200, 3000, 0.1, 200, 9000)
            res = summarize(hist, pf)
            res["sec"] = round(time.time() - t0, 1)
            out[f"w{w}_{name}"] = res
            print(f"w={w:2d} {name:4s} prime_frac={res['prime_frac']:.3f} mode={res['mode_Y']:3d} "
                  f"r13={res['rank13']:3d} r17={res['rank17']:3d} meanY={res['mean_Y']} top={res['top_Y'][:4]} ({res['sec']}s)")
    json.dump(out, open("results/4b.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] != "prey_only":
        {"check": check, "4a": main_4a, "4b": main_4b}[sys.argv[1]]()


# ---- 後から足した確認（4b の見立て: 捕食者のばらつきが素数を選ばせるか）----
@njit(parallel=True)
def run_prey_only(FY, prime, w, Xfixed, N, gens, mu, runs, seed):
    """捕食者を Xfixed に固定し、獲物だけ 4b と同じ GA で進化させる。"""
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
    out = np.zeros(runs)
    for r in prange(runs):
        np.random.seed(seed + r)
        Y = np.random.randint(2, 51, N).astype(np.int64)
        Yn = np.empty(N, np.int64)
        fy = np.empty(N)
        acc = 0.0
        for g in range(gens):
            for i in range(N):
                fy[i] = vy[Y[i]]
            if g >= gens // 2:
                pc = 0
                for i in range(N):
                    if prime[Y[i]]:
                        pc += 1
                acc += pc / N
            for i in range(N):
                Yn[i] = tourn(fy, Y, N, True)
                if np.random.random() < mu:
                    v = Yn[i] + mut(w)
                    if 2 <= v <= YMAX:
                        Yn[i] = v
            Y[:] = Yn
        out[r] = acc / (gens - gens // 2)
    return out


def main_prey_only():
    FX, FY = tables()
    sets = {
        "X=7のみ": np.full(200, 7, np.int64),
        "X=6のみ": np.full(200, 6, np.int64),
        "X=2..10一様": np.repeat(np.arange(2, 11), 22).astype(np.int64),
        "X=2..30一様": np.repeat(np.arange(2, 31), 7).astype(np.int64),
    }
    out = {}
    for name, xs in sets.items():
        for w in (3, 10):
            pf = run_prey_only(FY, PRIME, w, xs, 200, 3000, 0.1, 200, 11000)
            out[f"{name}_w{w}"] = round(float(pf.mean()), 4)
            print(name, "w=", w, "素数割合", out[f"{name}_w{w}"])
    json.dump(out, open("results/4b_prey_only.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__" and sys.argv[1] == "prey_only":
    main_prey_only()
