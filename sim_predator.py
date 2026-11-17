"""検証1: 天敵説（Markus, Schulz & Goles 2002 ScienceAsia 28:199）の自前実装。

1a  一様な世界: 捕食者 X・獲物 Y の突然変異＋選択（条件 K: 1<X<Y）
1b  空間のある世界: 格子のセルオートマトン（近傍9マスで最も適応度の高い値を写す）

使い方:
  python sim_predator.py check            # 式と年を数える版の突き合わせ
  python sim_predator.py 1a               # 一様な世界
  python sim_predator.py 1b L p runs [nodiv] [seed]
"""
import json
import math
import sys
import time

import numpy as np
from numba import njit, prange

import os
TIE = np.array([int(os.environ.get("CICADA_TIE", "0"))])  # 同点の扱い（0=既定）。JIT はこの値を定数として焼くので cache は使わない


def is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


# ---------------- 1a: 一様な世界 ----------------

def fit_formula(X, Y, divide=True):
    g = math.gcd(X, Y)
    if divide:
        return 2 * g / Y - 1, 1 - 2 * g / X   # (Fx, Fy) 論文の式
    # 世代数で割らない: XY 年の合計そのもの
    return 2 * g - Y, X - 2 * g


def fit_count(X, Y, divide=True):
    """年を1年ずつ数える素朴な版（t=0..XY-1、全員 t=0 に出る）。"""
    sx = sy = nx = ny = 0
    for t in range(X * Y):
        px, py = t % X == 0, t % Y == 0
        if px:
            nx += 1
            sx += 1 if py else -1
        if py:
            ny += 1
            sy += -1 if px else 1
    if divide:
        return sx / nx, sy / ny
    return sx, sy


def evolve_1a(rng, X, Y, steps, ymax=200, condK=True, divide=True):
    """X と Y を交互に突然変異（±1〜±3）。厳密に上回ったら入れ替え。"""
    for s in range(steps):
        d = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
        if s % 2 == 0:
            Xn = X + d
            if Xn < 2 or (condK and Xn >= Y) or Xn > ymax:
                continue
            if fit_formula(Xn, Y, divide)[0] > fit_formula(X, Y, divide)[0]:
                X = Xn
        else:
            Yn = Y + d
            if Yn < 2 or (condK and Yn <= X) or Yn > ymax:
                continue
            if fit_formula(X, Yn, divide)[1] > fit_formula(X, Y, divide)[1]:
                Y = Yn
    return X, Y


def is_locked(X, Y, ymax, condK, divide, reach=3):
    """これ以上どの1手（±1〜±reach）でも入れ替わらないか。"""
    fx, fy = fit_formula(X, Y, divide)
    for d in list(range(-reach, 0)) + list(range(1, reach + 1)):
        Xn, Yn = X + d, Y + d
        if 2 <= Xn <= ymax and not (condK and Xn >= Y):
            if fit_formula(Xn, Y, divide)[0] > fx:
                return False
        if 2 <= Yn <= ymax and not (condK and Yn <= X):
            if fit_formula(X, Yn, divide)[1] > fy:
                return False
    return True


def run_1a():
    out = {}
    for name, condK, divide in [("base", True, True), ("no_divide", True, False), ("no_condK", False, True)]:
        rng = np.random.default_rng(20261117)
        finals = []
        for _ in range(1000):
            while True:
                X, Y = int(rng.integers(2, 51)), int(rng.integers(2, 51))
                if X < Y:
                    break
            X, Y = evolve_1a(rng, X, Y, 20000, condK=condK, divide=divide)
            finals.append((X, Y, is_locked(X, Y, 200, condK, divide)))
        ys = [y for _, y, _ in finals]
        prime = sum(is_prime(y) for y in ys)
        locked = sum(l for *_, l in finals)
        hist = {}
        for y in ys:
            hist[y] = hist.get(y, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:10]
        xs_eq = sum(1 for x, y, _ in finals if x == y)
        out[name] = {"prime_Y": prime, "locked": locked, "n": len(finals), "top_Y": top, "X_eq_Y": xs_eq}
        print(name, out[name])
    return out


# ---------------- 1b: セルオートマトン ----------------

@njit(cache=True)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def p_none(ds, n):
    """k=1,2,... のうち、どの d も k を割り切らない割合（包除原理）。ds[:n] は互いに割り切らない集合。"""
    total = 0.0
    for mask in range(1 << n):
        l = 1
        bits = 0
        over = False
        for i in range(n):
            if mask >> i & 1:
                bits += 1
                g = gcd(l, ds[i])
                l = l // g * ds[i]
                if l > 10 ** 15:
                    over = True
                    break
        if over:
            continue
        if bits % 2 == 0:
            total += 1.0 / l
        else:
            total -= 1.0 / l
    return total


@njit(cache=True)
def fitness(me, others, p, divide):
    """me: 自分の周期。others: 近傍9マスの相手の周期。
    自分が出る年あたりの平均（divide）: 相手の数 ν の期待値 − p × P(ν=0)。
    捕食者はこれ、獲物は符号を反転して使う。"""
    s = 0.0
    ds = np.empty(9, np.int64)
    n = 0
    for j in range(9):
        g = gcd(me, others[j])
        s += g / others[j]
        d = others[j] // g
        if d == 1:
            n = -1
        if n >= 0:
            dup = False
            for k in range(n):
                if ds[k] == d:
                    dup = True
            if not dup:
                ds[n] = d
                n += 1
    if n < 0:
        p0 = 0.0
    else:
        # 他の d の倍数になっている d は条件として冗長なので落とす
        keep = np.empty(9, np.int64)
        m = 0
        for a in range(n):
            red = False
            for b in range(n):
                if a != b and ds[a] % ds[b] == 0 and (ds[a] != ds[b] or b < a):
                    red = True
            if not red:
                keep[m] = ds[a]
                m += 1
        p0 = p_none(keep, m)
    f = s - p * p0
    if not divide:
        f = f / me  # 1年あたりに直す（世代数で割らない＝出る回数の多さも得点になる）
    return f


@njit(cache=True)
def neigh(L, c, k):
    r, q = c // L, c % L
    dr = k // 3 - 1
    dq = k % 3 - 1
    if k == 4:
        return c
    return ((r + dr) % L) * L + (q + dq) % L


@njit
def step(L, X, Y, p, divide, Xn, Yn, Fx, Fy):
    N = L * L
    buf = np.empty(9, np.int64)
    for c in range(N):
        for k in range(9):
            buf[k] = Y[neigh(L, c, k)]
        Fx[c] = fitness(X[c], buf, p, divide)
        for k in range(9):
            buf[k] = X[neigh(L, c, k)]
        Fy[c] = -fitness(Y[c], buf, p, divide)
    for c in range(N):
        if TIE[0] == 0:
            # 自分を残す → 番号の小さい近傍
            bx, by = c, c
            for k in range(9):
                j = neigh(L, c, k)
                if Fx[j] > Fx[bx] + 1e-12:
                    bx = j
                if Fy[j] > Fy[by] + 1e-12:
                    by = j
        else:
            # TIE=1: 同点の中から無作為（自分も同じ扱い）／TIE=2: 自分を残す → 番号の大きい近傍
            mx, my = -1e300, -1e300
            for k in range(9):
                j = neigh(L, c, k)
                mx = max(mx, Fx[j])
                my = max(my, Fy[j])
            bx, by = -1, -1
            nx_, ny_ = 0, 0
            for k in range(9):
                j = neigh(L, c, k)
                if Fx[j] >= mx - 1e-12:
                    nx_ += 1
                    if TIE[0] == 1:
                        if np.random.random() * nx_ < 1.0:
                            bx = j
                    else:
                        bx = j
                if Fy[j] >= my - 1e-12:
                    ny_ += 1
                    if TIE[0] == 1:
                        if np.random.random() * ny_ < 1.0:
                            by = j
                    else:
                        by = j
            if TIE[0] == 2:
                if Fx[c] >= mx - 1e-12:
                    bx = c
                if Fy[c] >= my - 1e-12:
                    by = c
        Xn[c] = X[bx]
        Yn[c] = Y[by]


@njit
def run_one(L, X, Y, p, divide, maxsteps):
    N = L * L
    H = 64
    histX = np.zeros((H, N), np.int64)
    histY = np.zeros((H, N), np.int64)
    Xn = np.empty(N, np.int64)
    Yn = np.empty(N, np.int64)
    Fx = np.empty(N)
    Fy = np.empty(N)
    for s in range(maxsteps):
        histX[s % H] = X
        histY[s % H] = Y
        step(L, X, Y, p, divide, Xn, Yn, Fx, Fy)
        X[:] = Xn
        Y[:] = Yn
        lim = min(s + 1, H)
        for b in range(lim):
            same = True
            for c in range(N):
                if histX[b, c] != X[c] or histY[b, c] != Y[c]:
                    same = False
                    break
            if same:
                period = (s % H - b) % H + 1
                return s + 1, period
    return maxsteps, -1


@njit(parallel=True)
def run_many(L, p, divide, runs, seed, maxsteps, lo, hi):
    N = L * L
    finalY = np.zeros((runs, N), np.int64)
    finalX = np.zeros((runs, N), np.int64)
    initY = np.zeros((runs, N), np.int64)
    steps = np.zeros(runs, np.int64)
    periods = np.zeros(runs, np.int64)
    for r in prange(runs):
        np.random.seed(seed + r)
        X = np.random.randint(lo, hi + 1, N).astype(np.int64)
        Y = np.random.randint(lo, hi + 1, N).astype(np.int64)
        initY[r] = Y
        st, per = run_one(L, X, Y, p, divide, maxsteps)
        finalY[r] = Y
        finalX[r] = X
        steps[r] = st
        periods[r] = per
    return finalX, finalY, initY, steps, periods


def brute_fitness(me, others, p, divide):
    """年を数える版（t=0..lcm-1）。fitness() の検算用。"""
    l = me
    for o in others:
        l = l * o // math.gcd(l, o)
    tot = 0
    gens = 0
    for t in range(0, l, me):
        nu = sum(1 for o in others if t % o == 0)
        gens += 1
        tot += nu if nu > 0 else -p
    # t=0 は全員出ているので ν>0。lcm 周期の平均は時間平均と同じ
    f = tot / gens
    return f if divide else f / me


def check():
    bad = 0
    for X in range(2, 61):
        for Y in range(X + 1, 61):
            for dv in (True, False):
                a, b = fit_formula(X, Y, dv), fit_count(X, Y, dv)
                if abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9:
                    bad += 1
    print("1a 式と数え上げの不一致:", bad, "/", 59 * 58 // 2 * 2)
    rng = np.random.default_rng(1)
    bad = 0
    n = 0
    for _ in range(400):
        me = int(rng.integers(2, 21))
        others = rng.integers(2, 13, 9)
        l = me
        for o in others:
            l = l * int(o) // math.gcd(l, int(o))
        if l > 3_000_000:
            continue
        for p, dv in ((5, True), (0, True), (5, False)):
            n += 1
            a = fitness(me, others.astype(np.int64), p, dv)
            b = brute_fitness(me, [int(o) for o in others], p, dv)
            if abs(a - b) > 1e-9:
                bad += 1
                print("  ずれ", me, list(others), p, dv, a, b)
    print("1b 式と数え上げの不一致:", bad, "/", n)


def summarize_1b(finalY, initY, steps, periods, lo, hi):
    vals = finalY.ravel()
    cnt = np.bincount(vals, minlength=hi + 1)[lo:hi + 1]
    P = cnt / cnt.sum()
    ys = np.arange(lo, hi + 1)
    primes = np.array([is_prime(int(y)) for y in ys])
    order = np.argsort(-P)
    icnt = np.bincount(initY.ravel(), minlength=hi + 1)[lo:hi + 1]
    # 1 本ごとに「最終状態で一番多い Y」
    modal = [int(np.bincount(r).argmax()) for r in finalY]
    mcnt = np.bincount(modal, minlength=hi + 1)[lo:hi + 1]
    homog = int(sum(len(set(r.tolist())) == 1 for r in finalY))
    return {
        "P_top10": [(int(ys[i]), round(float(P[i]), 4)) for i in order[:10]],
        "argmax_Y": int(ys[order[0]]),
        "prime_share": round(float(P[primes].sum()), 4),
        "prime_share_initial": round(float((icnt[primes]).sum() / icnt.sum()), 4),
        "modal_top10": [(int(ys[i]), int(mcnt[i])) for i in np.argsort(-mcnt)[:10]],
        "P_all": {int(y): round(float(v), 5) for y, v in zip(ys, P)},
        "unconverged": int((periods < 0).sum()),
        "period_counts": {int(k): int(v) for k, v in zip(*np.unique(periods, return_counts=True))},
        "steps_median": float(np.median(steps)),
        "homogeneous_runs": homog,
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "1a":
        res = run_1a()
        json.dump(res, open("results/1a.json", "w"), ensure_ascii=False, indent=1)
    elif mode == "1b":
        L, p, runs = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        divide = not (len(sys.argv) > 5 and sys.argv[5] == "nodiv")
        seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
        lo = 2
        hi = int(sys.argv[7]) if len(sys.argv) > 7 else 50
        t0 = time.time()
        fX, fY, iY, st, per = run_many(L, p, divide, runs, seed, 3000, lo, hi)
        res = summarize_1b(fY, iY, st, per, lo, hi)
        res.update({"tie": int(TIE[0]), "hi": hi, "L": L, "p": p, "runs": runs, "divide": divide, "seed": seed, "sec": round(time.time() - t0, 1)})
        tag = ("" if TIE[0] == 0 else f"tie{TIE[0]}_") + f"1b_L{L}_p{int(p)}_{'div' if divide else 'nodiv'}_n{runs}_hi{hi}"
        json.dump(res, open(f"results/{tag}.json", "w"), ensure_ascii=False, indent=1)
        print(tag, {k: res[k] for k in ("argmax_Y", "prime_share", "prime_share_initial", "P_top10", "unconverged", "homogeneous_runs", "sec")})
