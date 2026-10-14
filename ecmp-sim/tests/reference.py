"""Deliberately slow, obviously-correct reference versions of the policies.

These work on ONE trial with plain Python lists. Ideally they are written by
someone who did not write ecmpsim/policies.py, straight from the policy
descriptions. Fed the same random draws, the fast versions must produce
exactly the same paths (tests/test_policies.py checks this).
"""


def ecmp(w, draws, n):
    return [int(p) for p in draws]


def dchoice(w, draws, n, by="bytes"):
    load = [0.0] * n
    paths = []
    for j, cand in enumerate(draws):
        best = int(cand[0])
        for c in cand[1:]:
            if load[c] < load[best]:          # strictly less: ties keep the earlier sample
                best = int(c)
        paths.append(best)
        load[best] += w[j] if by == "bytes" else 1.0
    return paths


def greedy(w, n):
    load = [0.0] * n
    paths = []
    for size in w:
        best = min(range(n), key=lambda b: load[b])     # lowest index among ties
        paths.append(best)
        load[best] += size
    return paths


def lpt(w, n):
    order = sorted(range(len(w)), key=lambda i: -w[i])  # stable: ties keep arrival order
    placed = greedy([w[i] for i in order], n)
    paths = [0] * len(w)
    for i, p in zip(order, placed):
        paths[i] = p
    return paths
