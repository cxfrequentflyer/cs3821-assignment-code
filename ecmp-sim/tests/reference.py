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


def drill(w, picks, n, by="bytes", memory=0, engines=1, keys=None, speeds=None):
    """DChoice with every option, one trial: DRILL(d, memory) on `engines`
    forwarding engines that decide in rounds from a shared snapshot.

    The candidates of flow j are its sampled paths followed by whatever its
    engine remembers. It goes to the candidate with the lowest utilization in
    the snapshot (ties: the earlier candidate, or the smaller key if keys are
    given). The engine then remembers the chosen path followed by the other
    distinct candidates from least to most loaded, up to `memory` of them.
    """
    inv = [1.0] * n if speeds is None else [1.0 / s for s in speeds]
    util = [0.0] * n
    mem = [[] for _ in range(engines)]
    snapshot = list(util)
    paths = []
    for j in range(len(w)):
        e = j % engines
        if e == 0:
            snapshot = list(util)
        cands = [int(c) for c in picks[j]] + mem[e]
        best = 0
        for i in range(1, len(cands)):
            a, b = snapshot[cands[i]], snapshot[cands[best]]
            if a < b or (a == b and keys is not None and keys[j][i] < keys[j][best]):
                best = i
        p = cands[best]
        paths.append(p)
        util[p] += (w[j] if by == "bytes" else 1.0) * inv[p]
        if memory:
            others = []
            for c in cands:
                if c != p and c not in others:
                    others.append(c)
            others.sort(key=lambda c: snapshot[c])          # stable: ties keep order
            mem[e] = ([p] + others)[:memory]
    return paths


def greedy_engines(w, n, engines=1, keys=None, speeds=None):
    """Greedy is DRILL with every path as a candidate and no memory."""
    return drill(w, [list(range(n))] * len(w), n, engines=engines, keys=keys, speeds=speeds)


def round_robin(m, n, starts):
    engines = len(starts)
    return [(int(starts[j % engines]) + j // engines) % n for j in range(m)]


def chunk_flows(w, c):
    """Chunks in 'flow' order: (size, flow, rank) for every chunk."""
    out = []
    for i, size in enumerate(w):
        rank = 0
        while size > c * (1 + 1e-9):
            out.append((c, i, rank))
            size -= c
            rank += 1
        out.append((size, i, rank))
    return out
