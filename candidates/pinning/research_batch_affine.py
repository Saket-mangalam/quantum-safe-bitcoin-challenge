#!/usr/bin/env python3
"""Batch-affine summation of the 15 fixed-base points: algebra and cost.

The kernel sums the 15 signed-digit table points with a deferred-Y XYZZ chain,
documented in pinning.cu as 3M+2S + 12*(7M+2S) + 8M+2S = 95M+28S.

Projective coordinates exist to avoid inversions, but an affine addition is only
2M+1S once 1/(x2-x1) is known, and the additions within one level of a binary
summation tree are independent, so their inverses batch by Montgomery's trick --
3(k-1) multiplies for k values -- on top of the block-wide inverse the kernel
already performs. Batch-affine addition beating projective for independent adds
is the same result multi-scalar-multiplication implementations rely on.

This script establishes two things on CPU:
  1. the batch-affine tree, and the batch-2 hybrid the kernel prototypes,
     both produce exactly the point the reference sum produces;
  2. the multiply/squaring counts, block-wide inversion rounds and live state
     of each structure.

It cannot establish that any of them is faster on an RTX 4090: extra inversion
rounds are barriers, and live state that will not fit in registers has to go to
shared memory and costs occupancy. That is what candidates/pinning/ab.sh is for.

Run: python3 candidates/pinning/research_batch_affine.py
"""

import random

P = P_MOD = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

CHUNKS = 15
BLOCK_INVERSE_MULS = 3     # 3(n-1)/n multiplies per lane for the block product tree
S_RATIO = 0.8              # squaring cost relative to a multiply
FIELD_BYTES = 32


def add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    if p[0] == q[0]:
        if (p[1] + q[1]) % P == 0:
            return None
        lam = (3 * p[0] * p[0]) * pow(2 * p[1], P - 2, P) % P
    else:
        lam = (q[1] - p[1]) * pow(q[0] - p[0], P - 2, P) % P
    x = (lam * lam - p[0] - q[0]) % P
    return (x, (lam * (p[0] - x) - p[1]) % P)


def mul(k, p):
    r, acc = None, p
    while k:
        if k & 1:
            r = add(r, acc)
        acc = add(acc, acc)
        k >>= 1
    return r


def reference_sum(points):
    total = None
    for p in points:
        total = add(total, p)
    return total


class Cost:
    """Multiplies, squarings, block-wide inversion rounds, peak live elements."""

    def __init__(self, name):
        self.name = name
        self.M = self.S = self.rounds = self.live = 0
        self.table_bytes = 0

    @property
    def total(self):
        return self.M + S_RATIO * self.S

    def chain(self, k):
        """The kernel's deferred-Y chain over k affine points, k >= 2.

        Seed mmadd 3M+2S, then k-3 deferred intermediates at 7M+2S, then the
        resolving addition at 8M+2S. k=15 reproduces the documented 95M+28S.
        """
        assert k >= 2
        if k == 2:
            self.M += 3
            self.S += 2
        else:
            self.M += 3 + 7 * (k - 3) + 8
            self.S += 2 + 2 * (k - 3) + 2
        # the chain holds an XYZZ accumulator, one table point and the y anchor
        self.live = max(self.live, 4 + 2 + 1)

    def block_inverse(self):
        self.M += BLOCK_INVERSE_MULS
        self.rounds += 1


def batch_inverse(vals, cost):
    """Montgomery's trick over k values: 3(k-1) multiplies plus one block round."""
    k = len(vals)
    partial = [vals[0]]
    for v in vals[1:]:
        partial.append(partial[-1] * v % P)
        cost.M += 1
    inv = pow(partial[-1], P - 2, P)     # supplied by the block-wide tree
    cost.block_inverse()
    out = [0] * k
    for i in range(k - 1, 0, -1):
        out[i] = inv * partial[i - 1] % P
        inv = inv * vals[i] % P
        cost.M += 2
    out[0] = inv
    return out


def affine_add(p, q, inv, cost):
    lam = (q[1] - p[1]) * inv % P
    cost.M += 1
    x = (lam * lam - p[0] - q[0]) % P
    cost.S += 1
    y = (lam * (p[0] - x) - p[1]) % P
    cost.M += 1
    return (x, y)


def strategy_chain(points):
    """What the kernel does today."""
    cost = Cost("XYZZ chain (today)")
    cost.chain(len(points))
    cost.block_inverse()                      # the finish's shared denominator
    cost.table_bytes = len(points) * 64
    return cost, reference_sum(points)


def strategy_full_tree(points):
    """Every level batch-affine."""
    cost = Cost("full affine tree")
    level = list(points)
    while len(level) > 1:
        pairs = [(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        carry = [level[-1]] if len(level) % 2 else []
        dens = [(b[0] - a[0]) % P for a, b in pairs]
        # every point feeding this level stays live alongside its denominator
        cost.live = max(cost.live, 2 * len(level) + len(dens))
        invs = batch_inverse(dens, cost)
        level = [affine_add(a, b, inv, cost)
                 for (a, b), inv in zip(pairs, invs)] + carry
    cost.block_inverse()                      # the finish still needs one
    cost.table_bytes = len(points) * 64
    return cost, level[0]


def strategy_hybrid(points, batch):
    """One batch-affine level in batches of `batch` pairs, then the chain.

    Two passes over the table: the first reads x halves only to form the
    denominators, the second re-reads full records once the inverses exist, so
    the level-1 outputs are consumed by the chain one at a time and never all
    exist together.
    """
    cost = Cost(f"1 affine level, batch {batch}")
    pairs = [(points[i], points[i + 1]) for i in range(0, len(points) - 1, 2)]
    carry = [points[-1]] if len(points) % 2 else []

    sums = []
    for start in range(0, len(pairs), batch):
        group = pairs[start:start + batch]
        dens = [(b[0] - a[0]) % P for a, b in group]
        # across the barrier: this batch's denominators plus the chain state
        cost.live = max(cost.live, len(dens) + 4 + 2)
        invs = batch_inverse(dens, cost)
        sums.extend(affine_add(a, b, inv, cost)
                    for (a, b), inv in zip(group, invs))

    cost.chain(len(sums) + len(carry))
    cost.block_inverse()
    cost.table_bytes = len(points) * 64 + len(pairs) * 2 * 32
    return cost, reference_sum(sums + carry)


def xyzz_mm(p1, p2):
    """_PointAddXYZZ_mm: X3,ZZ3,ZZZ3 ordinary, Y3 deferred against p1's y.

    Returns (X3, Y3_stored, ZZ3, ZZZ3, anchor) with the invariant
    Y_actual = Y3_stored - anchor*ZZZ3, anchor = y of the FIRST point.
    """
    (x1, y1), (x2, y2) = p1, p2
    P = (x2 - x1) % P_MOD
    R = (y2 - y1) % P_MOD
    ZZ3 = P * P % P_MOD                       # PP
    ZZZ3 = ZZ3 * P % P_MOD                    # PPP
    Q = x1 * ZZ3 % P_MOD
    X3 = (R * R - ZZZ3 - 2 * Q) % P_MOD
    Y3 = (Q - X3) * R % P_MOD                 # deferred: omits y1*ZZZ3
    return (X3, Y3, ZZ3, ZZZ3, y1)


def xyzz_add(state, p2, defer):
    """_PointAddXYZZ with the affine-anchor convention of GPUMath.h.

    The accumulator's stored Y is deferred against `anchor`; adding the affine
    point p2 re-anchors on p2's y when `defer` is set, and resolves Y exactly
    when it is not.
    """
    X1, Y1, ZZ1, ZZZ1, anchor = state
    x2, y2 = p2
    U2 = x2 * ZZ1 % P_MOD
    S2 = (y2 + anchor) * ZZZ1 % P_MOD
    P = (U2 - X1) % P_MOD
    R = (S2 - Y1) % P_MOD
    PP = P * P % P_MOD
    PPP = PP * P % P_MOD
    V = U2 * PP % P_MOD
    ZZ3 = ZZ1 * PP % P_MOD
    X3 = (R * R + PPP - 2 * V) % P_MOD
    ZZZ3 = ZZZ1 * PPP % P_MOD
    core = (V - X3) * R % P_MOD
    if defer:
        return (X3, core, ZZ3, ZZZ3, y2)
    return (X3, (core - y2 * ZZZ3) % P_MOD, ZZ3, ZZZ3, None)


def xyzz_to_affine(state):
    X, Y, ZZ, ZZZ, anchor = state
    assert anchor is None, "the final addition must resolve Y"
    return (X * pow(ZZ, P_MOD - 2, P_MOD) % P_MOD,
            Y * pow(ZZZ, P_MOD - 2, P_MOD) % P_MOD)


def prototype_sequence(points):
    """Exactly what _FixedBaseBatchAffine2Scalar does, in order.

    Pairs (0,1)...(12,13) are summed in affine coordinates; the seven pair sums
    seed and feed the deferred-Y chain, and chunk 14 closes it with the
    resolving addition. This is the check that the anchor bookkeeping is right:
    seeding anchors on the FIRST point, every deferred step re-anchors on the
    point just added, and only the last addition resolves.
    """
    cost = Cost("prototype sequence")
    pairs = [(points[i], points[i + 1]) for i in range(0, 14, 2)]
    carry = points[14]

    sums = []
    for start in range(0, len(pairs), 2):      # two pairs per block inverse
        group = pairs[start:start + 2]
        dens = [(b[0] - a[0]) % P_MOD for a, b in group]
        invs = batch_inverse(dens, cost)
        sums.extend(affine_add(a, b, inv, cost)
                    for (a, b), inv in zip(group, invs))
    assert len(sums) == 7

    state = xyzz_mm(sums[0], sums[1])
    for s in sums[2:]:
        state = xyzz_add(state, s, defer=True)
    state = xyzz_add(state, carry, defer=False)
    return xyzz_to_affine(state)


def main():
    rng = random.Random(20260917)

    # The prototype's own sequence, including the deferred-Y anchor handling.
    seq_checked = 0
    for _ in range(50):
        pts = [mul(rng.randrange(1, N), G) for _ in range(CHUNKS)]
        assert prototype_sequence(pts) == reference_sum(pts), \
            "prototype sequence (pairs + deferred-Y chain) is wrong"
        seq_checked += 1
    print(f"Prototype sequence: pair sums + deferred-Y chain + resolving add "
          f"equals the reference sum on {seq_checked} point sets")

    strategies = [
        strategy_chain,
        lambda pts: strategy_hybrid(pts, 2),
        lambda pts: strategy_hybrid(pts, 4),
        lambda pts: strategy_hybrid(pts, 7),
        strategy_full_tree,
    ]

    checked = 0
    for _ in range(100):
        pts = [mul(rng.randrange(1, N), G) for _ in range(CHUNKS)]
        want = reference_sum(pts)
        for fn in strategies:
            cost, got = fn(pts)
            assert got == want, f"{cost.name} disagrees with the reference sum"
        checked += 1

    pts = [mul(rng.randrange(1, N), G) for _ in range(CHUNKS)]
    base, _ = strategy_chain(pts)
    assert base.M - BLOCK_INVERSE_MULS == 95 and base.S == 28, (base.M, base.S)

    print(f"Correctness: every strategy equals the reference 15-point sum on "
          f"{checked} random point sets")
    print(f"Baseline reproduces the documented 95M+28S chain cost\n")

    print(f"{'strategy':<26}{'cost':<24}{'rounds':<8}{'live':<6}"
          f"{'shared@128':<13}{'table B':<9}delta")
    print("-" * 94)
    for fn in strategies:
        cost, _ = fn(pts)
        delta = (cost.total / base.total - 1) * 100
        shared = cost.live * FIELD_BYTES * 128 / 1024
        print(f"{cost.name:<26}{f'{cost.M}M+{cost.S}S = {cost.total:.1f}':<24}"
              f"{cost.rounds:<8}{cost.live:<6}{shared:>7.1f} KiB  "
              f"{cost.table_bytes:<9}{delta:+6.1f}%")

    print("\n  cost    squarings counted at %.1f of a multiply" % S_RATIO)
    print("  rounds  block-wide inversions; each is a barrier the chain does not pay")
    print("  live    peak field elements (32 B) that must be held per lane")
    print("  shared  that live state as shared memory at 128 lanes per block")
    print("\nThe full tree is the cheapest in multiplies and the least implementable:")
    print("its intermediate points must all survive each level. The batch-2 hybrid")
    print("keeps the state crossing a barrier to two denominators, which is what")
    print("QSB_BATCH_AFFINE prototypes in pinning.cu.")


if __name__ == "__main__":
    main()
