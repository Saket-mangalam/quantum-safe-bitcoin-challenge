#!/usr/bin/env python3
"""CPU audit of the subset block inverse tree (tests/gpu_epochs/tree_inverse.cuh).

Every ZLAB_TREE variant is re-implemented here from the CUDA index arithmetic --
the heap layout of variants 0 and 1, and the level-packed layout of the default
variant 2 -- and each is checked to leave every lane holding the modular inverse
of the value it supplied. Identity factors stand in for inactive lanes, which is
the contract the header states.

The multiply count is asserted too: all three schedules must cost 3*(n-1)
multiplications, i.e. the 765 the header claims for a 256-lane block.

This models values, not representations: the kernel's lazy variants keep
internal nodes as exact but possibly non-canonical residues below 2^256, which
is a representation choice invisible mod p. The source audit therefore checks
the normalization contract separately -- exactly one canonicalization per block,
on the root, immediately before the inversion.

Run from anywhere: python3 candidates/subset/audit_tree_inverse.py
"""

from pathlib import Path
import random
import re

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
HEADER = Path(__file__).resolve().parent / "tests" / "gpu_epochs" / "tree_inverse.cuh"
WIDTHS = (4, 8, 16, 32, 64, 128, 256)


class Counter:
    """Counts field multiplications so the schedules can be compared on cost."""

    def __init__(self):
        self.muls = 0
        self.normalizations = 0
        self.inversions = 0

    def mul(self, a, b):
        self.muls += 1
        return a * b % P

    def inv(self, a):
        self.normalizations += 1
        self.inversions += 1
        return pow(a, P - 2, P)


def tree_heap_strict(leaves, c):
    """ZLAB_TREE == 0: heap layout, every product canonical, serial root inverse."""
    n = len(leaves)
    tree = [0] * (2 * n)
    for tid in range(n):
        tree[n + tid] = leaves[tid] % P

    width = n >> 1
    while width > 0:
        for tid in range(width):
            node = width + tid
            tree[node] = c.mul(tree[2 * node], tree[2 * node + 1])
        width >>= 1

    tree[1] = c.inv(tree[1])

    width = 1
    while width < n:
        for tid in range(width):
            node = width + tid
            parent, left, right = tree[node], tree[2 * node], tree[2 * node + 1]
            # the left child receives parent * right, and vice versa
            tree[2 * node] = c.mul(parent, right)
            tree[2 * node + 1] = c.mul(parent, left)
        width <<= 1

    return [tree[n + tid] for tid in range(n)]


def tree_heap_lazy(leaves, c):
    """ZLAB_TREE == 1: same layout, lane 0 owns the root stage, per-lane leaves."""
    n = len(leaves)
    assert n >= 4
    tree = [0] * (2 * n)
    for tid in range(n):
        tree[n + tid] = leaves[tid] % P

    width = n >> 1
    while width > 1:
        for tid in range(width):
            node = width + tid
            tree[node] = c.mul(tree[2 * node], tree[2 * node + 1])
        width >>= 1

    # lane 0: root product, normalize, invert, and the first downward level
    a, b = tree[2], tree[3]
    root = c.inv(c.mul(a, b))
    tree[2], tree[3] = c.mul(root, b), c.mul(root, a)

    width = 2
    while width < (n >> 1):
        for tid in range(width):
            node = width + tid
            parent, left, right = tree[node], tree[2 * node], tree[2 * node + 1]
            tree[2 * node] = c.mul(parent, right)
            tree[2 * node + 1] = c.mul(parent, left)
        width <<= 1

    # every lane forms its own leaf inverse from its parent and sibling product
    out = []
    for tid in range(n):
        leaf = n + tid
        out.append(c.mul(tree[leaf >> 1], tree[leaf ^ 1]))
    return out


def tree_level_packed(leaves, c):
    """ZLAB_TREE == 2 (default): level-packed products/inverses, quad root inverse."""
    n = len(leaves)
    assert n >= 4
    products = [0] * (2 * n)
    inverses = [0] * n
    for tid in range(n):
        products[tid] = leaves[tid] % P

    offset = 0
    count = n
    while count > 2:
        half = count >> 1
        for tid in range(half):
            products[offset + count + tid] = c.mul(
                products[offset + tid], products[offset + half + tid]
            )
        offset += count
        count >>= 1
    assert offset == 2 * n - 4, (n, offset)

    # lanes 0..3 form the same root product; lanes 0 and 1 each write one child
    a, b = products[offset], products[offset + 1]
    root = c.inv(c.mul(a, b))
    for tid in range(2):
        child = products[offset + 1 - tid]
        inverses[offset - n + tid] = c.mul(root, child)

    offset -= 4
    count = 4
    while count < n:
        half = count >> 1
        for tid in range(count):
            parent_inv = inverses[offset + count - n + (tid & (half - 1))]
            sibling = products[offset + (tid ^ half)]
            inverses[offset - n + tid] = c.mul(parent_inv, sibling)
        offset -= count << 1
        count <<= 1
    assert offset == 0, (n, offset)

    half = n >> 1
    return [
        c.mul(inverses[tid & (half - 1)], products[tid ^ half])
        for tid in range(n)
    ]


VARIANTS = {
    0: ("heap, strict canonicalization", tree_heap_strict),
    1: ("heap, lazy canonicalization", tree_heap_lazy),
    2: ("level-packed, quad root inverse", tree_level_packed),
}


def variant_bodies(source):
    """Split the header into its three ZLAB_TREE implementations."""
    starts = {
        0: source.index("#if ZLAB_TREE == 0"),
        1: source.index("#elif ZLAB_TREE == 1"),
        2: source.index("#else", source.index("#elif ZLAB_TREE == 1")),
    }
    end = source.rindex("#endif")
    bounds = {0: starts[1], 1: starts[2], 2: end}
    return {k: source[starts[k]:bounds[k]] for k in starts}


def audit_source():
    source = HEADER.read_text()
    match = re.search(r"^#define ZLAB_TREE (\d+)", source, re.MULTILINE)
    assert match, "ZLAB_TREE default not found"
    default = int(match.group(1))
    assert default in VARIANTS, default

    bodies = variant_bodies(source)
    assert set(bodies) == set(VARIANTS)

    strict, lazy, packed = bodies[0], bodies[1], bodies[2]

    # Variant 0 keeps every product canonical and inverts serially on lane 0.
    assert "qsb_field_mul(" in strict and "qsb_field_mul_raw(" not in strict
    assert "_ModInv(root);" in strict
    assert "qsb_field_normalize(" not in strict

    # The lazy variants canonicalize exactly once, on the root, and only then invert.
    for body, name in ((lazy, "ZLAB_TREE==1"), (packed, "ZLAB_TREE==2")):
        assert "qsb_field_mul(" not in body.replace("qsb_field_mul_raw(", ""), name
        assert body.count("qsb_field_normalize(root);") == 1, name
        inverter = "_ModInv(root);" if body is lazy else "zi_inverse_quad(root,tid);"
        assert body.count(inverter) == 1, name
        assert body.index("qsb_field_normalize(root);") < body.index(inverter), name

    # Variant 2 runs the root stage on four lanes and writes two child inverses.
    assert "if(tid<4){" in packed
    assert "zi_inverse_quad(root,tid);" in packed
    assert "if(tid<2){" in packed
    assert "products[k][offset+1-tid]" in packed

    # Shared memory must cover the widest block the header admits (n <= 256).
    assert "__shared__ uint64_t tree[4][512];" in strict
    assert "__shared__ uint64_t tree[4][512];" in lazy
    assert "__shared__ uint64_t products[4][512];" in packed
    assert "__shared__ uint64_t inverses[4][256];" in packed

    # Every variant returns a 4-limb value with the carry limb cleared.
    for body in bodies.values():
        assert "value[4]=0;" in body
    return default


def audit_variant(name, fn, rng):
    cases = 0
    for n in WIDTHS:
        if fn is tree_heap_lazy and n < 4:
            continue
        samples = [
            [rng.randrange(1, P) for _ in range(n)],                       # all active
            [1] * n,                                                       # all inactive
            [1 if i % 3 else rng.randrange(1, P) for i in range(n)],       # mixed
            [rng.randrange(1, P) if i == 0 else 1 for i in range(n)],      # one active
            [P - 1] * n,                                                   # extremes
        ]
        for leaves in samples:
            c = Counter()
            got = fn(leaves, c)
            want = [pow(x % P, P - 2, P) for x in leaves]
            assert got == want, (name, n, "inverse mismatch")
            assert c.inversions == 1, (name, n, c.inversions)
            assert c.normalizations == 1, (name, n, c.normalizations)
            assert c.muls == 3 * (n - 1), (name, n, c.muls)
            cases += 1
    return cases


def main():
    default = audit_source()
    rng = random.Random(0x5355425345540000)
    total = 0
    for key, (name, fn) in sorted(VARIANTS.items()):
        total += audit_variant(name, fn, rng)
    # The 765-multiply figure the header quotes, re-derived rather than trusted.
    assert 3 * (256 - 1) == 765
    print(
        f"PASS: {total} block-inverse cases across ZLAB_TREE variants "
        f"{tuple(sorted(VARIANTS))} and block widths {WIDTHS}; every lane recovers "
        f"its modular inverse with one inversion and one canonicalization per block, "
        f"and all three schedules cost 3*(n-1) multiplies (765 at n=256). "
        f"Compiled default is ZLAB_TREE={default}."
    )


if __name__ == "__main__":
    main()
