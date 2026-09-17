#!/usr/bin/env python3
"""Audit the two-level 256x256 batch inversion of search-CTA roots."""

from pathlib import Path
import random

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
WIDTH = 256


def tree_up(values):
    assert len(values) == WIDTH
    products = [value % P for value in values] + [0] * (WIDTH - 1)
    offset = 0
    for count in (256, 128, 64, 32, 16, 8, 4, 2):
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = (
                products[offset + tid] * products[offset + half + tid] % P
            )
        offset += count
    assert offset == 510
    return products[:256], products[256:510], products[510]


def tree_down(leaves, internal, root_inverse):
    assert len(leaves) == WIDTH and len(internal) == 254
    products = list(leaves) + list(internal) + [None]
    inverses = [0] * 255
    inverses[254] = root_inverse % P
    offset = 508
    for count in (2, 4, 8, 16, 32, 64, 128):
        half = count // 2
        for tid in range(count):
            parent = offset + count - WIDTH + (tid & (half - 1))
            inverses[offset - WIDTH + tid] = (
                inverses[parent] * products[offset + (tid ^ half)] % P
            )
        offset -= 2 * count
    assert offset == 0
    return [
        inverses[tid & 127] * products[tid ^ 128] % P
        for tid in range(WIDTH)
    ]


def nested_inverse(values):
    """Model the two-level inversion, including the multi-block super level.

    qsb_invert_super_roots launches ceil(groups/WIDTH) blocks, each of which
    runs its own block product tree, one _ModInv and its own downward pass. A
    16M batch at QSB_TREE_N=128 produces 131,072 search CTAs and therefore 512
    groups, so more than one super block is the normal case, not an edge one.
    Returns (inverses, groups, modinvs).
    """
    count = len(values)
    assert count > 0
    groups = (count + WIDTH - 1) // WIDTH
    saved = []
    group_roots = []
    for start in range(0, count, WIDTH):
        leaves = [v % P for v in values[start:start + WIDTH]]
        leaves += [1] * (WIDTH - len(leaves))
        assert all(leaves)
        leaves, internal, root = tree_up(leaves)
        saved.append((leaves, internal))
        group_roots.append(root)

    # One independent block inversion per 256 group roots; inactive lanes in
    # the final block carry the identity, exactly as the kernel does.
    group_inverses = []
    modinvs = 0
    for start in range(0, groups, WIDTH):
        chunk = group_roots[start:start + WIDTH]
        super_leaves = chunk + [1] * (WIDTH - len(chunk))
        super_leaves, super_internal, super_root = tree_up(super_leaves)
        super_inverse = pow(super_root, P - 2, P)
        modinvs += 1
        group_inverses.extend(tree_down(super_leaves, super_internal, super_inverse)[:len(chunk)])

    result = []
    for group, (leaves, internal) in enumerate(saved):
        result.extend(tree_down(leaves, internal, group_inverses[group]))
    return result[:count], groups, modinvs


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    prepare = source.index("qsb_root_group_prepare(")
    super_inverse = source.index("qsb_invert_super_roots(", prepare)
    finish = source.index("qsb_root_group_finish(", super_inverse)
    recovery = source.index("/* Shared-denominator recovery", finish)
    assert "qsb_block_product_checkpoint<256>(r,super_roots,root_checkpoint);" in source[prepare:super_inverse]
    assert "qsb_block_inverse(r);" in source[super_inverse:finish]
    assert "qsb_block_inverse_checkpoint<256>(r,super_roots,root_checkpoint);" in source[finish:recovery]
    launch_begin = source.index("static void launch_pinning_pipeline(")
    launch_end = source.index(" * Fixed-base table construction", launch_begin)
    launches = source[launch_begin:launch_end]
    names = (
        "kernel_pinning_pipeline<FAST_TAIL,0>",
        "qsb_root_group_prepare<<<root_groups,256>>>",
        "qsb_invert_super_roots<<<(root_groups+255)/256,256>>>",
        "qsb_root_group_finish<<<root_groups,256>>>",
        "kernel_pinning_pipeline<FAST_TAIL,2>",
    )
    positions = [launches.index(name) for name in names]
    assert positions == sorted(positions)
    assert "int root_groups=(blocks+255)/256;" in launches
    assert "ROOT_GRDSZ=(GRDSZ+255)/256" in source


def main():
    audit_source()
    rng = random.Random(0x5355504552524F4F)
    # 65536 is one full super block; 65537 and 131072 exercise the multi-block
    # super level a default 16M batch at QSB_TREE_N=128 actually reaches.
    counts = (1, 2, 17, 255, 256, 257, 511, 512, 513, 4097, 65535, 65536,
              65537, 131072)
    checked = 0
    total_modinvs = 0
    for count in counts:
        values = []
        while len(values) < count:
            value = rng.getrandbits(256)
            if value % P:
                values.append(value)
        got, groups, modinvs = nested_inverse(values)
        assert groups == (count + WIDTH - 1) // WIDTH
        assert modinvs == (groups + WIDTH - 1) // WIDTH, (count, groups, modinvs)
        assert all((value % P) * inverse % P == 1 for value, inverse in zip(values, got))
        checked += count
        total_modinvs += modinvs
    print(f"PASS: two-level root inversion; {checked} roots across {len(counts)} "
          f"boundary sizes, including multi-block super inversion "
          f"({total_modinvs} _ModInv calls total)")


if __name__ == "__main__":
    main()
