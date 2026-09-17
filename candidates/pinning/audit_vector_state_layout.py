#!/usr/bin/env python3
"""Exact byte-layout and alignment audit for the vectorized pipeline state.

The kernel carries two state layouts behind QSB_SYM_FINISH:

  QSB_SYM_FINISH=1 (default)  Y, ZZZ and W cross the kernel boundary
                              -> 3 fields, 6 planes, 96 bytes/candidate
  QSB_SYM_FINISH=0            C, Y, W and ZZZ cross
                              -> 4 fields, 8 planes, 128 bytes/candidate

Both are audited here, and each is checked against its own branch of the
source, so neither layout can drift without this failing. W stays in planes
4-5 in both, which is what lets the tree kernels index it identically.
"""

from pathlib import Path


LIMBS = 4
LIMB_BYTES = 8
VECTOR_BYTES = 16
WARP = 32
PRODUCTION_BATCH = 16_777_216

# field order as stored, per configuration
SYM_FIELDS = ("qy", "qzzz", "qzz")
LEGACY_FIELDS = ("qx", "qy", "qzz", "qzzz")
CONFIGS = {1: SYM_FIELDS, 0: LEGACY_FIELDS}


def planes_for(fields):
    return len(fields) * LIMBS * LIMB_BYTES // VECTOR_BYTES


def old_offset(batch_size, field, limb, candidate, fields):
    plane = field * LIMBS + limb
    return (plane * batch_size + candidate) * LIMB_BYTES


def vector_offset(batch_size, field, limb, candidate, fields):
    plane = field * 2 + limb // 2
    return (plane * batch_size + candidate) * VECTOR_BYTES + (limb & 1) * LIMB_BYTES


def audit_batch(batch_size, fields):
    n_fields = len(fields)
    planes = planes_for(fields)
    old = {}
    vector = {}
    for field in range(n_fields):
        for limb in range(LIMBS):
            for candidate in range(batch_size):
                logical = (field, limb, candidate)
                old[old_offset(batch_size, *logical, fields)] = logical
                vector[vector_offset(batch_size, *logical, fields)] = logical

    cells = batch_size * n_fields * LIMBS
    expected_offsets = list(range(0, cells * LIMB_BYTES, LIMB_BYTES))
    assert len(old) == len(vector) == cells
    assert sorted(old) == sorted(vector) == expected_offsets

    packed = [None] * (batch_size * planes)
    for field in range(n_fields):
        for pair in range(2):
            plane = field * 2 + pair
            for candidate in range(batch_size):
                packed[plane * batch_size + candidate] = (
                    (field, pair * 2, candidate),
                    (field, pair * 2 + 1, candidate),
                )
    unpacked = {tag for pair in packed for tag in pair}
    expected = {
        (field, limb, candidate)
        for field in range(n_fields)
        for limb in range(LIMBS)
        for candidate in range(batch_size)
    }
    assert unpacked == expected

    for plane in range(planes):
        plane_base = plane * batch_size * VECTOR_BYTES
        assert plane_base % VECTOR_BYTES == 0
        for warp_first in range(0, batch_size, WARP):
            lanes = min(WARP, batch_size - warp_first)
            starts = [
                plane_base + candidate * VECTOR_BYTES
                for candidate in range(warp_first, warp_first + lanes)
            ]
            assert all(address % VECTOR_BYTES == 0 for address in starts)
            assert all(b - a == VECTOR_BYTES for a, b in zip(starts, starts[1:]))
            assert starts[-1] + VECTOR_BYTES - starts[0] == lanes * VECTOR_BYTES

    assert cells * LIMB_BYTES == batch_size * planes * VECTOR_BYTES
    return cells * LIMB_BYTES


def conditional_blocks(source, macro):
    """Every (if_body, else_body) pair guarded by `#if <macro>`, nesting-aware."""
    blocks = []
    marker = f"#if {macro}"
    at = source.find(marker)
    while at != -1:
        depth = 0
        if_body, else_body, target = [], [], None
        for line in source[at:].splitlines()[1:]:
            stripped = line.strip()
            if stripped.startswith("#if"):
                depth += 1
            elif stripped.startswith("#endif"):
                if depth == 0:
                    break
                depth -= 1
            elif stripped.startswith("#else") and depth == 0:
                target = else_body
                continue
            (if_body if target is None else else_body).append(line)
        blocks.append(("\n".join(if_body), "\n".join(else_body)))
        at = source.find(marker, at + 1)
    return blocks


def plane_addresses(body, op):
    """Plane indices touched by `op` (qsb_st_v2/qsb_ld_v2), in source order."""
    planes = []
    for line in body.splitlines():
        if op not in line:
            continue
        head, _, rest = line.partition("&saved[")
        if not rest:
            continue
        index, _, _ = rest.partition("u*state_plane_stride+state_idx]")
        if index.isdigit():
            planes.append(int(index))
    return planes


def audit_source():
    source = Path(__file__).with_name("pinning.cu").read_text()
    assert 'static_assert(sizeof(ulonglong2) == 16' in source
    assert 'static_assert(alignof(ulonglong2) == 16' in source
    assert "alignof(ulonglong2)-1u" in source

    # The allocation must be driven by QSB_STATE_PLANES, not a literal, so the
    # buffer cannot disagree with whichever layout is compiled in.
    assert "#define QSB_STATE_PLANES (QSB_SYM_FINISH ? 6u : 8u)" in source
    assert "(size_t)BATCH*QSB_STATE_PLANES*sizeof(ulonglong2)" in source
    assert planes_for(SYM_FIELDS) == 6 and planes_for(LEGACY_FIELDS) == 8

    # QSB_SYM_FINISH is the default, so the symmetric layout is what a ranked
    # build uses; assert the default rather than assuming it.
    assert "#define QSB_SYM_FINISH 1" in source

    blocks = conditional_blocks(source, "QSB_SYM_FINISH")
    stores = [b for b in blocks if "qsb_st_v2" in b[0]]
    loads = [b for b in blocks if "qsb_ld_v2" in b[0]]
    assert len(stores) == 1, len(stores)
    assert len(loads) == 1, len(loads)

    for (sym_body, legacy_body), op in ((stores[0], "qsb_st_v2"), (loads[0], "qsb_ld_v2")):
        assert plane_addresses(sym_body, op) == list(range(planes_for(SYM_FIELDS)))
        assert plane_addresses(legacy_body, op) == list(range(planes_for(LEGACY_FIELDS)))

    # Field order must match what this audit models, W last so the tree kernels
    # can address it at planes 4-5 in either configuration.
    for fields, body in ((SYM_FIELDS, stores[0][0]), (LEGACY_FIELDS, stores[0][1])):
        stored = []
        for line in body.splitlines():
            if "qsb_st_v2" not in line:
                continue
            _, _, args = line.partition("state_idx],")
            name = args.partition("[")[0].strip()
            if name and (not stored or stored[-1] != name):
                stored.append(name)
        assert stored == list(fields), (stored, fields)
    assert plane_addresses(stores[0][0], "qsb_st_v2")[-2:] == [4, 5]
    assert SYM_FIELDS[-1] == LEGACY_FIELDS[-2] == "qzz"   # W in planes 4-5


def audit_production_batch(fields):
    planes = planes_for(fields)
    total_bytes = PRODUCTION_BATCH * planes * VECTOR_BYTES
    assert total_bytes == planes // 2 * 512 * 1024**2
    for plane in range(planes):
        assert plane * PRODUCTION_BATCH * VECTOR_BYTES % VECTOR_BYTES == 0
    assert vector_offset(
        PRODUCTION_BATCH, len(fields) - 1, LIMBS - 1, PRODUCTION_BATCH - 1, fields
    ) == (total_bytes - LIMB_BYTES)
    for warp_first in (0, WARP, PRODUCTION_BATCH // 2, PRODUCTION_BATCH - WARP):
        starts = [
            vector_offset(PRODUCTION_BATCH, 0, 0, candidate, fields)
            for candidate in range(warp_first, warp_first + WARP)
        ]
        assert all(address % VECTOR_BYTES == 0 for address in starts)
        assert starts[-1] + VECTOR_BYTES - starts[0] == WARP * VECTOR_BYTES
    return total_bytes


def main():
    sizes = {}
    for sym_finish, fields in CONFIGS.items():
        per_candidate = len(fields) * LIMBS * LIMB_BYTES
        for batch_size in (1, 2, 3, 31, 32, 33, 255, 256, 257, 4096):
            assert audit_batch(batch_size, fields) == batch_size * per_candidate
        sizes[sym_finish] = audit_production_batch(fields)
    audit_source()
    assert sizes[1] == 1536 * 1024**2 and sizes[0] == 2 * 1024**3
    print(
        "PASS: both pipeline state layouts map bijectively onto ulonglong2 planes; "
        "QSB_SYM_FINISH=1 stores Y/ZZZ/W in 6 planes (96 bytes/candidate/direction, "
        "1.5 GiB at 16,777,216 candidates) and QSB_SYM_FINISH=0 stores C/Y/W/ZZZ in "
        "8 planes (128 bytes, 2 GiB); all vector elements are 16-byte aligned, warp "
        "addresses are contiguous, and W stays in planes 4-5 in both"
    )


if __name__ == "__main__":
    main()
