#!/usr/bin/env python3
"""CPU audit of the cooperative field inverse (tests/gpu_epochs/zinv32.cuh).

zinv32 is a delayed binary-GCD inverse over secp256k1's p, run by four lanes of
one warp: lane 0 owns u, lane 1 v, lane 2 r, lane 3 s, exchanging only matrix
rows, a sign flag, a zero flag and the partner vector. It replaces a serial
lane-0 _ModInv at the root of the subset block inverse tree, so a wrong result
there poisons every candidate in the block.

This models the whole thing on CPU at C semantics -- 32-bit wraparound, int64
accumulators, arithmetic shifts of signed limbs -- including the four lanes and
every __shfl_sync, and checks the returned value against pow(x, p-2, p).

The magic constants are re-derived rather than trusted: -p^-1 mod 2^32, the
little-endian limbs of p, and the 977 in p = 2^256 - 2^32 - 977.

Run from anywhere: python3 candidates/subset/audit_zinv32.py
"""

from pathlib import Path
import hashlib
import random
import re

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
HEADER = Path(__file__).resolve().parent / "tests" / "gpu_epochs" / "zinv32.cuh"

ZI_B = 30
ZI_MM32 = 0xD2253531
ZI_MASK30 = 0x3FFFFFFF
M32 = 0xFFFFFFFF


def body_digest(source, signature):
    """Stable digest of one function body: comments and whitespace removed."""
    at = source.index(signature)
    open_brace = source.index("{", at + len(signature) - 1)
    depth, i = 0, open_brace
    while True:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = source[open_brace:i + 1]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return hashlib.sha256("".join(body.split()).encode()).hexdigest()[:16]


# (name, signature, digest of the body this file's CPU model transcribes)
AUDITED_BODIES = (
    ("zi_divstep30", "void zi_divstep30(", "bd22f7855e4a4da7"),
    ("zi_row_ip", "void zi_row_ip(", "8c6d8c982629b533"),
    ("zi_condneg", "void zi_condneg(", "aaafa36e4f92e372"),
    ("zi_canon", "void zi_canon(", "91dfe34bf8992cc3"),
    ("zi_inverse_quad", "void zi_inverse_quad(", "2cfbd6fb0fed0b22"),
)


def u32(x):
    return x & M32


def i32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def s64(x):
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >> 63 else x


def ctz32(x):
    assert x & M32, "ctz32(0) is undefined in C"
    x &= M32
    n = 0
    while not x & 1:
        x >>= 1
        n += 1
    return n


def clz32(x):
    x &= M32
    if x == 0:
        return 32
    return 32 - x.bit_length()


def p_limbs():
    return [u32(P >> (32 * i)) for i in range(8)] + [0]


def divstep30(u0, v0, uh, vh):
    """zi_divstep30: 30 delayed divsteps; returns the two int32 matrix rows."""
    a, b, c, d = 1, 0, 0, 1
    S = 1 << ZI_B
    while True:
        z = ctz32(v0 | S)
        v0 = u32(v0 >> z)
        vh = u32(vh >> z)
        a = u32(a << z)
        b = u32(b << z)
        S = u32(S >> z)
        if S == 1:
            break
        if vh < uh:
            uh, vh = vh, uh
            u0, v0 = v0, u0
            a, c = c, a
            b, d = d, b
        vh = u32(vh - uh)
        v0 = u32(v0 - u0)
        d = u32(d - b)
        c = u32(c - a)
    return i32(a), i32(b), i32(c), i32(d)


def row_ip(X, Y, a, b, modp):
    """zi_row_ip: X = (a*X + b*Y [+ m*p]) >> 30, nine limbs, X[8] signed."""
    acc = s64(a * X[0] + b * Y[0])
    m = u32(u32(acc) * ZI_MM32) & ZI_MASK30 & u32(0 - modp)
    acc = s64(acc - 977 * m)
    X[0] = u32(acc)
    acc >>= 32
    acc = s64(acc + a * X[1] + b * Y[1] - m)
    X[1] = u32(acc)
    acc >>= 32
    for i in range(2, 8):
        acc = s64(acc + a * X[i] + b * Y[i])
        X[i] = u32(acc)
        acc >>= 32
    acc = s64(acc + a * i32(X[8]) + b * i32(Y[8]) + m)
    X[8] = u32(acc)
    for i in range(8):
        X[i] = u32((X[i] >> ZI_B) | (X[i + 1] << (32 - ZI_B)))
    X[8] = u32(i32(X[8]) >> ZI_B)


def condneg(X, neg):
    msk = u32(0 - neg)
    c = neg
    for i in range(9):
        c += X[i] ^ msk
        X[i] = u32(c)
        c >>= 32


def canon(X):
    """zi_canon: signed 288-bit congruent to r -> canonical r in X[0..7]."""
    PL = p_limbs()
    hi = i32(X[8])
    acc = s64(X[0] + hi * 977)
    X[0] = u32(acc)
    acc >>= 32
    acc = s64(acc + X[1] + hi)
    X[1] = u32(acc)
    acc >>= 32
    for i in range(2, 8):
        acc = s64(acc + X[i])
        X[i] = u32(acc)
        acc >>= 32
    X[8] = u32(acc)
    mneg = u32(i32(X[8]) >> 31)
    c = 0
    for i in range(9):
        c += X[i] + (PL[i] & mneg)
        X[i] = u32(c)
        c >>= 32
    T = [0] * 9
    c = 1
    for i in range(9):
        c += X[i] + u32(~PL[i])
        T[i] = u32(c)
        c >>= 32
    keep = u32(i32(T[8]) >> 31)
    for i in range(8):
        X[i] = (X[i] & keep) | (T[i] & u32(~keep))


class Lane:
    __slots__ = ("P", "Q", "odd", "rs", "pos", "a", "b", "c", "d", "neg", "nz")

    def __init__(self, lane, root):
        PL = p_limbs()
        self.odd = lane & 1
        self.rs = (lane >> 1) & 1
        self.P = [0] * 9
        self.Q = [0] * 9
        for i in range(9):
            xl = u32(root[i >> 1] >> (32 * (i & 1))) if i < 8 else 0
            own = (1 if i == 0 else 0) if self.rs else xl
            oth = 0 if self.rs else PL[i]
            self.P[i] = own if self.odd else oth
            self.Q[i] = oth if self.odd else own
        self.pos = 7
        self.a = self.b = self.c = self.d = 0
        self.neg = 0
        self.nz = 0


def inverse_quad(root_limbs):
    """zi_inverse_quad over four lanes; returns the canonical inverse as 4x64."""
    lanes = [Lane(l, root_limbs) for l in range(4)]
    guard = 0
    while True:
        guard += 1
        assert guard < 64, "decision loop did not terminate"
        for L in lanes:
            L.a = L.b = L.c = L.d = 0
        for idx in (0, 1):
            L = lanes[idx]
            while L.pos > 0 and (L.P[L.pos] | L.Q[L.pos]) == 0:
                L.pos -= 1
            ph, qh = L.P[L.pos], L.Q[L.pos]
            if L.pos > 0:
                sh = clz32(ph | qh)
                if sh:
                    ph = u32((ph << sh) | (L.P[L.pos - 1] >> (32 - sh)))
                    qh = u32((qh << sh) | (L.Q[L.pos - 1] >> (32 - sh)))
            u0 = L.Q[0] if L.odd else L.P[0]
            v0 = L.P[0] if L.odd else L.Q[0]
            uh = qh if L.odd else ph
            vh = ph if L.odd else qh
            L.a, L.b, L.c, L.d = divstep30(u0, v0, uh, vh)

        # each lane takes the row computed by lane (lane & 1)
        ks = []
        for lane, L in enumerate(lanes):
            src = lanes[lane & 1]
            ka = src.d if L.odd else src.a
            kb = src.c if L.odd else src.b
            ks.append((ka, kb))
        for (ka, kb), L in zip(ks, lanes):
            row_ip(L.P, L.Q, ka, kb, L.rs)

        negs = [1 if i32(L.P[8]) < 0 else 0 for L in lanes]
        for lane, L in enumerate(lanes):
            condneg(L.P, negs[lane & 1])

        for L in lanes:
            nz = 0
            for i in range(9):
                nz |= L.P[i]
            L.nz = nz
        if lanes[1].nz == 0:      # every lane reads lane 1's flag
            break
        snapshot = [list(L.P) for L in lanes]
        for lane, L in enumerate(lanes):
            L.Q = list(snapshot[lane ^ 1])

    for L in lanes:
        canon(L.P)
    result = lanes[2].P          # every lane takes lane 2's value
    return [result[2 * i] | (result[2 * i + 1] << 32) for i in range(4)]


def to_limbs(x):
    return [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]


def from_limbs(limbs):
    return sum(v << (64 * i) for i, v in enumerate(limbs))


def audit_constants():
    """The magic numbers, re-derived instead of copied."""
    assert P == 2**256 - 2**32 - 977
    assert ZI_MM32 == (-pow(P, -1, 1 << 32)) % (1 << 32)
    assert p_limbs()[:8] == [u32(P >> (32 * i)) for i in range(8)]
    assert ZI_MASK30 == (1 << ZI_B) - 1


def audit_source():
    source = HEADER.read_text()
    match = re.search(r"^#define ZI_B (\d+)", source, re.MULTILINE)
    assert match and int(match.group(1)) == ZI_B, "batch width changed"
    assert f"#define ZI_MM32 0x{ZI_MM32:08X}u" in source
    assert f"#define ZI_MASK30 0x{ZI_MASK30:X}u" in source
    limbs = re.search(r"#define ZI_PL_INIT \{([^}]*)\}", source)
    assert limbs, "p limbs not found"
    declared = [int(t.strip().rstrip("u"), 16) if t.strip().lower().startswith("0x")
                else int(t.strip().rstrip("u")) for t in limbs.group(1).split(",")]
    assert declared == p_limbs(), "declared p limbs are not secp256k1 p"

    # Lane roles and the collectives this model reproduces.
    assert "zi_inverse_quad(uint64_t *R,int lane)" in source
    assert "__shfl_sync(0xFu" in source
    assert "nz=zi_x(nz,1);" in source                 # zero flag comes from v
    assert "Q[i]=zi_x(P[i],lane^1);" in source        # partner exchange
    assert "P[i]=zi_x(P[i],2);" in source             # result comes from r
    assert "neg=zi_x(neg,lane&1);" in source
    assert "zi_canon(P);" in source
    # The 977 correction and the sparse m*p path.
    assert "(int64_t)977*(int64_t)m" in source
    assert "(int64_t)hi*977" in source

    # This audit is a hand transcription, so it stays honest only while the code
    # it transcribed is the code being compiled. Substring checks are not enough
    # -- a statement can be added next to one and still leave it intact -- so
    # digest each transcribed function body. Any edit, including an insertion,
    # breaks the digest. That is deliberate: the correct response is to
    # re-derive the model against the new code and then update the digest, never
    # to update the digest alone.
    for name, signature, digest in AUDITED_BODIES:
        actual = body_digest(source, signature)
        assert actual == digest, (
            f"{name} changed (digest {actual}, audited {digest}): re-derive the "
            f"CPU model in this file against the new code, then update the digest"
        )


def main():
    audit_constants()
    audit_source()
    rng = random.Random(0x7A696E763332)

    cases = 0
    # Zero must come back as zero: v starts at 0, one batch gives r = 0.
    assert from_limbs(inverse_quad(to_limbs(0))) == 0
    cases += 1

    fixed = [1, 2, 3, P - 1, P - 2, (P + 1) // 2, 1 << 255, (1 << 128) - 1,
             0xFFFFFFFF, 1 << 32, P - (1 << 32), 977]
    for x in fixed:
        got = from_limbs(inverse_quad(to_limbs(x)))
        assert got == pow(x, P - 2, P), (hex(x), hex(got))
        assert got < P
        assert x * got % P == 1
        cases += 1

    for _ in range(250):
        x = rng.randrange(1, P)
        got = from_limbs(inverse_quad(to_limbs(x)))
        assert got == pow(x, P - 2, P), hex(x)
        assert x * got % P == 1
        cases += 1

    # Values with long runs of zero limbs exercise the pos scan and the shift.
    for _ in range(50):
        x = rng.choice([rng.randrange(1, 1 << 32), rng.randrange(1, 1 << 64),
                        (rng.randrange(1, 1 << 32) << 224) % P])
        if x == 0:
            continue
        got = from_limbs(inverse_quad(to_limbs(x)))
        assert got == pow(x, P - 2, P), hex(x)
        cases += 1

    print(f"PASS: cooperative 4-lane zinv32 inverse reproduced on CPU and checked "
          f"against pow(x, p-2, p) on {cases} values, including 0 -> 0 and "
          f"sparse-limb inputs; -p^-1 mod 2^32, the declared p limbs and the 977 "
          f"correction all re-derived")


if __name__ == "__main__":
    main()
