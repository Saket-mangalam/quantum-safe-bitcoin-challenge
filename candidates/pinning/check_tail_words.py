"""Check a proposed one-block SHA tail against hashlib, without using the GPU.

This is a diagnostic experiment, not a replacement for the benchmark verifier.
It tests the byte packing and host-midstate scheme before changing the kernel.
Run from the repository root: python3 candidates/pinning/check_tail_words.py
"""
import hashlib
import random
import struct

# SHA-256 compression, in plain Python. The scheme under test needs a raw
# block compression with an explicit midstate, which hashlib does not expose.
# This used to call libcrypto's SHA256_Transform through ctypes; that is not
# portable -- macOS refuses to dlopen the unversioned libcrypto stub, and the
# LibreSSL copies it does ship disagree with OpenSSL here. The whole file is
# checked against hashlib on every case below, so a wrong compression cannot
# pass silently.

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
MASK = 0xFFFFFFFF


def _ror(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def compress(state, block):
    """One SHA-256 block compression. Returns the new 8-word state."""
    assert len(block) == 64
    w = list(struct.unpack(">16I", block))
    for i in range(16, 64):
        s0 = _ror(w[i - 15], 7) ^ _ror(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _ror(w[i - 2], 17) ^ _ror(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK)
    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + (_ror(e, 6) ^ _ror(e, 11) ^ _ror(e, 25))
              + (g ^ (e & (f ^ g))) + K[i] + w[i]) & MASK
        t2 = ((_ror(a, 2) ^ _ror(a, 13) ^ _ror(a, 22))
              + ((a & b) | (c & (a | b)))) & MASK
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    return [(x + y) & MASK for x, y in zip(state, [a, b, c, d, e, f, g, h])]


def check(seed):
    rng = random.Random(seed)
    prefix = rng.randbytes(9920)
    suffix = bytearray(rng.randbytes(75))
    prefix_state = list(IV)
    for off in range(0, len(prefix), 64):
        prefix_state = compress(prefix_state, prefix[off:off + 64])

    cases = 0
    seqs = [0, 1, 0x80000000, 0xFFFFFFFF, rng.getrandbits(32)]
    locktimes = [0, 1, 255, 256, 65535, 65536, 500000000, 1744599999,
                 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(20)]
    for seq in seqs:
        suffix[31:35] = seq.to_bytes(4, "little")
        state = compress(list(prefix_state), bytes(suffix[:64]))
        for lt in locktimes:
            # Proposed static GPU message words. Locktime crosses words 0/1.
            w0 = int.from_bytes(suffix[64:67], "big") << 8 | (lt & 255)
            w1 = ((lt >> 8) & 255) << 24 | ((lt >> 16) & 255) << 16
            w1 |= ((lt >> 24) & 255) << 8 | suffix[71]
            w2 = int.from_bytes(suffix[72:75], "big") << 8 | 0x80
            words = [w0, w1, w2] + [0] * 12 + [9995 * 8]
            out = compress(list(state), struct.pack(">16I", *words))
            got = hashlib.sha256(struct.pack(">8I", *out)).digest()
            suffix[67:71] = lt.to_bytes(4, "little")
            expected = hashlib.sha256(hashlib.sha256(prefix + suffix).digest()).digest()
            assert got == expected, (seed, seq, lt, got.hex(), expected.hex())
            cases += 1
    return cases


if __name__ == "__main__":
    count = sum(check(seed) for seed in range(16))
    print(f"PASS: {count} SHA256d comparisons across 16 synthetic prefixes; "
          "sequence and locktime boundaries included")
