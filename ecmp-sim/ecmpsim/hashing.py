"""Real hashes: what a switch actually computes instead of an ideal coin flip.

`policies.ECMP` gives every flow an independent uniform path. A switch
instead computes a fixed function of the packet's 5-tuple (source address,
destination address, source port, destination port, protocol) and takes it
modulo the number of next hops. Two things can then go wrong that the ideal
model cannot show:

  * structured inputs: real 5-tuples are far from uniform bits (a handful of
    subnets, well-known ports, consecutive ephemeral ports), and a weak hash
    maps that structure straight onto the paths;
  * correlated choices: every switch that runs the same function makes the
    same decision for the same flow, so a later tier only ever sees flows
    that already agreed at an earlier one (hash polarization, fattree.py).

    h = Hash("crc16")(headers)            # (..., 13) uint8 -> (...,) integers
    run(n, m, flows, [ECMP(), HashECMP("crc16", "datacenter"),
                      HashECMP("xor16", "datacenter")], ...)

Hashes (`Hash(kind)`), all table-driven and vectorized over flows:
    crc16       CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF
    crc16-arc   CRC-16/ARC (the "CRC16" of most switch chips, also called
                BISYNC or IBM): poly 0x8005, bit-reflected, init 0
    crc32       CRC-32 (Ethernet's), identical to zlib.crc32
    xor16       XOR of the 16-bit words of the header: the cheap fold

A CRC is linear over GF(2). For inputs of one length, a different initial
value, or seed bytes hashed before or after the header, only XOR a constant
into the result: crc(x, seed) = crc(x) ^ c(seed). So "the same CRC with a
per-switch seed" is modeled as exactly that, `h ^ seed`, and it is the reason
such seeds do not cure polarization when the fan-out is a power of two (the
buckets are permuted, not mixed). tests/test_hashing.py checks the identity.

5-tuple layout: 13 bytes in network order,
    src IP (4) | dst IP (4) | src port (2) | dst port (2) | protocol (1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADER_BYTES = 13
TCP = 6
SERVICES = np.array([80, 443, 53, 22, 3306, 5201, 8080, 9000])     # destination ports
EPHEMERAL = (32768, 61000)                                         # Linux's default range


# --------------------------------------------------------------------------
# Hash functions
# --------------------------------------------------------------------------

# kind -> (width, polynomial, initial value, reflected, final xor)
CRC_PARAMS = {
    "crc16": (16, 0x1021, 0xFFFF, False, 0x0000),
    "crc16-arc": (16, 0x8005, 0x0000, True, 0x0000),
    "crc32": (32, 0x04C11DB7, 0xFFFFFFFF, True, 0xFFFFFFFF),
}


def _reflect(x: int, bits: int) -> int:
    return int(f"{x:0{bits}b}"[::-1], 2)


def _crc_table(width: int, poly: int, reflected: bool) -> np.ndarray:
    mask = (1 << width) - 1
    table = np.empty(256, dtype=np.uint64)
    for byte in range(256):
        if reflected:
            r, p = byte, _reflect(poly, width)
            for _ in range(8):
                r = (r >> 1) ^ p if r & 1 else r >> 1
        else:
            r = byte << (width - 8)
            for _ in range(8):
                r = ((r << 1) ^ poly) & mask if r >> (width - 1) else (r << 1) & mask
        table[byte] = r
    return table


_TABLES = {kind: _crc_table(w, poly, refl) for kind, (w, poly, _, refl, _) in CRC_PARAMS.items()}


def crc(data: np.ndarray, kind: str = "crc16", init: int | None = None) -> np.ndarray:
    """CRC of every row of bytes: data (..., L) uint8 -> (...,) uint64.

    `init` overrides the initial register value (already reflected for the
    reflected CRCs, as the register holds it)."""
    width, _, init0, reflected, xorout = CRC_PARAMS[kind]
    table = _TABLES[kind]
    data = np.asarray(data, dtype=np.uint8)
    mask = np.uint64((1 << width) - 1)
    reg = np.full(data.shape[:-1], init0 if init is None else init, dtype=np.uint64)
    eight, top = np.uint64(8), np.uint64(width - 8)
    for i in range(data.shape[-1]):
        byte = data[..., i].astype(np.uint64)
        if reflected:
            reg = table[(reg ^ byte) & np.uint64(0xFF)] ^ (reg >> eight)
        else:
            reg = table[((reg >> top) ^ byte) & np.uint64(0xFF)] ^ ((reg << eight) & mask)
    return reg ^ np.uint64(xorout)


def xor16(data: np.ndarray) -> np.ndarray:
    """XOR of the big-endian 16-bit words of every row (an odd last byte is the
    high byte of a zero-padded word)."""
    data = np.asarray(data, dtype=np.uint8)
    if data.shape[-1] % 2:
        data = np.concatenate([data, np.zeros((*data.shape[:-1], 1), dtype=np.uint8)], axis=-1)
    words = (data[..., 0::2].astype(np.uint64) << np.uint64(8)) | data[..., 1::2].astype(np.uint64)
    return np.bitwise_xor.reduce(words, axis=-1)


@dataclass(frozen=True)
class Hash:
    """A switch's hash function over the 5-tuple.

    kind    "crc16", "crc16-arc", "crc32" or "xor16"
    shift   drop this many low bits first, so that two tiers can use
            different bits of one hash computation (a common remedy for
            polarization)
    """

    kind: str = "crc16"
    shift: int = 0

    def __post_init__(self):
        if self.kind != "xor16" and self.kind not in CRC_PARAMS:
            raise ValueError(f"unknown hash {self.kind!r}")
        if not 0 <= self.shift < self.bits:
            raise ValueError("shift must leave at least one bit")

    @property
    def bits(self) -> int:
        return 16 if self.kind == "xor16" else CRC_PARAMS[self.kind][0]

    @property
    def label(self) -> str:
        return self.kind + (f">>{self.shift}" if self.shift else "")

    def __call__(self, headers: np.ndarray, seed=0) -> np.ndarray:
        """(...,) int64 hash values. `seed` (a number or an array broadcastable
        to the flows) is XORed in before the shift: see the module docstring
        for why that is what seeding a CRC amounts to."""
        h = xor16(headers) if self.kind == "xor16" else crc(headers, self.kind)
        h = h ^ (np.asarray(seed).astype(np.uint64) & np.uint64((1 << self.bits) - 1))
        return (h >> np.uint64(self.shift)).astype(np.int64)


def as_hash(h) -> Hash:
    return h if isinstance(h, Hash) else Hash(h)


# --------------------------------------------------------------------------
# 5-tuples
# --------------------------------------------------------------------------

def pack(src_ip, dst_ip, src_port, dst_port, proto=TCP) -> np.ndarray:
    """Header bytes (..., 13) from integer fields (IPs as 32-bit integers)."""
    src_ip, dst_ip, src_port, dst_port, proto = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.int64) for x in (src_ip, dst_ip, src_port, dst_port, proto)))
    out = np.empty((*src_ip.shape, HEADER_BYTES), dtype=np.uint8)
    for k in range(4):
        out[..., k] = (src_ip >> (24 - 8 * k)) & 0xFF
        out[..., 4 + k] = (dst_ip >> (24 - 8 * k)) & 0xFF
    out[..., 8], out[..., 9] = src_port >> 8, src_port & 0xFF
    out[..., 10], out[..., 11] = dst_port >> 8, dst_port & 0xFF
    out[..., 12] = proto
    return out


def ip(a, b, c, d) -> np.ndarray:
    """The address a.b.c.d as an integer (arrays broadcast)."""
    a, b, c, d = (np.asarray(x, dtype=np.int64) for x in (a, b, c, d))
    return (a << 24) | (b << 16) | (c << 8) | d


PORT_KINDS = ("ephemeral", "sequential", "symmetric")


def sample_ports(rng: np.random.Generator, m: int, kind: str = "ephemeral"):
    """Source and destination ports of m flows.

    ephemeral    source uniform in Linux's ephemeral range, destination one of
                 a few service ports: the usual client-server mix
    sequential   the source port counts up from a random start (a client
                 opening connections one after another), one service port
    symmetric    source port = destination port (as BGP, NTP, IPsec/IKE and
                 many RPC meshes do): the two cancel in an XOR fold
    """
    lo, hi = EPHEMERAL
    if kind == "ephemeral":
        return rng.integers(lo, hi, size=m), SERVICES[rng.integers(0, len(SERVICES), size=m)]
    if kind == "sequential":
        src = lo + (rng.integers(0, hi - lo) + np.arange(m)) % (hi - lo)
        return src, np.full(m, SERVICES[rng.integers(0, len(SERVICES))])
    if kind == "symmetric":
        port = rng.integers(lo, hi, size=m)
        return port, port
    raise ValueError(f"unknown port pattern {kind!r}")


def _dc_hosts(rng, size):
    """Fat-tree style addresses 10.pod.edge.host: 16 pods x 8 switches x 8 hosts."""
    return ip(10, rng.integers(0, 16, size), rng.integers(0, 8, size), rng.integers(2, 10, size))


TRAFFIC_KINDS = ("random", "datacenter", "sequential", "symmetric", "incast")


def five_tuples(rng: np.random.Generator, m: int, traffic: str = "datacenter") -> np.ndarray:
    """(m, 13) header bytes for one ECMP group's flows.

    random       every bit uniform: the hash's best case, a sanity check
    datacenter   1024 hosts in 10.pod.edge.host, ephemeral source ports, a few
                 service ports: realistic, low entropy outside the source port
    sequential   one client, one server, one service; only the source port
                 changes, counting up by one per flow
    symmetric    datacenter hosts, source port = destination port
    incast       many clients to ONE server and service, ephemeral ports
    """
    if traffic == "random":
        return rng.integers(0, 256, size=(m, HEADER_BYTES), dtype=np.uint8)
    if traffic == "datacenter":
        return pack(_dc_hosts(rng, m), _dc_hosts(rng, m), *sample_ports(rng, m, "ephemeral"))
    if traffic == "sequential":
        return pack(_dc_hosts(rng, 1), _dc_hosts(rng, 1), *sample_ports(rng, m, "sequential"))
    if traffic == "symmetric":
        return pack(_dc_hosts(rng, m), _dc_hosts(rng, m), *sample_ports(rng, m, "symmetric"))
    if traffic == "incast":
        src_port, _ = sample_ports(rng, m, "ephemeral")
        return pack(_dc_hosts(rng, m), _dc_hosts(rng, 1), src_port,
                    SERVICES[rng.integers(0, len(SERVICES))])
    raise ValueError(f"unknown traffic {traffic!r}")


# --------------------------------------------------------------------------
# The policy for one ECMP group
# --------------------------------------------------------------------------

class HashECMP:
    """ECMP as a switch does it: path = hash(5-tuple) mod n.

    The policy's "random draws" are the flows' 5-tuples, so the placement is
    a deterministic function of the headers, and every hash sees the same
    headers for a given `traffic` (the stream is keyed by the traffic kind,
    not by the hash): comparisons between hashes are paired.
    """

    def __init__(self, hash="crc16", traffic: str = "datacenter"):
        if traffic not in TRAFFIC_KINDS:
            raise ValueError(f"unknown traffic {traffic!r}")
        self.hash, self.traffic = as_hash(hash), traffic
        self.seed_name = f"tuples-{traffic}"
        self.name = f"{self.hash.label}/{traffic}"

    def draw(self, rng: np.random.Generator, m: int, n: int) -> np.ndarray:
        return five_tuples(rng, m, self.traffic)

    def assign(self, w: np.ndarray, draws: np.ndarray, n: int, **_) -> np.ndarray:
        return self.hash(draws) % n

    def __repr__(self):
        return f"HashECMP({self.hash.label!r}, {self.traffic!r})"
