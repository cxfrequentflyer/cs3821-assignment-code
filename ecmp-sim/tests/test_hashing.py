"""Real hashes: the CRCs against published check values and a bit-at-a-time
reference, the properties the polarization story rests on, and the policy."""
import zlib

import numpy as np
import pytest

import reference as ref
from ecmpsim import ECMP, Hash, HashECMP, Unit, five_tuples, run
from ecmpsim.hashing import CRC_PARAMS, TRAFFIC_KINDS, crc, ip, pack, sample_ports, xor16

CHECK = np.frombuffer(b"123456789", dtype=np.uint8)


@pytest.mark.parametrize("kind, value", [("crc16", 0x29B1), ("crc16-arc", 0xBB3D),
                                         ("crc32", 0xCBF43926)])
def test_published_check_values(kind, value):
    assert int(crc(CHECK, kind)) == value


@pytest.mark.parametrize("kind", list(CRC_PARAMS))
def test_table_crc_matches_bitwise_reference(kind):
    width, poly, init, reflected, xorout = CRC_PARAMS[kind]
    data = np.random.default_rng(0).integers(0, 256, size=(200, 13), dtype=np.uint8)
    fast = crc(data, kind)
    for row, got in zip(data.tolist(), fast.tolist()):
        assert got == ref.crc_bitwise(row, width, poly, init, reflected, xorout)


def test_crc32_is_zlib():
    data = np.random.default_rng(1).integers(0, 256, size=(100, 13), dtype=np.uint8)
    assert crc(data, "crc32").tolist() == [zlib.crc32(bytes(row)) for row in data.tolist()]


@pytest.mark.parametrize("kind", list(CRC_PARAMS))
def test_seeding_a_crc_only_xors_a_constant(kind):
    # The linearity the module docstring claims: for inputs of one length, a
    # different initial value, or seed bytes before or after the header, change
    # the CRC by the same constant for every input. So Hash models a seed as h ^ seed.
    rng = np.random.default_rng(2)
    data = rng.integers(0, 256, size=(500, 13), dtype=np.uint8)
    plain = crc(data, kind)
    assert len(np.unique(crc(data, kind, init=0x1D0F) ^ plain)) == 1
    seed = np.broadcast_to(np.array([0xBE, 0xEF], dtype=np.uint8), (500, 2))
    zero = np.zeros((500, 2), dtype=np.uint8)
    for joined, base in [(np.hstack([seed, data]), np.hstack([zero, data])),
                         (np.hstack([data, seed]), np.hstack([data, zero]))]:
        assert len(np.unique(crc(joined, kind) ^ crc(base, kind))) == 1


def test_pack_layout_and_xor_fold():
    h = pack(ip(10, 1, 2, 3), ip(10, 4, 5, 6), 0x1234, 0x0050)
    assert h.tolist() == [10, 1, 2, 3, 10, 4, 5, 6, 0x12, 0x34, 0x00, 0x50, 6]
    expected = 0x0A01 ^ 0x0203 ^ 0x0A04 ^ 0x0506 ^ 0x1234 ^ 0x0050 ^ 0x0600
    assert int(xor16(h)) == expected
    assert int(Hash("xor16")(h)) == expected
    assert int(Hash("xor16", shift=4)(h)) == expected >> 4
    assert int(Hash("xor16")(h, seed=0xFFFF)) == expected ^ 0xFFFF


def test_symmetric_ports_cancel_in_xor_but_not_in_crc():
    rng = np.random.default_rng(3)
    port = rng.integers(1024, 65536, 2000)
    h = pack(ip(10, 0, 0, 2), ip(10, 1, 0, 2), port, port)
    assert len(np.unique(Hash("xor16")(h))) == 1
    assert len(np.unique(Hash("crc16")(h))) > 1900


def test_sequential_ports():
    src, dst = sample_ports(np.random.default_rng(4), 1000, "sequential")
    assert len(np.unique(dst)) == 1 and len(np.unique(src)) == 1000
    assert set(np.diff(src).tolist()) <= {1, 32768 - 61000 + 1}
    # consecutive ports through an XOR fold are a round robin: perfectly even
    paths = HashECMP("xor16", "sequential").assign(None, five_tuples(
        np.random.default_rng(5), 1024, "sequential"), 8)
    assert np.bincount(paths, minlength=8).tolist() == [128] * 8


@pytest.mark.parametrize("traffic", TRAFFIC_KINDS)
def test_tuples_have_the_right_shape(traffic):
    h = five_tuples(np.random.default_rng(6), 50, traffic)
    assert h.shape == (50, 13) and h.dtype == np.uint8
    if traffic != "random":
        assert (h[:, [0, 4]] == 10).all() and (h[:, 12] == 6).all()


@pytest.mark.parametrize("kind", ["crc16", "crc16-arc", "crc32"])
@pytest.mark.parametrize("n", [8, 7])
def test_crc_on_random_headers_looks_uniform(kind, n):
    paths = HashECMP(kind, "random").assign(None, five_tuples(
        np.random.default_rng(7), 80_000, "random"), n)
    counts = np.bincount(paths, minlength=n)
    chi2 = ((counts - 80_000 / n) ** 2 / (80_000 / n)).sum()
    assert chi2 < 30                       # chi-square with <= 7 d.o.f.: P(> 30) ~ 1e-4


def test_policy_runs_and_hashes_see_the_same_headers():
    pols = [ECMP(), HashECMP("crc16"), HashECMP("xor16"), HashECMP("crc16", "symmetric")]
    assert pols[1].seed_name == pols[2].seed_name != pols[3].seed_name
    df = run(8, 256, Unit(), pols, trials=50, seed=9)
    assert set(df.policy) == {p.name for p in pols}
    alone = run(8, 256, Unit(), [pols[2]], trials=50, seed=9)
    both = df[df.policy == pols[2].name].reset_index(drop=True)
    assert (alone.max_load == both.max_load).all()
    # a deterministic hash of random-enough headers behaves like ideal ECMP
    mean = df.groupby("policy").max_load.mean()
    assert abs(mean["crc16/datacenter"] - mean["ecmp"]) < 2.0


def test_bad_arguments():
    with pytest.raises(ValueError):
        Hash("md5")
    with pytest.raises(ValueError):
        Hash("crc16", shift=16)
    with pytest.raises(ValueError):
        HashECMP("crc16", "nonsense")
