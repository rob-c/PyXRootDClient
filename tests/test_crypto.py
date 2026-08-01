"""Crypto primitives, pinned to published vectors.

These have no second implementation to fall back on, so every one is checked
against a value from the standard that defines it.
"""

from __future__ import annotations

import hashlib
import hmac
import zlib

import pytest

from xrd.crypto.blowfish import Blowfish, _pi_fraction_words
from xrd.crypto.checksum import algorithms, checksum_bytes, checksum_file, new
from xrd.crypto.crc32c import PAGE_SIZE, _crc32c_py, crc32c, pack_pages, unpack_pages
from xrd.crypto.sigver import Signer, is_signed, sigver_hmac
from xrd.proto import constants as c
from xrd.proto import requests as r
from xrd.proto.frames import encode

# --------------------------------------------------------------------------
# CRC32C (Castagnoli) - RFC 3720 appendix B
# --------------------------------------------------------------------------


def test_crc32c_check_value():
    assert crc32c(b"123456789") == 0xE3069283


def test_crc32c_of_nothing_is_zero():
    assert crc32c(b"") == 0


@pytest.mark.parametrize(
    "data, expected",
    [
        (bytes(32), 0x8A9136AA),
        (b"\xff" * 32, 0x62A8AB43),
        (bytes(range(32)), 0x46DD794E),
    ],
)
def test_crc32c_rfc3720_vectors(data, expected):
    assert crc32c(data) == expected


def test_crc32c_is_chainable():
    whole = crc32c(b"123456789")
    piecewise = crc32c(b"56789", crc32c(b"1234"))
    assert piecewise == whole


def test_crc32c_is_not_the_ieee_crc32():
    assert crc32c(b"123456789") != zlib.crc32(b"123456789")


# --------------------------------------------------------------------------
# Paged I/O framing
# --------------------------------------------------------------------------


def test_pack_pages_prefixes_each_page_with_its_crc():
    data = b"a" * (PAGE_SIZE + 10)
    packed = pack_pages(data, 0)
    assert len(packed) == len(data) + 8
    assert packed[4:4 + PAGE_SIZE] == data[: PAGE_SIZE]


def test_pack_unpack_round_trips_on_a_page_boundary():
    data = bytes(range(256)) * 32  # exactly two pages
    back, corrupt = unpack_pages(pack_pages(data, 0), 0)
    assert back == data
    assert corrupt == ()


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 9000])
def test_pack_unpack_round_trips_at_every_size(size):
    data = bytes(i % 251 for i in range(size))
    back, corrupt = unpack_pages(pack_pages(data, 0), 0)
    assert back == data and corrupt == ()


def test_an_unaligned_start_offset_makes_a_short_first_page():
    offset = PAGE_SIZE - 10
    data = b"x" * 100
    packed = pack_pages(data, offset)
    back, corrupt = unpack_pages(packed, offset)
    assert back == data and corrupt == ()


def test_a_flipped_bit_is_reported_by_offset():
    data = b"y" * (2 * PAGE_SIZE)
    packed = bytearray(pack_pages(data, 0))
    packed[10] ^= 0xFF
    back, corrupt = unpack_pages(bytes(packed), 0)
    assert corrupt == (0,)
    assert len(back) == len(data)


def test_accelerated_and_pure_python_agree():
    payload = bytes(range(256)) * 7
    assert _crc32c_py(payload, 0) == crc32c(payload)


# --------------------------------------------------------------------------
# Blowfish - Schneier's published test vectors
# --------------------------------------------------------------------------


def test_the_pi_derivation_matches_the_published_p_array():
    assert _pi_fraction_words(4) == [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344]


@pytest.mark.parametrize(
    "key, plaintext, ciphertext",
    [
        ("0000000000000000", "0000000000000000", "4EF997456198DD78"),
        ("FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF", "51866FD5B85ECB8A"),
        ("3000000000000000", "1000000000000001", "7D856F9A613063F2"),
        ("1111111111111111", "1111111111111111", "2466DD878B963C9D"),
        ("0123456789ABCDEF", "1111111111111111", "61F9C3802281B096"),
        ("FEDCBA9876543210", "0123456789ABCDEF", "0ACEAB0FC6A0A28D"),
        ("7CA110454A1A6E57", "01A1D6D039776742", "59C68245EB05282B"),
    ],
)
def test_blowfish_ecb_vectors(key, plaintext, ciphertext):
    cipher = Blowfish(bytes.fromhex(key))
    assert cipher.encrypt_ecb(bytes.fromhex(plaintext)).hex().upper() == ciphertext


def test_blowfish_decrypt_inverts_encrypt():
    cipher = Blowfish(b"a secret key")
    assert cipher.decrypt_block(*cipher.encrypt_block(0x01234567, 0x89ABCDEF)) == (
        0x01234567,
        0x89ABCDEF,
    )
    assert cipher.decrypt_ecb(cipher.encrypt_ecb(b"12345678")) == b"12345678"


def test_blowfish_ecb_needs_whole_blocks():
    with pytest.raises(ValueError, match="multiple of 8"):
        Blowfish(b"k").encrypt_ecb(b"1234567")


def test_blowfish_decrypt_ecb_needs_whole_blocks_too():
    with pytest.raises(ValueError, match="multiple of 8"):
        Blowfish(b"k").decrypt_ecb(b"123456789")


@pytest.mark.parametrize(
    ("key", "message"),
    [(b"", "non-empty"), (b"k" * 57, "<= 56 bytes")],
)
def test_blowfish_refuses_a_key_the_algorithm_cannot_take(key, message):
    """448 bits is the ceiling Blowfish's key schedule is defined for."""
    with pytest.raises(ValueError, match=message):
        Blowfish(key)


def test_blowfish_cfb64_round_trips_at_any_length():
    cipher = Blowfish(b"another key")
    for length in (0, 1, 7, 8, 9, 100):
        data = bytes(range(length))
        iv = bytes(8)
        assert cipher.decrypt_cfb64(iv, cipher.encrypt_cfb64(iv, data)) == data


def test_blowfish_cfb64_is_a_stream_and_does_not_pad():
    cipher = Blowfish(b"k")
    assert len(cipher.encrypt_cfb64(bytes(8), b"abc")) == 3


def test_blowfish_repr_hides_the_key():
    assert repr(Blowfish(b"hunter2")) == "Blowfish(key=<redacted>)"
    assert "hunter2" not in repr(Blowfish(b"hunter2"))


def test_the_cfb64_iv_must_be_eight_bytes():
    with pytest.raises(ValueError, match="IV must be 8 bytes"):
        Blowfish(b"k").encrypt_cfb64(b"short", b"data")


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def test_the_advertised_algorithms_all_work():
    for name in algorithms():
        assert checksum_bytes(name, b"abc")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("adler32", f"{zlib.adler32(b'123456789'):08x}"),
        ("crc32", f"{zlib.crc32(b'123456789'):08x}"),
        ("crc32c", "e3069283"),
        ("md5", hashlib.md5(b"123456789").hexdigest()),
        ("sha256", hashlib.sha256(b"123456789").hexdigest()),
    ],
)
def test_checksum_values(name, expected):
    assert checksum_bytes(name, b"123456789") == expected


def test_checksum_names_are_case_insensitive():
    assert checksum_bytes("ADLER32", b"x") == checksum_bytes("adler32", b"x")


def test_an_unknown_algorithm_is_a_value_error():
    with pytest.raises(ValueError, match="unknown checksum algorithm"):
        new("md17")


def test_incremental_updates_match_a_single_shot():
    h = new("adler32")
    h.update(b"1234")
    h.update(b"56789")
    assert h.hexdigest() == checksum_bytes("adler32", b"123456789")


def test_checksum_file_consumes_an_iterable_of_chunks():
    assert checksum_file("sha1", [b"123", b"456789"]) == hashlib.sha1(b"123456789").hexdigest()


def test_digest_and_value_agree_for_native_checksums():
    h = new("crc32c")
    h.update(b"123456789")
    assert h.value == 0xE3069283
    assert h.digest() == (0xE3069283).to_bytes(4, "big")


def test_checksums_accept_memoryviews():
    h = new("crc32c")
    h.update(memoryview(bytearray(b"123456789")))
    assert h.value == 0xE3069283


# --------------------------------------------------------------------------
# kXR_sigver
# --------------------------------------------------------------------------


def test_sigver_hmac_is_sha256_over_seqno_header_payload():
    key, header, payload = b"k" * 32, b"h" * 24, b"body"
    expected = hmac.new(
        key, (1).to_bytes(8, "big") + header + payload, hashlib.sha256
    ).digest()
    assert sigver_hmac(key, 1, header, payload) == expected


@pytest.mark.parametrize("level", [c.kXR_secNone, c.kXR_secCompatible])
def test_nothing_is_signed_below_the_standard_level(level):
    assert is_signed(c.kXR_write, level) is False


def test_mutating_requests_are_signed_at_the_standard_level():
    assert is_signed(c.kXR_write, c.kXR_secStandard) is True
    assert is_signed(c.kXR_read, c.kXR_secStandard) is False
    # A clone writes to the destination without a byte of data on the wire,
    # so nothing but the signature says the request was not tampered with.
    assert is_signed(c.kXR_clone, c.kXR_secStandard) is True


def test_an_override_can_force_or_exempt_one_opcode():
    """The kXR_protocol security block overrides the level, either way."""
    assert is_signed(c.kXR_write, c.kXR_secStandard, {c.kXR_write: c.kXR_secNone}) is False
    assert is_signed(c.kXR_read, c.kXR_secStandard, {c.kXR_read: c.kXR_secStandard}) is True


def test_signer_increments_the_sequence_number():
    signer = Signer(b"k" * 32, c.kXR_secStandard, {})
    first = signer.sign(encode(r.Write(b"H", 0, b"a"), 5))
    second = signer.sign(encode(r.Write(b"H", 0, b"b"), 6))
    assert first is not None and second is not None
    assert second[0] == first[0] + 1


def test_signer_leaves_unsigned_requests_alone():
    signer = Signer(b"k" * 32, c.kXR_secStandard, {})
    assert signer.sign(encode(r.Read(b"H", 0, 1), 5)) is None


def test_signer_covers_the_header_and_the_payload():
    key = b"k" * 32
    signer = Signer(key, c.kXR_secStandard, {})
    frame = encode(r.Write(b"HDL0", 0, b"payload"), 5)
    seqno, mac = signer.sign(frame)
    assert mac == sigver_hmac(key, seqno, frame[:24], frame[24:])


def test_signer_repr_hides_the_session_key():
    assert "secret" not in repr(Signer(b"secretkey" * 4, c.kXR_secStandard, {}))


def test_a_signer_without_a_key_is_inert():
    assert Signer(b"", c.kXR_secStandard, {}).sign(encode(r.Write(b"H", 0, b"a"), 1)) is None


def test_a_native_checksum_prints_its_running_value():
    h = new("adler32")
    h.update(b"123456789")
    assert repr(h) == f"Checksum('adler32', {h.hexdigest()})"
    assert h.digest() == bytes.fromhex(h.hexdigest())


def test_a_hashlib_checksum_offers_the_same_surface_as_a_native_one():
    h, expected = new("sha256"), hashlib.sha256(b"123456789")
    h.update(b"123456789")
    assert h.hexdigest() == expected.hexdigest()
    assert h.digest() == expected.digest()
    assert h.value == int.from_bytes(expected.digest(), "big")
    assert repr(h) == f"Checksum('sha256', {expected.hexdigest()})"


def test_a_fresh_signer_starts_its_sequence_at_zero():
    """The counter is readable but not settable: ``sign`` is the only way up."""
    signer = Signer(b"k" * 32, c.kXR_secStandard, {})
    assert signer.seqno == 0
    signer.sign(encode(r.Write(b"H", 0, b"a"), 5))
    assert signer.seqno == 1
