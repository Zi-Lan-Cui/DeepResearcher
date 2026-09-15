import pytest

from deepresearcher.service.auth import (
    TokenCodec,
    TokenError,
    hash_password,
    normalize_email,
    password_policy_ok,
    verify_password,
)

SECRET = "s" * 40


def test_password_hash_roundtrip():
    stored = hash_password("correct horse battery staple")
    assert stored != "correct horse battery staple"
    assert verify_password("correct horse battery staple", stored)
    assert not verify_password("wrong", stored)


def test_verify_password_tolerates_garbage_hash():
    assert not verify_password("anything", "not-a-real-hash")


def test_email_normalization():
    assert normalize_email("  User@Test.COM ") == "user@test.com"


def test_password_policy_bounds():
    assert not password_policy_ok("x" * 7)
    assert password_policy_ok("x" * 8)
    assert password_policy_ok("x" * 128)
    assert not password_policy_ok("x" * 129)


def test_short_secret_rejected():
    with pytest.raises(ValueError, match="32"):
        TokenCodec("too-short")


def test_token_roundtrip_returns_user_id():
    codec = TokenCodec(SECRET)
    assert codec.decode(codec.encode(42)) == 42


def test_tampered_token_rejected():
    codec = TokenCodec(SECRET)
    token = codec.encode(1)
    with pytest.raises(TokenError):
        codec.decode(token[:-2] + "xx")


def test_token_from_other_secret_rejected():
    other = TokenCodec("t" * 40)
    with pytest.raises(TokenError):
        TokenCodec(SECRET).decode(other.encode(1))


def test_expired_token_rejected_with_injected_clock():
    # PyJWT 的 decode 用系统真实时钟（无法注入）；把“签发时刻”拨到过去
    # 两小时前，TTL 1h → 按真实时间判过期。
    import time

    real_now = time.time()
    codec = TokenCodec(SECRET, ttl_hours=1, now=lambda: real_now - 2 * 3600)
    token = codec.encode(7)
    with pytest.raises(TokenError):
        TokenCodec(SECRET, ttl_hours=1).decode(token)
