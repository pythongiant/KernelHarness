import pytest

from harness.launch import LaunchExprError, evaluate


def test_whitelisted_arithmetic():
    assert evaluate("ceildiv(1048576, 256)") == 4096
    assert evaluate("2*N + 3", {"N": 10}) == 23
    assert evaluate("min(4, 9) // 2") == 2
    assert evaluate("sqrt(81)") == 9
    assert evaluate("8 % 3") == 2


def test_rejects_arbitrary_code():
    with pytest.raises(LaunchExprError):
        evaluate("__import__('os')")
    with pytest.raises(LaunchExprError):
        evaluate("open('/etc/passwd')")
    with pytest.raises(LaunchExprError):
        evaluate("x for x in range(3)")
    with pytest.raises(LaunchExprError):
        evaluate("missing + 1")


def test_rejects_non_integral():
    assert evaluate("7 / 2") == 3  # integer division is the documented meaning of `/`
    with pytest.raises(LaunchExprError):
        evaluate("1.5 + 0.25")
