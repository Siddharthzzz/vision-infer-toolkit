"""CPU tests: cool-down target follows methodology §3.3 as amended by A9."""

from infer_toolkit.benchmark.session import cooldown_target


def test_floor_applies_when_idle_is_cool():
    assert cooldown_target(51.0) == 60.0  # idle + 5 = 56 would be below the floor
    assert cooldown_target(54.0) == 60.0


def test_never_below_idle_plus_margin_when_idle_is_hot():
    assert cooldown_target(58.0) == 63.0  # hot idle: the original idle + 5 rule wins
