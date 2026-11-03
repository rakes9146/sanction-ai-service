"""
floors.py - safety floors (guardrails G-01 ... G-11).

WHAT: a floor is a rule such as "a banned BIC must always be seen by a human".  It names a MINIMUM review level.
WHY:  a floor can only RAISE the level of a payment. Nothing, including the AI, can lower it.

Levels in order:  AUTO_CLEAR  <  L2  <  L3
    AUTO_CLEAR  nobody needs to look   |   L2  the AI L2 pass must run (and, by default, a human decides)   |   L3  a human officer decides
The table of floors lives in config/floors.yml.
"""
from __future__ import annotations


def level_rank(level: str, cfg: dict) -> int:
    """AUTO_CLEAR -> 0, L2 -> 1, L3 -> 2"""
    return cfg["levels"].index(level)


def max_level(a: str, b: str, cfg: dict) -> str:
    return a if level_rank(a, cfg) >= level_rank(b, cfg) else b


def floor_level(floor: str, cfg: dict) -> str:
    """The minimum level one floor demands. An unknown floor id is treated as L3 (the safe side)."""
    return cfg["floors"].get(floor, "L3")


def highest_floor_level(floors: list[str], cfg: dict) -> str:
    """The strictest level among the floors; AUTO_CLEAR when there are none."""
    level = "AUTO_CLEAR"
    for f in floors:
        level = max_level(level, floor_level(f, cfg), cfg)
    return level


def union(*lists: list[str]) -> list[str]:
    """Merge floor lists, no duplicates, stable order (G-01 before G-05 ...)."""
    seen: dict[str, None] = {}
    for items in lists:
        for item in items:
            seen[item] = None
    return sorted(seen)
