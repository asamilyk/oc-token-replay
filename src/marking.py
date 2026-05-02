"""
marking.py — Per-object token bag for OC-TBR.

Each object in the log has its own Marking instance that
tracks how many tokens it holds at each place.
"""

from __future__ import annotations
from collections import defaultdict
from .model import Place


class Marking:
    """
    Token bag for a single object: M_o : P -> N_0.

    Operations correspond directly to Algorithm 1 in the paper:
      - add(p)    : produce a token at place p
      - remove(p) : consume a token from p; returns missing count
      - total()   : count all tokens (used for r_o at end of case)
    """

    def __init__(self):
        self.tokens: dict[Place, int] = defaultdict(int)

    def add(self, place: Place, count: int = 1) -> None:
        """Produce `count` tokens at `place`."""
        self.tokens[place] += count

    def remove(self, place: Place, count: int = 1) -> int:
        """
        Consume `count` tokens from `place`.

        Returns the number of *missing* tokens (0 if sufficient).
        If tokens are missing, the caller is responsible for
        incrementing m_o and producing an artificial token
        (permissive replay, following van der Aalst 2016).
        """
        available = self.tokens.get(place, 0)
        if available >= count:
            self.tokens[place] -= count
            if self.tokens[place] == 0:
                del self.tokens[place]
            return 0
        # Not enough tokens
        missing = count - available
        if place in self.tokens:
            del self.tokens[place]
        return missing

    def count(self, place: Place) -> int:
        return self.tokens.get(place, 0)

    def total(self) -> int:
        """Total number of tokens held (= r_o contribution)."""
        return sum(self.tokens.values())

    def reset(self) -> None:
        self.tokens.clear()

    def __repr__(self) -> str:
        non_zero = {p.id: c for p, c in self.tokens.items() if c > 0}
        return f"Marking({non_zero})"
