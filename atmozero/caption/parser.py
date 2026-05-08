"""Caption parser that maps any captioner's output to a typed-claim list.

Used uniformly across baselines. Captions whose
terminating block fails to parse are scored as if every rule family were Omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from atmozero.caption.format import Claim, extract_claims


@dataclass
class ParsedCaption:
    raw: str
    claims: List[Claim]
    parse_failed: bool

    def as_q_list(self) -> List[Tuple[int, str]]:
        return [c.as_tuple() for c in self.claims]


def parse_caption(text: str) -> ParsedCaption:
    """Return the typed-claim list. `parse_failed=True` if no valid claims survive."""
    claims = extract_claims(text)
    return ParsedCaption(raw=text, claims=claims, parse_failed=len(claims) == 0)
