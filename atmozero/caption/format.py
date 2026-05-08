"""Structured caption format and shared prompt template.

Captions terminate with a CLAIMS block::

    CLAIMS:
    - (family_name, "value_in_lexicon")
    ...

Free prose may precede the CLAIMS block but is ignored by the verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from atmozero.verifier.lexicons import LEXICONS, is_in_lexicon


FAMILY_NAME_BY_ID = {
    1: "thermodynamic",
    2: "rain_moisture",
    3: "frontal",
    4: "wind_regime",
    5: "diurnal",
    6: "climatological",
    7: "spatial",
}
# Long-form aliases accepted by the parser (rule classes name family k=7
# "spatial_coherence"; the prompt template uses the short tag "spatial").
FAMILY_ID_BY_NAME = {v: k for k, v in FAMILY_NAME_BY_ID.items()}
FAMILY_ID_BY_NAME["spatial_coherence"] = 7
FAMILY_ID_BY_NAME["diurnal_cycle"] = 5


@dataclass
class Claim:
    family_id: int
    family_name: str
    value: str

    def as_tuple(self) -> Tuple[int, str]:
        return (self.family_id, self.value)


CLAIMS_HEADER = "CLAIMS:"
_CLAIM_LINE_RE = re.compile(
    r"-\s*\(?\s*(?P<family>[a-zA-Z_-]+)\s*,\s*[\"'“‘](?P<value>[^\"'”’]+)[\"'”’]\s*\)?",
    re.UNICODE,
)
_LOOSE_LINE_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:[-*]|\d+[\.)])[ \t]*[\"'“‘]?(?P<value>[^\n\"'”’()]+?)[\"'”’]?[ \t]*(?=\n|$)"
)


def render_claims_block(claims: List[Claim]) -> str:
    lines = [CLAIMS_HEADER]
    for c in claims:
        lines.append(f"- ({c.family_name}, \"{c.value}\")")
    return "\n".join(lines)


def _lookup_family_for_value(value: str) -> Optional[int]:
    v = value.strip().lower()
    for fid, lex in LEXICONS.items():
        if v in lex:
            return fid
    return None


def extract_claims(caption: str) -> List[Claim]:
    """Recover the typed-claim list from a free-form caption.

    Tries the strict ``- (family, "value")`` pattern, then falls back to a
    loose sweep over numbered/bulleted lines, then to a comma-separated
    inline pass. The fallback gives baseline captioners a fair shake even
    when they ignore the structured CLAIMS format.
    """
    if CLAIMS_HEADER in caption:
        block = caption.split(CLAIMS_HEADER, 1)[1]
    else:
        block = caption

    out: List[Claim] = []
    seen = set()

    for m in _CLAIM_LINE_RE.finditer(block):
        family = m.group("family").strip().lower().replace("-", "_")
        value = m.group("value").strip().lower()
        fid = FAMILY_ID_BY_NAME.get(family)
        if fid is None or not is_in_lexicon(fid, value):
            continue
        key = (fid, value)
        if key in seen:
            continue
        seen.add(key)
        out.append(Claim(family_id=fid, family_name=family, value=value))

    if not out:
        for m in _LOOSE_LINE_RE.finditer("\n" + block + "\n"):
            value = m.group("value").strip().lower()
            value = value.split(",")[0].strip(" \"'")
            fid = _lookup_family_for_value(value)
            if fid is None:
                continue
            key = (fid, value)
            if key in seen:
                continue
            seen.add(key)
            out.append(Claim(family_id=fid, family_name=FAMILY_NAME_BY_ID[fid], value=value))

    if not out:
        first_line = block.strip().split("\n")[0]
        for raw in first_line.split(","):
            value = raw.strip().lower().strip(" \"'.")
            if not value:
                continue
            fid = _lookup_family_for_value(value)
            if fid is None:
                continue
            key = (fid, value)
            if key in seen:
                continue
            seen.add(key)
            out.append(Claim(family_id=fid, family_name=FAMILY_NAME_BY_ID[fid], value=value))

    return out


SYSTEM_PROMPT = """You are an expert in weather science writing factually faithful captions for multivariate weather time series.

Each caption must end with a CLAIMS: block whose entries are drawn ONLY from the seven station-observable rule families and their controlled lexicons listed below. Do not invent claim values. Do not assert phenomena unsupported by the window.
""".strip()


PROMPT_TEMPLATE = """Window length: {T_w} hours.
Channels: {channels}.
Station: latitude {lat:.2f}, longitude {lon:.2f}, elevation {elev:.0f} m, Köppen zone {koppen}.

Numeric summary:
{numeric_summary}

Available rule families and admissible claim values:
{lexicon_block}

Write a brief prose summary (2--4 sentences) of the regime, then a typed-claim list:

CLAIMS:
- (<family>, "<value>")
- ...

Constraints:
- family ∈ {{thermodynamic, rain_moisture, frontal, wind_regime, diurnal, climatological, spatial}}.
- value ∈ V_family (the lexicon for that family).
- Omit a family when the data does not support any of its values.
- Do not assert claims contradicted by the numeric summary.
""".strip()


def render_lexicon_block() -> str:
    rows = []
    for fid, name in FAMILY_NAME_BY_ID.items():
        vals = ", ".join(f"\"{v}\"" for v in LEXICONS[fid])
        rows.append(f"- {name} (k={fid}): {vals}")
    return "\n".join(rows)
