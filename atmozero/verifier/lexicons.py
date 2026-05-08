"""Controlled lexicons V_k for the seven rule families."""

LEXICON_THERMODYNAMIC = (
    "warming trend",
    "cooling trend",
    "thermal stability",
    "thermal reversal",
    "dry air mass",
    "humid air mass",
)

LEXICON_RAIN_MOISTURE = (
    "light rain",
    "moderate rain",
    "heavy rain",
    "torrential rain",
    "moist advection",
    "dry advection",
    "post-frontal drying",
)

LEXICON_FRONTAL = (
    "cold front passage",
    "warm front passage",
    "occluded front",
    "pre-frontal trough",
    "post-frontal ridging",
)

LEXICON_WIND_REGIME = (
    "monsoonal southerly",
    "trade easterly",
    "westerly jet entrance",
    "post-frontal northerly",
    "gust-front transient",
    "calm regime",
)

LEXICON_DIURNAL = (
    "strong diurnal cycle",
    "suppressed diurnal cycle",
    "nocturnal heating",
    "afternoon convective peak",
    "daybreak temperature minimum",
)

LEXICON_CLIMATOLOGICAL = (
    "climatological warm anomaly",
    "climatological cold anomaly",
    "climatological wet anomaly",
    "climatological dry anomaly",
    "climatological pressure anomaly",
    "near-normal regime",
)

LEXICON_SPATIAL = (
    "regional regime",
    "locally anomalous",
    "contrasted gradient",
    "coherent advection",
    "isolated event",
)

LEXICONS = {
    1: LEXICON_THERMODYNAMIC,
    2: LEXICON_RAIN_MOISTURE,
    3: LEXICON_FRONTAL,
    4: LEXICON_WIND_REGIME,
    5: LEXICON_DIURNAL,
    6: LEXICON_CLIMATOLOGICAL,
    7: LEXICON_SPATIAL,
}


def is_in_lexicon(family_id: int, v_q: str) -> bool:
    return v_q in LEXICONS.get(family_id, ())
