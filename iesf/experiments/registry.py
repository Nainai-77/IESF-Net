"""Every multiscale configuration evaluated in the paper, as overrides of the full IESF-Net.

Representation variants (Table 12) hold fusion at ordinary bidirectional attention;
fusion variants (Table 13) keep the textual constructs (semantic, CSI, PII memory)
and change only how they are combined; module / scale-entry ablations (Table 14)
and window experiments (Table 15) modify one component of the complete model.
"""
from __future__ import annotations

FULL = dict(constructs=("sem", "csi", "pii"), fusion="full", router=None, text_scales=(0, 1, 2),
            shared_numeric=False, lookbacks=None)

VARIANTS: dict[str, dict] = {
    # --- core multiscale models (Table 7, Table 16) --------------------------------------------
    "Numeric": dict(constructs=()),
    "IESF-Net": dict(),
    # --- textual representations, ordinary bidirectional fusion (Table 12) ------------------------
    "Rep-NewsCount": dict(constructs=("count",), fusion="bidirectional"),
    "Rep-Lexical": dict(constructs=("lex",), fusion="bidirectional"),
    "Rep-Semantic": dict(constructs=("sem",), fusion="bidirectional"),
    "Rep-SemCSI": dict(constructs=("sem", "csi"), fusion="bidirectional"),
    "Rep-PIIAttr": dict(constructs=("sem", "csi", "pii_attr"), fusion="bidirectional"),
    "Rep-PIIMemory": dict(constructs=("sem", "csi", "pii"), fusion="bidirectional"),
    # --- fusion mechanisms (Table 13; controls of Tables 9-10) ----------------------------------
    "Fusion-Static": dict(fusion="static"),
    "Fusion-Gated": dict(fusion="gated"),
    "Fusion-Num2Text": dict(fusion="n2t"),
    "Fusion-Text2Num": dict(fusion="t2n"),
    "Bidirectional": dict(fusion="bidirectional"),
    "Fusion-FeatureCond": dict(fusion="feature"),
    "Fusion-MatchingCond": dict(fusion="matching"),
    # --- module and scale-entry ablations (Table 14) ---------------------------------------------
    "Abl-NoMemory": dict(constructs=("sem", "csi", "pii_attr")),
    "Abl-NoRouter": dict(router="equal"),
    "Abl-SharedEncoder": dict(shared_numeric=True),
    "Abl-NoText-S1": dict(text_scales=(1, 2)),
    "Abl-NoText-S2": dict(text_scales=(0, 2)),
    "Abl-NoText-S3": dict(text_scales=(0, 1)),
    # --- history windows (Table 15) ------------------------------------------------------------------
    "Win-Uniform10": dict(lookbacks="uniform10"),
    "Win-Uniform30": dict(lookbacks="uniform30"),
    "Win-Uniform60": dict(lookbacks="uniform60"),
    "Win-Uniform90": dict(lookbacks="uniform90"),
}

LABELS = {
    "Numeric": "Numeric", "IESF-Net": "IESF-Net",
    "Rep-NewsCount": "News count", "Rep-Lexical": "Lexical content", "Rep-Semantic": "Semantic content",
    "Rep-SemCSI": "Semantic + CSI", "Rep-PIIAttr": "+ PII attributes", "Rep-PIIMemory": "+ PII memory",
    "Fusion-Static": "Static fusion", "Fusion-Gated": "Gated fusion", "Fusion-Num2Text": "Numerical-to-text only",
    "Fusion-Text2Num": "Text-to-numerical only", "Bidirectional": "Ordinary bidirectional",
    "Fusion-FeatureCond": "Feature conditioning", "Fusion-MatchingCond": "Matching conditioning",
    "Abl-NoMemory": "No institutional memory", "Abl-NoRouter": "No construct router",
    "Abl-SharedEncoder": "Shared numerical encoder", "Abl-NoText-S1": "Remove Scale 1 text",
    "Abl-NoText-S2": "Remove Scale 2 text", "Abl-NoText-S3": "Remove Scale 3 text",
    "Win-Uniform10": "Uniform 10", "Win-Uniform30": "Uniform 30", "Win-Uniform60": "Uniform 60",
    "Win-Uniform90": "Uniform 90",
}
SHORT_LABELS = {  # labels used in the paired-comparison tables (Tables 9-10)
    "Fusion-Static": "Static fusion", "Fusion-Gated": "Gated fusion", "Fusion-Num2Text": "Num-to-text",
    "Fusion-Text2Num": "Text-to-num", "Bidirectional": "Bidirectional", "Fusion-FeatureCond": "Feature conditioning",
    "Fusion-MatchingCond": "Matching conditioning", "Numeric": "Numeric",
}


def variant_spec(name: str) -> dict:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name}; known: {sorted(VARIANTS)}")
    spec = dict(FULL)
    spec.update(VARIANTS[name])
    return spec


def is_multiscale(name: str) -> bool:
    return name in VARIANTS
