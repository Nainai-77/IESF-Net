"""Expert audit of the institutional-action extraction (Section 3.6.3, Table 6).

``sample_audit`` draws a stratified sample of documents (ten per candidate type
emitted by the rule layer) and writes an annotation sheet; the expert fills the
two decision columns blind to the model.  ``summarise_audit`` produces Table 6.

Decision codes
    expert_category:  correct | adjacent | not_qualifying
    expert_mechanism: correct | inconsistent
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORY_CODES = {"correct": "Correct", "adjacent": "Adjacent category preferred", "not_qualifying": "Not a qualifying action"}
MECHANISM_CODES = {"correct": "Correct", "inconsistent": "Inconsistent with expert judgement"}


def sample_audit(candidates: pd.DataFrame, docs: pd.DataFrame, per_type: int = 10, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    body = docs.set_index("document_id")["body"].to_dict()
    rows = []
    for ctype, grp in candidates.groupby("candidate_type"):
        first_per_doc = grp.drop_duplicates("document_id")
        take = min(per_type, len(first_per_doc))
        pick = first_per_doc.iloc[rng.choice(len(first_per_doc), size=take, replace=False)]
        for r in pick.to_dict("records"):
            primary = r["categories"][0]
            rows.append({
                "candidate_type": ctype, "document_id": r["document_id"], "candidate_id": r["candidate_id"],
                "title": r["title"], "evidence": r["evidence"], "full_text": body.get(r["document_id"], ""),
                "assigned_categories": "|".join(r["categories"]), "assigned_mechanism": r["mechanisms"][primary],
                "expert_category": "", "expert_mechanism": "", "expert_note": "",
            })
    return pd.DataFrame(rows)


def summarise_audit(sheet: pd.DataFrame) -> pd.DataFrame:
    n = len(sheet)
    rows = []
    for code, label in CATEGORY_CODES.items():
        k = int((sheet["expert_category"].str.strip() == code).sum())
        rows.append({"Decision": "Category assignment", "Outcome": label, "Documents": k, "Share (%)": round(100 * k / n, 1)})
    for code, label in MECHANISM_CODES.items():
        k = int((sheet["expert_mechanism"].str.strip() == code).sum())
        rows.append({"Decision": "Mechanism direction", "Outcome": label, "Documents": k, "Share (%)": round(100 * k / n, 1)})
    return pd.DataFrame(rows)
