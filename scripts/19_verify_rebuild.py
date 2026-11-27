"""Verify a rebuild of the model inputs against the published hashes.

After running stages 01-05 from the raw data, every array of every processed market file is
hashed (SHA-256 of its bytes, with shape and dtype) and compared with
evidence/processed_arrays_manifest.csv; the alignment, PII-event, component and grouping
files are compared with evidence/data_manifest.csv.  Array hashes do not depend on the
compression metadata of the .npz container.

Bytes can differ by floating-point rounding when the numerical libraries differ (for example
OpenBLAS instead of MKL).  For every array whose hash differs, the report therefore gives the
differences of its summary statistics (minimum, maximum, sums), and for every grouping file
that differs, the largest numerical difference from the published copy in evidence/metadata/.

    python scripts/19_verify_rebuild.py --evidence evidence
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _common import ROOT, config_from, parser

from iesf.evaluation.leakage import array_hashes, compare_summaries
from iesf.utils import get_logger, save_json

LOG = get_logger("verify")
FILE_PATTERNS = ("alignment/*_document_alignment.csv", "decomposition/*_components.csv",
                 "decomposition/*_grouping.json", "*_pii_events.csv")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_difference(a: dict, b: dict) -> float | None:
    """Largest absolute difference between the numerical entries of two flat JSON objects."""
    diffs = []
    for k in a.keys() | b.keys():
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float, list)) and isinstance(y, (int, float, list)) and not isinstance(x, bool):
            diffs.append(float(np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)))))
        elif x != y:
            return None                            # a non-numerical entry differs
    return max(diffs) if diffs else 0.0


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--evidence", default=str(ROOT / "evidence"))
    p.add_argument("--out", default=None, help="JSON report (default: <output_dir>/audit/rebuild_verification.json)")
    a = p.parse_args()
    cfg = config_from(a)
    ev = Path(a.evidence)
    proc, interim = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["interim_dir"])
    ref = pd.read_csv(ev / "processed_arrays_manifest.csv")
    got = array_hashes(proc)
    m = ref.merge(got, on=["file", "array"], how="outer", suffixes=("_published", "_rebuilt"), indicator=True)
    m["identical"] = (m["_merge"] == "both") & (m["sha256_published"] == m["sha256_rebuilt"])
    differing = []
    for r in m[~m["identical"]].to_dict("records"):
        pub = {k: r.get(f"{k}_published") for k in ("sum", "sum_abs", "min", "max")}
        reb = {k: r.get(f"{k}_rebuilt") for k in ("sum", "sum_abs", "min", "max")}
        differing.append({"file": r["file"], "array": r["array"],
                          "same_shape_and_dtype": r.get("shape_published") == r.get("shape_rebuilt")
                          and r.get("dtype_published") == r.get("dtype_rebuilt"),
                          "summary_differences": compare_summaries(pub, reb)})
    manifest = pd.read_csv(ev / "data_manifest.csv")
    files = []
    for pattern in FILE_PATTERNS:
        for f in sorted(list(interim.glob(pattern)) + list(proc.glob(pattern))):
            tail = "/".join(f.as_posix().split("/")[-2:]) if "/" in pattern else f.name
            hit = manifest[manifest["path"].str.endswith(tail)]
            row = {"file": tail, "published": hit["sha256"].iloc[0] if len(hit) else None, "rebuilt": file_sha(f)}
            published_copy = ev / "metadata" / f.name
            if row["published"] != row["rebuilt"] and f.suffix == ".json" and published_copy.exists():
                row["max_abs_numerical_difference"] = json_difference(json.loads(published_copy.read_text(encoding="utf-8")),
                                                                      json.loads(f.read_text(encoding="utf-8")))
            files.append(row)
    file_diff = [{k: v for k, v in r.items() if k not in ("published", "rebuilt")} for r in files if r["published"] != r["rebuilt"]]
    report = {"arrays_compared": int(len(m)), "arrays_identical": int(m["identical"].sum()),
              "arrays_differing": differing,
              "files_compared": len(files), "files_identical": len(files) - len(file_diff),
              "files_differing": file_diff}
    out = Path(a.out) if a.out else Path(cfg["paths"]["output_dir"]) / "audit" / "rebuild_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, out)
    LOG.info("arrays identical %d/%d, files identical %d/%d -> %s", report["arrays_identical"], report["arrays_compared"],
             report["files_identical"], report["files_compared"], out)
