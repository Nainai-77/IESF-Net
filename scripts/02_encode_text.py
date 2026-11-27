"""Frozen BERT semantic vectors and FinBERT tone probabilities for every document (Section 3.2.1)."""
from _common import config_from, parser

from iesf.pipeline import stage_encode, stage_import_encoded

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--mock-encoders", action="store_true", help="hashing/lexicon stand-ins (smoke test only)")
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--import-dir", default=None,
                   help="import encodings computed elsewhere with the same frozen encoders (revisions are checked)")
    a = p.parse_args()
    if a.import_dir:
        stage_import_encoded(config_from(a), a.import_dir)
    else:
        stage_encode(config_from(a), mock=a.mock_encoders, chunk=a.chunk)
