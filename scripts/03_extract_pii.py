"""Institutional-action extraction and canonical register (Sections 3.2.2, 3.6.2; Table 4)."""
from _common import config_from, parser

from iesf.pipeline import stage_pii

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--actions", default=None, help="use an externally reviewed action register (JSONL) instead of the extractor")
    a = p.parse_args()
    stage_pii(config_from(a), a.actions)
