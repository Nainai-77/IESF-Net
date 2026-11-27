"""Expert audit of the institutional-action extraction (Section 3.6.3, Table 6).

  sample:    write outputs/audit/audit_sheet.csv (ten documents per candidate type)
  summarise: read the annotated sheet and write Table 6
"""
from _common import config_from, parser

from iesf.pipeline import stage_audit

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("mode", choices=["sample", "summarise"])
    p.add_argument("--sheet", default=None)
    a = p.parse_args()
    stage_audit(config_from(a), a.mode, a.sheet)
