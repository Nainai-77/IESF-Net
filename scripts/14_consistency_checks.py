"""Timing, reconstruction and partition checks (Appendix B.2)."""
from _common import config_from, parser

from iesf.pipeline import stage_checks

if __name__ == "__main__":
    stage_checks(config_from(parser(__doc__).parse_args()))
