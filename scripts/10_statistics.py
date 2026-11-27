"""Metrics, DM-HAC tests, Holm correction and block-bootstrap intervals (Tables 7-16, C1-C2, A2)."""
from _common import config_from, parser

from iesf.pipeline import stage_statistics

if __name__ == "__main__":
    stage_statistics(config_from(parser(__doc__).parse_args()))
