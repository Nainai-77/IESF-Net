"""Figures 6-16 from the stored forecasts, tables and interpretability outputs."""
from _common import config_from, parser

from iesf.pipeline import stage_figures

if __name__ == "__main__":
    stage_figures(config_from(parser(__doc__).parse_args()))
