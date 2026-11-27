"""Table 3 (descriptive statistics, ADF, Ljung-Box), Figure 3 (descriptive CEEMDAN) and Figure 5 (partitions)."""
from _common import config_from, parser

from iesf.pipeline import stage_descriptive

if __name__ == "__main__":
    stage_descriptive(config_from(parser(__doc__).parse_args()))
