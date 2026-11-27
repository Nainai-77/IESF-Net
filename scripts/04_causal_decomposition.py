"""Origin-wise causal CEEMDAN and training-boundary frequency grouping (Section 3.3, Table A1)."""
from _common import config_from, parser

from iesf.pipeline import stage_decompose

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    stage_decompose(config_from(a), a.markets)
