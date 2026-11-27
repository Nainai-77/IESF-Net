"""Validation-only Optuna TPE selection of IESF-Net (Table A3, Table A4 steps 1-3)."""
from _common import config_from, parser

from iesf.pipeline import stage_tune

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--variants", nargs="*", default=["IESF-Net"])
    a = p.parse_args()
    stage_tune(config_from(a), a.markets, a.variants)
