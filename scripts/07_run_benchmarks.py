"""Numerical benchmarks: Optuna selection on validation, five-seed evaluation (Tables 7, 8, A2)."""
from _common import config_from, parser

from iesf.pipeline import stage_benchmarks

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--no-tune", action="store_true", help="skip Optuna and use the paper-selected configurations")
    a = p.parse_args()
    stage_benchmarks(config_from(a), a.markets, a.models, tune=not a.no_tune)
