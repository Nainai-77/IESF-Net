"""Train and forecast the multiscale suites: main, representation, fusion, ablation, window, cross_market."""
from _common import config_from, parser

from iesf.pipeline import stage_experiments

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--suites", nargs="*", default=None)
    p.add_argument("--force", action="store_true", help="re-run even if forecasts exist")
    a = p.parse_args()
    stage_experiments(config_from(a), a.suites, a.markets, a.force)
