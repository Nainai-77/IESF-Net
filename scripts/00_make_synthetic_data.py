"""Write synthetic raw data in the project schema (smoke test only)."""
from _common import config_from, parser

from iesf.pipeline import stage_synthetic

if __name__ == "__main__":
    stage_synthetic(config_from(parser(__doc__).parse_args()))
