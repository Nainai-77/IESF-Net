"""Assemble model-ready arrays per market: prices, components, semantic/CSI/lexical/count and PII tensors."""
from _common import config_from, parser

from iesf.pipeline import stage_build

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    stage_build(config_from(a), a.markets)
