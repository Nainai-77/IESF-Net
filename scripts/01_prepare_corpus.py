"""Clean and deduplicate the document corpus and align it to every market calendar (Section 3.6.2, Table 5)."""
from _common import config_from, parser

from iesf.pipeline import stage_corpus

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    stage_corpus(config_from(a), a.markets)
