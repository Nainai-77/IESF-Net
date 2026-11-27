"""Attention maps, routing trajectories and the c = -0.9 / +0.9 contrast of the trained IESF-Net (Figures 14-16)."""
from _common import config_from, parser

from iesf.pipeline import stage_interpret

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    stage_interpret(config_from(a), a.markets)
