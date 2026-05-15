# Strategies have been split into individual files.
# This file is kept only for backwards compatibility.
# Import from the strategies package directly instead:
#
#   from strategies import get_all_strategies, get_strategy
#   from strategies.momentum import MomentumStrategy

from strategies import get_all_strategies, get_strategy, REGISTRY

__all__ = ["get_all_strategies", "get_strategy", "REGISTRY"]
