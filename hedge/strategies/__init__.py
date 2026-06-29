"""Monte Carlo strategies live here.

Each strategy subclasses `Strategy` (see base.py) and returns `Signal`s. Drop a
new file in this package, implement `evaluate`, and register it. See CLAUDE.md
for the full plug-and-play guide.
"""

from hedge.strategies.base import Strategy

__all__ = ["Strategy"]
