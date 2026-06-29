"""Shared weather data + Monte Carlo machinery for the temperature-hedge strategies.

Strategies in ``hedge/strategies/weather_*.py`` stay thin: they call into this
package to fetch forecasts, build a predictive distribution of the official daily
high, and turn that into per-bucket ``P(YES)``. Keeping the cross-cutting logic
here (not inside any one strategy) follows the repo house rule that shared/
framework code lives outside strategy files, so every strategy is backtestable
against the same data layer.
"""

from hedge.weather.stations import STATIONS, Station, station_for_series
from hedge.weather.markets import TempMarket, parse_temp_market

__all__ = [
    "STATIONS",
    "Station",
    "station_for_series",
    "TempMarket",
    "parse_temp_market",
]
