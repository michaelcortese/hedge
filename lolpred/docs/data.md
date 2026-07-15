# Data: Oracle's Elixir match CSVs

Raw data is Oracle's Elixir (OE) per-game match exports: 12 rows per game
(participantid 1–10 player rows + 100/200 team rows, side Blue/Red), one CSV
per season year, ~160 columns in recent years (older years have fewer).

## Sources

| source | what | notes |
|---|---|---|
| Google Drive (canonical) | per-year files `oe_{year}.csv`, 2014–2026 | file ids in `data/raw/oracleselixir_gdrive_manifest.tsv` (columns: year, file_id, bytes, filename). **Quota caveat:** Drive rate-limits downloads; when exceeded it serves an HTML page instead of the CSV. `download_data.py` detects this, deletes the bad file and tells you to retry later. |
| Hugging Face | combined 2014–2023 (`oe_2014_2023.csv`, ~517 MB) | `eligrayy/OE-LoL-Esports-Dataset` — reliable bulk history, same schema family. |
| GitHub mirror | 2025 season (`oe_2025.csv`) | `arthurcvl/LeagueCompetitiveStats` repo mirror of the 2025 OE file. |
| Leaguepedia | live/current results | for inference-time updates, not backtesting; no OE-format stats. |

Overlapping years across sources (e.g. 2023 in both the HF dump and a
per-year file) are safe: the loader dedupes on `gameid`, first file wins.

## Downloading

```bash
# everything (HF bulk + 2025 mirror + gdrive for 2024/2026):
.venv/bin/python scripts/download_data.py --dest data/raw --years 2014-2026 --source auto

# a specific source / years:
.venv/bin/python scripts/download_data.py --source gdrive --years 2024,2026
```

Idempotent: existing valid files (first line starts with `gameid`; the HF
bulk file additionally must be >400 MB) are never re-downloaded. Downloads
stream with progress every ~50 MB and are validated before being kept.

## What `load_games` guarantees

`lolpred.data.loader.load_games(paths, min_datacompleteness="any", verbose=False)`
accepts one path, many paths, a directory (all `*.csv` under it) or a glob,
and returns the canonical table of CONTRACTS.md §1:

- exactly **one row per game**, blue/red oriented (`blue_*` / `red_*`),
  deduplicated on `gameid` (first file wins), sorted by `(date, gameid)`,
  index reset;
- `blue_win` int 0/1 consistent with both team rows (`blue+red result == 1`
  enforced); `date` a Timestamp; `year`/`playoffs`/`game_in_series` ints;
  `gamelength` float seconds; stat columns float (`errors="coerce"`);
- `series_id = f"{date.date()}|{league}|{'|'.join(sorted([blue, red]))}"`;
- `blue_players`/`red_players`: sorted `"|"`-joined starter names from the
  player rows, `""` when player rows are unavailable;
- games are **dropped** (and counted in `df.attrs["drop_counts"]`, logged
  when `verbose=True`) when they lack exactly one Blue and one Red team row,
  have a missing/`unknown team` team name, missing or inconsistent results,
  or a missing date / year 0.

`filter_games(df, leagues=None, min_year=None, complete_only=False)` narrows
by league / year; `complete_only` keeps `datacompleteness == "complete"`.

## Known quirks

- **`datacompleteness == "partial"`**: games scraped without full timeline
  data; the @15 columns are NaN there. Kept by default — use
  `min_datacompleteness="complete"` or `filter_games(complete_only=True)`
  when a model needs the @15 features.
- **Missing @15 columns** in some leagues/years entirely (notably older
  years and some minor leagues): loader adds them as NaN so the schema is
  stable across the full 2014→present range.
- **Google Drive quota-exceeded HTML** masquerading as a CSV — handled by
  the downloader (see above); if you download by hand, check the first line
  starts with `gameid`.
- **Duplicate `gameid`s across files** when year coverage overlaps between
  sources — deduped by the loader, first file wins.
- **`unknown team` / missing team names** and **year 0 / missing dates**
  appear in early/partial data — those games are dropped and counted.
- **Season labels vs calendar dates**: a season file can contain games
  labeled with the *next* season's `year` (e.g. ~164 late-2025 games in
  `oe_2025.csv` carry `year == 2026`, HLL). Filter by `date` when you mean
  calendar time, `year` when you mean season.
