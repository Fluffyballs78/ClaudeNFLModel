# NFL Power Rating & Spread Prediction Engine

A Python-based NFL power rating system that generates predicted spreads from play-by-play EPA data, compares them to market lines, and identifies betting edges.

## How It Works

The engine computes a single power rating per team (points above/below league average) using four components:

- **Offensive EPA per play** — opponent-adjusted, recency-weighted
- **Defensive EPA per play** — opponent-adjusted, recency-weighted
- **Turnover-adjusted efficiency** — regressed toward league average to filter noise
- **Special teams EPA** — smaller weight, but captures hidden field position value

Ratings are opponent-adjusted through 10 iterations and blended with prior-season data early in the year. The final predicted spread for any matchup is:

```
Predicted Spread = (Away Rating - Home Rating) - Home Field Advantage
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/nfl-spread-engine.git
cd nfl-spread-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run with real NFL data
python3 main.py

# Run with synthetic data (no download needed)
python3 main.py --synthetic
```

## Project Structure

```
nfl-spread-engine/
├── main.py                  # Entry point — run this
├── engine/
│   ├── __init__.py
│   ├── power_ratings.py     # Core rating engine
│   ├── data_loader.py       # Real + synthetic data loading
│   └── config.py            # All tunable parameters
├── requirements.txt
└── README.md
```

## Configuration

All tunable parameters are in `engine/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recency_decay` | 0.92 | Weight decay per game back (0.92 = last game is 100%, game before is 92%, etc.) |
| `home_field_advantage` | 2.5 | League-wide HFA in points |
| `turnover_regression_factor` | 0.50 | How much to regress turnovers toward league avg (1.0 = ignore turnovers) |
| `min_edge_threshold` | 2.0 | Minimum model-vs-market gap to flag as a bet |
| `weight_off_epa` | 0.35 | Offensive EPA weight in composite rating |
| `weight_def_epa` | 0.35 | Defensive EPA weight in composite rating |
| `weight_turnover_adj` | 0.20 | Turnover adjustment weight |
| `weight_special_teams` | 0.10 | Special teams weight |

## Data Sources

- **Play-by-play**: [nflverse/nflfastR](https://github.com/nflverse/nflverse-data) via `nfl_data_py`
- **Spreads**: Included in nflfastR data (`spread_line` column = Vegas closing spread)

## Roadmap

- [ ] Tune parameters via grid search backtest
- [ ] Add line movement data (The Odds API)
- [ ] Add public betting percentage overlay
- [ ] Weekly automated picks report
- [ ] Injury report integration
- [ ] Weather adjustment for outdoor games
