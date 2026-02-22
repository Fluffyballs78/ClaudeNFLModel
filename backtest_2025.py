"""
2025 Out-of-Sample Backtest
This is the real test — 2025 data was never used to build or tune the model.
Uses 2020-2024 as historical data, backtests 2025.
"""

import pandas as pd
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load all 6 seasons
print("Loading 2020-2025 play-by-play data...")
games = load_real_data([2020, 2021, 2022, 2023, 2024, 2025])

engine = NFLPowerRatingEngine(CONFIG)
engine.load_data(games)

# Build historical ratings (2020-2024)
print("\nBuilding historical ratings...")
for season in [2020, 2021, 2022, 2023, 2024]:
    max_wk = games[games['season'] == season]['week'].max()
    engine.compute_ratings(season, min(18, max_wk))
    print(f"  {season}: Computed through Week {min(18, max_wk)}")

# Show 2025 power ratings
print("\n")
engine.print_ratings(2025, 18)

# Backtest 2025
print("\n[BACKTEST] Running 2025 out-of-sample test...")
results = engine.backtest(2025)
decided = results[results['ats_won'].notna()].copy()

engine.print_backtest_summary(results)

# Weekly breakdown
if len(decided) > 0:
    print(f"\n  Weekly breakdown:")
    print(f"  {'Week':<6} {'Bets':>5} {'W-L':>7} {'Cum W%':>8} {'Cum ROI':>9}")
    print(f"  {'-'*40}")
    
    cum_w = 0
    cum_l = 0
    for week in sorted(decided['week'].unique()):
        wk = decided[decided['week'] == week]
        w = wk['ats_won'].sum()
        l = len(wk) - w
        cum_w += w
        cum_l += l
        cum_wp = cum_w / (cum_w + cum_l) * 100
        cum_roi = (cum_w * 100 - cum_l * 110) / ((cum_w + cum_l) * 110) * 100
        print(f"  {int(week):<6} {len(wk):>5} {int(w)}-{int(l):>4} {cum_wp:>7.1f}% {cum_roi:>+8.1f}%")

# Full season comparison
print(f"\n{'='*60}")
print(f"  ALL SEASONS COMPARISON (with trending filter)")
print(f"{'='*60}")
print(f"  {'Season':<8} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
print(f"  {'-'*30}")

for season in [2021, 2022, 2023, 2024, 2025]:
    r = engine.backtest(season)
    d = r[r['ats_won'].notna()]
    if len(d) > 0:
        t = len(d)
        w = d['ats_won'].sum()
        l = t - w
        wp = w / t * 100
        roi = (w * 100 - l * 110) / (t * 110) * 100
        print(f"  {season:<8} {t:>5} {wp:>6.1f}% {roi:>+7.1f}%")
