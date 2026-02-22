"""
Multi-Season Backtest
Run this to see how the model performs across 2023 and 2024.
2022 is used as the prior-building season.
"""

import pandas as pd
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load all three seasons
games = load_real_data([2022, 2023, 2024])
engine = NFLPowerRatingEngine(CONFIG)
engine.load_data(games)

# Build 2022 ratings as priors
max_wk_2022 = games[games['season'] == 2022]['week'].max()
engine.compute_ratings(2022, min(18, max_wk_2022))

# Backtest each season
all_results = []

for season in [2023, 2024]:
    results = engine.backtest(season)
    if len(results) == 0:
        print(f"\n=== {season} SEASON ===")
        print("No edges found")
        continue
    
    decided = results[results['ats_won'].notna()].copy()
    decided['season'] = season
    all_results.append(decided)
    
    total = len(decided)
    wins = decided['ats_won'].sum()
    losses = total - wins
    wp = wins / total * 100
    roi = (wins * 100 - losses * 110) / (total * 110) * 100
    
    print(f"\n{'='*55}")
    print(f"  {season} SEASON")
    print(f"{'='*55}")
    print(f"  Bets: {total}  Record: {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    for lo, hi, lab in [(5, 8, "5-8 pts"), (8, 99, "8+ pts")]:
        s = decided[(decided['edge'] >= lo) & (decided['edge'] < hi)]
        if len(s) > 0:
            sw = s['ats_won'].sum()
            sl = len(s) - sw
            swp = sw / len(s) * 100
            sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
            print(f"    {lab}: {len(s)} bets  Win%: {swp:.1f}%  ROI: {sroi:+.1f}%")

# Combined results
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    total = len(combined)
    wins = combined['ats_won'].sum()
    losses = total - wins
    wp = wins / total * 100
    roi = (wins * 100 - losses * 110) / (total * 110) * 100
    
    print(f"\n{'='*55}")
    print(f"  COMBINED 2023-2024")
    print(f"{'='*55}")
    print(f"  Bets: {total}  Record: {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    for lo, hi, lab in [(5, 8, "5-8 pts"), (8, 99, "8+ pts")]:
        s = combined[(combined['edge'] >= lo) & (combined['edge'] < hi)]
        if len(s) > 0:
            sw = s['ats_won'].sum()
            sl = len(s) - sw
            swp = sw / len(s) * 100
            sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
            print(f"    {lab}: {len(s)} bets  Win%: {swp:.1f}%  ROI: {sroi:+.1f}%")
    
    # Weekly distribution across both seasons
    print(f"\n  Bets per week (both seasons combined):")
    weekly = combined.groupby('week')['game_id'].count()
    print(f"  {weekly.to_string()}")
