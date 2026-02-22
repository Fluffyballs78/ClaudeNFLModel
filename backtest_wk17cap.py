"""
5-Season Backtest — Week 17 Cap
Excludes Week 18 (rest/meaningless games) from betting.
Tests 2021-2025 with 2020 as prior.
"""

import pandas as pd
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load all 6 seasons
games = load_real_data([2020, 2021, 2022, 2023, 2024, 2025])

# Test both week 17 and week 18 caps side by side
for max_week, label in [(18, "Through Week 18 (current)"), (17, "Through Week 17 (exclude Wk 18)")]:
    engine = NFLPowerRatingEngine(CONFIG)
    engine.load_data(games)
    
    # Build 2020 priors
    max_wk_2020 = games[games['season'] == 2020]['week'].max()
    engine.compute_ratings(2020, min(18, max_wk_2020))
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    for season in [2021, 2022, 2023, 2024, 2025]:
        results = engine.backtest(season, max_week=max_week)
        if len(results) == 0:
            print(f"  {season}: No edges found")
            continue
        
        decided = results[results['ats_won'].notna()].copy()
        decided['season'] = season
        all_results.append(decided)
        
        total = len(decided)
        wins = decided['ats_won'].sum()
        losses = total - wins
        wp = wins / total * 100
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        print(f"  {season}: {total} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        total = len(combined)
        wins = combined['ats_won'].sum()
        losses = total - wins
        wp = wins / total * 100
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        prof = 0
        for s in [2021, 2022, 2023, 2024, 2025]:
            sv = combined[combined['season'] == s]
            if len(sv) > 0:
                sw = sv['ats_won'].sum()
                sl = len(sv) - sw
                if sw * 100 - sl * 110 > 0:
                    prof += 1
        
        print(f"\n  COMBINED: {total} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
        print(f"  Avg bets/season: {total/5:.0f}")
        print(f"  Profitable seasons: {prof}/5")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Week 18 games are often meaningless (resting starters,")
print(f"  teams locked into seeds, tanking for draft position).")
print(f"  The model can't account for team motivation.")
