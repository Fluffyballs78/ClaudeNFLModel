"""
QB Adjustment Impact Test
Compares backtest results with and without QB adjustments
across 2023 and 2024 seasons.
"""

import pandas as pd
from engine import NFLPowerRatingEngine, QBAdjuster, CONFIG
from engine.data_loader import load_real_data

# Load data WITH raw play-by-play (needed for QB profiles)
games, pbp = load_real_data([2022, 2023, 2024], return_pbp=True)

# Build QB profiles
print("\nBuilding QB profiles...")
qb_adj = QBAdjuster(min_attempts=30)
qb_adj.build_qb_profiles(pbp)

# Show QB report for 2023 (the problem season)
qb_adj.get_qb_report(2023, through_week=18)

# =====================================================================
# Run backtests: WITHOUT vs WITH QB adjustment
# =====================================================================

for label, adjuster in [("WITHOUT QB Adjustment", None), ("WITH QB Adjustment", qb_adj)]:
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {label}")
    print(f"{'='*60}")
    
    engine = NFLPowerRatingEngine(CONFIG)
    engine.load_data(games)
    
    # Build 2022 priors
    max_wk_2022 = games[games['season'] == 2022]['week'].max()
    engine.compute_ratings(2022, min(18, max_wk_2022))
    
    all_results = []
    
    for season in [2023, 2024]:
        results = engine.backtest(season, qb_adjuster=adjuster)
        if len(results) == 0:
            print(f"\n  {season}: No edges found")
            continue
        
        decided = results[results['ats_won'].notna()].copy()
        decided['season'] = season
        all_results.append(decided)
        
        total = len(decided)
        wins = decided['ats_won'].sum()
        losses = total - wins
        wp = wins / total * 100
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        print(f"\n  {season}: {total} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        total = len(combined)
        wins = combined['ats_won'].sum()
        losses = total - wins
        wp = wins / total * 100
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        print(f"\n  COMBINED: {total} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
        
        for lo, hi, lab in [(5, 8, "5-8 pts"), (8, 99, "8+ pts")]:
            s = combined[(combined['edge'] >= lo) & (combined['edge'] < hi)]
            if len(s) > 0:
                sw = s['ats_won'].sum()
                sl = len(s) - sw
                swp = sw / len(s) * 100
                sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
                print(f"    {lab}: {len(s)} bets  Win%: {swp:.1f}%  ROI: {sroi:+.1f}%")
