"""
5-Season Backtest with Trending Filter
Compares results WITH and WITHOUT the trending filter
across 2021-2024 (2020 as prior).
"""

import pandas as pd
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load 5 seasons
games = load_real_data([2020, 2021, 2022, 2023, 2024])

def run_backtest(games, config, label):
    """Run a full 4-season backtest with given config."""
    engine = NFLPowerRatingEngine(config)
    engine.load_data(games)
    
    max_wk_2020 = games[games['season'] == 2020]['week'].max()
    engine.compute_ratings(2020, min(18, max_wk_2020))
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    for season in [2021, 2022, 2023, 2024]:
        results = engine.backtest(season)
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
        
        print(f"\n  COMBINED: {total} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
        
        # Avg bets per season
        print(f"  Avg bets/season: {total/4:.0f}")
        
        # Profitable seasons
        profitable = 0
        for s in [2021, 2022, 2023, 2024]:
            sv = combined[combined['season'] == s]
            if len(sv) > 0:
                sw = sv['ats_won'].sum()
                sl = len(sv) - sw
                if sw * 100 - sl * 110 > 0:
                    profitable += 1
        print(f"  Profitable seasons: {profitable}/4")
    
    return combined if all_results else pd.DataFrame()


# Test 1: WITHOUT trending filter (baseline)
config_no_filter = dict(CONFIG)
config_no_filter['use_trend_filter'] = False
results_no_filter = run_backtest(games, config_no_filter, "WITHOUT TRENDING FILTER (8+ pt edges)")

# Test 2: WITH trending filter (decline threshold -0.5)
config_with_filter = dict(CONFIG)
config_with_filter['use_trend_filter'] = True
config_with_filter['trend_decline_threshold'] = -0.5
results_with_filter = run_backtest(games, config_with_filter, "WITH TRENDING FILTER (skip declining teams, threshold -0.5)")

# Test 3: Tighter filter (decline threshold -0.25)
config_tight_filter = dict(CONFIG)
config_tight_filter['use_trend_filter'] = True
config_tight_filter['trend_decline_threshold'] = -0.25
results_tight = run_backtest(games, config_tight_filter, "TIGHTER TRENDING FILTER (threshold -0.25)")

# Summary comparison
print(f"\n{'='*60}")
print(f"  SUMMARY COMPARISON")
print(f"{'='*60}")
print(f"  {'Config':<40} {'Bets':>5} {'Win%':>7} {'ROI':>8} {'Prof Szns':>10}")
print(f"  {'-'*72}")

for label, results in [("No filter", results_no_filter), 
                        ("Filter -0.5", results_with_filter),
                        ("Filter -0.25", results_tight)]:
    if len(results) > 0:
        t = len(results)
        w = results['ats_won'].sum()
        l = t - w
        wp = w / t * 100
        roi = (w * 100 - l * 110) / (t * 110) * 100
        
        prof = 0
        for s in [2021, 2022, 2023, 2024]:
            sv = results[results['season'] == s]
            if len(sv) > 0:
                sw = sv['ats_won'].sum()
                sl = len(sv) - sw
                if sw * 100 - sl * 110 > 0:
                    prof += 1
        
        print(f"  {label:<40} {t:>5} {wp:>6.1f}% {roi:>+7.1f}% {prof:>7}/4")
