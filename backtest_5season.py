"""
5-Season Backtest (2021-2024)
2020 used as prior-building season.
Tests model across 4 full seasons of betting to determine
if there's real signal or just variance.
"""

import pandas as pd
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load 5 seasons
games = load_real_data([2020, 2021, 2022, 2023, 2024])

engine = NFLPowerRatingEngine(CONFIG)
engine.load_data(games)

# Build 2020 ratings as priors
max_wk_2020 = games[games['season'] == 2020]['week'].max()
engine.compute_ratings(2020, min(18, max_wk_2020))

# Backtest 2021-2024
all_results = []

for season in [2021, 2022, 2023, 2024]:
    results = engine.backtest(season)
    if len(results) == 0:
        print(f"\n{season}: No edges found")
        continue
    
    decided = results[results['ats_won'].notna()].copy()
    decided['season'] = season
    all_results.append(decided)
    
    total = len(decided)
    wins = decided['ats_won'].sum()
    losses = total - wins
    wp = wins / total * 100
    roi = (wins * 100 - losses * 110) / (total * 110) * 100
    
    # Favorite vs underdog breakdown
    fav_bets = 0
    fav_wins = 0
    dog_bets = 0
    dog_wins = 0
    for _, row in decided.iterrows():
        if row['market_spread'] == 0:
            continue
        betting_home = row['bet_side'] == row['home_team']
        home_is_fav = row['market_spread'] > 0
        if (betting_home and home_is_fav) or (not betting_home and not home_is_fav):
            fav_bets += 1
            fav_wins += row['ats_won']
        else:
            dog_bets += 1
            dog_wins += row['ats_won']
    
    print(f"\n{'='*60}")
    print(f"  {season} SEASON")
    print(f"{'='*60}")
    print(f"  Bets: {total}  Record: {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    for lo, hi, lab in [(5, 8, "5-8 pts"), (8, 99, "8+ pts")]:
        s = decided[(decided['edge'] >= lo) & (decided['edge'] < hi)]
        if len(s) > 0:
            sw = s['ats_won'].sum()
            sl = len(s) - sw
            swp = sw / len(s) * 100
            sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
            print(f"    {lab}: {len(s)} bets  Win%: {swp:.1f}%  ROI: {sroi:+.1f}%")
    
    if dog_bets > 0:
        print(f"    Underdogs: {dog_bets} bets  Win%: {dog_wins/dog_bets*100:.1f}%")
    if fav_bets > 0:
        print(f"    Favorites: {fav_bets} bets  Win%: {fav_wins/fav_bets*100:.1f}%")

# Combined results
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    total = len(combined)
    wins = combined['ats_won'].sum()
    losses = total - wins
    wp = wins / total * 100
    roi = (wins * 100 - losses * 110) / (total * 110) * 100
    
    print(f"\n{'='*60}")
    print(f"  COMBINED 2021-2024 (4 seasons)")
    print(f"{'='*60}")
    print(f"  Bets: {total}  Record: {int(wins)}-{int(losses)}  Win%: {wp:.1f}%  ROI: {roi:+.1f}%")
    
    for lo, hi, lab in [(5, 8, "5-8 pts"), (8, 99, "8+ pts")]:
        s = combined[(combined['edge'] >= lo) & (combined['edge'] < hi)]
        if len(s) > 0:
            sw = s['ats_won'].sum()
            sl = len(s) - sw
            swp = sw / len(s) * 100
            sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
            print(f"    {lab}: {len(s)} bets  Win%: {swp:.1f}%  ROI: {sroi:+.1f}%")
    
    # Trending analysis across all seasons
    print(f"\n  Trending team performance (all seasons):")
    trending_bets = []
    for _, row in combined.iterrows():
        year = row['season']
        week = row['week']
        bet_team = row['bet_side']
        ratings_week = max(1, week - 1)
        
        if year in engine.power_ratings and ratings_week in engine.power_ratings[year]:
            current = engine.power_ratings[year][ratings_week].get(bet_team, 0)
        else:
            continue
        
        earlier_week = max(1, ratings_week - 4)
        if earlier_week in engine.power_ratings.get(year, {}):
            earlier = engine.power_ratings[year][earlier_week].get(bet_team, 0)
        else:
            continue
        
        trend = current - earlier
        trending_bets.append({
            'won': row['ats_won'],
            'trend': trend,
            'trending_up': trend > 0.5,
            'trending_down': trend < -0.5,
            'stable': abs(trend) <= 0.5,
        })
    
    tb = pd.DataFrame(trending_bets)
    for cat, col in [("Trending UP", 'trending_up'),
                     ("Trending DOWN", 'trending_down'),
                     ("STABLE", 'stable')]:
        subset = tb[tb[col]]
        if len(subset) > 0:
            w = subset['won'].sum()
            l = len(subset) - w
            wp_t = w / len(subset) * 100
            roi_t = (w * 100 - l * 110) / (len(subset) * 110) * 100
            print(f"    {cat}: {len(subset)} bets  {int(w)}-{int(l)}  Win%: {wp_t:.1f}%  ROI: {roi_t:+.1f}%")
    
    # Year-over-year consistency
    print(f"\n  Season-by-season summary:")
    print(f"  {'Season':<8} {'Bets':>5} {'Win%':>7} {'ROI':>8}")
    print(f"  {'-'*30}")
    for season in [2021, 2022, 2023, 2024]:
        s = combined[combined['season'] == season]
        if len(s) > 0:
            sw = s['ats_won'].sum()
            sl = len(s) - sw
            swp = sw / len(s) * 100
            sroi = (sw * 100 - sl * 110) / (len(s) * 110) * 100
            print(f"  {season:<8} {len(s):>5} {swp:>6.1f}% {sroi:>+7.1f}%")
