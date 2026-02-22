"""
2023 vs 2024: Why Did the Model Work in One and Not the Other?

This script digs into the structural differences between the two seasons
from the model's perspective to understand what drives profitability.
"""

import pandas as pd
import numpy as np
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load data
games = load_real_data([2022, 2023, 2024])
engine = NFLPowerRatingEngine(CONFIG)
engine.load_data(games)

# Build priors
max_wk_2022 = games[games['season'] == 2022]['week'].max()
engine.compute_ratings(2022, min(18, max_wk_2022))

# Run backtests
results_2023 = engine.backtest(2023)
d23 = results_2023[results_2023['ats_won'].notna()].copy()

results_2024 = engine.backtest(2024)
d24 = results_2024[results_2024['ats_won'].notna()].copy()

# =====================================================================
# TEST 1: Direction of disagreement
# When model says team is BETTER than market thinks vs WORSE
# =====================================================================
print("=" * 60)
print("  TEST 1: Which direction did the model disagree?")
print("=" * 60)
print("  (Model favors home MORE = model thinks home is better than market does)")

for label, decided in [("2023", d23), ("2024", d24)]:
    # model_spread > market_spread means model thinks home is better
    model_higher_on_home = decided[decided['model_spread'] > decided['market_spread']]
    model_lower_on_home = decided[decided['model_spread'] <= decided['market_spread']]
    
    h_w = model_higher_on_home['ats_won'].sum()
    h_l = len(model_higher_on_home) - h_w
    l_w = model_lower_on_home['ats_won'].sum()
    l_l = len(model_lower_on_home) - l_w
    
    print(f"\n  --- {label} ---")
    if len(model_higher_on_home) > 0:
        print(f"  Model favors home MORE than market: {len(model_higher_on_home)} bets  {int(h_w)}-{int(h_l)}  Win%: {h_w/len(model_higher_on_home)*100:.1f}%")
    if len(model_lower_on_home) > 0:
        print(f"  Model favors away MORE than market: {len(model_lower_on_home)} bets  {int(l_w)}-{int(l_l)}  Win%: {l_w/len(model_lower_on_home)*100:.1f}%")

# =====================================================================
# TEST 2: Favorite vs underdog bias
# Is the model systematically betting on favorites or underdogs?
# =====================================================================
print(f"\n{'=' * 60}")
print("  TEST 2: Are we betting favorites or underdogs?")
print("=" * 60)

for label, decided in [("2023", d23), ("2024", d24)]:
    # If market_spread > 0, home is the favorite
    # If we bet home when home is favorite, we're betting the favorite
    fav_bets = []
    dog_bets = []
    
    for _, row in decided.iterrows():
        if row['market_spread'] > 0:
            home_is_fav = True
        elif row['market_spread'] < 0:
            home_is_fav = False
        else:
            continue  # pick'em
        
        betting_home = row['bet_side'] == row['home_team']
        
        if (betting_home and home_is_fav) or (not betting_home and not home_is_fav):
            fav_bets.append(row['ats_won'])
        else:
            dog_bets.append(row['ats_won'])
    
    fav_wins = sum(fav_bets)
    dog_wins = sum(dog_bets)
    
    print(f"\n  --- {label} ---")
    if fav_bets:
        print(f"  Betting FAVORITES: {len(fav_bets)} bets  {int(fav_wins)}-{int(len(fav_bets)-fav_wins)}  Win%: {fav_wins/len(fav_bets)*100:.1f}%")
    if dog_bets:
        print(f"  Betting UNDERDOGS: {len(dog_bets)} bets  {int(dog_wins)}-{int(len(dog_bets)-dog_wins)}  Win%: {dog_wins/len(dog_bets)*100:.1f}%")

# =====================================================================
# TEST 3: How did the market move between seasons?
# Was 2023 just a tighter market?
# =====================================================================
print(f"\n{'=' * 60}")
print("  TEST 3: Market characteristics by season")
print("=" * 60)

for year in [2023, 2024]:
    season_games = games[(games['season'] == year) & (games['week'] <= 18)]
    
    # How often did favorites cover?
    fav_covered = season_games[season_games['result'] > season_games['spread_line']]
    dog_covered = season_games[season_games['result'] < season_games['spread_line']]
    pushes = season_games[season_games['result'] == season_games['spread_line']]
    total = len(season_games) - len(pushes)
    
    # Average spread size
    avg_spread = season_games['spread_line'].abs().mean()
    
    # How accurate was the market?
    season_games_copy = season_games.copy()
    season_games_copy['market_error'] = abs(season_games_copy['result'] - season_games_copy['spread_line'])
    avg_error = season_games_copy['market_error'].mean()
    
    # Correlation between spread and result
    corr = season_games['spread_line'].corr(season_games['result'])
    
    print(f"\n  --- {year} ---")
    print(f"  Avg spread size:        {avg_spread:.1f} pts")
    print(f"  Avg market error:       {avg_error:.1f} pts")
    print(f"  Spread-result corr:     {corr:.3f}")
    print(f"  Favorites covered:      {len(fav_covered)}/{total} ({len(fav_covered)/total*100:.1f}%)")
    print(f"  Underdogs covered:      {len(dog_covered)}/{total} ({len(dog_covered)/total*100:.1f}%)")

# =====================================================================
# TEST 4: Model accuracy on the GAMES WE BET vs games we didn't
# Is the model worse specifically on games it flags as edges?
# =====================================================================
print(f"\n{'=' * 60}")
print("  TEST 4: Model accuracy on bet vs non-bet games")
print("=" * 60)

for label, decided, year in [("2023", d23, 2023), ("2024", d24, 2024)]:
    season_games = games[(games['season'] == year) & (games['week'] <= 18) & (games['week'] >= 2)]
    
    # Get all predictions
    all_preds = []
    for _, game in season_games.iterrows():
        ratings_week = max(1, game['week'] - 1)
        pred = engine.predict_spread(game['home_team'], game['away_team'], year, ratings_week)
        if pred is not None and pd.notna(game['spread_line']):
            edge = abs(pred - game['spread_line'])
            model_err = abs(pred - game['result'])
            market_err = abs(game['spread_line'] - game['result'])
            all_preds.append({
                'game_id': game['game_id'],
                'edge': edge,
                'model_error': model_err,
                'market_error': market_err,
                'is_bet': edge >= CONFIG['min_edge_threshold'],
            })
    
    all_preds = pd.DataFrame(all_preds)
    bets = all_preds[all_preds['is_bet']]
    no_bets = all_preds[~all_preds['is_bet']]
    
    print(f"\n  --- {year} ---")
    print(f"  Games with edge (bet):     Model err: {bets['model_error'].mean():.1f}  Market err: {bets['market_error'].mean():.1f}  Gap: {bets['model_error'].mean() - bets['market_error'].mean():+.1f}")
    print(f"  Games without edge (skip): Model err: {no_bets['model_error'].mean():.1f}  Market err: {no_bets['market_error'].mean():.1f}  Gap: {no_bets['model_error'].mean() - no_bets['market_error'].mean():+.1f}")

# =====================================================================
# TEST 5: Recency signal — teams improving vs declining
# Did the model profit by catching teams trending up/down?
# =====================================================================
print(f"\n{'=' * 60}")
print("  TEST 5: Was the edge from catching trending teams?")
print("=" * 60)

for label, decided, year in [("2023", d23, 2023), ("2024", d24, 2024)]:
    trending_bets = []
    
    for _, row in decided.iterrows():
        week = row['week']
        bet_team = row['bet_side']
        ratings_week = max(1, week - 1)
        
        # Get team's current rating
        if year in engine.power_ratings and ratings_week in engine.power_ratings[year]:
            current_rating = engine.power_ratings[year][ratings_week].get(bet_team, 0)
        else:
            continue
        
        # Get rating from 4 weeks earlier (if available)
        earlier_week = max(1, ratings_week - 4)
        if earlier_week in engine.power_ratings.get(year, {}):
            earlier_rating = engine.power_ratings[year][earlier_week].get(bet_team, 0)
        else:
            continue
        
        trend = current_rating - earlier_rating
        trending_bets.append({
            'won': row['ats_won'],
            'trend': trend,
            'trending_up': trend > 0.5,
            'trending_down': trend < -0.5,
            'stable': abs(trend) <= 0.5,
        })
    
    tb = pd.DataFrame(trending_bets)
    
    print(f"\n  --- {year} ---")
    for cat, col in [("Trending UP (rating improved)", 'trending_up'), 
                     ("Trending DOWN (rating declined)", 'trending_down'),
                     ("STABLE (little change)", 'stable')]:
        subset = tb[tb[col]]
        if len(subset) > 0:
            w = subset['won'].sum()
            l = len(subset) - w
            wp = w / len(subset) * 100
            print(f"  {cat}: {len(subset)} bets  {int(w)}-{int(l)}  Win%: {wp:.1f}%")

# =====================================================================
# TEST 6: Week-over-week model improvement
# Does the model get better as the season goes on? At what rate?
# =====================================================================
print(f"\n{'=' * 60}")
print("  TEST 6: Cumulative ATS record by week")
print("=" * 60)

for label, decided, year in [("2023", d23, 2023), ("2024", d24, 2024)]:
    print(f"\n  --- {year} ---")
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
