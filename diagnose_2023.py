"""
2023 Season Diagnostic
Breaks down where and why the model underperformed in 2023.
"""

import pandas as pd
import numpy as np
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data

# Load data
games = load_real_data([2022, 2023, 2024])
engine = NFLPowerRatingEngine(CONFIG)
engine.load_data(games)

# Build 2022 priors
max_wk_2022 = games[games['season'] == 2022]['week'].max()
engine.compute_ratings(2022, min(18, max_wk_2022))

# Run 2023 backtest
results_2023 = engine.backtest(2023)
decided_2023 = results_2023[results_2023['ats_won'].notna()].copy()

# Also run 2024 for comparison
results_2024 = engine.backtest(2024)
decided_2024 = results_2024[results_2024['ats_won'].notna()].copy()

# =====================================================================
# DIAGNOSTIC 1: Performance by week (early vs late season)
# =====================================================================
print("=" * 60)
print("  DIAGNOSTIC 1: Performance by Week Phase")
print("=" * 60)

for label, decided, year in [("2023", decided_2023, 2023), ("2024", decided_2024, 2024)]:
    print(f"\n  --- {year} ---")
    for phase, wk_lo, wk_hi in [("Early (Wk 2-6)", 2, 6), ("Mid (Wk 7-12)", 7, 12), ("Late (Wk 13-18)", 13, 18)]:
        subset = decided[(decided['week'] >= wk_lo) & (decided['week'] <= wk_hi)]
        if len(subset) > 0:
            w = subset['ats_won'].sum()
            l = len(subset) - w
            wp = w / len(subset) * 100
            print(f"  {phase}: {len(subset)} bets  {int(w)}-{int(l)}  Win%: {wp:.1f}%")

# =====================================================================
# DIAGNOSTIC 2: Which teams were we betting on/against most?
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 2: Most Bet Teams (2023)")
print("=" * 60)

bet_teams = decided_2023['bet_side'].value_counts().head(10)
print(f"\n  Most frequently bet ON:")
for team, count in bet_teams.items():
    team_bets = decided_2023[decided_2023['bet_side'] == team]
    wins = team_bets['ats_won'].sum()
    losses = count - wins
    wp = wins / count * 100
    print(f"    {team}: {count} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%")

# Which teams were we betting AGAINST?
print(f"\n  Most frequently bet AGAINST:")
against = []
for _, row in decided_2023.iterrows():
    other = row['away_team'] if row['bet_side'] == row['home_team'] else row['home_team']
    against.append(other)
decided_2023['bet_against'] = against
against_teams = decided_2023['bet_against'].value_counts().head(10)
for team, count in against_teams.items():
    team_bets = decided_2023[decided_2023['bet_against'] == team]
    wins = team_bets['ats_won'].sum()
    losses = count - wins
    wp = wins / count * 100
    print(f"    {team}: {count} bets  {int(wins)}-{int(losses)}  Win%: {wp:.1f}%")

# =====================================================================
# DIAGNOSTIC 3: Direction of edges — home vs away
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 3: Bet Direction (Home vs Away)")
print("=" * 60)

for label, decided, year in [("2023", decided_2023, 2023), ("2024", decided_2024, 2024)]:
    home_bets = decided[decided['bet_side'] == decided['home_team']]
    away_bets = decided[decided['bet_side'] == decided['away_team']]
    
    hw = home_bets['ats_won'].sum() if len(home_bets) > 0 else 0
    hl = len(home_bets) - hw
    aw = away_bets['ats_won'].sum() if len(away_bets) > 0 else 0
    al = len(away_bets) - aw
    
    print(f"\n  --- {year} ---")
    if len(home_bets) > 0:
        print(f"  Bet HOME: {len(home_bets)} bets  {int(hw)}-{int(hl)}  Win%: {hw/len(home_bets)*100:.1f}%")
    if len(away_bets) > 0:
        print(f"  Bet AWAY: {len(away_bets)} bets  {int(aw)}-{int(al)}  Win%: {aw/len(away_bets)*100:.1f}%")

# =====================================================================
# DIAGNOSTIC 4: Edge size vs actual margin miss
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 4: How Wrong Was the Model? (2023)")
print("=" * 60)

decided_2023['model_miss'] = abs(decided_2023['model_spread'] - decided_2023['actual_result'])
decided_2023['market_miss'] = abs(decided_2023['market_spread'] - decided_2023['actual_result'])

print(f"\n  Average absolute error:")
print(f"    Model predicted spread vs actual result:  {decided_2023['model_miss'].mean():.1f} pts")
print(f"    Market spread vs actual result:           {decided_2023['market_miss'].mean():.1f} pts")

decided_2024_temp = decided_2024.copy()
decided_2024_temp['model_miss'] = abs(decided_2024_temp['model_spread'] - decided_2024_temp['actual_result'])
decided_2024_temp['market_miss'] = abs(decided_2024_temp['market_spread'] - decided_2024_temp['actual_result'])

print(f"\n  Comparison to 2024:")
print(f"    2024 model error:   {decided_2024_temp['model_miss'].mean():.1f} pts")
print(f"    2024 market error:  {decided_2024_temp['market_miss'].mean():.1f} pts")

# =====================================================================
# DIAGNOSTIC 5: Power ratings at end of 2022 (the priors feeding 2023)
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 5: 2022 End-of-Season Ratings (2023 Priors)")
print("=" * 60)

ratings_2022 = engine.power_ratings[2022][min(18, max_wk_2022)]
sorted_2022 = sorted(ratings_2022.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 10:")
for i, (team, rating) in enumerate(sorted_2022[:10], 1):
    print(f"    {i}. {team}: {rating:+.2f}")

print(f"\n  Bottom 10:")
for i, (team, rating) in enumerate(sorted_2022[-10:], 23):
    print(f"    {i}. {team}: {rating:+.2f}")

# =====================================================================
# DIAGNOSTIC 6: Biggest losses in 2023 — what went wrong?
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 6: Biggest Losses in 2023 (by edge size)")
print("=" * 60)

losses_2023 = decided_2023[decided_2023['ats_won'] == False].sort_values('edge', ascending=False).head(15)
print(f"\n  {'Week':<6} {'Matchup':<16} {'Model':>7} {'Market':>7} {'Edge':>5} {'Result':>7}")
print(f"  {'-'*52}")
for _, row in losses_2023.iterrows():
    matchup = f"{row['away_team']}@{row['home_team']}"
    print(f"  {int(row['week']):<6} {matchup:<16} {row['model_spread']:>+7.1f} {row['market_spread']:>+7.1f} {row['edge']:>5.1f} {int(row['actual_result']):>+7}")

# =====================================================================
# DIAGNOSTIC 7: Were 2023 games just more random?
# =====================================================================
print(f"\n{'=' * 60}")
print("  DIAGNOSTIC 7: Game Outcome Volatility")
print("=" * 60)

for year in [2022, 2023, 2024]:
    year_games = games[games['season'] == year]
    reg_season = year_games[year_games['week'] <= 18]
    margin_std = reg_season['result'].std()
    avg_margin = reg_season['result'].abs().mean()
    upsets = len(reg_season[
        ((reg_season['spread_line'] > 0) & (reg_season['result'] < 0)) |
        ((reg_season['spread_line'] < 0) & (reg_season['result'] > 0))
    ])
    total = len(reg_season)
    print(f"\n  {year}:")
    print(f"    Margin std dev:      {margin_std:.1f} pts")
    print(f"    Avg absolute margin: {avg_margin:.1f} pts")
    print(f"    Outright upsets:     {upsets}/{total} ({upsets/total*100:.1f}%)")
