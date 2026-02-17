"""
NFL Power Rating & Spread Prediction Engine
=============================================
Run this file to generate power ratings, predict spreads, and backtest.

Usage:
    python3 main.py              # Load real NFL data (requires internet)
    python3 main.py --synthetic  # Use synthetic data (offline testing)
"""

import sys
from engine import NFLPowerRatingEngine, CONFIG
from engine.data_loader import load_real_data, load_synthetic_data


def main():
    use_synthetic = "--synthetic" in sys.argv
    
    print("NFL Power Rating Engine v0.1")
    print("=" * 55)
    
    # -----------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------
    seasons = CONFIG['seasons']
    
    if use_synthetic:
        print("\n[1] Generating synthetic NFL data...")
        games = load_synthetic_data(seasons)
    else:
        print("\n[1] Loading real NFL play-by-play data...")
        games = load_real_data(seasons)
    
    print(f"    {len(games)} games loaded across {len(seasons)} seasons")
    
    # -----------------------------------------------------------------
    # Step 2: Initialize engine and build historical ratings
    # -----------------------------------------------------------------
    engine = NFLPowerRatingEngine(CONFIG)
    engine.load_data(games)
    
    print("\n[2] Building historical ratings...")
    for season in seasons[:-1]:  # All seasons except the last
        max_week = games[games['season'] == season]['week'].max()
        engine.compute_ratings(season, max_week)
        print(f"    {season}: Computed through Week {max_week}")
    
    # -----------------------------------------------------------------
    # Step 3: Show current ratings for most recent season
    # -----------------------------------------------------------------
    current_season = seasons[-1]
    max_week = games[games['season'] == current_season]['week'].max()
    
    print(f"\n[3] Computing {current_season} ratings...")
    engine.print_ratings(current_season, max_week)
    
    # -----------------------------------------------------------------
    # Step 4: Backtest most recent season
    # -----------------------------------------------------------------
    print(f"\n[4] Running {current_season} backtest...")
    results = engine.backtest(current_season)
    summary = engine.print_backtest_summary(results)
    
    # -----------------------------------------------------------------
    # Step 5: Example spread predictions
    # -----------------------------------------------------------------
    ratings = engine.power_ratings[current_season][max_week]
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*55}")
    print(f"  EXAMPLE PREDICTIONS ({current_season} Week {max_week} ratings)")
    print(f"{'='*55}")
    
    best = sorted_teams[0][0]
    worst = sorted_teams[-1][0]
    mid = sorted_teams[15][0]
    
    for home, away in [(best, worst), (best, mid), (mid, worst)]:
        spread = engine.predict_spread(home, away, current_season, max_week)
        hr, ar = ratings[home], ratings[away]
        print(f"\n  {away} ({ar:+.1f}) @ {home} ({hr:+.1f})")
        print(f"  Predicted spread: {home} {spread:+.1f}")
    
    # -----------------------------------------------------------------
    # Step 6: Sample edge analysis
    # -----------------------------------------------------------------
    sample_week = min(10, max_week)
    print(f"\n{'='*55}")
    print(f"  SAMPLE EDGE ANALYSIS — {current_season} Week {sample_week}")
    print(f"{'='*55}")
    
    week_edges = engine.find_edges(current_season, sample_week)
    if len(week_edges) > 0:
        for _, edge in week_edges.iterrows():
            result_str = "W" if edge['ats_won'] else "L"
            print(f"  {edge['away_team']} @ {edge['home_team']}")
            print(f"    Model: {edge['model_spread']:+.1f}  Market: {edge['market_spread']:+.1f}  Edge: {edge['edge']:.1f} pts")
            print(f"    Bet: {edge['bet_side']}  Result: {result_str}")
    else:
        print(f"  No edges >= {CONFIG['min_edge_threshold']} pts found for Week {sample_week}")
    
    print(f"\n{'='*55}")
    print("  Done. Edit engine/config.py to tune parameters and re-run.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
