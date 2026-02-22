"""
NFL Power Rating Engine — Data Loading
========================================
Handles loading real nflfastR data via nfl_data_py
and generating synthetic data for testing.
"""

import pandas as pd
import numpy as np


def load_real_data(seasons=[2022, 2023, 2024], return_pbp=False):
    """
    Load real NFL play-by-play data and aggregate to game-level EPA summaries.
    
    Downloads data from nflverse GitHub releases via nfl_data_py.
    First run will take 1-2 minutes (downloads ~500MB of play-by-play data).
    Subsequent runs use cached parquet files.
    
    Args:
        seasons: List of NFL seasons to load
        return_pbp: If True, also return the raw play-by-play DataFrame
                   (needed for QB adjustment module)
    
    Returns:
        DataFrame with one row per game and EPA metrics
        If return_pbp=True, returns (games_df, pbp_df)
    """
    import nfl_data_py as nfl
    
    print(f"Downloading play-by-play data for {seasons}...")
    print("(First run takes 1-2 minutes, subsequent runs are faster)")
    pbp = nfl.import_pbp_data(seasons)
    print(f"Downloaded {len(pbp):,} plays across {len(seasons)} seasons")
    
    games = compute_game_epa_from_pbp(pbp)
    print(f"Aggregated to {len(games)} games")
    
    if return_pbp:
        return games, pbp
    return games


def compute_game_epa_from_pbp(pbp_df):
    """
    Convert raw nflfastR play-by-play to per-game EPA summaries.
    
    This is the bridge between raw PBP data and what the engine expects.
    Filters to pass/run plays, computes per-game offensive and defensive
    EPA per play, and pulls market spreads from the PBP data.
    
    Args:
        pbp_df: Raw nflfastR play-by-play DataFrame
    
    Returns:
        DataFrame with game-level EPA metrics matching engine input format
    """
    # Filter to real plays (exclude penalties, timeouts, kickoffs, etc.)
    plays = pbp_df[pbp_df['play_type'].isin(['pass', 'run'])].copy()
    
    # Special teams plays for ST EPA
    st_plays = pbp_df[pbp_df['play_type'].isin(['punt', 'kickoff', 'field_goal', 'extra_point'])].copy()
    
    games = []
    
    for game_id, game_plays in plays.groupby('game_id'):
        game_info = game_plays.iloc[0]
        home = game_info['home_team']
        away = game_info['away_team']
        
        # Split by possession team
        home_off = game_plays[game_plays['posteam'] == home]
        away_off = game_plays[game_plays['posteam'] == away]
        
        # Compute EPA per play
        home_off_epa = home_off['epa'].mean() if len(home_off) > 0 else 0
        away_off_epa = away_off['epa'].mean() if len(away_off) > 0 else 0
        
        # Defensive EPA = negative of opponent's offensive EPA (lower = better D)
        home_def_epa = -away_off_epa
        away_def_epa = -home_off_epa
        
        # Turnovers
        home_turnovers = 0
        away_turnovers = 0
        
        if 'interception' in home_off.columns:
            home_turnovers += home_off['interception'].sum()
        if 'fumble_lost' in home_off.columns:
            home_turnovers += home_off['fumble_lost'].sum()
        if 'interception' in away_off.columns:
            away_turnovers += away_off['interception'].sum()
        if 'fumble_lost' in away_off.columns:
            away_turnovers += away_off['fumble_lost'].sum()
        
        # Special teams EPA
        home_st_epa = 0
        away_st_epa = 0
        game_st = st_plays[st_plays['game_id'] == game_id] if game_id in st_plays['game_id'].values else pd.DataFrame()
        
        if len(game_st) > 0:
            home_st = game_st[game_st['posteam'] == home]
            away_st = game_st[game_st['posteam'] == away]
            home_st_epa = home_st['epa'].mean() if len(home_st) > 0 and 'epa' in home_st.columns else 0
            away_st_epa = away_st['epa'].mean() if len(away_st) > 0 and 'epa' in away_st.columns else 0
        
        # Get final score
        # nflfastR stores cumulative scores — take the max
        home_score = game_plays['total_home_score'].max() if 'total_home_score' in game_plays.columns else game_plays['home_score'].max()
        away_score = game_plays['total_away_score'].max() if 'total_away_score' in game_plays.columns else game_plays['away_score'].max()
        
        # Market spread (from nflfastR — negative means home favored)
        spread_line = game_info.get('spread_line', np.nan)
        
        games.append({
            'game_id': game_id,
            'season': game_info['season'],
            'week': game_info['week'],
            'home_team': home,
            'away_team': away,
            'home_score': int(home_score) if pd.notna(home_score) else 0,
            'away_score': int(away_score) if pd.notna(away_score) else 0,
            'result': int(home_score - away_score) if pd.notna(home_score) and pd.notna(away_score) else 0,
            'spread_line': spread_line,
            'home_off_epa_per_play': home_off_epa,
            'home_def_epa_per_play': home_def_epa,
            'away_off_epa_per_play': away_off_epa,
            'away_def_epa_per_play': away_def_epa,
            'home_turnovers': int(home_turnovers),
            'away_turnovers': int(away_turnovers),
            'home_st_epa_per_play': home_st_epa if pd.notna(home_st_epa) else 0,
            'away_st_epa_per_play': away_st_epa if pd.notna(away_st_epa) else 0,
            'home_plays': len(home_off),
            'away_plays': len(away_off),
        })
    
    return pd.DataFrame(games)


def load_synthetic_data(seasons=[2022, 2023, 2024]):
    """
    Generate realistic synthetic NFL data for testing without network access.
    
    Produces ~288 games per season (32 teams, 18 weeks, 16 games/week).
    Team strengths, scores, EPA values, and market spreads are all generated
    from realistic statistical distributions.
    
    Args:
        seasons: List of seasons to generate
    
    Returns:
        DataFrame matching the same format as load_real_data()
    """
    np.random.seed(42)
    
    teams = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ]
    
    all_games = []
    
    for season in seasons:
        # Generate base team strengths (points above/below average)
        team_strength = {team: np.random.normal(0, 5) for team in teams}
        
        # Make a few teams clearly elite/bad for realism
        elite = np.random.choice(teams, 4, replace=False)
        bad = np.random.choice([t for t in teams if t not in elite], 4, replace=False)
        for t in elite:
            team_strength[t] += np.random.uniform(3, 7)
        for t in bad:
            team_strength[t] -= np.random.uniform(3, 7)
        
        for week in range(1, 19):
            shuffled = teams.copy()
            np.random.shuffle(shuffled)
            
            for i in range(0, len(shuffled), 2):
                home, away = shuffled[i], shuffled[i + 1]
                
                # Within-season strength drift
                home_str = team_strength[home] + np.random.normal(0, 1.5)
                away_str = team_strength[away] + np.random.normal(0, 1.5)
                
                hfa = 2.5
                true_diff = home_str - away_str + hfa
                actual_diff = true_diff + np.random.normal(0, 13.5)
                
                home_score = max(0, int(23 + actual_diff / 2 + np.random.normal(0, 4)))
                away_score = max(0, int(23 - actual_diff / 2 + np.random.normal(0, 4)))
                
                market_spread = -(true_diff + np.random.normal(0, 1.5))
                market_spread = round(market_spread * 2) / 2
                
                all_games.append({
                    'game_id': f"{season}_{week:02d}_{away}_{home}",
                    'season': season,
                    'week': week,
                    'home_team': home,
                    'away_team': away,
                    'home_score': home_score,
                    'away_score': away_score,
                    'result': home_score - away_score,
                    'spread_line': market_spread,
                    'home_off_epa_per_play': (home_str / 30) + np.random.normal(0, 0.08),
                    'home_def_epa_per_play': -(away_str / 30) + np.random.normal(0, 0.08),
                    'away_off_epa_per_play': (away_str / 30) + np.random.normal(0, 0.08),
                    'away_def_epa_per_play': -(home_str / 30) + np.random.normal(0, 0.08),
                    'home_turnovers': max(0, int(np.random.poisson(1.2))),
                    'away_turnovers': max(0, int(np.random.poisson(1.2))),
                    'home_st_epa_per_play': np.random.normal(0, 0.02),
                    'away_st_epa_per_play': np.random.normal(0, 0.02),
                    'home_plays': int(np.random.normal(64, 6)),
                    'away_plays': int(np.random.normal(64, 6)),
                })
    
    return pd.DataFrame(all_games)
