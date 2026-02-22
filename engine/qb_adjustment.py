"""
NFL Power Rating Engine — QB Adjustment Module
================================================
Tracks QB-level EPA and adjusts team offensive ratings
based on who is actually starting.

The core insight: a team's offensive EPA with their starter
vs their backup can differ by 0.15-0.30 EPA/play, which
translates to roughly 5-10 points per game. The market
prices this in; our base model doesn't. This module fixes that.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class QBAdjuster:
    """
    Tracks QB performance and provides adjustments to team offensive ratings
    based on which QB is starting.
    
    Usage:
        adjuster = QBAdjuster()
        adjuster.build_qb_profiles(pbp_df)
        adjustment = adjuster.get_team_qb_adjustment("ARI", season=2024, week=10)
    """
    
    def __init__(self, min_attempts=30):
        """
        Args:
            min_attempts: Minimum pass attempts for a QB to have a reliable EPA estimate.
                         Below this, we regress heavily toward team average.
        """
        self.min_attempts = min_attempts
        self.qb_game_logs = None       # Per-game QB stats
        self.qb_season_stats = None    # Season-level QB summaries
        self.team_primary_qbs = {}     # {(season, team): passer_player_id}
        self.league_avg_epa = {}       # {season: float}
    
    def build_qb_profiles(self, pbp_df):
        """
        Build QB profiles from raw play-by-play data.
        
        Args:
            pbp_df: Raw nflfastR play-by-play DataFrame
        """
        # Filter to pass plays with a known passer
        passes = pbp_df[
            (pbp_df['play_type'] == 'pass') & 
            (pbp_df['passer_player_name'].notna())
        ].copy()
        
        # Build per-game QB logs
        game_logs = []
        for (game_id, passer_id), group in passes.groupby(['game_id', 'passer_player_id']):
            game_info = group.iloc[0]
            game_logs.append({
                'game_id': game_id,
                'season': game_info['season'],
                'week': game_info['week'],
                'team': game_info['posteam'],
                'passer_player_id': passer_id,
                'passer_player_name': game_info['passer_player_name'],
                'attempts': len(group),
                'qb_epa_per_play': group['qb_epa'].mean() if 'qb_epa' in group.columns else group['epa'].mean(),
                'total_epa': group['qb_epa'].sum() if 'qb_epa' in group.columns else group['epa'].sum(),
            })
        
        self.qb_game_logs = pd.DataFrame(game_logs)
        
        # Identify primary QB per game (most attempts)
        self.qb_game_logs['is_primary'] = False
        for game_id, game_group in self.qb_game_logs.groupby(['game_id', 'team']):
            primary_idx = game_group['attempts'].idxmax()
            self.qb_game_logs.loc[primary_idx, 'is_primary'] = True
        
        # Compute league average EPA per season
        for season in self.qb_game_logs['season'].unique():
            season_data = self.qb_game_logs[self.qb_game_logs['season'] == season]
            self.league_avg_epa[season] = season_data['qb_epa_per_play'].mean()
        
        # Identify team primary QBs per season (most total attempts)
        for (season, team), group in self.qb_game_logs.groupby(['season', 'team']):
            qb_attempts = group.groupby('passer_player_id')['attempts'].sum()
            primary_qb = qb_attempts.idxmax()
            self.team_primary_qbs[(season, team)] = primary_qb
        
        total_qbs = self.qb_game_logs['passer_player_id'].nunique()
        total_games = self.qb_game_logs[self.qb_game_logs['is_primary']]['game_id'].nunique()
        print(f"QB Adjuster: Built profiles for {total_qbs} QBs across {total_games} games")
    
    def get_qb_epa(self, passer_id, season, through_week, decay=0.92):
        """
        Get a QB's recency-weighted EPA/play through a given week.
        
        Args:
            passer_id: The QB's player ID
            season: NFL season
            through_week: Include games through this week
            decay: Recency weighting factor
        
        Returns:
            tuple: (epa_per_play, total_attempts, is_reliable)
        """
        logs = self.qb_game_logs[
            (self.qb_game_logs['passer_player_id'] == passer_id) &
            (self.qb_game_logs['season'] == season) &
            (self.qb_game_logs['week'] <= through_week) &
            (self.qb_game_logs['is_primary'] == True)
        ].sort_values('week')
        
        if len(logs) == 0:
            # Check previous season
            prev_logs = self.qb_game_logs[
                (self.qb_game_logs['passer_player_id'] == passer_id) &
                (self.qb_game_logs['season'] == season - 1) &
                (self.qb_game_logs['is_primary'] == True)
            ]
            if len(prev_logs) > 0:
                # Use previous season, heavily regressed
                prev_epa = prev_logs['qb_epa_per_play'].mean()
                league_avg = self.league_avg_epa.get(season - 1, 0)
                regressed = prev_epa * 0.5 + league_avg * 0.5
                return regressed, prev_logs['attempts'].sum(), False
            return self.league_avg_epa.get(season, 0), 0, False
        
        # Recency-weighted EPA
        weights = []
        for _, log in logs.iterrows():
            weeks_ago = through_week - log['week']
            weights.append(decay ** weeks_ago)
        
        weights = np.array(weights)
        epa = np.average(logs['qb_epa_per_play'].values, weights=weights)
        total_attempts = logs['attempts'].sum()
        
        # Regress toward league average based on sample size
        reliability = min(1.0, total_attempts / (self.min_attempts * 3))
        league_avg = self.league_avg_epa.get(season, 0)
        regressed_epa = epa * reliability + league_avg * (1 - reliability)
        
        return regressed_epa, total_attempts, total_attempts >= self.min_attempts
    
    def get_starter_for_game(self, team, season, week):
        """
        Identify who started (most attempts) for a team in a specific game.
        
        Returns:
            tuple: (passer_player_id, passer_name, attempts) or None
        """
        logs = self.qb_game_logs[
            (self.qb_game_logs['team'] == team) &
            (self.qb_game_logs['season'] == season) &
            (self.qb_game_logs['week'] == week) &
            (self.qb_game_logs['is_primary'] == True)
        ]
        
        if len(logs) == 0:
            return None
        
        row = logs.iloc[0]
        return row['passer_player_id'], row['passer_player_name'], row['attempts']
    
    def _get_primary_qb_through_week(self, team, season, through_week):
        """
        Identify the primary QB (most attempts) for a team using only
        data available through the given week. No future data leakage.
        
        Returns:
            tuple: (passer_player_id, passer_player_name) or (None, None)
        """
        logs = self.qb_game_logs[
            (self.qb_game_logs['team'] == team) &
            (self.qb_game_logs['season'] == season) &
            (self.qb_game_logs['week'] <= through_week)
        ]
        
        if len(logs) == 0:
            return None, None
        
        qb_attempts = logs.groupby('passer_player_id')['attempts'].sum()
        primary_id = qb_attempts.idxmax()
        
        primary_name = logs[logs['passer_player_id'] == primary_id].iloc[0]['passer_player_name']
        return primary_id, primary_name
    
    def get_team_qb_adjustment(self, team, season, through_week, upcoming_week=None):
        """
        Calculate the QB adjustment for a team's offensive rating.
        
        Compares the team's most recent starter to the primary QB
        (most attempts through the given week — NO future data used).
        If the starter has changed, returns the EPA difference as an adjustment.
        
        Args:
            team: Team abbreviation
            season: NFL season
            through_week: Data available through this week
            upcoming_week: If predicting a future game, which week.
                          If None, uses through_week.
        
        Returns:
            dict with:
                'adjustment': float (EPA/play adjustment to apply to team offense)
                'current_qb': str (name of most recent starter)
                'primary_qb': str (name of primary QB through this week)
                'qb_changed': bool (whether a QB change has occurred)
                'current_qb_epa': float
                'primary_qb_epa': float
        """
        # Find the most recent starter (using only data through through_week)
        recent_logs = self.qb_game_logs[
            (self.qb_game_logs['team'] == team) &
            (self.qb_game_logs['season'] == season) &
            (self.qb_game_logs['week'] <= through_week) &
            (self.qb_game_logs['is_primary'] == True)
        ].sort_values('week', ascending=False)
        
        if len(recent_logs) == 0:
            return {
                'adjustment': 0,
                'current_qb': 'Unknown',
                'primary_qb': 'Unknown',
                'qb_changed': False,
                'current_qb_epa': 0,
                'primary_qb_epa': 0,
            }
        
        current_qb_id = recent_logs.iloc[0]['passer_player_id']
        current_qb_name = recent_logs.iloc[0]['passer_player_name']
        
        # Get the primary QB using ONLY data through this week (no leakage)
        primary_qb_id, primary_qb_name = self._get_primary_qb_through_week(
            team, season, through_week
        )
        
        if primary_qb_id is None:
            return {
                'adjustment': 0,
                'current_qb': current_qb_name,
                'primary_qb': 'Unknown',
                'qb_changed': False,
                'current_qb_epa': 0,
                'primary_qb_epa': 0,
            }
        
        # Get EPA for both QBs (using only data through this week)
        current_epa, current_att, current_reliable = self.get_qb_epa(
            current_qb_id, season, through_week
        )
        primary_epa, primary_att, primary_reliable = self.get_qb_epa(
            primary_qb_id, season, through_week
        )
        
        # Calculate adjustment
        qb_changed = current_qb_id != primary_qb_id
        
        if qb_changed:
            # The team's base rating was built largely on the primary QB's performance.
            # If a different QB is starting, adjust by the difference.
            # Dampen the adjustment based on reliability of the current QB's sample.
            raw_adjustment = current_epa - primary_epa
            
            # Dampen: if we don't have much data on the current QB, 
            # don't fully trust the difference
            confidence = min(1.0, current_att / (self.min_attempts * 2))
            adjustment = raw_adjustment * confidence
        else:
            # Same QB starting — no adjustment needed.
            # The team's EPA already reflects this QB's performance.
            adjustment = 0
        
        return {
            'adjustment': round(adjustment, 4),
            'current_qb': current_qb_name,
            'primary_qb': primary_qb_name,
            'qb_changed': qb_changed,
            'current_qb_epa': round(current_epa, 4),
            'primary_qb_epa': round(primary_epa, 4),
        }
    
    def get_qb_report(self, season, through_week, top_n=15):
        """
        Print a report of QB rankings and team QB situations.
        """
        teams = self.qb_game_logs[self.qb_game_logs['season'] == season]['team'].unique()
        
        # Collect all starting QB EPAs
        qb_rankings = []
        qb_changes = []
        
        for team in sorted(teams):
            adj = self.get_team_qb_adjustment(team, season, through_week)
            
            # Get current starter's EPA
            recent = self.qb_game_logs[
                (self.qb_game_logs['team'] == team) &
                (self.qb_game_logs['season'] == season) &
                (self.qb_game_logs['week'] <= through_week) &
                (self.qb_game_logs['is_primary'] == True)
            ].sort_values('week', ascending=False)
            
            if len(recent) > 0:
                current_id = recent.iloc[0]['passer_player_id']
                epa, att, reliable = self.get_qb_epa(current_id, season, through_week)
                qb_rankings.append({
                    'team': team,
                    'qb': adj['current_qb'],
                    'epa_per_play': epa,
                    'attempts': att,
                    'reliable': reliable,
                })
            
            if adj['qb_changed']:
                qb_changes.append(adj | {'team': team})
        
        # Print QB rankings
        qb_rankings = sorted(qb_rankings, key=lambda x: x['epa_per_play'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"  QB RANKINGS — {season} Through Week {through_week}")
        print(f"{'='*60}")
        print(f"  {'Rank':<6} {'Team':<5} {'QB':<20} {'EPA/play':>9} {'Att':>5}")
        print(f"  {'-'*50}")
        
        for i, qb in enumerate(qb_rankings[:top_n], 1):
            flag = " *" if not qb['reliable'] else ""
            print(f"  {i:<6} {qb['team']:<5} {qb['qb']:<20} {qb['epa_per_play']:>+9.3f} {qb['attempts']:>5}{flag}")
        
        print(f"\n  ...bottom 5:")
        for i, qb in enumerate(qb_rankings[-5:], len(qb_rankings) - 4):
            flag = " *" if not qb['reliable'] else ""
            print(f"  {i:<6} {qb['team']:<5} {qb['qb']:<20} {qb['epa_per_play']:>+9.3f} {qb['attempts']:>5}{flag}")
        
        if qb_changes:
            print(f"\n  QB CHANGES DETECTED:")
            print(f"  {'-'*50}")
            for change in qb_changes:
                direction = "upgrade" if change['adjustment'] > 0 else "downgrade"
                print(f"  {change['team']}: {change['primary_qb']} → {change['current_qb']}")
                print(f"    EPA: {change['primary_qb_epa']:+.3f} → {change['current_qb_epa']:+.3f}")
                print(f"    Adjustment: {change['adjustment']:+.4f} EPA/play ({direction})")
        else:
            print(f"\n  No QB changes detected.")
        
        print(f"\n  * = small sample, regressed toward league average")
        
        return qb_rankings
