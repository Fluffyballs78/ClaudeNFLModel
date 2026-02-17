"""
NFL Power Rating Engine — Core
================================
Computes team power ratings from EPA efficiency metrics,
applies opponent adjustments and recency weighting,
and generates predicted spreads for any matchup.
"""

import pandas as pd
import numpy as np
from .config import CONFIG


class NFLPowerRatingEngine:
    """
    NFL team power rating system.
    
    Computes a single number per team representing points above/below
    league average. Combines offensive EPA, defensive EPA, turnover-adjusted
    efficiency, and special teams — all opponent-adjusted and recency-weighted.
    
    Usage:
        engine = NFLPowerRatingEngine()
        engine.load_data(games_df)
        ratings = engine.compute_ratings(2024, through_week=12)
        spread = engine.predict_spread("KC", "BUF", 2024, 12)
        backtest = engine.backtest(2024)
    """
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.games_df = None
        self.ratings = {}
        self.power_ratings = {}
        self.league_averages = {}
    
    def load_data(self, games_df):
        """Load game-level EPA data."""
        self.games_df = games_df.sort_values(['season', 'week']).reset_index(drop=True)
        seasons = sorted(self.games_df['season'].unique())
        total = len(self.games_df)
        print(f"Loaded {total} games across seasons: {seasons}")
    
    # =========================================================================
    # INTERNAL: DATA PREPARATION
    # =========================================================================
    
    def _get_team_games(self, team, season, through_week):
        """Get all games for a team in a season through a given week, normalized to team perspective."""
        df = self.games_df
        mask = (df['season'] == season) & (df['week'] <= through_week)
        home = df[mask & (df['home_team'] == team)].copy()
        away = df[mask & (df['away_team'] == team)].copy()
        
        home_records = home.assign(
            team=team, opponent=home['away_team'], is_home=True,
            off_epa=home['home_off_epa_per_play'],
            def_epa=home['home_def_epa_per_play'],
            turnovers=home['home_turnovers'],
            opp_turnovers=home['away_turnovers'],
            st_epa=home['home_st_epa_per_play'],
            plays=home['home_plays'],
        )
        
        away_records = away.assign(
            team=team, opponent=away['home_team'], is_home=False,
            off_epa=away['away_off_epa_per_play'],
            def_epa=away['away_def_epa_per_play'],
            turnovers=away['away_turnovers'],
            opp_turnovers=away['home_turnovers'],
            st_epa=away['away_st_epa_per_play'],
            plays=away['away_plays'],
        )
        
        return pd.concat([home_records, away_records]).sort_values('week')
    
    def _apply_recency_weights(self, games_series, current_week):
        """Apply exponential decay — more recent games count more."""
        decay = self.config['recency_decay']
        weights = []
        for _, game in games_series.iterrows():
            weeks_ago = current_week - game['week']
            weights.append(decay ** weeks_ago)
        return np.array(weights)
    
    def _compute_league_averages(self, season, through_week):
        """Compute league average metrics for turnover regression baseline."""
        df = self.games_df
        mask = (df['season'] == season) & (df['week'] <= through_week)
        data = df[mask]
        
        self.league_averages[(season, through_week)] = {
            'avg_turnovers_per_game': (data['home_turnovers'].mean() + data['away_turnovers'].mean()) / 2,
            'avg_off_epa': (data['home_off_epa_per_play'].mean() + data['away_off_epa_per_play'].mean()) / 2,
            'avg_def_epa': (data['home_def_epa_per_play'].mean() + data['away_def_epa_per_play'].mean()) / 2,
        }
    
    # =========================================================================
    # CORE: RATING COMPUTATION
    # =========================================================================
    
    def _compute_raw_ratings(self, season, through_week):
        """Compute raw (non-opponent-adjusted) ratings for all teams."""
        teams = set(self.games_df[self.games_df['season'] == season]['home_team'].unique())
        self._compute_league_averages(season, through_week)
        league_avg = self.league_averages[(season, through_week)]
        
        raw_ratings = {}
        
        for team in teams:
            games = self._get_team_games(team, season, through_week)
            
            if len(games) == 0:
                raw_ratings[team] = {
                    'off_epa': 0, 'def_epa': 0, 'turnover_adj': 0,
                    'st_epa': 0, 'games_played': 0, 'opponents': []
                }
                continue
            
            weights = self._apply_recency_weights(games, through_week)
            
            off_epa = np.average(games['off_epa'].values, weights=weights)
            def_epa = np.average(games['def_epa'].values, weights=weights)
            
            # Turnover regression
            reg = self.config['turnover_regression_factor']
            actual_to = np.average(games['turnovers'].values, weights=weights)
            actual_opp_to = np.average(games['opp_turnovers'].values, weights=weights)
            
            regressed_to = actual_to * (1 - reg) + league_avg['avg_turnovers_per_game'] * reg
            regressed_opp_to = actual_opp_to * (1 - reg) + league_avg['avg_turnovers_per_game'] * reg
            
            # Net turnover advantage scaled to per-play
            turnover_pts = (regressed_opp_to - regressed_to) * 3.5
            avg_plays = games['plays'].mean()
            turnover_adj = turnover_pts / avg_plays if avg_plays > 0 else 0
            
            st_epa = np.average(games['st_epa'].values, weights=weights)
            
            raw_ratings[team] = {
                'off_epa': off_epa,
                'def_epa': def_epa,
                'turnover_adj': turnover_adj,
                'st_epa': st_epa,
                'games_played': len(games),
                'opponents': list(games['opponent'].values),
            }
        
        return raw_ratings
    
    def _opponent_adjust(self, raw_ratings, season, through_week):
        """
        Iterative opponent adjustment.
        
        Adjusts offensive EPA based on defensive quality of opponents faced,
        and vice versa. Repeats until convergence.
        """
        teams = list(raw_ratings.keys())
        iterations = self.config['opponent_adj_iterations']
        adjusted = {team: dict(raw_ratings[team]) for team in teams}
        
        for _ in range(iterations):
            new_adjusted = {}
            
            for team in teams:
                raw = raw_ratings[team]
                opponents = raw['opponents']
                
                if len(opponents) == 0:
                    new_adjusted[team] = dict(raw)
                    continue
                
                opp_def = [adjusted[opp]['def_epa'] for opp in opponents if opp in adjusted]
                opp_off = [adjusted[opp]['off_epa'] for opp in opponents if opp in adjusted]
                
                avg_opp_def = np.mean(opp_def) if opp_def else 0
                avg_opp_off = np.mean(opp_off) if opp_off else 0
                
                new_adjusted[team] = {
                    'off_epa': raw['off_epa'] - avg_opp_def,
                    'def_epa': raw['def_epa'] + avg_opp_off,
                    'turnover_adj': raw['turnover_adj'],
                    'st_epa': raw['st_epa'],
                    'games_played': raw['games_played'],
                    'opponents': raw['opponents'],
                }
            
            adjusted = new_adjusted
        
        return adjusted
    
    def _blend_preseason_prior(self, current_ratings, prior_ratings, week):
        """Blend current-season ratings with prior-season ratings (regressed to mean)."""
        w1 = self.config['prior_full_weight_week']
        w0 = self.config['prior_zero_weight_week']
        
        if week >= w0 or prior_ratings is None:
            return current_ratings
        
        prior_weight = max(0, (w0 - week) / (w0 - w1))
        current_weight = 1 - prior_weight
        prior_regression = 0.67  # Regress prior toward league average
        
        blended = {}
        for team in current_ratings:
            if team in prior_ratings:
                prior = prior_ratings[team]
                current = current_ratings[team]
                blended[team] = {
                    'off_epa': current_weight * current['off_epa'] + prior_weight * prior['off_epa'] * prior_regression,
                    'def_epa': current_weight * current['def_epa'] + prior_weight * prior['def_epa'] * prior_regression,
                    'turnover_adj': current_weight * current['turnover_adj'] + prior_weight * prior['turnover_adj'] * prior_regression,
                    'st_epa': current_weight * current['st_epa'] + prior_weight * prior['st_epa'] * prior_regression,
                    'games_played': current['games_played'],
                    'opponents': current['opponents'],
                }
            else:
                blended[team] = current_ratings[team]
        
        return blended
    
    def _compute_power_rating(self, team_ratings):
        """Convert component ratings into a single power number (points above average)."""
        cfg = self.config
        plays_per_game = 65
        
        off_pts = team_ratings['off_epa'] * plays_per_game
        def_pts = team_ratings['def_epa'] * plays_per_game
        to_pts = team_ratings['turnover_adj'] * plays_per_game
        st_pts = team_ratings['st_epa'] * plays_per_game
        
        power = (
            cfg['weight_off_epa'] * off_pts +
            cfg['weight_def_epa'] * (-def_pts) +
            cfg['weight_turnover_adj'] * to_pts +
            cfg['weight_special_teams'] * st_pts
        )
        
        return round(power, 2)
    
    # =========================================================================
    # PUBLIC: MAIN INTERFACE
    # =========================================================================
    
    def compute_ratings(self, season, through_week):
        """
        Compute power ratings for all teams at a given point in the season.
        
        Returns:
            dict: {team_abbrev: power_rating} where rating is points above average
        """
        raw = self._compute_raw_ratings(season, through_week)
        adjusted = self._opponent_adjust(raw, season, through_week)
        
        # Blend with prior season if available and early in year
        prior = None
        if season - 1 in self.ratings:
            max_prior_week = max(self.ratings[season - 1].keys())
            prior = self.ratings[season - 1][max_prior_week]
        
        blended = self._blend_preseason_prior(adjusted, prior, through_week)
        
        # Store detailed ratings
        if season not in self.ratings:
            self.ratings[season] = {}
        self.ratings[season][through_week] = blended
        
        # Compute scalar ratings and center around 0
        if season not in self.power_ratings:
            self.power_ratings[season] = {}
        
        team_powers = {team: self._compute_power_rating(r) for team, r in blended.items()}
        avg = np.mean(list(team_powers.values()))
        team_powers = {team: round(p - avg, 2) for team, p in team_powers.items()}
        
        self.power_ratings[season][through_week] = team_powers
        return team_powers
    
    def predict_spread(self, home_team, away_team, season, through_week):
        """
        Predict the spread for a specific matchup.
        
        Returns:
            float: Predicted spread (positive = home favored), matching nflfastR convention.
                   Rounded to nearest 0.5.
        """
        if season not in self.power_ratings or through_week not in self.power_ratings[season]:
            self.compute_ratings(season, through_week)
        
        ratings = self.power_ratings[season][through_week]
        
        if home_team not in ratings or away_team not in ratings:
            return None
        
        # Positive = home favored (matches nflfastR spread_line convention)
        predicted_spread = ratings[home_team] - ratings[away_team] + self.config['home_field_advantage']
        
        return round(predicted_spread * 2) / 2
    
    def find_edges(self, season, week):
        """
        Compare model predictions to market spreads for a given week.
        
        Spread convention (matching nflfastR): positive = home favored.
        Result convention (matching nflfastR): positive = home won by X.
        
        ATS cover logic:
            Home covers if: result > spread (home won by more than spread)
            Away covers if: result < spread (home won by less than spread, or lost)
        
        Returns:
            DataFrame with columns: game_id, home_team, away_team, model_spread,
            market_spread, edge, bet_side, actual_result, ats_won
        """
        ratings_week = max(1, week - 1)
        self.compute_ratings(season, ratings_week)
        
        df = self.games_df
        week_games = df[(df['season'] == season) & (df['week'] == week)]
        
        edges = []
        for _, game in week_games.iterrows():
            predicted = self.predict_spread(game['home_team'], game['away_team'], season, ratings_week)
            market = game['spread_line']
            
            if predicted is None or pd.isna(market):
                continue
            
            # Edge: difference between model and market
            # Both use same convention: positive = home favored
            edge = predicted - market
            
            # If edge > 0: model thinks home is stronger than market does → bet home
            # If edge < 0: model thinks away is stronger than market does → bet away
            if abs(edge) >= self.config['min_edge_threshold']:
                if edge > 0:
                    bet_side = game['home_team']
                else:
                    bet_side = game['away_team']
                
                # ATS result: did the bet win?
                # Home covers when: result > spread (won by more than expected)
                # Away covers when: result < spread (home didn't cover)
                home_covered = game['result'] > market
                
                if edge > 0:
                    won = home_covered       # we bet home
                else:
                    won = not home_covered   # we bet away
                
                # Handle pushes (result == spread)
                if game['result'] == market:
                    won = None  # push
                
                edges.append({
                    'game_id': game['game_id'],
                    'week': week,
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'model_spread': predicted,
                    'market_spread': market,
                    'edge': round(abs(edge), 1),
                    'bet_side': bet_side,
                    'actual_result': game['result'],
                    'ats_won': won,
                })
        
        return pd.DataFrame(edges)
    
    def backtest(self, season):
        """
        Run a full-season backtest and return all flagged edges with ATS results.
        
        Returns:
            DataFrame with all bets and outcomes for the season
        """
        all_edges = []
        max_week = self.games_df[self.games_df['season'] == season]['week'].max()
        
        for week in range(2, max_week + 1):
            week_edges = self.find_edges(season, week)
            if len(week_edges) > 0:
                all_edges.append(week_edges)
        
        if not all_edges:
            return pd.DataFrame()
        
        return pd.concat(all_edges, ignore_index=True)
    
    # =========================================================================
    # OUTPUT: DISPLAY & REPORTING
    # =========================================================================
    
    def print_ratings(self, season, through_week, top_n=32):
        """Pretty print current power ratings."""
        ratings = self.compute_ratings(season, through_week)
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*55}")
        print(f"  NFL POWER RATINGS — {season} Season, Through Week {through_week}")
        print(f"{'='*55}")
        print(f"  {'Rank':<6} {'Team':<6} {'Rating':>8}  {'Tier'}")
        print(f"  {'-'*45}")
        
        for i, (team, rating) in enumerate(sorted_teams[:top_n], 1):
            if rating > 5:
                tier = "Elite"
            elif rating > 2:
                tier = "Contender"
            elif rating > -2:
                tier = "Average"
            elif rating > -5:
                tier = "Below Avg"
            else:
                tier = "Rebuild"
            print(f"  {i:<6} {team:<6} {rating:>+8.2f}  {tier}")
        
        return sorted_teams
    
    def print_backtest_summary(self, results):
        """Print backtest performance summary with ROI breakdown."""
        if len(results) == 0:
            print("No edges found in backtest.")
            return
        
        # Filter out pushes
        decided = results[results['ats_won'].notna()].copy()
        pushes = len(results) - len(decided)
        
        total = len(decided)
        wins = decided['ats_won'].sum()
        losses = total - wins
        win_pct = wins / total * 100 if total > 0 else 0
        breakeven = 52.38
        
        # ROI at -110 juice
        profit = wins * 100 - losses * 110
        roi = profit / (total * 110) * 100
        
        print(f"\n{'='*55}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*55}")
        print(f"  Total bets:        {total} ({pushes} pushes excluded)")
        print(f"  Record:            {int(wins)}-{int(losses)}")
        print(f"  Win %:             {win_pct:.1f}%")
        print(f"  Breakeven:         {breakeven:.1f}%")
        print(f"  Estimated ROI:     {roi:+.1f}%")
        print(f"  Avg edge:          {decided['edge'].mean():.1f} pts")
        print(f"  Min edge filter:   {self.config['min_edge_threshold']} pts")
        
        print(f"\n  Performance by Edge Size:")
        print(f"  {'Edge Range':<15} {'Bets':>6} {'Win%':>8} {'ROI':>8}")
        print(f"  {'-'*40}")
        
        for low, high, label in [(2, 3, "2-3 pts"), (3, 5, "3-5 pts"), (5, 99, "5+ pts")]:
            subset = decided[(decided['edge'] >= low) & (decided['edge'] < high)]
            if len(subset) > 0:
                sw = subset['ats_won'].sum()
                sl = len(subset) - sw
                wp = sw / len(subset) * 100
                sub_roi = (sw * 100 - sl * 110) / (len(subset) * 110) * 100
                print(f"  {label:<15} {len(subset):>6} {wp:>7.1f}% {sub_roi:>+7.1f}%")
        
        return {
            'total_bets': total,
            'wins': int(wins),
            'losses': int(losses),
            'win_pct': win_pct,
            'roi': roi,
        }
