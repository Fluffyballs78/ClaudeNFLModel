"""
NFL Power Rating Engine — Configuration
=========================================
All tunable parameters in one place.
Adjust these and re-run main.py to see how they affect backtest results.
"""

CONFIG = {
    # -------------------------------------------------------------------------
    # RECENCY WEIGHTING
    # -------------------------------------------------------------------------
    # Each game back gets multiplied by this factor.
    # 0.92 means: game N is 100%, N-1 is 92%, N-2 is 84.6%, N-3 is 77.9%...
    # Lower = more aggressive recency bias. Range: 0.80 - 0.98
    "recency_decay": 0.92,

    # -------------------------------------------------------------------------
    # HOME FIELD ADVANTAGE
    # -------------------------------------------------------------------------
    # League-wide HFA in points. Historically ~3.0, trending toward 2.0-2.5
    # since 2020. You could make this team-specific later.
    "home_field_advantage": 2.5,

    # -------------------------------------------------------------------------
    # OPPONENT ADJUSTMENT
    # -------------------------------------------------------------------------
    # Number of iterations for opponent strength adjustment.
    # Converges quickly — 10 is more than enough.
    "opponent_adj_iterations": 10,

    # -------------------------------------------------------------------------
    # PRESEASON PRIOR BLENDING
    # -------------------------------------------------------------------------
    # Early in the season, current-year data is noisy. These control how
    # much weight goes to last year's ratings (regressed toward the mean).
    #
    # prior_full_weight_week: week where prior has maximum influence
    # prior_zero_weight_week: week where current season fully takes over
    "prior_full_weight_week": 1,
    "prior_zero_weight_week": 9,

    # -------------------------------------------------------------------------
    # TURNOVER REGRESSION
    # -------------------------------------------------------------------------
    # Turnovers are highly random. This controls how much to regress toward
    # league average. 0.0 = trust actual turnovers, 1.0 = ignore them entirely.
    # Research suggests 0.4 - 0.6 is optimal.
    "turnover_regression_factor": 0.50,

    # -------------------------------------------------------------------------
    # COMPONENT WEIGHTS
    # -------------------------------------------------------------------------
    # How much each component contributes to the final power rating.
    # Must sum to 1.0.
    "weight_off_epa": 0.35,
    "weight_def_epa": 0.35,
    "weight_turnover_adj": 0.20,
    "weight_special_teams": 0.10,

    # -------------------------------------------------------------------------
    # EDGE DETECTION
    # -------------------------------------------------------------------------
    # Minimum difference (in points) between model spread and market spread
    # to flag as a betting opportunity. Higher = fewer but stronger signals.
    "min_edge_threshold": 5.0,

    # -------------------------------------------------------------------------
    # SEASONS
    # -------------------------------------------------------------------------
    # Which seasons to load. More history = better opponent adjustments
    # and preseason priors, but slower data download.
    "seasons": [2022, 2023, 2024],
}
