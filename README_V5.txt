NFL Parlay Prediction Engine — V5 upgrade

FILES TO UPLOAD TO THE ROOT OF YOUR EXISTING GITHUB REPOSITORY:
- app.py (replace)
- model_engine.py (replace; uploaded original was named model_engine(2).py)
- v3_context.py (replace or leave unchanged)
- v4_parlays.py (replace or leave unchanged)
- v5_probabilities.py (NEW)
- requirements.txt (replace or leave unchanged)

GITHUB STEPS:
1. Download and unzip this archive.
2. Open your nfl-parlay-engine GitHub repository.
3. Choose Add file > Upload files.
4. Drag the SIX .py/.txt requirements files listed above into the upload page
   (do not upload README_V5.txt unless you want it).
5. Commit changes to your main branch.
6. Open the existing Streamlit deployment; allow it to restart.
7. In the V3 Training Lab near the bottom, click Train models. You must
   retrain once per Streamlit session to create the 2025 holdout residuals.
8. Open V5 Auto Prop Research. Select a game, defense, market and player.
9. Enter the ACTUAL FanDuel line and individual American odds.
10. Add props from DIFFERENT games to the watchlist, then screen 10x-25x.

IMPORTANT LIMITATIONS:
- V5 is exploratory, not a validated or calibrated betting model.
- Probability estimates are derived from 2025 holdout residuals, assuming
  the errors transfer across players and seasons. This assumption is untested.
- The 2025 holdout is used for both model comparison and residual estimation;
  a separate later calibration and testing period is needed.
- It does not confirm starters, injuries, playing time, or current sportsbook odds.
- It excludes same-game parlays and treats cross-game legs as independent.
- Product-of-individual odds is not necessarily the actual combined FanDuel price.
- No claim of positive expected value or profitability is made.
