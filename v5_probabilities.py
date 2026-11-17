"""V5 exploratory prop probabilities from 2025 out-of-sample residuals.

These are not calibrated probabilities, injury forecasts, or sportsbook quotes.
Same-game combinations are deliberately excluded.
"""
import itertools
import math
import numpy as np
import pandas as pd
import streamlit as st
from model_engine import forecasts, MARKETS
from v3_context import schedule_matchups
from v4_parlays import decimal_odds


def residual_hit_probability(projected, line, side, residuals):
    """Empirical 2025 holdout residual CDF with half-count smoothing.

    An observed residual e corresponds to hypothetical outcome projected + e.
    This assumes residual distributions transfer across players, lines and years;
    it has NOT been validated for probability calibration.
    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 30:
        raise ValueError('At least 30 historical out-of-sample residuals are required.')
    simulated = np.maximum(0.0, float(projected) + e)
    # Strict comparisons: exact equality with the sportsbook line is a push.
    wins = int(np.count_nonzero(simulated > line if side == 'Over' else simulated < line))
    pushes = int(np.count_nonzero(simulated == line))
    p = (wins + 0.5) / (len(e) + 1.0)
    return p, pushes / len(e), len(e)


def candidate_combinations(props, low, high, max_legs):
    """Cross-game independent screening only; never a verified FanDuel quote."""
    rows = []
    if len(props) > 20:
        raise ValueError('Please keep the watchlist to 20 props or fewer.')
    for n in range(2, min(max_legs, len(props)) + 1):
        for idx in itertools.combinations(range(len(props)), n):
            sub = props.iloc[list(idx)]
            if sub['Game'].nunique() != n or sub['Player'].nunique() != n:
                continue
            mult = math.prod(sub['Decimal odds'].astype(float))
            if not low <= mult <= high:
                continue
            prob = math.prod(sub['Estimated hit probability'].astype(float))
            labels = ' | '.join(
                f"{r['Player']} {r['Market']} {r['Side']} {r['Line']:g}"
                for _, r in sub.iterrows()
            )
            rows.append({'Legs': n, 'Selections': labels,
                         'Illustrative payout ×': round(mult, 2),
                         'Independence-based chance': round(prob, 5),
                         'Exploratory ROI': round(prob * mult - 1, 3),
                         'Break-even chance': round(1 / mult, 5)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Exploratory ROI', ascending=False).head(50)


def v5_tab():
    st.subheader('V5 · Automatic prop probability research')
    st.caption('Research/paper trading only · 2025 holdout residual estimates · no live FanDuel feed · no calibrated betting edge')
    needed = ['nfl_models', 'nfl_training_data', 'nfl_schedule', 'nfl_holdout_residuals']
    if any(k not in st.session_state for k in needed):
        st.info('First open **Train historical NFL player-stat models** below and click **Train models**. Once training finishes, return to this tab.')
        return
    schedule = st.session_state['nfl_schedule']
    models = st.session_state['nfl_models']
    d = st.session_state['nfl_training_data']
    residuals = st.session_state['nfl_holdout_residuals']
    games = schedule_matchups(schedule, 2026)
    if games.empty:
        st.warning('No unplayed 2026 regular-season games found in the schedule. Verify the schedule feed before generating props.')
        return
    labels = {f"Week {int(r.week)}: {r.away_team} @ {r.home_team}": r for r in games.itertuples()}
    chosen_label = st.selectbox('Upcoming game', list(labels), key='v5_game')
    chosen = labels[chosen_label]
    market = st.selectbox('Prop market', list(MARKETS), key='v5_market')
    opponent = st.selectbox('Defense / opponent', [chosen.home_team, chosen.away_team], key='v5_opponent')
    game_key = f"{int(chosen.week)}:{chosen.away_team}@{chosen.home_team}"
    try:
        projections = forecasts(models, d, market, 2026, opponent=opponent, schedule=schedule)
    except (ValueError, KeyError) as exc:
        st.error(f'Could not generate matchup projections: {exc}')
        return
    if projections.empty:
        st.info('No players with historical appearances found for that offense and market.')
        return
    st.dataframe(projections.head(30), hide_index=True, use_container_width=True)
    names = projections['Player'].dropna().astype(str).tolist()
    selected = st.selectbox('Player (confirm starter and availability yourself)', names, key='v5_player')
    projected = float(projections.loc[projections['Player'].eq(selected), 'Projected next-game value'].iloc[0])
    a, b, c = st.columns(3)
    side = a.selectbox('Prop side', ['Over', 'Under'], key='v5_side')
    line = b.number_input('Actual FanDuel prop line', min_value=0.0, value=float(max(0.5, round(projected) + 0.5)), step=0.5, key='v5_line')
    american = c.number_input('Actual FanDuel American odds', value=-110, step=5, key='v5_odds')
    try:
        decimal = decimal_odds(american)
        p, push_rate, sample = residual_hit_probability(projected, line, side, residuals[market])
    except (ValueError, TypeError) as exc:
        st.error(f'Cannot calculate probability: {exc}')
        return
    m1, m2, m3 = st.columns(3)
    m1.metric('Projected performance', f'{projected:.1f}')
    m2.metric('Exploratory hit probability', f'{p:.1%}')
    m3.metric('Illustrative single-leg ROI', f'{(p * decimal - 1):+.1%}')
    st.caption(f'Based on {sample:,} out-of-sample 2025 residuals for {market}. Estimated push frequency: {push_rate:.1%}. Estimates are not calibrated, and the ROI ignores pushes. Player status, usage, opponent and uncertainty may differ from the holdout sample.')
    if st.button('Add prop to research watchlist', type='primary', key='v5_add'):
        new = {'Game': game_key, 'Player': selected, 'Market': market,
               'Side': side, 'Line': float(line), 'American odds': int(american),
               'Decimal odds': float(decimal), 'Projected value': projected,
               'Estimated hit probability': float(p)}
        current = st.session_state.get('v5_watchlist', [])
        key = (new['Game'], new['Player'], new['Market'], new['Side'], new['Line'])
        current = [x for x in current if (x['Game'], x['Player'], x['Market'], x['Side'], x['Line']) != key]
        current.append(new)
        st.session_state['v5_watchlist'] = current
        st.success(f'Added {selected} {side} {line:g} to your watchlist.')
    st.divider()
    st.markdown('#### Flexible parlay screening · 10×–25×')
    watch = pd.DataFrame(st.session_state.get('v5_watchlist', []))
    if watch.empty:
        st.info('Add at least two props from different games to begin screening.')
        return
    st.dataframe(watch, hide_index=True, use_container_width=True)
    if st.button('Clear watchlist', key='v5_clear'):
        st.session_state['v5_watchlist'] = []
        st.rerun()
    x, y, z = st.columns(3)
    low = x.number_input('Minimum payout (×)', min_value=2.0, max_value=1000.0, value=10.0, step=1.0, key='v5_low')
    high = y.number_input('Maximum payout (×)', min_value=low, max_value=1000.0, value=max(low, 25.0), step=1.0, key='v5_high')
    max_legs = z.slider('Maximum legs', 2, 8, 6, key='v5_maxlegs')
    if st.button('Screen cross-game combinations', key='v5_screen'):
        try:
            result = candidate_combinations(watch, low, high, max_legs)
            if result.empty:
                st.info('No cross-game combinations within your payout target. Add more props from different games or adjust the range.')
            else:
                st.dataframe(result, hide_index=True, use_container_width=True)
                st.download_button('Download research combinations', result.to_csv(index=False),
                                   file_name='v5_parlay_research.csv', mime='text/csv')
        except ValueError as exc:
            st.error(str(exc))
    st.warning('Screening odds multiply individual prices; they are NOT confirmed combined FanDuel quotes. Cross-game results are treated as independent for illustration, but may still be correlated. No bet is warranted solely by a positive exploratory ROI. Verify starters, lines, odds and quoted combined payout before any decision.')
