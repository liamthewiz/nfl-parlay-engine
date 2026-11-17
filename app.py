import math
import streamlit as st
import pandas as pd

st.set_page_config(page_title='NFL Parlay Prediction Engine', page_icon='🏈', layout='wide')
st.title('🏈 NFL Parlay Prediction Engine')
st.caption('FanDuel · 5–8 legs · target 25×+ · prototype / paper trading')

with st.sidebar:
    st.header('Settings')
    min_legs = st.slider('Minimum legs', 5, 8, 5)
    stake = st.number_input('Hypothetical stake ($)', min_value=0.0, value=10.0, step=5.0)
    target = st.number_input('Minimum payout multiplier', min_value=25.0, value=25.0)
    st.info('This prototype does not fetch live games or predict outcomes. Enter your own estimates; results are illustrative.')

if 'legs' not in st.session_state:
    st.session_state.legs = pd.DataFrame([
        {'Game':'', 'Player':'', 'Market':'Passing yards over', 'Line':0.0, 'Estimated hit probability':0.60},
        {'Game':'', 'Player':'', 'Market':'Rushing yards over', 'Line':0.0, 'Estimated hit probability':0.60},
        {'Game':'', 'Player':'', 'Market':'Receiving yards over', 'Line':0.0, 'Estimated hit probability':0.60},
        {'Game':'', 'Player':'', 'Market':'Receptions over', 'Line':0.0, 'Estimated hit probability':0.60},
        {'Game':'', 'Player':'', 'Market':'Anytime touchdown', 'Line':0.0, 'Estimated hit probability':0.30},
    ])

t1,t2,t3 = st.tabs(['Parlay Builder','Game Research','Results Tracker'])
with t1:
    st.subheader('Build a candidate parlay')
    st.write('Edit the legs and your probability estimates. These estimates are **not model-generated** yet.')
    df = st.data_editor(st.session_state.legs, num_rows='dynamic', use_container_width=True,
        column_config={'Estimated hit probability':st.column_config.NumberColumn(min_value=0.001,max_value=0.999,step=0.01,format='%.3f')}, key='editor')
    combined = st.number_input('Actual FanDuel combined decimal odds (e.g. 25.0)', min_value=1.01, value=25.0, step=1.0)
    st.warning('Multiplying individual probabilities assumes independence. Same-game NFL props are often correlated. The result below is a rough illustration, NOT a calibrated joint probability or betting edge.')
    valid = min_legs <= len(df) <= 8 and df['Estimated hit probability'].between(0,1,inclusive='neither').all()
    if not valid:
        st.error(f'Enter {min_legs}–8 legs with probabilities strictly between 0 and 1.')
    else:
        p = math.prod(df['Estimated hit probability'].astype(float))
        ev = p*combined-1
        a,b,c = st.columns(3)
        a.metric('Independence-based hit chance', f'{p:.2%}')
        b.metric('Total payout', f'${stake*combined:,.2f}')
        c.metric('Illustrative expected ROI', f'{ev:+.1%}')
        if combined < target:
            st.error(f'Below target: {combined:.1f}× vs {target:.1f}× minimum.')
        else:
            st.success('Payout threshold met. This does NOT mean the parlay has positive value.')
        st.caption(f'Break-even probability at these odds: {1/combined:.2%}. The true joint probability must be estimated with a validated correlated simulation model.')
with t2:
    st.subheader('Game Research — matchup research')
    st.write('Scheduled matchups and opponent-adjusted research are in the V3 Training Lab below. Injuries, weather and odds are not connected.')
    st.info('Choose a scheduled game in the forecast section. Verify schedule freshness before relying on projections.')
with t3:
    st.subheader('Paper-trading log')
    st.write('Record completed test parlays here; keep odds and probability estimates captured before kickoff.')
    st.dataframe(pd.DataFrame(columns=['Date','Matchup','Legs','Quoted odds','Predicted joint probability','Result','Stake','Profit / loss']), use_container_width=True)
    st.caption('Persistent storage and backtesting will be added in the next development stage.')

st.divider()
st.header('🧠 NFL Game Context Training Lab — V3')
st.caption('Experimental opponent + scoring-context forecasts. Schedule is downloaded; injuries, weather, odds and correlated simulations are not connected.')
with st.expander('Train historical NFL player-stat models', expanded=False):
    st.write('Use the official nflverse player-stats CSV, or upload a previously downloaded copy. Training runs in memory on this Streamlit session.')
    source = st.radio('Data source', ['Download nflverse CSV', 'Upload nflverse CSV'], horizontal=True)
    upload = st.file_uploader('Upload player_stats.csv', type='csv') if source.startswith('Upload') else None
    if st.button('Train models', type='primary'):
        try:
            from model_engine import download_data, train_models, data_diagnostics
            from v3_context import download_schedule
            with st.spinner('Loading historical games and fitting four models...'):
                raw = pd.read_csv(upload,low_memory=False) if upload is not None else download_data()
                st.session_state["nfl_data_diagnostics"] = data_diagnostics(raw)
                schedule = download_schedule()
                report, models, clean = train_models(raw, schedule=schedule)
                st.session_state['nfl_schedule'] = schedule
                st.session_state['nfl_model_report'] = report
                st.session_state['nfl_models'] = models
                st.session_state['nfl_training_data'] = clean
            st.success('Training complete. Review the 2025 holdout comparison below.')
        except Exception as exc:
            st.error(f'Training could not complete: {exc}')
            if 'nfl_data_diagnostics' in st.session_state:
                st.write('Available records by season and position:')
                st.dataframe(st.session_state['nfl_data_diagnostics'])
if 'nfl_model_report' in st.session_state:
    st.subheader('Historical holdout validation')
    st.dataframe(st.session_state['nfl_model_report'], hide_index=True, use_container_width=True)
    st.caption('MAE is mean absolute error in yards (or catches for receptions). Lower is better. The 2025 holdout uses only earlier seasons for model fitting. V3 adds lagged team scoring context; it does not yet use EPA or pressure metrics.')
    market = st.selectbox('Forecast market', list(st.session_state['nfl_models']))
    season = st.selectbox('Latest season to forecast from', [2026,2025])
    from model_engine import forecasts
    from v3_context import schedule_matchups
    schedule = st.session_state.get('nfl_schedule')
    games = schedule_matchups(schedule, season) if schedule is not None else pd.DataFrame()
    if games.empty:
        st.warning('No unplayed regular-season games found in the schedule feed. Check its update date.')
        fc = pd.DataFrame()
    else:
        game_labels = {f"Week {int(r.week)}: {r.away_team} @ {r.home_team}": r for r in games.itertuples()}
        selected_game = st.selectbox('Upcoming scheduled matchup', list(game_labels))
        chosen = game_labels[selected_game]
        opponent = st.selectbox('Defensive opponent', [chosen.home_team, chosen.away_team])
        fc = forecasts(st.session_state['nfl_models'], st.session_state['nfl_training_data'], market,
                       season, opponent=opponent, schedule=schedule)
    if fc.empty:
        st.info('No games found for that season. Try 2025, or upload newer weekly data.')
    else:
        st.dataframe(fc.head(100),hide_index=True,use_container_width=True)
        st.download_button('Download forecast CSV',fc.to_csv(index=False),file_name='nfl_forecasts.csv',mime='text/csv')
    st.warning('These are experimental next-appearance estimates. Without selecting the actual upcoming opponent, projections are not matchup-specific. They are not calibrated FanDuel betting probabilities or verified 25× parlay edges.')
else:
    st.info('Open the training panel above and click Train models to start.')
