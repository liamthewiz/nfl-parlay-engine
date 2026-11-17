import math
import streamlit as st
import pandas as pd

st.set_page_config(page_title='NFL Parlay Prediction Engine', page_icon='🏈', layout='wide')
st.title('🏈 NFL Parlay Prediction Engine')
st.caption('FanDuel research · flexible 2–8 legs · target 10×–25× · paper trading')

from v4_parlays import research_tab

t1,t2,t3 = st.tabs(['V4 Parlay Research','Game Research','Results Tracker'])
with t1:
    research_tab()
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
