"""V4 research-only prop and parlay workspace. No verified betting edge."""
import itertools
import math
import pandas as pd
import streamlit as st

COLUMNS = ['Game', 'Player', 'Market', 'Side', 'Line', 'American odds', 'Estimated hit probability']

def decimal_odds(american):
    a = float(american)
    if not math.isfinite(a) or a == 0 or -100 < a < 100:
        raise ValueError('Use valid American odds: +100 or greater, or -100 or lower.')
    return 1 + (a / 100 if a > 0 else 100 / abs(a))

def research_tab():
    st.subheader('V4 · FanDuel prop and flexible parlay research')
    st.caption('Research and paper trading only. Probabilities are manually entered, not calibrated model outputs. No live FanDuel connection.')
    if 'v4_props' not in st.session_state:
        st.session_state.v4_props = pd.DataFrame([
            {'Game':'', 'Player':'', 'Market':'Passing yards', 'Side':'Over', 'Line':225.5, 'American odds':-110, 'Estimated hit probability':0.52},
            {'Game':'', 'Player':'', 'Market':'Rushing yards', 'Side':'Over', 'Line':65.5, 'American odds':-110, 'Estimated hit probability':0.52},
            {'Game':'', 'Player':'', 'Market':'Receptions', 'Side':'Over', 'Line':4.5, 'American odds':-110, 'Estimated hit probability':0.52},
        ])
    c1,c2,c3 = st.columns(3)
    low = c1.number_input('Minimum total payout (×)', 2.0, 1000.0, 10.0, 1.0)
    high = c2.number_input('Maximum total payout (×)', min_value=float(low), max_value=1000.0, value=max(25.0,float(low)), step=1.0)
    max_legs = c3.slider('Maximum legs', 2, 8, 6)
    stake = st.number_input('Hypothetical stake ($)', 0.0, 10000.0, 10.0, 5.0)
    st.write('Enter props, **actual individual FanDuel odds**, and your own provisional hit-probability estimates. Use different games for the independence-based screening calculation.')
    edited = st.data_editor(st.session_state.v4_props, num_rows='dynamic', hide_index=True, use_container_width=True,
        column_config={'Side': st.column_config.SelectboxColumn(options=['Over','Under'], required=True),
        'Estimated hit probability':st.column_config.NumberColumn(min_value=0.001,max_value=0.999,format='%.3f'),
        'American odds':st.column_config.NumberColumn(format='%d')}, key='v4_editor')
    if st.button('Screen candidate combinations', type='primary'):
        try:
            d = edited.dropna(subset=['Game','Player','Market','Side','Line','American odds','Estimated hit probability']).copy()
            for col in ['Game','Player','Market']:
                d[col] = d[col].astype(str).str.strip()
            d = d[(d.Game!='') & (d.Player!='') & (d.Market!='')]
            if len(d) < 2:
                raise ValueError('Enter at least two complete prop rows with a game identifier.')
            if len(d)>20:
                raise ValueError('Limit screening to 20 candidate props to avoid excessive combinations.')
            if not d['Estimated hit probability'].between(0,1,inclusive='neither').all():
                raise ValueError('Every estimated probability must be between 0 and 1.')
            d['Decimal odds'] = d['American odds'].map(decimal_odds)
            records=[]
            for n in range(2,min(max_legs,len(d))+1):
                for inds in itertools.combinations(range(len(d)),n):
                    sub=d.iloc[list(inds)]
                    # Exclude all same-game combinations until a joint simulation exists.
                    if sub.Game.nunique()!=len(sub):
                        continue
                    # Avoid two selections for the same player in the same parlay.
                    if sub.Player.nunique()!=len(sub):
                        continue
                    mult=math.prod(sub['Decimal odds'])
                    if not low<=mult<=high:
                        continue
                    p=math.prod(sub['Estimated hit probability'])
                    label=' | '.join(f"{r.Player} {r.Market} {r.Side} {r.Line:g}" for r in sub.itertuples(index=False))
                    records.append({'Legs':n,'Selections':label,'Screening payout ×':round(mult,2),
                        'Assumed independent hit chance':round(p,5),'Illustrative ROI':round(p*mult-1,3),
                        'Break-even chance':round(1/mult,5)})
            result=pd.DataFrame(records)
            if result.empty:
                st.info('No eligible cross-game combinations at these settings. Try other lines or payout limits.')
            else:
                result=result.sort_values('Illustrative ROI',ascending=False).head(50)
                st.dataframe(result,hide_index=True,use_container_width=True)
                st.download_button('Download screening results',result.to_csv(index=False),file_name='v4_parlay_screen.csv',mime='text/csv')
                st.warning('These are screening calculations, NOT recommended bets. They assume independent games and user-entered probabilities; real outcomes can remain dependent. The payout is the product of individual quoted odds, not a verified FanDuel parlay quote.')
            st.session_state['v4_last_screen']=result
        except (ValueError,TypeError,OverflowError) as exc:
            st.error(f'Could not screen props: {exc}')
    st.divider()
    st.markdown('#### Verify a specific FanDuel parlay quote')
    actual=st.number_input('Actual combined FanDuel decimal odds (×)',min_value=1.01,value=10.0,step=0.5)
    joint=st.number_input('Independently validated JOINT hit probability (0–1)',min_value=0.0,max_value=1.0,value=0.0,step=0.005,format='%.3f')
    st.metric('Break-even joint probability',f'{1/actual:.2%}')
    st.metric('Total payout at quoted odds',f'${stake*actual:,.2f}')
    if joint>0:
        st.metric('Hypothetical expected ROI',f'{(joint*actual-1):+.1%}')
        st.info('This calculation is only as reliable as the independently validated joint probability you enter.')
    else:
        st.info('No validated joint probability supplied — expected value is intentionally not calculated. The current V3 models do not yet produce calibrated joint probabilities.')
