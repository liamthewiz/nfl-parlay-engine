"""NFL weekly player-prop research baselines. No validated betting edge."""
import io
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/stats_player/"
    "stats_player_week_{season}.csv"
)
MARKETS = {'Passing yards':('passing_yards',['QB']), 'Rushing yards':('rushing_yards',['RB','QB']), 'Receiving yards':('receiving_yards',['WR','TE','RB']), 'Receptions':('receptions',['WR','TE','RB'])}
FEATURES = ['lag1','avg3','avg5','avg10','games_played','season_week']

def download_data():
    frames = []
    failed = []

    for season in range(2021, 2027):
        url = DATA_URL.format(season=season)

        try:
            response = requests.get(
                url,
                timeout=90,
                headers={
                    "User-Agent": "nfl-parlay-research/1.0"
                }
            )
            response.raise_for_status()

            df = pd.read_csv(
                io.BytesIO(response.content),
                low_memory=False
            )

            if df.empty:
                raise ValueError("Empty season file")

            df["season"] = season
            frames.append(df)

        except (requests.RequestException, ValueError) as e:
            failed.append(f"{season}: {e}")

    if failed:
        raise ValueError(
            "NFL season downloads failed: "
            + " | ".join(failed)
        )

    data = pd.concat(frames, ignore_index=True)

    print(
        "Downloaded records by season:",
        data.groupby("season").size().to_dict()
    )

    return data


def prepare(df):
    required=['player_id','player_display_name','season','week','position']+[v[0] for v in MARKETS.values()]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError('Missing CSV columns: '+', '.join(missing))
    d=df.copy()
    for c in ['season','week']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['player_id','season','week'])
    d=d[(d.season>=2021)&(d.week.between(1,18))].copy()
    d=d.sort_values(['player_id','season','week']).drop_duplicates(['player_id','season','week'])
    for target,_ in MARKETS.values():
        d[target]=pd.to_numeric(d[target],errors='coerce').fillna(0).clip(lower=0)
    return d.reset_index(drop=True)

def data_diagnostics(raw):
    d=prepare(raw)
    return d.groupby(['season','position'],dropna=False).size().unstack(fill_value=0)

def feature_frame(d,target):
    x=d[['player_id','player_display_name','season','week',target]].copy()
    g=x.groupby('player_id',sort=False)[target]
    x['lag1']=g.shift(1)
    for w in (3,5,10):
        x[f'avg{w}']=g.transform(lambda s:s.shift(1).rolling(w,min_periods=1).mean())
    x['games_played']=x.groupby('player_id').cumcount()
    x['season_week']=x.week
    return x

def train_models(raw):
    d=prepare(raw)
    seasons=sorted(int(s) for s in d.season.unique())
    eligible=[s for s in seasons if s>=2023 and s<=2026]
    if not eligible: raise ValueError(f'No eligible holdout season (2023–2026). Available seasons: {seasons}')
   holdout = 2025

if (d.season == 2025).sum() < 100:
    raise ValueError(
        "Insufficient 2025 data. Check the season downloads."
    )
    results=[]; models={}
    for market,(target,positions) in MARKETS.items():
        subset=d[d.position.isin(positions)].copy()
        f=feature_frame(subset,target)
        f=f[f.games_played>=3].copy()
        train=f[f.season<holdout].copy(); valid=f[f.season==holdout].copy()
        if len(train)<100 or len(valid)<30:
            counts=f.groupby('season').size().to_dict()
            raise ValueError(f'{market}: insufficient rows for holdout {holdout}. Rows by season: {counts}. Check that the CSV contains multiple full NFL seasons.')
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=10,random_state=42)
        model.fit(train[FEATURES],train[target]); pred=np.maximum(0,model.predict(valid[FEATURES]))
        baseline=valid.avg5.fillna(valid.avg3).to_numpy()
        residual=valid[target].to_numpy()-pred
        results.append({'Market':market,'Holdout season':holdout,'Train rows':len(train),'Holdout rows':len(valid),'Model MAE':round(mean_absolute_error(valid[target],pred),2),'Rolling-5 MAE':round(mean_absolute_error(valid[target],baseline),2),'Residual SD':round(float(np.std(residual)),2)})
        models[market]=(model,float(np.std(residual)))
    return pd.DataFrame(results),models,d

def forecasts(models,d,market,season=2026):
    target,positions=MARKETS[market]
    subset=d[d.position.isin(positions)].copy()
    f=feature_frame(subset,target)
    latest=f[f.season==season].sort_values('week').groupby('player_id').tail(1).copy()
    if latest.empty:return pd.DataFrame()
    g=subset.sort_values(['player_id','season','week']).groupby('player_id')[target]
    latest['lag1']=latest[target]
    for w in (3,5,10):
        latest[f'avg{w}']=latest.player_id.map(g.apply(lambda s:s.tail(w).mean()))
    latest['games_played']+=1
    latest['season_week']=(latest.week+1).clip(upper=19)
    model,sd=models[market]
    latest['Projected next-game value']=np.maximum(0,model.predict(latest[FEATURES])).round(1)
    latest['Validation residual SD']=round(sd,1)
    return latest[['player_display_name','season','week','Projected next-game value','Validation residual SD']].rename(columns={'player_display_name':'Player','week':'Last recorded week'}).sort_values('Projected next-game value',ascending=False)
