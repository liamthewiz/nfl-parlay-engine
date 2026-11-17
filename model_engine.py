"""Leakage-aware NFL weekly player-stat baseline models. No betting recommendations."""
import io
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_URL = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv'
MARKETS = {
    'Passing yards': ('passing_yards', 'passing_yards'),
    'Rushing yards': ('rushing_yards', 'rushing_yards'),
    'Receiving yards': ('receiving_yards', 'receiving_yards'),
    'Receptions': ('receptions', 'receptions'),
}
FEATURES = ['lag1', 'avg3', 'avg5', 'avg10', 'games_played', 'season_week']

def download_data():
    r = requests.get(DATA_URL, timeout=80, headers={'User-Agent':'nfl-parlay-research/1.0'})
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)

def prepare(df):
    required = ['player_id','player_display_name','season','week'] + [x[0] for x in MARKETS.values()]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing) + '. Use nflverse weekly player stats CSV.')
    d = df.copy()
    d['season'] = pd.to_numeric(d['season'], errors='coerce')
    d['week'] = pd.to_numeric(d['week'], errors='coerce')
    d = d.dropna(subset=['player_id','season','week'])
    d = d[(d.season >= 2021) & (d.week.between(1,18))]
    d = d.sort_values(['player_id','season','week']).drop_duplicates(['player_id','season','week'])
    for target, _ in MARKETS.values():
        d[target] = pd.to_numeric(d[target], errors='coerce').fillna(0).clip(lower=0)
    return d.reset_index(drop=True)

def feature_frame(d, target):
    x = d[['player_id','player_display_name','season','week',target]].copy()
    # The current game never contributes to its own predictors.
    g = x.groupby('player_id', sort=False)[target]
    x['lag1'] = g.shift(1)
    for window in (3,5,10):
        x[f'avg{window}'] = x.groupby('player_id', sort=False)[target].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    x['games_played'] = x.groupby('player_id').cumcount()
    x['season_week'] = x['week']
    return x

def train_models(raw):
    d = prepare(raw)
    results, models = [], {}
    for market, (target, _) in MARKETS.items():
        f = feature_frame(d, target)
        f = f[f.games_played >= 3].copy()
        train = f[f.season.between(2021,2024)]
        valid = f[f.season == 2025]
        if len(train) < 100 or len(valid) < 30:
            raise ValueError('Need 2021–2025 historical records to train and validate. Found insufficient data for '+market)
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, l2_regularization=10, random_state=42)
        model.fit(train[FEATURES], train[target])
        pred = np.maximum(0, model.predict(valid[FEATURES]))
        baseline = valid['avg5'].fillna(valid['avg3']).to_numpy()
        residuals = valid[target].to_numpy()-pred
        results.append({'Market':market,'Train rows':len(train),'2025 test rows':len(valid),
                        'Model MAE':round(mean_absolute_error(valid[target],pred),2),
                        'Rolling-5 MAE':round(mean_absolute_error(valid[target],baseline),2),
                        'Residual SD':round(float(np.std(residuals)),2)})
        models[market] = (model, float(np.std(residuals)))
    return pd.DataFrame(results), models, d

def forecasts(models, d, market, season=2026):
    target = MARKETS[market][0]
    f = feature_frame(d, target)
    # Construct NEXT-game features from games completed through the latest recorded week.
    latest = f[f.season == season].sort_values('week').groupby('player_id').tail(1).copy()
    if latest.empty:
        return pd.DataFrame()
    player_history = d.sort_values(['player_id','season','week']).groupby('player_id')[target]
    latest['lag1'] = latest[target]
    for w in (3,5,10):
        tail_mean = player_history.apply(lambda s: s.tail(w).mean())
        latest[f'avg{w}'] = latest['player_id'].map(tail_mean)
    latest['games_played'] += 1
    latest['season_week'] = (latest['week']+1).clip(upper=19)
    model, sd = models[market]
    latest['Projected next-game value'] = np.maximum(0,model.predict(latest[FEATURES])).round(1)
    latest['Validation residual SD'] = round(sd,1)
    return latest[['player_display_name','season','week','Projected next-game value','Validation residual SD']].rename(
        columns={'player_display_name':'Player','week':'Last recorded week'}).sort_values('Projected next-game value',ascending=False)
