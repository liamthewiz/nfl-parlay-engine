"""NFL weekly player-stat and matchup research. Not a validated betting edge."""
import io
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from v3_context import team_context

DATA_URL = ('https://github.com/nflverse/nflverse-data/releases/download/'
            'stats_player/stats_player_week_{season}.csv')
MARKETS = {
    'Passing yards': ('passing_yards', ['QB']),
    'Rushing yards': ('rushing_yards', ['RB', 'QB']),
    'Receiving yards': ('receiving_yards', ['WR', 'TE', 'RB']),
    'Receptions': ('receptions', ['WR', 'TE', 'RB']),
}
BASE_FEATURES = ['lag1', 'avg3', 'avg5', 'avg10', 'games_played', 'season_week']
MATCH_FEATURES = BASE_FEATURES + ['opp_allowed_avg5', 'h2h_avg', 'h2h_games']
CONTEXT_FEATURES = MATCH_FEATURES + ['team_points_avg5', 'opponent_points_allowed_avg5', 'home_flag']
FEATURES = BASE_FEATURES  # Compatibility with existing dashboard code.


def download_data():
    frames, errors = [], []
    for season in range(2021, 2027):
        try:
            response = requests.get(DATA_URL.format(season=season), timeout=90,
                                    headers={'User-Agent': 'nfl-parlay-research/2.0'})
            response.raise_for_status()
            frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
            if frame.empty:
                raise ValueError('Empty CSV')
            frame['season'] = season
            frames.append(frame)
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            errors.append(f'{season}: {exc}')
    if errors:
        raise ValueError('Season downloads failed: ' + ' | '.join(errors))
    return pd.concat(frames, ignore_index=True)


def prepare(df):
    required = ['player_id', 'player_display_name', 'season', 'week', 'position',
                'opponent_team'] + [v[0] for v in MARKETS.values()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(missing) +
                         '. Matchup analysis requires opponent_team in the weekly data.')
    d = df.copy()
    for col in ['season', 'week']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna(subset=['player_id', 'season', 'week'])
    d = d[(d.season >= 2021) & d.week.between(1, 18)].copy()
    d['opponent_team'] = d['opponent_team'].astype('string').str.upper().str.strip()
    d = d.sort_values(['player_id', 'season', 'week']).drop_duplicates(
        ['player_id', 'season', 'week'])
    for target, _ in MARKETS.values():
        d[target] = pd.to_numeric(d[target], errors='coerce').fillna(0).clip(lower=0)
    return d.reset_index(drop=True)


def data_diagnostics(raw):
    d = prepare(raw)
    return d.groupby(['season', 'position'], dropna=False).size().unstack(fill_value=0)


def feature_frame(d, target):
    x = d[['player_id', 'player_display_name', 'season', 'week',
           'opponent_team', target]].copy().sort_values(['player_id', 'season', 'week'])
    group = x.groupby('player_id', sort=False)[target]
    x['lag1'] = group.shift(1)
    for window in (3, 5, 10):
        x[f'avg{window}'] = group.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    x['games_played'] = x.groupby('player_id').cumcount()
    x['season_week'] = x.week
    return x


def matchup_frame(d, target):
    """Opponent allowed is a historical proxy, not position-adjusted defensive EPA."""
    x = feature_frame(d, target)
    # Sum player production faced by each opponent in each completed game.
    allowed = (d.dropna(subset=['opponent_team'])
               .groupby(['opponent_team', 'season', 'week'], as_index=False)[target]
               .sum().sort_values(['opponent_team', 'season', 'week']))
    allowed['opp_allowed_avg5'] = allowed.groupby('opponent_team')[target].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    x = x.merge(allowed[['opponent_team', 'season', 'week', 'opp_allowed_avg5']],
                on=['opponent_team', 'season', 'week'], how='left', validate='many_to_one')
    x = x.sort_values(['player_id', 'opponent_team', 'season', 'week'])
    h2h = x.groupby(['player_id', 'opponent_team'], dropna=True)[target]
    x['h2h_avg'] = h2h.transform(lambda s: s.shift(1).expanding().mean())
    x['h2h_games'] = x.groupby(['player_id', 'opponent_team']).cumcount()
    x.loc[x.opponent_team.isna(), 'h2h_games'] = 0
    x['h2h_avg'] = x['h2h_avg'].fillna(x['avg5'])
    return x.sort_values(['player_id', 'season', 'week'])


def context_frame(d, target, schedule):
    """Pregame context joined by exact team/opponent/week, never current scores."""
    if 'recent_team' not in d.columns:
        raise ValueError('Player stats missing recent_team; needed for V3 game context')
    x = matchup_frame(d, target)
    teams = team_context(schedule)
    offense = teams[['season','week','team','opponent_team','is_home','points_for_avg5']].rename(
        columns={'team':'recent_team','is_home':'home_flag','points_for_avg5':'team_points_avg5'})
    defense = teams[['season','week','team','points_against_avg5']].rename(
        columns={'team':'opponent_team','points_against_avg5':'opponent_points_allowed_avg5'})
    x = x.merge(d[['player_id','season','week','recent_team']],on=['player_id','season','week'],how='left',validate='one_to_one')
    x = x.merge(offense,on=['season','week','recent_team','opponent_team'],how='left',validate='many_to_one')
    x = x.merge(defense,on=['season','week','opponent_team'],how='left',validate='many_to_one')
    return x


def _model():
    return HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                         l2_regularization=10, random_state=42)


def train_models(raw, schedule=None):
    d = prepare(raw)
    if (d.season == 2025).sum() < 100:
        raise ValueError('2025 holdout data unavailable. Seasons: ' +
                         str(sorted(d.season.unique().tolist())))
    report, models = [], {}
    if schedule is None:
        raise ValueError('V3 requires the historical schedule. Download it before training.')
    for market, (target, positions) in MARKETS.items():
        subset = d[d.position.isin(positions)].copy()
        base = feature_frame(subset, target)
        match = matchup_frame(subset, target)
        context = context_frame(subset, target, schedule)
        base = base[base.games_played >= 3].copy()
        match = match[match.games_played >= 3].copy()
        context = context[context.games_played >= 3].copy()
        train_mask = base.season < 2025
        valid_mask = base.season == 2025
        if train_mask.sum() < 100 or valid_mask.sum() < 30:
            raise ValueError(f'Insufficient {market} data for 2025 holdout')
        baseline = base.loc[valid_mask, 'avg5'].fillna(base.loc[valid_mask, 'lag1'])
        scores = {}
        for label, frame, columns in [('Baseline ML', base, BASE_FEATURES),
                                      ('Matchup ML', match, MATCH_FEATURES),
                                      ('Context V3', context, CONTEXT_FEATURES)]:
            train = frame[frame.season < 2025]
            valid = frame[frame.season == 2025]
            m = _model()
            m.fit(train[columns], train[target])
            pred = np.maximum(0, m.predict(valid[columns]))
            scores[label] = (mean_absolute_error(valid[target], pred),
                             float(np.std(valid[target].to_numpy() - pred)))
            if label == 'Context V3':
                final = _model()
                final.fit(frame[columns], frame[target])
                models[market] = (final, scores[label][1])
        report.append({
            'Market': market, 'Holdout season': 2025,
            'Train rows': int(train_mask.sum()), 'Holdout rows': int(valid_mask.sum()),
            'Model MAE': round(scores['Context V3'][0], 2),
            'Rolling-5 MAE': round(mean_absolute_error(base.loc[valid_mask, target], baseline), 2),
            'Residual SD': round(scores['Context V3'][1], 2),
            'Old model MAE': round(scores['Baseline ML'][0], 2),
            'Matchup model MAE': round(scores['Matchup ML'][0], 2),
            'Context V3 MAE': round(scores['Context V3'][0], 2),
        })
    return pd.DataFrame(report), models, d


def forecasts(models, d, market, season=2026, opponent=None, schedule=None):
    """Next-appearance projections. Set opponent for a specific defensive matchup."""
    if market not in MARKETS:
        raise ValueError(f'Unknown market: {market}')
    target, positions = MARKETS[market]
    subset = d[d.position.isin(positions)].copy()
    f = feature_frame(subset, target)
    latest = (f[f.season == season].sort_values(['player_id', 'week'])
              .groupby('player_id').tail(1).copy())
    if latest.empty:
        return pd.DataFrame()
    # Each player's own historical production, including their most recent game.
    history = subset.sort_values(['player_id', 'season', 'week'])
    for window in (3, 5, 10):
        averages = history.groupby('player_id')[target].apply(
            lambda s: s.tail(window).mean())
        latest[f'avg{window}'] = latest.player_id.map(averages)
    latest['lag1'] = latest[target]
    latest['games_played'] += 1
    latest['season_week'] = (latest.week + 1).clip(upper=19)
    # The opponent must be selected for an actual matchup projection.
    if opponent is None:
        latest['opp_allowed_avg5'] = np.nan
        latest['h2h_avg'] = latest['avg5']
        latest['h2h_games'] = 0
    else:
        opponent = str(opponent).upper().strip()
        latest['opponent_team'] = opponent
        allowed = (subset[subset.opponent_team == opponent]
                   .groupby(['season', 'week'])[target].sum().sort_index())
        latest['opp_allowed_avg5'] = allowed.tail(5).mean() if len(allowed) else np.nan
        prior_h2h = (subset[subset.opponent_team == opponent]
                     .groupby('player_id')[target].agg(['mean', 'count']))
        latest['h2h_avg'] = latest.player_id.map(prior_h2h['mean']).fillna(latest.avg5)
        latest['h2h_games'] = latest.player_id.map(prior_h2h['count']).fillna(0)
    if schedule is None or opponent is None:
        raise ValueError('V3 forecasts require the schedule and a confirmed opponent')
    if 'recent_team' not in subset.columns:
        raise ValueError('Missing recent_team in player stats')
    team_map = subset.sort_values(['season','week']).groupby('player_id').recent_team.last()
    latest['recent_team'] = latest.player_id.map(team_map)
    upcoming = schedule[(schedule.season==season) & (schedule.home_score.isna() | schedule.away_score.isna())]
    upcoming = upcoming[(upcoming.home_team==opponent) | (upcoming.away_team==opponent)]
    if upcoming.empty:
        raise ValueError('No unplayed scheduled game found for selected opponent')
    next_game = upcoming.sort_values('week').iloc[0]
    home, away = next_game.home_team, next_game.away_team
    latest = latest[latest.recent_team.isin([home,away])].copy()
    latest = latest[latest.recent_team.ne(opponent)].copy()
    if latest.empty:
        return pd.DataFrame()
    hist = team_context(schedule)
    for index, row in latest.iterrows():
        team = row.recent_team
        opponent_actual = away if team == home else home
        team_hist = hist[(hist.team==team)&(hist.points_for.notna())&(hist.season<=season)].sort_values(['season','week'])
        opp_hist = hist[(hist.team==opponent_actual)&(hist.points_against.notna())&(hist.season<=season)].sort_values(['season','week'])
        latest.loc[index,'team_points_avg5'] = team_hist.points_for.tail(5).mean()
        latest.loc[index,'opponent_points_allowed_avg5'] = opp_hist.points_against.tail(5).mean()
        latest.loc[index,'home_flag'] = int(team==home)
    latest['season_week'] = next_game.week
    model, sd = models[market]
    latest['Projected next-game value'] = np.maximum(
        0, model.predict(latest[CONTEXT_FEATURES])).round(1)
    latest['Validation residual SD'] = round(sd, 1)
    result = latest[['player_display_name', 'season', 'week',
                     'Projected next-game value', 'Validation residual SD',
                     'opp_allowed_avg5', 'h2h_avg', 'h2h_games']].rename(columns={
                         'player_display_name': 'Player', 'week': 'Last recorded week',
                         'opp_allowed_avg5': 'Opponent allowed (last 5)',
                         'h2h_avg': 'Previous H2H average',
                         'h2h_games': 'Previous H2H games'})
    return result.sort_values('Projected next-game value', ascending=False)
