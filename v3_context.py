"""Pregame schedule and game-script features. Research only; no live odds."""
import io
import numpy as np
import pandas as pd
import requests

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'

def download_schedule():
    r = requests.get(SCHEDULE_URL, timeout=45, headers={'User-Agent': 'nfl-parlay-research/3.0'})
    r.raise_for_status()
    games = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    needed = {'season','week','home_team','away_team','home_score','away_score'}
    if not needed.issubset(games.columns):
        raise ValueError('Schedule missing columns: '+str(sorted(needed-set(games.columns))))
    games['season'] = pd.to_numeric(games.season, errors='coerce')
    games['week'] = pd.to_numeric(games.week, errors='coerce')
    for c in ('home_score','away_score'):
        games[c] = pd.to_numeric(games[c], errors='coerce')
    if 'game_type' in games:
        games = games[games.game_type.eq('REG')]
    games = games[games.week.between(1,18)].copy()
    return games

def team_context(games):
    """Lagged team points for/against and scoring margin; no future results used."""
    home = games[['season','week','home_team','away_team','home_score','away_score']].rename(columns={
        'home_team':'team','away_team':'opponent_team','home_score':'points_for','away_score':'points_against'})
    home['is_home'] = 1
    away = games[['season','week','away_team','home_team','away_score','home_score']].rename(columns={
        'away_team':'team','home_team':'opponent_team','away_score':'points_for','home_score':'points_against'})
    away['is_home'] = 0
    all_games = pd.concat([home,away], ignore_index=True).sort_values(['team','season','week'])
    for c in ('points_for','points_against'):
        all_games[c+'_avg5'] = all_games.groupby('team')[c].transform(
            lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    all_games['expected_margin_proxy'] = all_games.points_for_avg5-all_games.points_against_avg5
    return all_games

def schedule_matchups(games, season=2026):
    g = games[(games.season==season) & (games.home_score.isna() | games.away_score.isna())].copy()
    if g.empty:
        return pd.DataFrame(columns=['season','week','home_team','away_team'])
    first_week = g.week.min()
    return g[g.week==first_week][['season','week','home_team','away_team']].sort_values('home_team')
