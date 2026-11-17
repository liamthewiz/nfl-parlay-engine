# NFL Parlay Prediction Engine — Web Dashboard Prototype

This is an interactive **prototype**, not a trained prediction model. It does not pull live FanDuel odds, NFL stats, or injuries. The entered leg probabilities are user-provided and multiplied under an independence assumption, which is **not valid for many same-game parlays**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit in your browser. For phone access, deploy to Streamlit Community Cloud or another hosting service; this package is **not hosted** yet.

## Next engineering milestones

1. Ingest nflverse historical play-by-play and weekly player data; build strictly pre-game features.
2. Train and calibrate predictive distributions for pass attempts/yards, carries/rushing yards, targets/receptions/receiving yards.
3. Simulate game-level correlated player outcomes and estimate joint parlay hit probabilities with uncertainty intervals.
4. Integrate licensed odds snapshots and obtain **actual FanDuel same-game parlay quotes** before evaluating EV.
5. Implement walk-forward testing, paper trading, closing-line comparisons, and persisted records.
6. Reject parlays without reliable estimates or without a verified 25×+ quote.

Never treat prototype estimates as actionable betting advice.
