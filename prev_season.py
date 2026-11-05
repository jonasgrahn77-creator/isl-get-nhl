#!/usr/bin/env python3
"""Sparar forra sasongens nyckeltal till sidan. Anvandning: python training/prev_season.py mapp
Mappen ska innehalla skaters.csv, goalies.csv, lines.csv och teams.csv for en avslutad sasong."""
import gzip, json, pathlib, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline as P

d = pathlib.Path(sys.argv[1])
s, _ = P.load(d / "skaters.csv"); g = P.load_goalies(d / "goalies.csv"); l = P.load_lines(d / "lines.csv"); t = P.load_teams(d / "teams.csv")
problems = P.validate(s) + P.validate_goalies(g) + P.validate_lines(l) + P.validate_teams(t) + P.cross_checks(s, g, l, t)
if problems:
    sys.exit("Filerna godkandes inte: " + "; ".join(problems))
gm = P.goalie_metrics(g); gm = gm[gm.situation == "all"]
tm = P.team_metrics(t, P.goalie_metrics(g))
lm = P.line_metrics(l)
a = s[s.situation == "all"]
out = {"season": int(s.season.iloc[0]),
       "skaters": {str(r.playerId): {"gp": int(r.games_played), "p": int(r.I_F_points), "g": int(r.I_F_goals)} for r in a.itertuples()},
       "goalies": {str(r.playerId): {"gp": int(r.games_played), "gsax": round(r.gsax, 1), "sv": round(r.save_pct, 1) if pd.notna(r.save_pct) else None} for r in gm.itertuples()},
       "teams": {r.team: {"xg": round(r.xg_pct_5on5, 1), "gd": int(r.gf - r.ga)} for r in tm.itertuples()},
       "lines": {r.lineId: {"min": round(r.minutes), "xg": round(r.xg_pct, 1)} for r in lm.itertuples() if pd.notna(r.xg_pct)}}
with gzip.open(ROOT / "model" / "prev_season.json.gz", "wt", encoding="utf-8") as f:
    json.dump(out, f)
print("Sparat", out["season"], {k: len(v) for k, v in out.items() if isinstance(v, dict)})
