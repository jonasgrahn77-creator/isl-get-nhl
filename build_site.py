"""Bygger den statiska hemsidan fran site_template.html."""
import gzip
import json
import math
import pathlib
import pipeline

HERE = pathlib.Path(__file__).parent
SITS = ["all", "5on5", "5on4", "4on5"]


def load_prev(season):
    """Forra sasongens varden, bara om filen verkligen galler forra sasongen."""
    path = HERE / "model" / "prev_season.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        prev = json.load(f)
    return prev if prev.get("season") == season - 1 else {}


def _c(v, digits=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), digits)


def _skaters(raw, metrics, rated, deltas, top_line, team_xg, prev):
    raw_by = {(r.playerId, r.situation): r for r in raw.itertuples()}
    met_by = {(r.playerId, r.situation): r for r in metrics.itertuples()}
    out = []
    for p in rated.itertuples():
        sits = {}
        for s in SITS:
            r, m = raw_by[(p.playerId, s)], met_by[(p.playerId, s)]
            sits[s] = {"min": _c(m.minutes, 0), "g": _c(r.I_F_goals, 0), "p": _c(r.I_F_points, 0),
                       "xg": _c(r.I_F_xGoals), "sog": _c(r.I_F_shotsOnGoal, 0)}
        out.append({
            "id": int(p.playerId), "name": p.name, "team": p.team, "pos": p.position, "group": p.group,
            "gp": int(p.games_played), "rating": _c(p.rating, 0), "conf": p.confidence,
            "delta": deltas.get(int(p.playerId)), "min": _c(p.minutes, 0),
            "cxg": _c(p.created_xg_60), "pa": _c(p.primary_a_60), "xga": _c(p.xga_60),
            "rel": _c(p.rel_xg_pct, 1), "gax": _c(p.goals_minus_xg), "oz": _c(p.ozone_start_pct, 0),
            "pn": _c(p.penalty_net, 0), "puck": _c(p.puck_net, 0),
            "pts": sits["all"]["p"], "teamXg": _c(team_xg.get(p.team), 1), "line": top_line.get(int(p.playerId)),
            "prev": prev.get(str(int(p.playerId))), "p60": _c(p.points_60), "fc": {"cxg": _c(p.f_cxg60), "p60": _c(p.f_p60), "xga": _c(p.f_xga60), "rel": _c(p.f_relxg, 1)},
            "pct": {"cxg": _c(p.pct_cxg60, 0), "p60": _c(p.pct_p60, 0), "xga": _c(p.pct_xga60, 0), "rel": _c(p.pct_relxg, 0)},
            "sits": sits})
    return out


def _goalies(gm, gr, deltas, prev):
    by = {(r.playerId, r.situation): r for r in gm.itertuples()}
    out = []
    for g in gr.itertuples():
        sits = {s: {"min": _c(by[(g.playerId, s)].minutes, 0), "ga": _c(by[(g.playerId, s)].goals, 0),
                    "sog": _c(by[(g.playerId, s)].shots, 0), "gsax": _c(by[(g.playerId, s)].gsax, 1),
                    "sv": _c(by[(g.playerId, s)].save_pct, 1)} for s in SITS}
        out.append({
            "id": int(g.playerId), "name": g.name, "team": g.team, "gp": int(g.games_played),
            "rating": _c(g.rating, 0), "conf": g.confidence, "delta": deltas.get(int(g.playerId)),
            "prevGsax": (prev.get(str(int(g.playerId))) or {}).get("gsax"), "prevSv": (prev.get(str(int(g.playerId))) or {}).get("sv"),
            "min": _c(g.minutes, 0), "gsax": _c(g.gsax, 1), "g60": _c(g.gsax_60), "sv": _c(g.save_pct, 1),
            "ga": _c(g.goals, 0), "sog": _c(g.shots, 0), "sits": sits,
            "danger": {d: {"shots": _c(getattr(g, f"{d}_shots"), 0), "goals": _c(getattr(g, f"{d}_goals"), 0),
                           "gsax": _c(getattr(g, f"{d}_gsax"), 1)} for d in ("low", "medium", "high")}})
    return out


def _lines(lm, names, prev):
    out = []
    for l in lm.itertuples():
        out.append({"id": l.lineId, "name": l.name, "team": l.team, "type": l.position, "gp": int(l.games_played),
                    "min": _c(l.minutes, 0), "xg": _c(l.xg_pct, 1), "xgf": _c(l.xgf_60), "xga": _c(l.xga_60),
                    "gf": _c(l.gf, 0), "ga": _c(l.ga, 0), "prevXg": (prev.get(l.lineId) or {}).get("xg"), "prevMin": (prev.get(l.lineId) or {}).get("min"),
                    "players": [names.get(i, str(i)) for i in pipeline.line_players(l.lineId)]})
    return out


def _teams(tm, prev):
    return [{"team": t.team, "gp": int(t.games_played), "gf": _c(t.gf, 0), "ga": _c(t.ga, 0), "diff": _c(t.gf - t.ga, 0),
             "xg": _c(t.xg_pct_5on5, 1), "xgf": _c(t.xgf_60_5on5), "xga": _c(t.xga_60_5on5), "pp": _c(t.pp_xgf_60),
             "ppg": _c(t.pp_goals, 0), "pk": _c(t.pk_xga_60), "pkga": _c(t.pk_goals_against, 0),
             "fin": _c(t.finishing, 1), "gsax": _c(t.goalie_gsax, 1), "prevXg": (prev.get(t.team) or {}).get("xg")} for t in tm.itertuples()]


def render(d, updated, deltas=None, note=None):
    """d innehaller raw, metrics, rated, goalie_metrics, goalie_ratings, line_metrics, team_metrics."""
    deltas = deltas or {}
    names = dict(zip(d["raw"].playerId, d["raw"].name))
    top_line = {}
    for l in d["line_metrics"].sort_values("minutes").itertuples():  # langst istid skrivs sist och vinner
        for pid in pipeline.line_players(l.lineId):
            top_line[pid] = {"name": l.name, "min": _c(l.minutes, 0), "xg": _c(l.xg_pct, 1)}
    team_xg = dict(zip(d["team_metrics"].team, d["team_metrics"].xg_pct_5on5))
    season = int(d["raw"].season.iloc[0])
    prev = load_prev(season)
    data = {"updated": updated, "season": f"{season}–{str(season + 1)[-2:]}", "note": note,
            "prevLabel": f"{str(season - 1)[-2:]}–{str(season)[-2:]}" if prev else None,
            "skaters": _skaters(d["raw"], d["metrics"], d["rated"], deltas, top_line, team_xg, prev.get("skaters", {})),
            "goalies": _goalies(d["goalie_metrics"], d["goalie_ratings"], deltas, prev.get("goalies", {})),
            "lines": _lines(d["line_metrics"], names, prev.get("lines", {})), "teams": _teams(d["team_metrics"], prev.get("teams", {}))}
    # "</" far inte avsluta script-taggen, oavsett vad filen innehaller
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return (HERE / "site_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)


def write_site(fragment, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "index.html.tmp"
    tmp.write_text('<!doctype html>\n<html lang="sv"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   + fragment + "\n</html>\n", encoding="utf-8")
    tmp.replace(out / "index.html")
