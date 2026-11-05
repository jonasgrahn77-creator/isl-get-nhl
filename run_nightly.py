#!/usr/bin/env python3
"""Nattkorningen: hamta -> kontrollera -> modell -> historik -> hemsida.
Stoppar med felkod utan att skriva nagot om datan inte ar godkand, sa senaste sidan ligger kvar."""
import argparse
import datetime as dt
import pathlib
import sys
import tempfile
import duckdb
import pandas as pd
import build_site
import fetch_data
import forecast
import model
import pipeline


class DataError(Exception):
    pass


def snapshots(history):
    return sorted(pathlib.Path(history).glob("*.csv"))


def gate(skaters, dropped, goalies, lines, teams, history, today, min_players):
    """Stoppar korningen om nagon fil ser trasig ut eller inte stammer med de andra."""
    problems = (pipeline.validate(skaters) + pipeline.validate_goalies(goalies) + pipeline.validate_lines(lines)
                + pipeline.validate_teams(teams))
    if problems:
        raise DataError("; ".join(problems))
    problems = pipeline.cross_checks(skaters, goalies, lines, teams)
    if problems:
        raise DataError("; ".join(problems))
    n = skaters.playerId.nunique()
    if n < min_players:
        raise DataError(f"bara {n} spelare, kraver minst {min_players}")
    if len(dropped) > 0.02 * (n + len(dropped)):
        raise DataError(f"{len(dropped)} ofullstandiga spelare")
    old = [p for p in snapshots(history) if p.stem < today]
    if old:
        prev = pd.read_csv(old[-1])
        same = prev[(prev.season == skaters.season.iloc[0]) & (prev.kind == "skater")]
        if len(same) and n < 0.9 * len(same):
            raise DataError(f"antalet spelare foll fran {len(same)} till {n}")


def deltas(rated, history, today):
    """Betygsforandring mot senaste historikfil som ar minst 7 dagar gammal."""
    cutoff = (dt.date.fromisoformat(today) - dt.timedelta(days=7)).isoformat()
    old = [p for p in snapshots(history) if p.stem <= cutoff]
    if not old:
        return {}
    prev = pd.read_csv(old[-1]).set_index("playerId").rating
    out = {}
    for r in rated.itertuples():
        before = prev.get(r.playerId)
        if before is not None and pd.notna(before) and pd.notna(r.rating):
            out[int(r.playerId)] = int(round(r.rating - before))
    return out


def run(a):
    today = a.date or dt.datetime.now(dt.timezone.utc).date().isoformat()
    season = a.season or fetch_data.current_season(dt.date.fromisoformat(today))
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(a.input_dir) if a.input_dir else fetch_data.download_all(season, tmp)
        raw, dropped = pipeline.load(src / "skaters.csv")
        goalies = pipeline.load_goalies(src / "goalies.csv")
        lines = pipeline.load_lines(src / "lines.csv")
        teams = pipeline.load_teams(src / "teams.csv")
    if raw.empty or goalies.empty or teams.empty:
        raise DataError("minst en fil saknar data")
    gate(raw, dropped, goalies, lines, teams, a.history, today, a.min_players)

    raw = forecast.fix_names(raw)
    metrics = pipeline.metrics(raw)
    rated = forecast.rate(raw, metrics)
    gm = pipeline.goalie_metrics(goalies)
    gr = model.rate_goalies(gm)
    lm = pipeline.line_metrics(lines)
    tm = pipeline.team_metrics(teams, gm)
    snap = pd.concat([rated.assign(kind="skater"), gr.assign(kind="goalie")])[
        ["playerId", "season", "kind", "games_played", "minutes", "rating"]].round(2)
    d = deltas(snap, a.history, today)
    fragment = build_site.render({"raw": raw, "metrics": metrics, "rated": rated, "goalie_metrics": gm, "goalie_ratings": gr,
                                  "line_metrics": lm, "team_metrics": tm}, today, d, a.note)

    # Allt skrivs forst nar datan ar godkand
    build_site.write_site(fragment, a.out)
    if a.fragment:
        pathlib.Path(a.fragment).write_text(fragment, encoding="utf-8")
    hist = pathlib.Path(a.history)
    hist.mkdir(parents=True, exist_ok=True)
    snap.to_csv(hist / f"{today}.csv", index=False)
    data = pathlib.Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(data / "nhl.duckdb"))
    for name, frame in {"skaters": raw, "skater_metrics": metrics, "skater_ratings": rated, "goalies": goalies,
                        "goalie_metrics": gm, "goalie_ratings": gr, "lines": lines, "line_metrics": lm,
                        "teams": teams, "team_metrics": tm}.items():
        con.register("frame", frame)
        con.execute(f"create or replace table {name} as select * from frame")
        con.unregister("frame")
    con.close()
    print(f"Klart: {raw.playerId.nunique()} utespelare ({rated.rating.notna().sum()} med betyg), "
          f"{goalies.playerId.nunique()} malvakter ({gr.rating.notna().sum()} med betyg), "
          f"{len(lines)} kedjor och backpar, {teams.team.nunique()} lag. Borttagna: {dropped}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", help="mapp med skaters.csv, goalies.csv, lines.csv och teams.csv")
    p.add_argument("--season", type=int)
    p.add_argument("--date", help="YYYY-MM-DD, standard ar dagens datum")
    p.add_argument("--out", default="site")
    p.add_argument("--history", default="history")
    p.add_argument("--data", default="data")
    p.add_argument("--min-players", type=int, default=100)
    p.add_argument("--note", help="meddelande overst pa sidan")
    p.add_argument("--fragment", help="spara ocksa sidan utan html-skal")
    a = p.parse_args()
    try:
        run(a)
    except (DataError, OSError, ValueError, KeyError, pd.errors.ParserError) as e:
        print(f"STOPP, sidan uppdateras inte: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
