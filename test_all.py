import pathlib, subprocess, sys
import numpy as np, pandas as pd, pytest
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import build_site, features, fetch_data, forecast, model, pipeline, run_nightly

COLS = """games_played icetime I_F_xGoals_with_earned_rebounds I_F_primaryAssists I_F_secondaryAssists I_F_points I_F_goals I_F_xGoals
I_F_shotsOnGoal I_F_shotAttempts OnIce_F_goals OnIce_A_goals OnIce_F_xGoals OnIce_A_xGoals OffIce_F_xGoals OffIce_A_xGoals I_F_oZoneShiftStarts I_F_dZoneShiftStarts
penaltiesDrawn penalties I_F_takeaways I_F_giveaways""".split()


def frame(n=40, minutes=300):
    rows = []
    for i in range(n):
        for s, share in {"all": 1.0, "5on5": .8, "5on4": .1, "4on5": .05, "other": .05}.items():
            r = dict.fromkeys(set(COLS) | {c for _, c in features.METRICS.values()}, 0.0)
            r.update(playerId=8000000 + i, season=2026, name=f"Spelare {i}", team="UMU", position="D" if i % 4 == 0 else "C",
                     situation=s, games_played=15, icetime=minutes * 60 * share, I_F_xGoals_with_earned_rebounds=(1 + i / 10) * share,
                     I_F_xGoals=(1 + i / 10) * share, OnIce_F_xGoals=10 * share, OnIce_A_xGoals=10 * share, OffIce_F_xGoals=10 * share, OffIce_A_xGoals=10 * share)
            rows.append(r)
    return pd.DataFrame(rows)


GOALIE_COLS = """xGoals goals unblocked_shot_attempts ongoal lowDangerShots mediumDangerShots highDangerShots lowDangerxGoals mediumDangerxGoals
highDangerxGoals lowDangerGoals mediumDangerGoals highDangerGoals""".split()
TEAM_COLS = "xGoalsFor xGoalsAgainst goalsFor goalsAgainst".split()
SHARE = {"all": 1.0, "5on5": .8, "5on4": .1, "4on5": .05, "other": .05}


def goalies(n=4, minutes=600):
    rows = []
    for i in range(n):
        for s, share in SHARE.items():
            r = dict.fromkeys(GOALIE_COLS, 0.0)
            r.update(playerId=8100000 + i, season=2026, name=f"Malvakt {i}", team="UMU", position="G", situation=s, games_played=10,
                     icetime=minutes * 60 * share, xGoals=(20 + i) * share, ongoal=200 * share)
            rows.append(r)
    return pd.DataFrame(rows)


def teams(n=32):
    rows = []
    for i in range(n):
        for s, share in SHARE.items():
            r = dict.fromkeys(TEAM_COLS, 0.0)
            r.update(team=f"T{i}", season=2026, name=f"T{i}", position="Team Level", situation=s, games_played=15, iceTime=3600 * 15 * share,
                     xGoalsFor=30 * share, xGoalsAgainst=30 * share)
            rows.append(r)
    return pd.DataFrame(rows)


def lines():
    return pd.DataFrame([dict(lineId="800000180000028000003", season=2026, name="A-B-C", team="UMU", position="line", situation="5on5",
                              games_played=9, icetime=3000.0, xGoalsFor=3.0, xGoalsAgainst=1.0, goalsFor=2.0, goalsAgainst=1.0)])


def folder(tmp, name="in", s=None, g=None, l=None, t=None):
    d = tmp / name
    d.mkdir(exist_ok=True)
    (frame(40) if s is None else s).to_csv(d / "skaters.csv", index=False)
    (goalies() if g is None else g).to_csv(d / "goalies.csv", index=False)
    (lines() if l is None else l).to_csv(d / "lines.csv", index=False)
    (teams() if t is None else t).to_csv(d / "teams.csv", index=False)
    return d


def test_load_removes_truncated_and_incomplete(tmp_path):
    df = frame(3)
    text = df.to_csv(index=False)
    p = tmp_path / "f.csv"
    p.write_text(text.rstrip()[: -len(text.rstrip().split("\n")[-1]) + 20])
    out, dropped = pipeline.load(p)
    assert out.playerId.nunique() == 2 and dropped == ["Spelare 2"]


def test_validate_finds_duplicates_and_bad_points():
    df = pd.concat([frame(2), frame(2).iloc[[0]]])
    assert any("dubbletter" in x for x in pipeline.validate(df))
    df = frame(2); df.loc[0, "I_F_points"] = 9
    assert any("poang" in x for x in pipeline.validate(df))


def test_validate_allows_small_gap_but_not_large():
    df = frame(30)
    df["I_F_shotAttempts"] = 100.0 * df.situation.map({"all": 1, "5on5": .8, "5on4": .1, "4on5": .05, "other": .05})
    df.loc[(df.playerId == 8000000) & (df.situation == "5on5"), "I_F_shotAttempts"] -= 5
    assert pipeline.validate(df) == []
    df.loc[(df.playerId == 8000001) & (df.situation == "5on5"), "I_F_shotAttempts"] -= 40
    assert any("for laga" in x for x in pipeline.validate(df))
    df.loc[(df.playerId == 8000002) & (df.situation == "5on5"), "I_F_shotAttempts"] += 5
    assert any("hogre" in x for x in pipeline.validate(df))


def test_metrics_per60():
    df = frame(1); df.loc[df.situation == "5on5", "I_F_primaryAssists"] = 4
    m = pipeline.metrics(df); r = m[m.situation == "5on5"].iloc[0]
    assert r.primary_a_60 == pytest.approx(4 / 240 * 60)


def scaled(n=40, minutes=800):
    """Spelare dar allt offensivt och lagets xG vaxer med numret."""
    df = frame(n, minutes)
    q = df.playerId - 8000000
    share = df.situation.map(SHARE)
    df["gameScore"] = (5 + q) * share
    df["I_F_xGoals_with_earned_rebounds"] = (2 + q) * share
    df["I_F_xGoals"] = (2 + q) * share
    df["I_F_shotsOnGoal"] = (20 + 3 * q) * share
    df["I_F_primaryAssists"] = (q // 2 + 1) * 1.0 * (df.situation.isin(["all", "5on5"]))
    df["I_F_points"] = df.I_F_primaryAssists
    df["OnIce_F_xGoals"] = (10 + q) * share
    return df


def test_forecast_ranks_better_players_higher():
    df = scaled()
    r = forecast.rate(df, pipeline.metrics(df))
    f = r[r.group == "F"].sort_values("playerId")
    assert f.rating.iloc[-5:].mean() > f.rating.iloc[:5].mean() + 40
    assert r.rating.notna().all() and set(r.pct_xga60.dropna().round(0)) != set()


def test_forecast_shrinks_small_samples():
    """Samma per-minut-varden ska ge ett mer extremt betyg med mycket istid an med lite."""
    big, small = scaled(minutes=1000), scaled(minutes=1000)
    five = small.playerId == 8000039
    for c in small.select_dtypes("number").columns.drop(["playerId", "season", "games_played"]):
        small.loc[five, c] = small.loc[five, c] * 0.12
    a = forecast.rate(big, pipeline.metrics(big)).set_index("playerId")
    b = forecast.rate(small, pipeline.metrics(small)).set_index("playerId")
    assert b.loc[8000039, "f_gs60"] < a.loc[8000039, "f_gs60"]


def test_too_few_minutes_gets_no_rating():
    df = frame(10, minutes=50)
    r = forecast.rate(df, pipeline.metrics(df))
    assert r.rating.isna().all() and set(r.confidence) == {"För lite data"}


def test_forecast_uses_real_history_and_fixes_names():
    """McDavid har historik i modellen. En kort sasong ska inte radera den."""
    df = frame(30); df.loc[df.playerId == 8000000, "playerId"] = 8478402
    df.loc[df.playerId == 8478402, "name"] = "Connor McDavid"; df.loc[df.playerId == 8478402, "position"] = "C"
    r = forecast.rate(df, pipeline.metrics(df)).set_index("playerId")
    assert r.loc[8478402, "effective_minutes"] > r.loc[8000001, "effective_minutes"] + 500
    assert r.loc[8478402, "f_gs60"] > r.loc[8000001, "f_gs60"]
    assert r.loc[8478402, "confidence"] == "Hög"
    raw = pd.DataFrame({"playerId": [8482116, 8000001], "name": ["Tim Sttzle", "Vem Som"]})
    assert list(forecast.fix_names(raw).name) == ["Tim Stutzle", "Vem Som"]


def args(tmp_path, input_dir, **kw):
    a = dict(input_dir=str(input_dir), season=None, date="2026-11-05", out=str(tmp_path / "site"), history=str(tmp_path / "h"),
             data=str(tmp_path / "d"), min_players=10, note=None, fragment=None)
    a.update(kw)
    return type("A", (), a)


def test_full_run_writes_site_history_and_delta(tmp_path):
    run_nightly.run(args(tmp_path, folder(tmp_path), date="2026-11-01"))
    df = frame(40); five = df.situation == "5on5"
    df.loc[five & (df.playerId == 8000001), "I_F_xGoals_with_earned_rebounds"] = 50.0
    run_nightly.run(args(tmp_path, folder(tmp_path, "b", s=df), date="2026-11-09"))
    html = (tmp_path / "site/index.html").read_text()
    assert html.startswith("<!doctype html>") and "__DATA__" not in html and "Malvakt 3" in html and "A-B-C" in html
    snap = pd.read_csv(tmp_path / "h/2026-11-09.csv")
    assert set(snap.kind) == {"skater", "goalie"}
    assert 8000001 in run_nightly.deltas(pd.DataFrame({"playerId": [8000001], "rating": [90.0]}), tmp_path / "h", "2026-11-09")


@pytest.mark.parametrize("bad", ["few", "drop", "goalie_dup", "few_teams", "bad_line", "season", "goals"])
def test_gate_keeps_last_site(tmp_path, bad):
    run_nightly.run(args(tmp_path, folder(tmp_path), date="2026-11-01"))
    before = (tmp_path / "site/index.html").read_text()
    kw = {}
    if bad == "few": kw["s"] = frame(5)
    if bad == "drop": kw["s"] = frame(30)
    if bad == "goalie_dup": kw["g"] = pd.concat([goalies(), goalies().iloc[[0]]])
    if bad == "few_teams": kw["t"] = teams(31)
    if bad == "bad_line":
        l = lines(); l["lineId"] = "123"; kw["l"] = l
    if bad == "season":
        kw["t"] = teams().assign(season=2025)
    if bad == "goals":
        s = frame(40); s.loc[s.situation == "all", "I_F_goals"] = 5.0; s.loc[s.situation == "all", "I_F_points"] = 5.0
        s.loc[s.situation == "5on5", "I_F_goals"] = 5.0; s.loc[s.situation == "5on5", "I_F_points"] = 5.0
        kw["s"] = s
    with pytest.raises(run_nightly.DataError):
        run_nightly.run(args(tmp_path, folder(tmp_path, "bad", **kw), date="2026-11-02"))
    assert (tmp_path / "site/index.html").read_text() == before
    assert not (tmp_path / "h/2026-11-02.csv").exists()


def test_html_in_data_cannot_close_script(tmp_path):
    df = frame(12); df["name"] = "</script><img src=x>"
    run_nightly.run(args(tmp_path, folder(tmp_path, s=df)))
    assert "</script><img" not in (tmp_path / "site/index.html").read_text()


def test_goalie_rating_order_and_minimum():
    g = goalies(6)
    r = model.rate_goalies(pipeline.goalie_metrics(g)).sort_values("playerId")
    assert r.rating.is_monotonic_increasing and r.iloc[-1].gsax == pytest.approx(25)
    r = model.rate_goalies(pipeline.goalie_metrics(goalies(4, minutes=100)))
    assert r.rating.isna().all()


def test_goalie_save_pct_and_line_ids():
    g = goalies(1); g.loc[g.situation == "all", "goals"] = 20.0
    m = pipeline.goalie_metrics(g); a = m[m.situation == "all"].iloc[0]
    assert a.save_pct == pytest.approx(90) and a.gsax == 0
    assert pipeline.line_players("800000180000028000003") == [8000001, 8000002, 8000003]


def test_line_and_team_metrics():
    m = pipeline.line_metrics(lines()).iloc[0]
    assert m.xg_pct == pytest.approx(75) and m.xgf_60 == pytest.approx(3.6)
    tm = pipeline.team_metrics(teams(32), pipeline.goalie_metrics(goalies()))
    assert len(tm) == 32 and tm.iloc[0].xg_pct_5on5 == pytest.approx(50)


def test_fetch_rejects_wrong_content(tmp_path, monkeypatch):
    import io
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: R(b"<html>nope"))
    with pytest.raises(ValueError):
        fetch_data.download(2026, "lines", tmp_path / "x.csv")
    assert not (tmp_path / "x.csv").exists()


def test_prev_season_only_used_for_right_season():
    assert build_site.load_prev(2026)["skaters"]["8478402"]["p"] == 138
    assert build_site.load_prev(2027) == {}


def test_season_rollover():
    import datetime as dt
    assert fetch_data.current_season(dt.date(2026, 11, 5)) == 2026
    assert fetch_data.current_season(dt.date(2027, 4, 1)) == 2026
