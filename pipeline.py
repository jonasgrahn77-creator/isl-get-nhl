#!/usr/bin/env python3
"""Laser, kontrollerar och raknar nyckeltal fran MoneyPucks spelarfil.
Data: MoneyPuck.com (fritt for icke-kommersiellt bruk, ange kallan)."""
import pandas as pd

KEY = ["playerId", "season", "situation"]
SITUATIONS = {"all", "5on5", "5on4", "4on5", "other"}
PARTS_MIN_GAP = 5  # storsta tillatna avvikelse for spelare med fa handelser


def load(path):
    """Las filen och kasta rader som ar avklippta (ofullstandiga)."""
    df = pd.read_csv(path)
    broken = df[df.isna().any(axis=1)]
    df = df.drop(broken.index).reset_index(drop=True)
    # En spelare som saknar nagon situation ar ocksa ofullstandig
    n = df.groupby(["playerId", "season"]).situation.transform("nunique")
    incomplete = df[n != len(SITUATIONS)]
    df = df[n == len(SITUATIONS)].reset_index(drop=True)
    return df, sorted(set(broken.name.dropna()) | set(incomplete.name.dropna()))


def validate(df):
    """Returnerar en lista med hittade datafel. Tom lista = godkand."""
    p = []
    if df.duplicated(KEY).any():
        return [f"{df.duplicated(KEY).sum()} dubbletter pa {KEY}"]
    if set(df.situation) - SITUATIONS:
        p.append(f"okanda situationer: {set(df.situation) - SITUATIONS}")
    n = df.groupby(["playerId", "season"]).situation.nunique()
    if (n != 5).any():
        p.append(f"{(n != 5).sum()} spelare saknar nagon situation: {n[n != 5].index.tolist()}")
    # 'all' ar facit for totaler. MoneyPucks uppdelning saknar ibland nagra handelser,
    # sa delarna far vara nagot lagre, men aldrig hogre och aldrig mycket lagre.
    parts = df[df.situation != "all"].groupby(["playerId", "season"]).sum(numeric_only=True)
    whole = df[df.situation == "all"].set_index(["playerId", "season"]).select_dtypes("number")
    for c in ["icetime", "I_F_goals", "I_F_points", "I_F_shotAttempts", "OnIce_F_goals", "OnIce_A_goals"]:
        gap = (whole[c] - parts[c]).dropna()
        if (gap < -1e-6).any():
            p.append(f"{c}: delarna ar hogre an all for {(gap < -1e-6).sum()} spelare")
        limit = (0.1 * whole[c]).clip(lower=PARTS_MIN_GAP).reindex(gap.index)
        if (gap > limit).any():
            p.append(f"{c}: delarna ar for laga for {(gap > limit).sum()} spelare")
        if gap.sum() > 0.02 * whole[c].sum():
            p.append(f"{c}: delarna saknar over 2 % av ligans total")
    bad = df[(df.I_F_goals + df.I_F_primaryAssists + df.I_F_secondaryAssists - df.I_F_points).abs() > 1e-6]
    if len(bad):
        p.append(f"{len(bad)} rader dar mal + assist != poang")
    if (df.icetime < 0).any():
        p.append("negativ istid")
    return p


def metrics(df):
    """Beraknar nyckeltal fran rakolumnerna, inte fran de avrundade procenten."""
    m = df[["playerId", "season", "name", "team", "position", "situation", "games_played"]].copy()
    m["minutes"] = df.icetime / 60
    per60 = 3600 / df.icetime.where(df.icetime > 0)
    m["created_xg_60"] = df.I_F_xGoals_with_earned_rebounds * per60
    m["primary_a_60"] = df.I_F_primaryAssists * per60
    m["points_60"] = df.I_F_points * per60
    m["goals_minus_xg"] = df.I_F_goals - df.I_F_xGoals
    m["xga_60"] = df.OnIce_A_xGoals * per60
    on = df.OnIce_F_xGoals / (df.OnIce_F_xGoals + df.OnIce_A_xGoals)
    off = df.OffIce_F_xGoals / (df.OffIce_F_xGoals + df.OffIce_A_xGoals)
    m["xg_pct"] = on * 100
    m["rel_xg_pct"] = (on - off) * 100
    m["ozone_start_pct"] = 100 * df.I_F_oZoneShiftStarts / (df.I_F_oZoneShiftStarts + df.I_F_dZoneShiftStarts)
    m["penalty_net"] = df.penaltiesDrawn - df.penalties
    m["puck_net"] = df.I_F_takeaways - df.I_F_giveaways
    return m


# --- Malvakter, kedjor och lag ---

def _parts_check(df, keys, columns):
    """Delarna far vara nagot lagre an 'all', aldrig hogre. Samma regel som for utespelare."""
    p = []
    parts = df[df.situation != "all"].groupby(keys).sum(numeric_only=True)
    whole = df[df.situation == "all"].set_index(keys).select_dtypes("number")
    for c in columns:
        gap = (whole[c] - parts[c]).dropna()
        if (gap < -1e-6).any():
            p.append(f"{c}: delarna ar hogre an all for {(gap < -1e-6).sum()} rader")
        if gap.sum() > 0.02 * whole[c].sum():
            p.append(f"{c}: delarna saknar over 2 % av totalen")
    return p


def load_goalies(path):
    return pd.read_csv(path)


def load_teams(path):
    # Filen har kolumnen team tva ganger
    return pd.read_csv(path).drop(columns=["team.1"], errors="ignore")


def load_lines(path):
    return pd.read_csv(path, dtype={"lineId": str})


def validate_goalies(g):
    key = ["playerId", "season", "situation"]
    if g.duplicated(key).any():
        return [f"malvakter: {g.duplicated(key).sum()} dubbletter"]
    p = []
    n = g.groupby("playerId").situation.nunique()
    if (n != len(SITUATIONS)).any():
        p.append(f"malvakter: {(n != len(SITUATIONS)).sum()} saknar nagon situation")
    if (g.goals > g.ongoal).any():
        p.append("malvakter: fler mal an skott pa mal")
    return p + [f"malvakter: {x}" for x in _parts_check(g, ["playerId"], ["icetime", "goals", "ongoal"])]


def validate_teams(t):
    key = ["team", "season", "situation"]
    if t.duplicated(key).any():
        return [f"lag: {t.duplicated(key).sum()} dubbletter"]
    p = []
    if t.team.nunique() < 32:
        p.append(f"lag: bara {t.team.nunique()} lag")
    n = t.groupby("team").situation.nunique()
    if (n != len(SITUATIONS)).any():
        p.append("lag: nagon situation saknas")
    return p + [f"lag: {x}" for x in _parts_check(t, ["team"], ["iceTime", "goalsFor", "goalsAgainst"])]


def line_players(line_id):
    """lineId ar spelarnas id, sju siffror vardera, skrivna efter varandra."""
    return [int(line_id[i:i + 7]) for i in range(0, len(line_id), 7)]


def validate_lines(l):
    key = ["lineId", "season", "situation"]
    if l.duplicated(key).any():
        return [f"kedjor: {l.duplicated(key).sum()} dubbletter"]
    p = []
    if l.empty:
        return ["kedjor: filen ar tom"]
    size = l.position.map({"line": 21, "pairing": 14})
    if size.isna().any() or (l.lineId.str.len() != size).any():
        p.append("kedjor: lineId har fel langd")
    if (l.icetime < 0).any():
        p.append("kedjor: negativ istid")
    return p


def cross_checks(skaters, goalies, lines, teams):
    """Filerna ska beskriva samma sasong och samma matcher."""
    p = []
    if len({int(d.season.iloc[0]) for d in (skaters, goalies, lines, teams)}) != 1:
        p.append("filerna galler olika sasonger")
    gs, gt = skaters[skaters.situation == "all"].I_F_goals.sum(), teams[teams.situation == "all"].goalsFor.sum()
    if abs(gs - gt) > max(5, 0.01 * gt):
        p.append(f"spelarnas mal ({gs:.0f}) stammer inte med lagens ({gt:.0f})")
    ga, tga = goalies[goalies.situation == "all"].goals.sum(), teams[teams.situation == "all"].goalsAgainst.sum()
    if ga > tga + 1e-6 or (tga and ga < 0.8 * tga):
        p.append(f"malvakternas insläppta ({ga:.0f}) stammer inte med lagens ({tga:.0f})")
    known = set(skaters.playerId)
    ids = [i for line in lines.lineId for i in line_players(line)]
    if ids and sum(i in known for i in ids) < 0.95 * len(ids):
        p.append("kedjorna innehaller spelare som saknas i spelarfilen")
    return p


def goalie_metrics(g):
    m = g[["playerId", "season", "name", "team", "situation", "games_played"]].copy()
    m["minutes"] = g.icetime / 60
    m["goals"], m["xgoals"], m["shots"] = g.goals, g.xGoals, g.ongoal
    m["gsax"] = g.xGoals - g.goals
    m["gsax_60"] = m.gsax * 3600 / g.icetime.where(g.icetime > 0)
    m["save_pct"] = 100 * (1 - g.goals / g.ongoal.where(g.ongoal > 0))
    for d, name in {"low": "lowDanger", "medium": "mediumDanger", "high": "highDanger"}.items():
        m[f"{d}_shots"], m[f"{d}_goals"] = g[f"{name}Shots"], g[f"{name}Goals"]
        m[f"{d}_gsax"] = g[f"{name}xGoals"] - g[f"{name}Goals"]
    return m


def line_metrics(l):
    m = l[["lineId", "season", "name", "team", "position", "games_played"]].copy()
    m["minutes"] = l.icetime / 60
    per60 = 3600 / l.icetime.where(l.icetime > 0)
    m["xg_pct"] = 100 * l.xGoalsFor / (l.xGoalsFor + l.xGoalsAgainst).where(lambda s: s > 0)
    m["xgf_60"], m["xga_60"] = l.xGoalsFor * per60, l.xGoalsAgainst * per60
    m["gf"], m["ga"] = l.goalsFor, l.goalsAgainst
    return m


def team_metrics(t, goalie_m):
    rows = []
    sit = {s: d.set_index("team") for s, d in t.groupby("situation")}
    gsax = goalie_m[goalie_m.situation == "all"].groupby("team").gsax.sum()
    for team, a in sit["all"].iterrows():
        e, pp, pk = sit["5on5"].loc[team], sit["5on4"].loc[team], sit["4on5"].loc[team]
        rows.append({
            "team": team, "season": a.season, "games_played": a.games_played, "gf": a.goalsFor, "ga": a.goalsAgainst,
            "xg_pct_5on5": 100 * e.xGoalsFor / (e.xGoalsFor + e.xGoalsAgainst),
            "xgf_60_5on5": e.xGoalsFor * 3600 / e.iceTime, "xga_60_5on5": e.xGoalsAgainst * 3600 / e.iceTime,
            "pp_xgf_60": pp.xGoalsFor * 3600 / pp.iceTime, "pp_goals": pp.goalsFor,
            "pk_xga_60": pk.xGoalsAgainst * 3600 / pk.iceTime, "pk_goals_against": pk.goalsAgainst,
            "finishing": a.goalsFor - a.xGoalsFor, "goalie_gsax": gsax.get(team, 0.0)})
    return pd.DataFrame(rows)
