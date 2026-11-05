"""Betyget for utespelare: prognos fran modellen som tranats pa historiken (se training/train_model.py).
Pagaende sasong och tidigare sasonger vags ihop, och fa minuter dras mot positionens snitt."""
import json
import pathlib
import numpy as np
import pandas as pd
import features as F

MODEL_DIR = pathlib.Path(__file__).parent / "model"
MIN_EFFECTIVE = 100  # minsta viktade minuter 5 mot 5 for betyg
# Grans for sakerhet, fran slutprovet: rangkorrelationen vaxer tydligt med mangden data
CONF_LOW, CONF_HIGH = 300, 1000
TARGETS = ["gs60", "p60", "cxg60", "relxg", "xga60"]


def confidence(minutes):
    if minutes < MIN_EFFECTIVE:
        return "För lite data"
    if minutes < CONF_LOW:
        return "Låg"
    return "Medel" if minutes < CONF_HIGH else "Hög"


def fix_names(raw, model_dir=MODEL_DIR):
    """MoneyPucks filer tappar ibland tecken. Finns en langre stavning av samma id i historiken anvands den."""
    known = pd.read_csv(model_dir / "players.csv.gz", index_col="playerId").name

    def is_sub(short, long):
        it = iter(long)
        return all(ch in it for ch in short)

    def better(pid, name):
        other = known.get(pid)
        return other if isinstance(other, str) and len(other) > len(name) and is_sub(name, other) else name

    fixed = {pid: better(pid, name) for pid, name in zip(raw.playerId, raw.name)}
    return raw.assign(name=raw.playerId.map(fixed))


def rate(raw, metrics, model_dir=MODEL_DIR):
    """Returnerar en rad per spelare med dagens nyckeltal, prognoser, betyg och percentiler."""
    model = json.loads((model_dir / "player_model.json").read_text())
    season = int(raw.season.iloc[0])
    hist = pd.read_csv(model_dir / "history.csv.gz")
    hist = hist[hist.season < season]
    cur = F.wide(raw)
    allw = pd.concat([hist, cur.drop(columns=["team", "games_played"])], ignore_index=True)
    allw = allw[allw.toi_all > 0]
    norm, _ = F.normalize(allw, model["ref"])
    D = F.Data(norm)
    first = pd.read_csv(model_dir / "players.csv.gz", index_col="playerId")["first"]
    D.first = np.array([min(first.get(p, s), s) for p, s in zip(D.players, D.first)])

    current = D.counts(season)
    st = D.stack(season, current)
    X = F.design(st, D.isD, D.first, season)
    if list(X.columns) != model["columns"]:
        raise ValueError("Variablerna stammer inte med den tranade modellen")
    Z = (X.values - np.array(model["mu"])) / np.array(model["sd"])
    pred = {t: np.zeros(len(Z)) for t in TARGETS}
    for tag, flag in (("D", True), ("F", False)):
        i = D.isD == flag
        for t in TARGETS:
            c = model["coef"][tag][t]
            pred[t][i] = c["intercept"] + Z[i] @ np.array(c["coef"])
    effective = np.asarray(F.effective_minutes(st))
    out = pd.DataFrame({f"f_{t}": pred[t] for t in TARGETS}, index=D.players)
    out["effective_minutes"] = effective
    out["value"] = model["weights"]["gs60"] * pred["gs60"] + model["weights"]["relxg"] * pred["relxg"]

    x = metrics[metrics.situation == "5on5"].merge(out, left_on="playerId", right_index=True, how="left")
    x["group"] = np.where(x.position == "D", "D", "F")
    x["confidence"] = x.effective_minutes.map(confidence)
    x["rating"] = np.nan
    pcts = {"f_cxg60": 1, "f_p60": 1, "f_xga60": -1, "f_relxg": 1}
    for c in pcts:
        x["pct_" + c[2:]] = np.nan
    for _, idx in x.groupby("group").groups.items():
        g = x.loc[idx]
        q = g[g.effective_minutes >= MIN_EFFECTIVE]
        if len(q) < 2:
            continue
        x.loc[q.index, "rating"] = (q.value.rank() - 0.5) / len(q) * 100
        for c, sign in pcts.items():
            x.loc[q.index, "pct_" + c[2:]] = ((sign * q[c]).rank() - 0.5) / len(q) * 100
    return x
