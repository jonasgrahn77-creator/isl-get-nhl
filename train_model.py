#!/usr/bin/env python3
"""Tranar spelarmodellen. Anvandning: python training/train_model.py historik1.csv [historik2.csv ...]
Testsasongerna 2022-2025 anvands inte till nagra modellval. For varje testsasong tranas modellen bara pa aren fore."""
import gzip, json, pathlib, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import features as F
warnings.filterwarnings("ignore")
TARGETS = ["gs60", "p60", "cxg60", "relxg", "xga60"]
DEV, TEST = [2016, 2017, 2018, 2019, 2020, 2021], [y for y in [2022, 2023, 2024, 2025] if True]
MIN_TARGET = 100 * 60

raw = pd.concat([pd.read_csv(p) for p in sys.argv[1:]]).drop_duplicates(["playerId", "season", "situation"])
w0 = F.wide(raw); w0 = w0[w0.toi_all > 0]
w, ref = F.normalize(w0)
D = F.Data(w); n = len(D.players)
_c = {}
def rows(y, skip=()):
    if (y, skip) not in _c:
        st = D.stack(y, skip=skip)
        X = F.design(st, D.isD, D.first, y)
        _c[(y, skip)] = (X.values, F.targets(D.counts(y)), D.counts(y)["toi"], np.array(F.effective_minutes(st)) * np.ones(n), list(X.columns))
    return _c[(y, skip)]
def train_set(years):
    A = [rows(y) for y in years]; K = [(a[2] >= MIN_TARGET) & (a[3] > 0) for a in A]
    return (np.vstack([a[0][k] for a, k in zip(A, K)]), {t: np.concatenate([a[1][t][k] for a, k in zip(A, K)]) for t in TARGETS},
            np.concatenate([np.minimum(a[2][k] / 60, 1000) for a, k in zip(A, K)]), np.concatenate([D.isD[k] for k in K]))
def fit(years, alpha):
    Xa, Ya, Wa, Da = train_set(years); mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    out = {}
    for flag in (True, False):
        i = Da == flag
        out[flag] = {t: Ridge(alpha=alpha).fit((Xa[i] - mu) / sd, Ya[t][i], sample_weight=Wa[i]) for t in TARGETS}
    return out, mu, sd
def predict(m, mu, sd, X):
    P = {t: np.zeros(len(X)) for t in TARGETS}
    for flag in (True, False):
        i = D.isD == flag
        for t in TARGETS:
            P[t][i] = m[flag][t].predict((X[i] - mu) / sd)
    return P
def wr2(y, p, g):
    return float(1 - np.average((y - p) ** 2, weights=g) / np.average((y - np.average(y, weights=g)) ** 2, weights=g))
def hand(c):
    T = F.targets(c); toi = c["toi"] / 60; pa = c["a1"] / np.maximum(c["toi"], 1) * 3600
    out = np.full(n, np.nan)
    for flag in (True, False):
        g = (D.isD == flag) & (toi >= 100)
        if g.sum() < 2: continue
        s = 0
        for v, sign, wt in [(T["cxg60"], 1, .30), (pa, 1, .15), (T["xga60"], -1, .25), (T["relxg"], 1, .30)]:
            pr = np.average(v[g], weights=toi[g]); sh = (v * toi + pr * 300) / (toi + 300) * sign
            s = s + wt * (sh - sh[g].mean()) / (sh[g].std() + 1e-9)
        out[g] = s[g]
    return out
def team_table(year, P, hd=None):
    cur = w0[(w0.season == year) & (w0.toi > 0)].set_index("playerId"); idx = pd.Index(D.players)
    rows_ = []
    for team, r in cur.groupby("team"):
        pos = idx.get_indexer(r.index); row = {"team": team, "year": year, "xgd60": (r.oxgf.sum() - r.oxga.sum()) / r.toi.sum() * 3600, "gd60": (r.ogf.sum() - r.oga.sum()) / r.toi.sum() * 3600}
        for t, v in P.items(): row[t] = np.average(v[pos], weights=r.toi.values)
        if hd is not None: row["hand"] = np.average(np.nan_to_num(hd[pos]), weights=r.toi.values)
        rows_.append(row)
    return rows_

# 1) val pa utvecklingssasonger: alpha och betygets vikter
print("Val av regularisering", flush=True)
best, best_r2 = None, -9
for alpha in (5000, 15000, 50000, 150000):
    s = []
    for y in DEV:
        m, mu, sd = fit(range(2011, y), alpha); P = predict(m, mu, sd, rows(y)[0]); _, T, ft, h, _ = rows(y); k = (ft >= MIN_TARGET) & (h > 0)
        for flag in (True, False):
            ib = k & (D.isD == flag)
            s += [wr2(T[t][ib], P[t][ib], np.minimum(ft[ib] / 60, 1000)) for t in TARGETS]
    print(alpha, round(np.mean(s), 4), flush=True)
    if np.mean(s) > best_r2: best, best_r2 = alpha, float(np.mean(s))
ALPHA = best
tt = []
for y in DEV:
    m, mu, sd = fit(range(2011, y), ALPHA); tt += team_table(y, predict(m, mu, sd, rows(y)[0]))
tt = pd.DataFrame(tt)
L = LinearRegression().fit(tt[["gs60", "relxg"]], tt.xgd60)
W = {"gs60": float(L.coef_[0]), "relxg": float(L.coef_[1])}
print("Alpha", ALPHA, "vikter", W, "R2 lag dev", round(L.score(tt[["gs60", "relxg"]], tt.xgd60), 3), flush=True)

# 2) slutprov, en gang
print("Slutprov", TEST, flush=True)
TEST = [y for y in TEST if y in D.seasons]
rec, teamrows, buckets = [], [], []
for y in TEST:
    m, mu, sd = fit(range(2011, y), ALPHA); X, T, ft, h, _ = rows(y); P = predict(m, mu, sd, X); k = (ft >= MIN_TARGET) & (h > 0)
    prev = F.targets(D.counts(y - 1)); hd = hand(D.counts(y - 1)); g = np.minimum(ft / 60, 1000)
    T = dict(T); T["value"] = W["gs60"] * T["gs60"] + W["relxg"] * T["relxg"]
    P = dict(P); P["value"] = W["gs60"] * P["gs60"] + W["relxg"] * P["relxg"]
    prev = dict(prev); prev["value"] = W["gs60"] * prev["gs60"] + W["relxg"] * prev["relxg"]
    # samma spelare utan fjolarets data
    Xg, _, _, hg, _ = rows(y, (y - 1,)); Pg = predict(m, mu, sd, Xg); Pg["value"] = W["gs60"] * Pg["gs60"] + W["relxg"] * Pg["relxg"]
    for flag, name in [(True, "D"), (False, "F")]:
        ib = k & (D.isD == flag)
        played = ib & (D.counts(y - 1)["toi"] >= MIN_TARGET)
        both = played & (hg > 0) & ~np.isnan(hd)
        for t in TARGETS + ["value"]:
            sign = -1 if t == "xga60" else 1
            rec.append(dict(year=y, group=name, target=t, n=int(ib.sum()),
                            model_r2=wr2(T[t][ib], P[t][ib], g[ib]), model_rho=spearmanr(T[t][ib], P[t][ib])[0],
                            last_r2=wr2(T[t][played], prev[t][played], g[played]), last_rho=spearmanr(T[t][played], prev[t][played])[0],
                            hand_rho=spearmanr(T[t][both], sign * hd[both])[0], model_rho_same=spearmanr(T[t][both], P[t][both])[0],
                            gap_r2=wr2(T[t][both], Pg[t][both], g[both]), gap_rho=spearmanr(T[t][both], Pg[t][both])[0], model_r2_same=wr2(T[t][both], P[t][both], g[both])))
        # traffsakerhet for toppen och per datamangd, for det sammanvagda malet
        top_true = T["value"][ib] >= np.quantile(T["value"][ib], .8); top_pred = P["value"][ib] >= np.quantile(P["value"][ib], .8)
        buckets.append(dict(year=y, group=name, b="top20_hit", v=float((top_true & top_pred).sum() / top_true.sum())))
        for lo, hi, lab in [(0, 300, "<300"), (300, 1000, "300-1000"), (1000, 1e9, ">=1000")]:
            s = ib & (h >= lo) & (h < hi)
            if s.sum() > 15: buckets.append(dict(year=y, group=name, b=lab, v=float(spearmanr(T["value"][s], P["value"][s])[0]), n=int(s.sum())))
    teamrows += team_table(y, {t: P[t] for t in TARGETS}, hd)
rec = pd.DataFrame(rec); teamT = pd.DataFrame(teamrows); B = pd.DataFrame(buckets)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
sumr = rec.groupby(["target", "group"]).mean(numeric_only=True).drop(columns=["year", "n"]).round(3)
print(sumr.to_string())
print(rec.groupby("target").mean(numeric_only=True).drop(columns=["year", "n"]).round(3).to_string())
top = B[B.b == "top20_hit"].groupby("year").v.mean().round(3).to_dict(); print("Topp 20 per ar", top)
print("Per ar value", rec[rec.target == "value"].groupby("year")[["model_r2", "model_rho", "hand_rho", "model_rho_same"]].mean().round(3).to_string())
teamT["value"] = W["gs60"] * teamT.gs60 + W["relxg"] * teamT.relxg
tc = {c: {o: float(teamT[o].corr(teamT[c])) for o in ["xgd60", "gd60"]} for c in ["value", "gs60", "relxg", "hand"]}
tc_year = {int(y): {c: {o: round(float(g[o].corr(g[c])), 3) for o in ["xgd60", "gd60"]} for c in ["value", "hand"]} for y, g in teamT.groupby("year")}
print("Lag per ar", json.dumps(tc_year), flush=True)
print("Lag", json.dumps(tc), flush=True)
print(B.groupby("b").v.mean().round(3).to_dict(), B.groupby("b").n.sum().to_dict() if "n" in B else "")

# 3) slutlig modell pa all tillganglig data
years = [y for y in range(2011, max(D.seasons) + 1)]
m, mu, sd = fit(years, ALPHA)
cols = rows(years[0])[4]
out = ROOT / "model"; out.mkdir(exist_ok=True)
model = {"trained_years": [years[0], years[-1]], "alpha": ALPHA, "columns": cols, "mu": mu.tolist(), "sd": sd.tolist(), "ref": ref, "weights": W,
         "coef": {("D" if flag else "F"): {t: {"intercept": float(m[flag][t].intercept_), "coef": m[flag][t].coef_.tolist()} for t in TARGETS} for flag in (True, False)}}
(out / "player_model.json").write_text(json.dumps(model))
# de senaste sasongerna behovs vid prognos, plus forsta sasong och namn for alla
keep = max(D.seasons) - 4
hist = w0[w0.season >= keep].drop(columns=["team", "games_played"])
hist.round(2).to_csv(out / "history.csv.gz", index=False, compression="gzip")
first = w0.groupby("playerId").season.min()
names = w0.groupby("playerId").name.agg(lambda s: max(s, key=len))
pd.DataFrame({"first": first, "name": names}).to_csv(out / "players.csv.gz", compression="gzip")
report = {"top20_by_year": {int(k): v for k, v in top.items()}, "alpha": ALPHA, "weights": W, "summary_by_target": rec.groupby("target").mean(numeric_only=True).drop(columns=["year", "n"]).round(3).to_dict("index"), "team": tc, "team_by_year": tc_year, "value_by_year": rec[rec.target == "value"].groupby("year")[["model_r2", "model_rho", "hand_rho", "model_rho_same"]].mean().round(3).to_dict("index"), "buckets": B.groupby("b").v.mean().round(3).to_dict()}
(out / "report.json").write_text(json.dumps(report, indent=1))
print("Sparat", [p.name for p in out.iterdir()], "seasons kept from", keep, flush=True)
