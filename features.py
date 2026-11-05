"""Gemensam kod for traning och nattkorning: gor sasonger jamforbara och bygger prognosvariabler."""
import numpy as np
import pandas as pd

# namn: (situation, kolumn)
METRICS = {
    "toi": ("5on5", "icetime"), "g": ("5on5", "I_F_goals"), "a1": ("5on5", "I_F_primaryAssists"), "a2": ("5on5", "I_F_secondaryAssists"),
    "xg": ("5on5", "I_F_xGoals"), "cxg": ("5on5", "I_F_xGoals_with_earned_rebounds"), "sog": ("5on5", "I_F_shotsOnGoal"),
    "sa": ("5on5", "I_F_shotAttempts"), "hd": ("5on5", "I_F_highDangerShots"), "uba": ("5on5", "I_F_unblockedShotAttempts"),
    "oxgf": ("5on5", "OnIce_F_xGoals"), "oxga": ("5on5", "OnIce_A_xGoals"), "ogf": ("5on5", "OnIce_F_goals"), "oga": ("5on5", "OnIce_A_goals"),
    "ocf": ("5on5", "OnIce_F_shotAttempts"), "oca": ("5on5", "OnIce_A_shotAttempts"), "ouf": ("5on5", "OnIce_F_unblockedShotAttempts"),
    "oua": ("5on5", "OnIce_A_unblockedShotAttempts"), "fxgf": ("5on5", "OffIce_F_xGoals"), "fxga": ("5on5", "OffIce_A_xGoals"),
    "fcf": ("5on5", "OffIce_F_shotAttempts"), "fca": ("5on5", "OffIce_A_shotAttempts"), "ozs": ("5on5", "I_F_oZoneShiftStarts"),
    "dzs": ("5on5", "I_F_dZoneShiftStarts"), "hits": ("5on5", "I_F_hits"), "tk": ("5on5", "I_F_takeaways"), "gv": ("5on5", "I_F_giveaways"),
    "blk": ("5on5", "shotsBlockedByPlayer"), "pen": ("5on5", "penalties"), "pend": ("5on5", "penaltiesDrawn"),
    "fow": ("5on5", "faceoffsWon"), "fol": ("5on5", "faceoffsLost"),
    "toi_all": ("all", "icetime"), "gs": ("all", "gameScore"), "p_all": ("all", "I_F_points"), "g_all": ("all", "I_F_goals"),
    "toi_pp": ("5on4", "icetime"), "p_pp": ("5on4", "I_F_points"), "xg_pp": ("5on4", "I_F_xGoals"), "oxgf_pp": ("5on4", "OnIce_F_xGoals"),
    "toi_pk": ("4on5", "icetime"), "oxga_pk": ("4on5", "OnIce_A_xGoals"),
}
TOI = ["toi", "toi_all", "toi_pp", "toi_pk"]
COUNT = [m for m in METRICS if m not in TOI]
BASE = {"p_all": "toi_all", "g_all": "toi_all", "gs": "toi_all", "p_pp": "toi_pp", "xg_pp": "toi_pp", "oxgf_pp": "toi_pp", "oxga_pk": "toi_pk"}
# Hur snabbt ett matt stabiliseras: minuter av positionssnitt som laggs till innan rate beraknas
KS = {"default": 300.0, "g": 700, "a1": 400, "a2": 700, "xg": 150, "cxg": 150, "sog": 100, "sa": 100, "uba": 100, "hd": 150, "oxgf": 400, "oxga": 400,
      "ogf": 700, "oga": 700, "ocf": 300, "oca": 300, "ouf": 300, "oua": 300, "fxgf": 300, "fxga": 300, "fcf": 300, "fca": 300, "gs": 300, "p_all": 400,
      "g_all": 700, "p_pp": 150, "xg_pp": 100, "oxgf_pp": 150, "oxga_pk": 150}
# Vid simulerad gallring foljer icke-heltal ett raknevarde
THIN_WITH = {"xg": "uba", "cxg": "uba", "oxgf": "ouf", "oxga": "oua", "fxgf": "fcf", "fxga": "fca", "xg_pp": "p_pp", "oxgf_pp": "p_pp", "oxga_pk": "toi_pk"}


def base(m):
    return BASE.get(m, "toi")


def wide(df):
    """En rad per spelare och sasong med alla matt i platt form."""
    out = {name: df[df.situation == sit].set_index(["playerId", "season"])[col].astype(float) for name, (sit, col) in METRICS.items()}
    w = pd.DataFrame(out).fillna(0.0)
    meta = df[df.situation == "all"].set_index(["playerId", "season"])[["name", "team", "position", "games_played"]]
    return w.join(meta).reset_index()


def season_factors(w, ref=None):
    """Faktor per sasong och matt sa att alla sasonger blir jamforbara med ref-nivan."""
    a = w.groupby("season")[COUNT + TOI].sum()
    if ref is None:
        ref = {m: float(a[m].sum() / a[base(m)].sum()) for m in COUNT}
    fac = {m: (ref[m] / (a[m] / a[base(m)])).replace([np.inf, -np.inf], 1.0).fillna(1.0) for m in COUNT}
    return ref, fac


def normalize(w, ref=None):
    ref, fac = season_factors(w, ref)
    w = w.copy()
    for m in COUNT:
        w[m] = w[m] * w.season.map(fac[m])
    return w, ref


class Data:
    """Matriser spelare x sasong. first_seen kan komma fran tidigare historik."""

    def __init__(self, w):
        self.piv = {m: w.pivot(index="playerId", columns="season", values=m).fillna(0.0) for m in COUNT + TOI}
        self.players = self.piv["toi"].index
        self.seasons = list(self.piv["toi"].columns)
        info = w.sort_values("season").groupby("playerId").agg(position=("position", "last"), first=("season", "min"))
        self.isD = (info.loc[self.players, "position"].values == "D")
        self.first = info.loc[self.players, "first"].values

    def counts(self, year):
        return {m: self.piv[m][year].values for m in COUNT + TOI}

    def stack(self, year, frag=None, skip=(), depth=4):
        """Lista med (ar sedan, varden). frag ar den pagaende sasongens forsta del."""
        st = [] if frag is None else [(0, frag)]
        for k in range(1, depth + 1):
            if year - k in self.seasons and year - k not in skip:
                st.append((k, self.counts(year - k)))
        return st

    def thin(self, year, f, seed):
        """Delar sasongen slumpvis i en borjan (andel f av istiden) och resten."""
        g = np.random.default_rng(seed)
        tot = self.counts(year)
        frag, rest = {}, {}
        for m in TOI:
            frag[m] = tot[m] * f
        for m in COUNT:
            if m in THIN_WITH or m == "gs":
                continue
            n = np.round(tot[m]).astype(int).clip(min=0)
            frag[m] = np.where(n > 0, tot[m] * g.binomial(n, f) / np.maximum(n, 1), 0.0)
        for m, by in THIN_WITH.items():
            ratio = np.where(tot[by] > 0, frag[by] / np.maximum(tot[by], 1e-9), f)
            frag[m] = tot[m] * ratio
        pts = frag["g"] + frag["a1"] + frag["a2"]
        ptot = tot["g"] + tot["a1"] + tot["a2"]
        frag["gs"] = tot["gs"] * np.where(ptot > 0, pts / np.maximum(ptot, 1e-9) * .6 + .4 * f, f)
        rest = {m: tot[m] - frag[m] for m in COUNT + TOI}
        return frag, rest


def pooled(st, decay, n, ks_scale=1.0, mask=None, isD=None):
    """Viktad summa over sasonger, regresserad mot positionssnittet."""
    agg = {m: sum((decay ** k) * c[m] for k, c in st) if st else np.zeros(n) for m in COUNT + TOI}
    rates = {}
    for m in COUNT:
        toi = np.maximum(agg[base(m)], 0)
        k = KS.get(m, KS["default"]) * ks_scale * 60
        mean = np.zeros(n)
        for flag in (True, False):
            idx = isD == flag
            mean[idx] = agg[m][idx].sum() / max(toi[idx].sum(), 1e-9)
        rates[m] = (agg[m] + mean * k) / np.maximum(toi + k, 1e-9) * 3600
    return agg, rates


DECAYS = (0.5, 0.9)


def design(st, isD, first, year, decays=DECAYS, ks_scale=1.0):
    """Prognosvariabler. Alla varden ar begransade till rimliga intervall."""
    n = len(isD)
    X = {}
    for d in decays:
        tag = f"d{int(d * 100)}"
        agg, r = pooled(st, d, n, ks_scale, isD=isD)
        for m in COUNT:
            X[f"{tag}_{m}"] = r[m]
        on = r["oxgf"] / (r["oxgf"] + r["oxga"]); off = r["fxgf"] / (r["fxgf"] + r["fxga"])
        X[f"{tag}_rel_xg"] = (on - off) * 100
        X[f"{tag}_rel_cf"] = (r["ocf"] / (r["ocf"] + r["oca"]) - r["fcf"] / (r["fcf"] + r["fca"])) * 100
        X[f"{tag}_oz"] = r["ozs"] / (r["ozs"] + r["dzs"])
        X[f"{tag}_fo"] = r["fow"] / (r["fow"] + r["fol"])
        X[f"{tag}_p60"] = r["g"] + r["a1"] + r["a2"]
        X[f"{tag}_sh"] = (agg["g"] + 5) / (agg["sog"] + 62.5)
    agg, _ = pooled(st, decays[0], n, ks_scale, isD=isD)
    toi = agg["toi"] / 3600
    X["log_toi"] = np.log1p(toi)
    X["pp_share"] = np.clip(agg["toi_pp"] / np.maximum(agg["toi_all"], 3600), 0, .5)
    X["pk_share"] = np.clip(agg["toi_pk"] / np.maximum(agg["toi_all"], 3600), 0, .5)
    X["toi_per_season"] = np.clip(toi / max(sum(decays[0] ** k for k, _ in st), 1), 0, 2000) / 1000 if st else np.zeros(n)
    X["exp"] = np.clip(year - first, 0, 20)
    return pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def effective_minutes(st, decay=DECAYS[1]):
    """Hur mycket data prognosen vilar pa, i viktade minuter 5 mot 5."""
    return sum((decay ** k) * c["toi"] for k, c in st) / 60 if st else 0.0


def targets(c):
    """Det vi vill forutsaga, per 60 minuter."""
    toi = np.maximum(c["toi"], 1)
    ta = np.maximum(c["toi_all"], 1)
    on = c["oxgf"] / np.maximum(c["oxgf"] + c["oxga"], 1e-6)
    off = c["fxgf"] / np.maximum(c["fxgf"] + c["fxga"], 1e-6)
    return {"gs60": c["gs"] / ta * 3600, "p60": (c["g"] + c["a1"] + c["a2"]) / toi * 3600, "g60": c["g"] / toi * 3600, "cxg60": c["cxg"] / toi * 3600,
            "a160": c["a1"] / toi * 3600, "xga60": c["oxga"] / toi * 3600, "relxg": (on - off) * 100}
