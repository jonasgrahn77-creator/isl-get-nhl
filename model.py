"""Malvaktsbetyget. Regelbaserat tills det finns malvaktshistorik att trana pa. Utespelarnas betyg finns i forecast.py."""
import numpy as np

# Malvakter: sparade mal over forvantan per 60 i alla situationer. Mycket brusigt, sa stark dragning mot snittet.
GOALIE_PRIOR_MINUTES = 1500
GOALIE_MIN_MINUTES = 200


def goalie_confidence(minutes):
    if minutes < GOALIE_MIN_MINUTES:
        return "För lite data"
    if minutes < 600:
        return "Låg"
    return "Medel" if minutes < 1500 else "Hög"


def rate_goalies(metrics):
    x = metrics[metrics.situation == "all"].copy()
    x["rating"] = np.nan
    x["confidence"] = x.minutes.map(goalie_confidence)
    q = x[x.minutes >= GOALIE_MIN_MINUTES]
    if len(q) >= 2:
        prior = q.gsax.sum() * 60 / q.minutes.sum()
        shrunk = (q.gsax_60 * q.minutes + prior * GOALIE_PRIOR_MINUTES) / (q.minutes + GOALIE_PRIOR_MINUTES)
        x.loc[q.index, "rating"] = (shrunk.rank() - 0.5) / len(q) * 100
    return x
