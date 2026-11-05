"""Hamtar MoneyPucks spelarfil for aktuell grundserie."""
import pathlib
import urllib.request

URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/{file}.csv"
# Forsta kolumnen i varje fil, for att kanna igen ratt innehall
FILES = {"skaters": b"playerId,", "goalies": b"playerId,", "lines": b"lineId,", "teams": b"team,"}


def current_season(today):
    """MoneyPuck anger sasongens startar. Grundserien startar i oktober."""
    return today.year if today.month >= 10 else today.year - 1


def download(season, file, dest, timeout=120):
    req = urllib.request.Request(URL.format(season=season, file=file), headers={"User-Agent": "hockey-spelaragent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    if not body.startswith(FILES[file]):
        raise ValueError(f"Svaret ser inte ut som MoneyPucks {file}-fil")
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def download_all(season, folder):
    """Hamtar alla filer till en tillfallig plats forst, sa en halv hamtning inte blandas med gammal data."""
    folder = pathlib.Path(folder)
    for f in FILES:
        download(season, f, folder / f"{f}.csv")
    return folder
