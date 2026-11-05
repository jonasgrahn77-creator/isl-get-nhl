# Isläget: hockeyspelare, uppdaterade varje natt

Hämtar MoneyPucks filer för utespelare, målvakter, kedjor och lag, kontrollerar dem, räknar betyg och bygger en statisk hemsida.
Data: [MoneyPuck.com](https://moneypuck.com/data.htm). Fritt för icke-kommersiellt bruk och källan ska anges. Fråga MoneyPuck före annan användning.

## Kom igång
1. Skapa ett GitHub-repo och lägg in alla filer, även mappen `.github`.
2. Välj **Settings > Pages > Source: GitHub Actions**.
3. Välj **Actions > Nattlig uppdatering > Run workflow**. Sedan körs den varje dag 09:17 UTC.

Sidan hamnar på `https://<ditt-konto>.github.io/<repo>/`. På GitHubs gratiskonto behöver repot vara publikt för att använda Pages.

## Köra lokalt
```
pip install -r requirements.txt
pytest -q
python run_nightly.py
```
Med nedladdade filer: `python run_nightly.py --input-dir mapp`. Mappen ska innehålla `skaters.csv`, `goalies.csv`, `lines.csv` och `teams.csv`.

## Filer
- `fetch_data.py`: hämtar filerna
- `pipeline.py`: läser, kontrollerar och räknar nyckeltal
- `forecast.py`, `features.py`: utespelarnas prognos
- `model.py`: målvaktsbetyget
- `model/`: den tränade modellen och den historik den behöver
- `training/`: träning och utvärdering
- `build_site.py`, `site_template.html`: hemsidan
- `run_nightly.py`: kör allt. Stoppar utan att skriva något om datan inte godkänns, så senaste sidan ligger kvar
- `history/`: en fil per körning, används för förändring över 7 dagar
- `data/nhl.duckdb`: databasen, byggs vid varje körning och är tänkt för agenten

## Om betyget
Utespelarnas betyg är en prognos som tränats på säsongerna 2008-09 till 2025-26 (`training/train_model.py`).
Modellen är en ridge-regression per positionsgrupp. Den förutsäger Game Score per 60 och relativ xG%, och vägningen mellan dem anpassades mot lagens xG-skillnad på säsonger före slutprovet.
Säsongen hittills och upp till fyra tidigare säsonger räknas ihop, och lite data dras mot positionens snitt. Sammanfattningen av slutprovet finns i `model/report.json`.

Så läser du proven. Modellen förutsäger nästa säsong och tränas varje gång bara på åren före. Inga modellval gjordes på 2022-2025.
- 2025-26 testades innan säsongen lades till i träningen: rangkorrelationen för spelarvärde blev 0,56 mot 0,41 för den gamla handviktade modellen, på samma spelare. Modellen hittade 52 % av säsongens 20 % bästa, där slumpen ger 20 %.
- Alla fyra testsäsongerna: 0,59 mot 0,44.
- På lagnivå finns ingen tydlig skillnad: 0,70 mot 0,72 för xG-skillnad och 0,53 mot 0,52 för målskillnad.
- Utan föregående säsong föll rangkorrelationen från 0,59 till 0,53.
- Prognosen är sämre med lite data: 0,18 under 300 viktade minuter, 0,34 upp till 1 000 och 0,61 därefter.

Begränsningar:
- Det finns ingen träffsäkerhet uppmätt mitt i en säsong, eftersom filerna saknar matcher för matcher. Provet gäller nästa säsong, och att väga in pågående säsong är ett antagande.
- Ålder, skador och nya roller ingår inte. Antal säsonger i ligan används som ersättning för ålder.
- Målvaktsbetyget i `model.py` är regelbaserat och otestat. Förra säsongens GSAx visas men används inte i betyget, eftersom 2025-26 nästan inte sa något om starten av 2026-27 (rangkorrelation 0,11 på 51 målvakter).

## Träna om modellen
```
pip install -r training/requirements.txt
python training/train_model.py skaters_2008_to_2024.csv skaters_2025.csv
python training/prev_season.py mapp_med_forra_sasongens_fyra_filer
```
Gör det efter varje avslutad säsong, annars saknar prognosen den senaste hela säsongen. Skriptet skriver om filerna i `model/`. Lägg alltid till nya säsonger som egna filer, med alla fem situationer för varje spelare.

## Andra ligor
SHL och HockeyAllsvenskan kräver en egen hämtning och egna nyckeltal, eftersom kolumnerna är MoneyPucks.
Kontrollen, historiken och hemsidan går att återanvända när en tillåten datakälla finns.
