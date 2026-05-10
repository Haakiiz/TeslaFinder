# FinnFinder

En Python-app som overvåker Finn.no for ting du vil kjøpe (brukt Tesla, barnestol, gitar, hva som helst), og bruker en LLM (ChatGPT/Claude/Grok) til å rangere de beste annonsene for deg.

---

## Hvordan det funker (det store bildet)

Du har én app med to steg:

1. **`scrape.py`** — går til Finn.no, henter annonser, lagrer i en lokal database, og skriver ut bare det som er nytt eller endret siden sist.
2. **`LLM_Summariser.py`** — leser ut det nye, sender til en LLM med din "brief" (f.eks. "jeg leter etter familievennlig elbil"), og får tilbake en rangert toppliste.

Det viktigste å skjønne: **du har én app, ikke flere. Alt du vil søke etter står i én konfigurasjonsfil — `searches.yaml`.**

```
                ┌─────────────────┐
                │  searches.yaml  │  ← Her står alle søkene dine
                └────────┬────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      ┌──────────┐              ┌─────────────────┐
      │ scrape   │ ───────────▶ │  LLM_Summariser │
      │   .py    │  delta_*.json│       .py       │
      └──────────┘              └────────┬────────┘
                                         │
                                         ▼
                                  summary - <søk> - <dato>.txt
```

---

## Førstegangs-oppsett

Disse kommandoene kjører du én gang når du klona repoet:

```bash
python3 -m venv venv               # Lager et "isolert" Python-miljø
source venv/bin/activate           # Aktiverer det (Windows: venv\Scripts\activate)
pip install -r requirements.txt    # Installerer alle Python-bibliotekene
playwright install chromium        # Installerer en headless nettleser (brukes til å scrape Finn.no)
```

Deretter må du lage en fil som heter `.env` i hovedmappen, med API-nøklene dine:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROK_API_KEY=xai-...
```

Du trenger bare nøkkelen for de LLM-providerne du faktisk bruker. Hvis alle søkene dine bruker `provider: openai`, holder det med `OPENAI_API_KEY`.

---

## Hvordan legge til et nytt søk (steg for steg)

La oss si du vil overvåke barnestoler på Finn.no.

### Steg 1 — Finn URL-en på Finn.no

Gå til [finn.no](https://www.finn.no), skriv inn det du leter etter (f.eks. "barnestol"), og sett alle filtrene du vil ha (pris, sted osv.). Når du er fornøyd, **kopier hele URL-en fra adressefeltet i nettleseren**.

Eksempel: `https://www.finn.no/recommerce/forsale/search?q=barnestol&location=0.20002`

### Steg 2 — Åpne `searches.yaml` og legg til en ny blokk

Filen ser slik ut etter at du har lagt til barnestol:

```yaml
searches:

  - name: tesla_model_y
    description: "Tesla Model Y, Østlandet, maks 500k NOK"
    finn_url: "https://www.finn.no/mobility/search/car?body_type=2&..."
    filters:
      price_max: 500000
      year_min: 2019
      mileage_max: 100000
      exclude_keywords: ["import", "skadet"]
    llm:
      provider: openai
    llm_context: |
      Evaluate used Tesla Model Y in Norway for a family with a young child.

  # ← Det er her du legger inn det nye søket:
  - name: barnestol
    description: "Brukt barnestol, Østlandet, maks 1500 NOK"
    finn_url: "https://www.finn.no/recommerce/forsale/search?q=barnestol&location=0.20002"
    filters:
      price_max: 1500
      exclude_keywords: ["skadet", "ødelagt"]
    llm:
      provider: claude
    llm_context: |
      Brukt barnestol til 1-åring. Prioriter: god stand, trygghet, god pris.
      Skip annonser som er åpenbart slitne eller mangler bilder.
```

**Forklaring av feltene:**

| Felt | Hva det gjør | Påkrevd? |
|------|--------------|----------|
| `name` | Kort navn (brukes i filnavn). Ingen mellomrom — bruk underscore. | Ja |
| `description` | En setning som forteller deg selv hva søket er for | Nei (men greit å ha) |
| `finn_url` | URL-en du kopierte fra Finn.no | Ja |
| `filters.price_max` | Maks pris i NOK | Nei |
| `filters.year_min` | Tidligste årsmodell (kun for biler) | Nei |
| `filters.mileage_max` | Maks kilometer (kun for biler) | Nei |
| `filters.exclude_keywords` | Hvis annonsen inneholder noen av disse ordene → hopp over | Nei |
| `llm.provider` | Hvilken AI som skal rangere: `openai`, `claude` eller `grok` | Ja |
| `llm.model` | Spesifikk modell (f.eks. `gpt-4o`). Hvis du ikke skriver noe, brukes default. | Nei |
| `llm_context` | "Brief"-en til AI-en — hva den skal se etter. Skriv naturlig på norsk eller engelsk. | Ja |

### Steg 3 — Kjør scraperen

```bash
# Kjør bare det nye søket:
python scrape.py --search barnestol

# Eller kjør alle søk (Tesla + barnestol):
python scrape.py
```

Dette lager en fil `delta_barnestol.json` med alle nye/endrede annonser.

### Steg 4 — Be AI-en rangere

```bash
python LLM_Summariser.py --search barnestol
```

Resultatet vises i terminalen og lagres i `summary - barnestol - <dato>.txt` (datoen settes automatisk til når du kjørte den).

Legg til `--verbose` hvis du vil se framdrift mens den jobber.

---

## Bytte mellom OpenAI, Claude og Grok

Bare endre `provider`-linjen i `searches.yaml`:

```yaml
llm:
  provider: claude     # bytt fra openai til claude
```

Hver søk kan ha sin egen provider. Tesla kan bruke OpenAI mens barnestol bruker Claude — det er helt opp til deg.

**Default-modeller** (brukes hvis du ikke spesifiserer `model`):
- `openai` → `gpt-5`
- `claude` → `claude-opus-4-7`
- `grok` → `grok-3`

> Merk: Det ekstra "bagasjeromsstørrelse"-søket (kun relevant for biler) bruker en spesiell OpenAI-funksjon (web search). Det hoppes automatisk over hvis du bruker Claude eller Grok.

---

## Filer du vil møte

| Fil | Hva det er |
|-----|------------|
| `searches.yaml` | **Konfigurasjon — den eneste filen du trenger å redigere** |
| `scrape.py` | Selve scraperen (du kjører den, men trenger ikke endre den) |
| `LLM_Summariser.py` | LLM-rangereren |
| `listings.db` | SQLite-database der alle annonser huskes (slik at neste kjøring vet hva som er nytt) |
| `delta_<navn>.json` | Output fra scraperen — sendt til summariseren |
| `listing_cache.json` | Mellomlagring av annonsetekster (12 timer) for å spare API-kall |
| `summary - <navn> - <dato>.txt` | LLM-rangeringen — det endelige resultatet |
| `.env` | API-nøklene dine (ikke committ denne!) |

Annonser eldre enn 30 dager slettes automatisk fra databasen.

---

## Branch-forklaring (hvis du er forvirret over Git)

Repoet har én hovedbranch — `master` — som inneholder den allsidige koden.

**Du trenger IKKE en branch per søk.** Tesla, barnestol, gitar — alt ligger på samme branch og styres av `searches.yaml`. Det er hele poenget med refaktoret: én kodebase, mange søk.
