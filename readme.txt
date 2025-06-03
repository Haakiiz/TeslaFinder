# Tesla Model Y Hunter

**Overview**

Tesla Model Y Hunter is a Python-based scraper and intelligence pipeline for Finn.no, Norway’s leading classified ads site. The goal is to automatically collect and filter used Tesla Model Y listings, apply a user-defined “buy box” filter, and persist only new or updated listings in a local SQLite database. Later, these filtered results are fed into an LLM summarizer for daily commentary, price trend analysis, and notifications.

**Project Components**

1. **scrape.py**: Fetches all Tesla Model Y listings from Finn.no, page by page. For each listing, it extracts key fields such as ad ID, URL, price, year, mileage, color, and location by parsing the search–results HTML. It applies a strict YAML-configurable buy‑box filter (price cap, minimum year, maximum mileage, color, etc.). New or changed listings are stored in `listings.db` and written to `delta_listings.json` for downstream processing.

2. **buy\_box.yaml**: Defines the user’s non‑negotiable filter criteria. Examples include:

   * `price_max`: maximum price in NOK
   * `year_min`: earliest model year
   * `mileage_max`: maximum odometer reading
   * `color`: list of allowed colors
   * `exclude_keywords`: terms that disqualify a listing (e.g., “import,” “skadet”)

3. **SQLite Database (`listings.db`)**: Keeps one row per ad ID. Columns include `ad_id`, `url`, `price`, `year`, `mileage`, `color`, `location`, and timestamps. When a listing’s price or mileage changes, the record is updated, ensuring only fresh data is flagged.

4. **delta\_listings.json**: JSON dump of all newly inserted or updated listings since the last run. This file is consumed by the next component (LLM summarizer).

**Environment & Prerequisites**

* Python 3.11+ (tested on Python 3.12)
* Virtual environment setup:

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  playwright install chromium
  ```
* Dependencies (in `requirements.txt`):

  ```
  playwright
  pyyaml
  loguru
  tiktoken
  sqlite-utils
  telegram-send   # (optional, for notifications)
  openai          # (for LLM summarizer)
  ```

**Usage**

1. **Configure `buy_box.yaml`** with desired thresholds (price\_max, year\_min, etc.).
2. **Run the scraper**:

   ```bash
   python scrape.py
   ```

   * The first run populates `listings.db` and writes any matching listings to `delta_listings.json`.
   * Subsequent runs detect changes in price/mileage and update `listings.db`.
3. **Inspect `delta_listings.json`** for new or changed ads. These results are then fed into `openai_summarise.py` (not yet implemented) to generate daily insights.

**Planned & Not Yet Completed**

1. **LLM Summarizer (`openai_summarise.py`)**

   * Read `delta_listings.json`, batch listings into a single prompt, and call GPT‑4 to shortlist top deals with justifications.
   * Generate weekly trend analysis (average price, volume changes) and detect anomalies.
   * Output a human‑readable summary (email, Telegram, or other notification).

2. **Notifier (`notifier.py`)**

   * Accepts LLM output and sends an email or Telegram message to the user with the daily shortlist.
   * Configurable templates and scheduling.

3. **Trend Analysis Module**

   * Aggregate historical data in `listings.db` to compute rolling averages, standard deviations, or price distributions.
   * Allow the LLM to reference these stats in its prompt for deeper market insights.

4. **Geo‑Filtering & Radius Logic**

   * Convert `location_center` (e.g., “Oslo”) into latitude/longitude.
   * Compute haversine distance and drop listings outside `location_radius_km`.
   * (Currently stubbed out; latitude/longitude not scraped.)

5. **Extras & Feature Tags**

   * Parse optional fields (e.g., “Long Range,” “Performance,” “FSD”) from the thumbnail description.
   * Add filters like `fsd_required: true` and `must_include_images: true` by checking listing detail pages or embedded metadata.

6. **Robust Error Handling & Scheduling**

   * Implement retries, exponential backoff, and robust logging.
   * Add a `cron` entry (or use systemd timer) on a Raspberry Pi to run `scrape.py` daily at a fixed time.

**Future Enhancements**

* **Interactive Dashboard**: A simple web interface (React + Tailwind) to view current listings, trends, and modify the buy‑box without editing YAML.
* **Seller Language Analysis**: Use LLM to scan the ad description for negotiation hints (e.g., urgency, overpricing) and surface red flags.
* **Price Forecasting**: Build a lightweight ML model to predict short‑term price movement for Tesla Model Y in Norway.
* **Multi‑Model Support**: Extend to other EV models by abstracting the model code (e.g., Model 3, Model X).

---

**Maintainers**

* Håkon Granheim (Primary developer)
* Dr. Peter Norvig (Conceptual design & oversight)

**License**
MIT (or your preferred license).
