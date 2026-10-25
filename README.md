# HanuAi - Text Mining & Exploration Dashboard

A Streamlit-based web dashboard designed for data quality exploration, text enrichment, and text mining analysis.

## 🚀 Quick Start / Local Installation

Follow these steps to set up the environment and run the application locally:

### 1. Clone the Repository
bash
git clone <YOUR_GITHUB_REPOSITORY_URL_HERE>
cd HanuAi


### 2. Set Up a Virtual Environment
It is highly recommended to isolate your dependencies using a virtual environment:
bash
# Create the environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate


### 3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt


### 4. Run the Dashboard
Launch the interface locally via the Streamlit CLI:
bash
streamlit run app.py

The dashboard will automatically open in your web browser at http://localhost:8501.

---

## 📂 Project Structure Overview
* app.py: Main execution entry point for the Streamlit interface.
* src/: Core source folder containing modular text mining utilities (data_quality, enrich_text, etc.).
* data/raw/: Directory holding raw datasets for analysis.
* outputs/: Destination folder for processed/exported results.
*

# HanuAi ML Assessment

Local Python project for the two tasks in the HanuAi ML Assessment PDF:

1. **BestBuy Canada reviews:** collect publicly visible reviews for a product with at least 50 reviews, score sentiment, and assign aspect tags.
2. **Vehicle complaint dataset:** profile and clean the supplied data, mine complaint text, tag failure entities, cluster issue narratives, and produce stakeholder-ready summaries and charts.

The supplied `Dataset.csv` is a vehicle complaint dataset for Task 2. It is not a BestBuy review export. Best Buy Canada's current [Conditions of Use](https://www.bestbuy.ca/en-ca/help/policies-and-terms-and-conditions/conditions-of-use) expressly prohibit scraping and data mining. The Task 1 collector is disabled unless you provide a written permission document that authorizes automated collection. Without permission, use a review export supplied through an authorized route and run sentiment/reporting only. Do not use CAPTCHA bypass, stealth, IP rotation, or other access-control evasion.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer
- Internet access only for installing packages and the optional BestBuy scraper
- A BestBuy Canada product review page with at least 50 reviews for Task 1

## Quick start (Windows PowerShell)

Open this project folder in VS Code. In the VS Code terminal, run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. You can also invoke `.venv\Scripts\python.exe` directly without activation.

Put the provided source file at `data/raw/Dataset.csv` (the included copy is already in that location). Then run:

```powershell
python -m src.pipeline --input data/raw/Dataset.csv --output-dir outputs --clusters 6
```

Launch the interactive dashboard:

```powershell
streamlit run app.py
```

After activation, use **Terminal → Run Task → HanuAi: Analyze Dataset** to regenerate Task 2 outputs, or **Terminal → Run Task → HanuAi: Run Dashboard** to launch the dashboard. Open and run the annotated assignment notebooks with:

```powershell
jupyter lab
```

The Task 1 notebook requires an authorized review export or written permission for automated collection. The Task 2 notebook uses `data/raw/Dataset.csv` and writes analysis artifacts to `outputs/`.

Open the local URL printed by Streamlit, normally `http://localhost:8501`. In the dashboard, select the CSV, review data quality and charts, and download the enriched records.

## Task 2 outputs

The analysis command writes these files to `outputs/`:

- `enriched_dataset.csv` — original fields plus parsed dates, cleaned text, mined entity tags, issue category, and cluster assignment.
- `data_quality.json` — dimensions, data types, missing values, and duplicate counts.
- `summary.json` — key descriptive statistics and leading issue patterns.
- `cluster_terms.json` — representative TF-IDF terms for each complaint cluster.
- `stakeholder_report.md` — generated synthesis, recommended actions, and method limitations.
- `figures/` — issue category, model, temporal trend, and cluster frequency charts.

The authorized Task 1 review workflow writes `bestbuy_reviews.csv` and `task1_stakeholder_report.md` when you have a written permission document. An authorized CSV processed separately produces `reviews_scored.csv` and the report path you select.

Dates are parsed with coercion; missing optional tag fields remain missing in the raw columns and mined tags are produced from narrative text where possible. Exact duplicate rows are counted and flagged; event IDs are also checked for duplication. No source rows are silently discarded.

Text mining combines `CUSTOMER_VERBATIM`, `CAUSAL_VERBATIM`, and `CORRECTION_VERBATIM`, with structured issue description fields as supporting signals. Rule-based dictionaries identify common symptoms, components, electrical/software/communication clues and repair actions. TF-IDF with K-Means groups similar narratives; cluster labels are data-driven top terms, not verified engineering root causes. Adjust `--clusters` to compare groupings.

## Task 1: BestBuy reviews

Use a page URL for a product with at least 50 public reviews. Review the site's policies and page behavior before running the browser automation.

```powershell
python -m src.bestbuy_scraper --url "https://www.bestbuy.ca/en-ca/product/PRODUCT-SLUG/12345678" --pages 5 --output outputs/bestbuy_reviews.csv
```

Best Buy Canada's Conditions of Use prohibit scraping and data mining. The automated collector is included only for use after you receive written permission that specifically authorizes it. Without that permission, do not run the collector; use an authorized review export. The script requires a non-empty permission document path and, when used under that permission, opens a visible Chrome browser, attempts sort controls, and writes a deduplicated CSV plus sentiment tags and a stakeholder report. Current page markup may require selector maintenance. It uses bounded retry/backoff for transient page-load failures and must stop at any access challenge.

For sentiment tagging of an existing review CSV with columns `title` and `body` (and optional `rating`), run:

```powershell
python -m src.review_sentiment --input path\to\reviews.csv --output outputs\reviews_scored.csv
```

An authorized input should use `title` and `body` columns; it may also provide `date`, `rating`, `source`, `reviewer`, and `id`. Missing metadata columns are added as blank or labeled as a provided review dataset, so the CSV schema is complete but the missing values are not inferred. Then create the stakeholder summary:

```powershell
python -m src.review_report --input outputs\reviews_scored.csv --output outputs\task1_stakeholder_report.md
```

This reproducible local baseline uses VADER lexicon polarity and heuristic aspect labels. It is not a fine-tuned model; validate it against a manually labeled sample before using it for decisions.

## Project layout

```text
app.py                     Streamlit dashboard
src/pipeline.py            Task 2 CLI analysis and output generation
src/text_mining.py         Entity rules, issue labels, and TF-IDF clustering
src/bestbuy_scraper.py     Optional visible-browser review collector
src/review_sentiment.py    Sentiment and aspect-tagging utility
src/review_report.py       Task 1 stakeholder report generator
notebooks/                 Annotated Task 1 and Task 2 notebooks
data/raw/Dataset.csv       Supplied vehicle complaint data
outputs/                   Generated analysis artifacts (created on run)
```

## Reproducibility and limitations

- Keep the input CSV unchanged; generated files are written under `outputs/`.
- Scrape only publicly accessible content at a modest pace. The collector uses normal UI interactions and retry/backoff for transient failures. It does not use rotating IPs, stealth fingerprinting, or CAPTCHA bypass.
- The dataset's sample is limited to the supplied 1,000 records. Trends describe this file and should not be generalized without checking coverage and reporting periods.
- Automated entity tags and clusters are candidate signals for human review. They do not prove causal failure mechanisms.
- Task 1 collection is gated because Best Buy Canada's current site terms prohibit scraping; obtain specific written permission or use an authorized review export.
- For an auditable submission, retain the exact input file, package versions, command arguments, and any manual validation notes with the exported results.

