# SEC 8-K New-Product Extraction with a Local LLM

Extracts **newly announced products** from SEC Form 8-K filings of S&P 500 companies
using a locally hosted large language model (Ollama + Llama 3 8B), performs
NER-style identification of the filing company (ORGANIZATION) and the products
(PRODUCT), validates the model output against the filing text, and writes a CSV:

```
company_name, stock_name, filing_time, new_product, product_description
```

GitHub: https://github.com/alexhubenthal/sec-8k-product-extraction

## Repository layout

| Path | What it is |
|---|---|
| `run_extraction.py` | Command-line entry point (run this). |
| `sec8k/edgar.py` | SEC EDGAR client: ticker → CIK, 8-K Atom feed, filing documents → plain text. Rate-limited, cached. |
| `sec8k/llm.py` | Ollama client, NER prompt, JSON schema, validation of the model's answer. |
| `sec8k/pipeline.py` | Orchestration, JSONL checkpointing, CSV / table / summary outputs. |
| `notebooks/sec_8k_product_extraction.ipynb` | Walk-through notebook: runs the pipeline step by step and shows the results. |
| `data/sp500_constituents.csv` | Company universe (S&P 500 constituents, public dataset snapshot). |
| `data/processed/extractions.jsonl` | Checkpoint: one JSON record per filing analysed (LLM answer, validation, timings). |
| `output/new_products.csv` | **Main deliverable**: one row per newly announced product. |
| `output/new_products_table.txt` | Same table in the pipe-separated layout from the assignment. |
| `output/filings_processed.csv` | One row per filing analysed, including filings without a product. |
| `output/summary.json` | Statistics of the run (used in the report). |
| `report/report.md`, `report/report.pdf` | Methodology report. |

## Quick start

Requirements: Python 3.10+, [Ollama](https://ollama.com) with the `llama3` model, and
internet access to sec.gov.

```bash
pip install -r requirements.txt
ollama pull llama3                # once; the Ollama server must be running
export SEC_USER_AGENT="Your Name your.email@example.com"   # SEC requires this
python run_extraction.py          # stops after 100 product rows (~2-3 h on an M-series Mac)
```

SEC's [fair-access policy](https://www.sec.gov/os/accessing-edgar-data) requires every
automated request to carry a `User-Agent` with your name and e-mail and to stay
under 10 requests per second; the client enforces both. The script exits with a
message if `SEC_USER_AGENT` is not set.

Useful options (`python run_extraction.py --help` lists all):

```bash
# quick test on a few companies
python run_extraction.py --tickers AAPL,MSFT,NVDA --per-company 1 --target-rows 3 --max-filings 3
# bigger run, different model (--max-filings caps the filings sent to the LLM per run)
python run_extraction.py --target-rows 200 --max-filings 800 --model llama3.1
# process filings in pure recency order instead of cue-phrase order
python run_extraction.py --no-cue-sort
# after changing the validation rules: re-apply them to the stored model answers (no new model calls)
python run_extraction.py --revalidate
# after changing the prompt: drop some companies' filings so they are extracted again on the next run
python run_extraction.py --forget CHD, META, ABT
```

Runs are resumable: every analyzed filing is appended to
`data/processed/extractions.jsonl` and skipped on the next run; the output files
are regenerated from that checkpoint each time.

## How it works

1. **Company universe** – `data/sp500_constituents.csv` (503 tickers; companies listed
   under two share classes such as GOOGL/GOOG are processed once). Each ticker is
   mapped to its CIK with `https://www.sec.gov/files/company_tickers.json`.
2. **Filing discovery** – for every CIK, the browse-edgar Atom feed
   `.../cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=8-K&count=20&output=atom`
   lists the latest 8-Ks with their filing timestamp and *item* numbers. Only original
   8-Ks (no 8-K/A) that report item 2.02, 7.01 or 8.01 are kept, because those are
   the items under which press releases are attached; up to two per company.
3. **Text extraction** – the filing index page is parsed with BeautifulSoup; the
   primary 8-K document and every EX-99 press-release exhibit are downloaded and
   converted to plain text (hidden inline-XBRL metadata is stripped). The text sent
   to the model is capped at 10,000 characters: the 8-K cover page and item
   heading plus the press release.
4. **Prioritisation** – filings are ordered by the number of product-announcement
   cue phrases they contain (launch, introduce, unveil, FDA approval, …) so the
   100-row target is reached sooner. This only changes the *order*; every filing
   that is processed is classified by the LLM.
5. **LLM extraction (NER)** – Llama 3 is prompted as a named-entity recognizer with
   two entity types, ORGANIZATION (the registrant) and PRODUCT (a product announced
   as new by the registrant), and is constrained by a JSON schema (Ollama structured
   outputs) to return `company_name`, `has_new_product` and, per product, a verbatim
   `evidence` sentence, an `announcement_type`, the `new_product` name and a
   `product_description`.
6. **Validation** – deterministic checks against the filing text: the product name
   must occur in the text (its character span is recorded); the evidence sentence
   must occur in the text, and the sentence containing it must mention the product
   and contain a launch / approval / availability cue; names that are the company
   itself, a financial instrument, a programme/initiative/partnership/opening or a
   generic phrase ("new products") are dropped, as are products the sentence
   describes as pending approval or planned for the future; descriptions are
   clamped to < 180 characters; duplicates are removed. Every record stores the
   prompt and validation rule versions that produced it.
7. **Verification** – every surviving product is re-checked by a second, short LLM
   call that sees the passage around the mention and must confirm that the filer is
   announcing it as new (disable with `--no-verify`).
8. **Outputs** – CSV (one row per product; the same product announced in two filings
   of a company is listed once), pipe-separated table, per-filing log and summary
   statistics. Filings whose LLM call failed are retried on the next run.

## Results

Run of 1 November 2026 with Llama 3 8B (full log in `output/run.log`, statistics in
`output/summary.json`, methodology and discussion in `report/report.pdf`):

| | |
|---|---|
| Companies resolved to a CIK | 500 of 503 S&P 500 tickers (3 dual-class duplicates skipped) |
| Candidate 8-K filings listed | 1,000 (items 2.02 / 7.01 / 8.01, up to 2 per company) |
| Filings analysed by the LLM | 213 from 185 companies, filed 30 Apr – 30 Oct 2026 |
| Filings announcing a new product | 58 |
| Rows in `output/new_products.csv` | **120** products from 48 companies |
| Product entities proposed / rejected by validation | 438 / 316 |
| Model time | ≈ 23 s per extraction, 6 s per verification, 104 min in total |
| Manual check of all rows | 3 rows wrong, ≈ 12 borderline (precision 87 – 97 %) |

Filings with three or more launch cue phrases yielded 1.19 products each, filings with
one or two 0.28, filings with none 0.02, so the cue-phrase ordering is what made the run
feasible on a laptop.

## Author

Alexander Hubenthal – Assignment 2, CAP5619 (Fall 2026).
