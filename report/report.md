---
title: "Extracting New-Product Announcements from SEC Form 8-K Filings with a Local LLM"
author: "Alex Hubenthal — CAP5619, Fall 2026"
date: "November 2026"
---

**Code and data:** https://github.com/alexhubenthal/sec-8k-product-extraction

## 1. Goal

Form 8-K is the "current report" that U.S. public companies file whenever a material
event happens. Many of those events are product launches, announced in a press release
attached to the filing. The goal of this project was to process 8-K filings of S&P 500
companies, use a large language model to recognise the filing company and any newly
announced products, and deliver a table with five columns: `company_name`,
`stock_name`, `filing_time`, `new_product` and `product_description` (under 180
characters), with at least 100 entries.

## 2. Data source

Two public SEC EDGAR endpoints drive the whole pipeline:

1. `https://www.sec.gov/files/company_tickers.json` maps a ticker to the company's
   Central Index Key (CIK), e.g. AAPL → 0000320193.
2. `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=8-K&count=20&output=atom`
   returns an Atom feed of the company's latest 8-K filings with their timestamp,
   accession number, the 8-K *item* numbers and a link to the filing index page.

The company universe is the S&P 500 constituent list (503 tickers; companies listed
under two share classes such as GOOGL/GOOG share one CIK and are processed once).
From the index page the primary 8-K document and every EX-99 exhibit (the press
release) are downloaded and converted to plain text with BeautifulSoup. Only
original 8-Ks (no 8-K/A) that report item 2.02 (results of operations), 7.01
(Regulation FD) or 8.01 (other events) are considered, because those are the items
under which press releases are attached; items such as 5.02 (officer changes) never
announce products. SEC's fair-access rules are respected: every request carries an
identifying `User-Agent`, the client stays under 10 requests per second, and every
response is cached on disk.

## 3. Methodology

**Pipeline** (`run_extraction.py`, package `sec8k/`): ticker → CIK → 8-K feed →
candidate filings (up to two per company, newest first, round-robin across companies)
→ document text (capped at 10,000 characters: 8-K cover page and item heading, then the
press release) → LLM extraction → validation → LLM verification → CSV. Filings are
ordered by the number of product-announcement cue phrases they contain (launch,
introduce, unveil, FDA approval, …) so that the row target is reached sooner; this
only changes the processing order, every processed filing is classified by the LLM.
Each processed filing is appended to a JSONL checkpoint, so an interrupted run resumes.

**NER with the LLM.** The model is prompted as a named-entity recognition system with
two entity types: ORGANIZATION (the registrant) and PRODUCT (a product, service, device,
drug, software or model that *this* filing announces as new and that belongs to the
registrant). The prompt lists what is *not* a product (financial results, existing
products merely reported on, drug candidates in trials, future products, partners'
products, financing, M&A, executive changes). Ollama's structured-output feature
constrains the answer to a JSON schema, and the schema deliberately asks for a
verbatim `evidence` sentence *before* the product name, so the model commits to a
quote first and extracts the entity from it. Every kept entity is grounded in the
source text and its character span is recorded, which is the span-level output an
NER system produces.

**Validation.** Small local models hallucinate, so deterministic checks follow:
the product name must occur in the filing text (word-boundary match); the evidence
sentence must occur in the text; the sentence containing the evidence must mention
the product and contain a launch/approval/availability cue; names equal to the
company, to a financial instrument ("senior notes", "quarterly dividend"), to a
programme, initiative, partnership or opening, or consisting only of generic words
("new products") are dropped, as are products whose sentence says they are pending
approval or planned for later; descriptions are clamped to 179 characters at a word
boundary; duplicates within a filing and the same product announced in two filings of
one company are removed. A second, short LLM call then classifies each surviving
product into one of five categories (new launch or approval / existing product
reported / pipeline or future product / third-party product / not a product) and only
the first category is kept. Validation rules can be re-applied to the stored model
answers without new model calls (`--revalidate`), and every record carries the prompt
and rule versions that produced it.

**Outputs.** `output/new_products.csv` (the deliverable), the same table in the
pipe-separated layout from the assignment, `output/filings_processed.csv` with one row
per filing analysed (including filings without a product), and `output/summary.json`.

## 4. LLM specifics

| | |
|---|---|
| Model | Llama 3 8B Instruct (`llama3:latest`, Q4_0 quantisation, 4.7 GB) served locally by Ollama 0.30.10 |
| Hardware | Apple M4 MacBook, 24 GB unified memory, no discrete GPU |
| Decoding | temperature 0, context window 8,192 tokens, JSON-schema constrained output, up to 2,500 output tokens |
| Prompt size | ≈ {{PROMPT_TOKENS}} tokens per filing (10,000 characters of filing text plus instructions) |
| Speed | ≈ {{LLM_SECONDS}} s per extraction call and ≈ {{VERIFY_SECONDS}} s per verification call; {{LLM_TOTAL_MIN}} minutes of model time in total |

## 5. Results

{{RESULTS_PARAGRAPH}}

{{SAMPLE_TABLE}}

## 6. Challenges

The project turned out to be less about calling the model and more about making a
small local model's output trustworthy.

1. **EDGAR mechanics.** SEC rejects anonymous clients (an identifying `User-Agent` and
   the 10 requests/second limit are mandatory). The index page links the primary 8-K
   through the inline-XBRL viewer (`/ix?doc=…`), which needs JavaScript, so the prefix
   has to be stripped to reach the raw HTML. Inline-XBRL documents start with a hidden
   `<ix:header>` block; until it was removed, about 600 characters of tagging metadata
   (`lly:A2.125NotesDueJune32030Member …`) led every 8-K body sent to the model.
2. **"New product" is a fuzzy target.** Earnings releases (item 2.02) mention existing
   products constantly. The first version of the extractor returned *Eliquis* and
   *Padcev* from a Pfizer sentence about revenue growth and *Hugging Face's platform*
   from an NVIDIA investment announcement. The remedies were an explicit exclusion
   list in the prompt, the evidence-first schema, grounding of both the product name
   and the evidence sentence in the text, and the requirement that the sentence
   containing the evidence carries a launch/approval/availability cue.
3. **Press-release formatting.** Lilly packs several approvals into one
   semicolon-separated sentence ("FDA approvals of Mounjaro …; Onswik, a once-weekly
   basal insulin …; Inluriyo …"). The model's quote for *Onswik* omitted the drug's
   name and the word "approvals" that governs the whole sentence, so the checks were
   widened from the quote to the sentence that contains it; the sentence splitter must
   not treat "U.S." or "Inc." as a sentence end.
4. **A yes/no verifier flip-flops.** Asked a binary question, Llama 3 8B rejected
   every product in one framing (its own explanation said "Next Gen Roadster was
   unveiled, which indicates a new launch" and then answered *false*) and accepted
   every product, including a cost-cutting programme, in another. Emitting the
   explanation before the decision and replacing the boolean by a five-way category
   gave stable, sensible verdicts on the same test cases.
5. **Company names.** The 8-K cover page prints the registrant in capitals and
   EDGAR's title is upper case ("ELI LILLY & Co"); the model echoed them. When the
   model returns the all-caps form, a ticker or a placeholder, the proper-cased S&P 500
   security name is used instead.
6. **Dual-class tickers.** GOOGL/GOOG, FOXA/FOX and NWSA/NWS resolve to one CIK, so the
   same filing was queued, sent to the model and counted twice until the universe was
   de-duplicated by CIK and candidates by accession number.
7. **Throughput.** About {{LLM_SECONDS}} s per filing on a laptop without a discrete GPU
   makes 500 companies × 2 filings infeasible in one sitting, hence the cue-phrase
   ordering, the per-filing checkpoint and the on-disk response cache.

## 7. Limitations and possible extensions

* **Coverage is biased towards product-rich filings.** Filings are processed in
  cue-phrase order until the row target is met, so companies whose recent 8-Ks contain
  no launch vocabulary are under-represented; the per-filing log shows exactly which
  filings were analysed.
* **Recency window.** Only the ~20 most recent 8-Ks per company were listed and at
  most two per company analysed, so the table is a snapshot ({{DATE_RANGE}}).
* **Truncation.** The model sees 10,000 characters; an announcement buried deep in a
  long earnings release can be missed.
* **Definition of "new".** A new FDA indication for an existing drug and an updated
  version of an existing vehicle are counted as new products; a stricter definition
  would drop them.
* **Model.** A single 8B model was used. The backend is pluggable (`--model`), so a
  larger local model or a hosted API would likely raise precision further; a labelled
  sample would allow a proper precision/recall measurement instead of the manual check
  reported above. EDGAR's full-text search API could also target launch announcements
  directly instead of sampling by company.

## 8. Reproducing the run

```bash
pip install -r requirements.txt
ollama pull llama3                                   # Ollama server running
export SEC_USER_AGENT="Your Name your.email@example.com"
python run_extraction.py --target-rows 110
```
