---
title: "Extracting New-Product Announcements from SEC Form 8-K Filings with a Local LLM"
author: "Alex Hubenthal — CAP5619, Fall 2026"
date: "November 2026"
---

**Code, data and outputs:** https://github.com/alexhubenthal/sec-8k-product-extraction

## 1. Goal and data

Form 8-K is the "current report" a U.S. public company files when something material
happens; product launches are announced there in attached press releases. The task was to
process 8-K filings of S&P 500 companies, use a large language model to recognise the filing
company and any newly announced product, and deliver a table with the columns
`company_name, stock_name, filing_time, new_product, product_description` (description
under 180 characters, at least 100 entries).

Two public EDGAR endpoints drive the pipeline. `https://www.sec.gov/files/company_tickers.json`
maps a ticker to the company's CIK (AAPL → 0000320193), and the Atom feed
`https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=8-K&count=20&output=atom`
lists the latest 8-Ks with timestamp, accession number, 8-K item numbers and a link to the
filing index page. The universe is the S&P 500 constituent list (503 tickers; the three
companies listed under two share classes are processed once). From the index page the primary
8-K and every EX-99 exhibit (the press release) are downloaded and converted to plain text
with BeautifulSoup. Only original 8-Ks reporting item 2.02 (results of operations), 7.01
(Regulation FD) or 8.01 (other events) are considered, since those are the items under which
press releases are attached. Requests carry the identifying `User-Agent` SEC requires, stay
below 10 per second and are cached on disk.

## 2. Methodology

**Pipeline** (`run_extraction.py`, package `sec8k/`): ticker → CIK → 8-K feed → up to two
candidate filings per company → document text, capped at 10,000 characters (8-K cover page
and item heading, then the press release, hidden inline-XBRL metadata stripped) → LLM
extraction → validation → LLM verification → CSV. Filings are ordered by the number of
product-announcement cue phrases they contain (launch, introduce, unveil, FDA approval, …)
so the row target is reached sooner; this changes only the order, every processed filing is
classified by the LLM. Each filing is appended to a JSONL checkpoint, so interrupted runs
resume and outputs are regenerated from the checkpoint.

**NER with the LLM.** The model is prompted as a named-entity recognition system with two
entity types: ORGANIZATION (the registrant) and PRODUCT (a product, service, device, drug,
software or model that *this* filing announces as new and that belongs to the registrant),
with an explicit list of what is not a product (financial results, existing products merely
reported on, drug candidates in trials, future products, partners' products, financing,
M&A, executive changes, openings). Ollama's structured output constrains the answer to a
JSON schema that asks for a verbatim `evidence` sentence *before* the product name, so the
model commits to a quote and extracts the entity from it. Every kept entity is grounded in
the source text and its character span is recorded, i.e. the span-level output of an NER
system.

**Validation and verification.** Deterministic checks follow the model: the product name must
occur in the text (word-boundary match); the quoted evidence must occur in the text and the
sentence containing it must mention the product and contain a launch/approval/availability
cue; names equal to the company, to a financial instrument, to a programme, initiative,
partnership or opening, or consisting only of generic words ("new products") are dropped, as
are products whose quote says they are pending approval or planned for later; descriptions
are clamped to 179 characters; duplicates within a filing and across a company's filings are
merged. A second, short LLM call then classifies each surviving product into one of five
categories (new launch or approval / existing product reported / pipeline or future product /
third-party product / not a product) and only the first is kept. Rules can be re-applied to
the stored model answers without new model calls (`--revalidate`); every record carries the
prompt and rule versions that produced it.

## 3. LLM specifics

| | |
|---|---|
| Model | Llama 3 8B Instruct (`llama3:latest`, Q4_0, 4.7 GB) served locally by Ollama 0.30.10 |
| Hardware | Apple M4 MacBook, 24 GB unified memory, no discrete GPU |
| Decoding | temperature 0, 8,192-token context, JSON-schema constrained output, ≤ 2,500 output tokens |
| Prompt size | ≈ 2,700 tokens per filing (10,000 characters of filing text plus instructions) |
| Speed | ≈ 23 s per extraction call, ≈ 6 s per verification call; 104 minutes of model time in total |

## 4. Results

All 500 registrants were resolved to a CIK, 1,000 candidate filings were listed, 825 texts
were downloaded (about 2,500 EDGAR requests, all cached) and **213 filings from 185
companies** were analysed, filed between 30 April and 30 October 2026 (160 earnings
releases, 51 Regulation FD, 34 other-event filings). 58 filings (27%) announced at least one
new product. The model proposed 438 PRODUCT entities and 316 were rejected: 91 by the LLM
verifier, 84 for a sentence without launch cue, 63 because the quote did not mention the
product, 25 because the name is not in the text, 17 as programmes, partnerships or
openings, 12 as generic names, 5 as pending or future, the rest as duplicates, company
names or financial instruments. The deliverable `output/new_products.csv` has **120 rows
from 48 companies** (Health Care 18, Information Technology 17, Financials 6, Industrials
5, Consumer Discretionary 5, Consumer Staples 5, Communication Services 2); the longest
description has 178 characters. Cue-phrase ordering is what made the run feasible on a
laptop:

| Cue phrases in the filing | Filings analysed | Products kept | Products per filing |
|---|---|---|---|
| 3 or more | 83 | 99 | 1.19 |
| 1 to 2 | 80 | 22 | 0.28 |
| 0 | 50 | 1 | 0.02 |

**Table 1 – six of the 120 rows** (the file holds the full ISO-8601 filing timestamp).

| company_name | stock_name | filing_time | new_product | product_description |
|---|---|---|---|---|
| AbbVie Inc. | ABBV | 2026-10-30 | Skyrizi | Subcutaneous (SC) induction in the treatment of moderately to severely active Crohn's disease (CD) |
| Amazon.com, Inc. | AMZN | 2026-10-29 | AWS Continuum | An AI-powered security service that automatically discovers, prioritizes, validates, and remediates software vulnerabilities. |
| Eli Lilly and Company | LLY | 2026-10-29 | Onswik | Once-weekly basal insulin for adults with type 2 diabetes. |
| Tesla, Inc. | TSLA | 2026-10-21 | Next Gen Roadster | New roadster model |
| Nvidia | NVDA | 2026-08-26 | NVIDIA Vera CPU | CPU for AI agents |
| Apple Inc. | AAPL | 2026-04-30 | iPhone 17e | A new iPhone model |

**Quality.** Every row was read against its evidence sentence. The first version of the table
(111 rows) contained 18 rows (16%) that were not products in the intended sense: initiatives
presented as "launches" (CRH "Velocity", McDonald's "Make It Golden"), partnerships and
supply agreements, four Hilton hotel openings, two devices still awaiting approval, future
launch plans, generic names and a duplicate. The validation rules were extended for these
classes, re-applied to the stored answers, and the run was resumed to refill the table. In
the final table 3 rows (2.5%) are still wrong (United's new Airbus A321XLR aircraft type, a
CVS patient-support programme, a Ford pickup announced for a future year) and about 12 (10%)
are borderline: new features of existing platforms, a geographic launch (Omnipod 5 in Spain),
an internal tool (Block's "Buzz") and a pricing model. Estimated precision is 87–97%
depending on how strictly "new product" is read; recall was not measured.

## 5. Challenges

1. **EDGAR mechanics.** SEC rejects anonymous clients; the index page links the primary 8-K
   through the inline-XBRL viewer (`/ix?doc=…`), which needs JavaScript, so the prefix must
   be stripped; and inline-XBRL documents start with a hidden `<ix:header>` block that put
   about 600 characters of tagging metadata at the top of every 8-K body until it was removed.
2. **"New product" is a fuzzy target.** Earnings releases mention existing products
   constantly; the first extractor returned *Eliquis* and *Padcev* from a Pfizer sentence
   about revenue growth and a partner's platform from an NVIDIA investment release. The
   evidence-first schema, grounding of name and quote, and the launch-cue requirement fixed
   this; a later review added the programme, pending and generic-name rules.
3. **Press-release formatting.** Lilly packs several approvals into one semicolon-separated
   sentence, and the model's quote for *Onswik* omitted both the drug's name and the word
   "approvals", so positive checks were widened to the sentence containing the quote. Negative
   rules, however, must stay on the quote itself: judged on the sentence, one pending item
   condemned its approved neighbours (J&J's OTTAVA and IMAAVY, four AMD launches).
4. **A yes/no verifier flip-flops.** Asked a binary question, Llama 3 8B rejected every
   product in one framing (its explanation said "Next Gen Roadster was unveiled, which
   indicates a new launch" and then answered *false*) and accepted everything, including a
   cost-cutting programme, in another. Writing the explanation first and choosing among five
   categories gave stable verdicts.
5. **The prompt's own example leaked.** A concrete example description in the prompt was
   copied verbatim for six unrelated products; it was replaced by an abstract instruction and
   the affected filings were extracted again.
6. **Bookkeeping.** Dual-class tickers (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) share a CIK and were
   processed twice until de-duplicated; cover pages print registrant names in capitals, so the
   S&P 500 name is used when the model echoes them; at about 30 s per filing, cue ordering,
   checkpointing and response caching were essential.

## 6. Limitations

Coverage favours product-rich filings (processing stops at the row target); only the 20 most
recent 8-Ks per company and at most two per company were considered; the model sees 10,000
characters, so an announcement deep in a long release can be missed; a new indication for an
existing drug counts as a new product; and a single 8B model was used. The backend is
pluggable (`--model`), a larger model would likely raise precision, and a labelled sample would
allow a proper precision/recall measurement.

## 7. Reproducing the run

```bash
pip install -r requirements.txt
ollama pull llama3                                   # Ollama server running
export SEC_USER_AGENT="Your Name your.email@example.com"
python run_extraction.py --target-rows 110 --max-filings 1000
```

The submitted table was produced in several steps recorded in `output/run.log`: a first run
(400 texts) stopped at 78 rows when its queue reached the zero-cue filings, was resumed with
`--max-filings 1000` to 111 rows, cleaned with `--forget CHD,META,ABT` and `--revalidate`
after the prompt fix, resumed to 112 rows and revalidated with the final rules (120 rows).
