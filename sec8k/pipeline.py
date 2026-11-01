"""
sec8k.pipeline
==============

End-to-end orchestration:

1. Load the company universe (S&P 500) and resolve each ticker to a CIK.
2. For every company, list recent 8-K filings from the EDGAR Atom feed and keep
   the ones whose items can carry a press release (2.02 / 7.01 / 8.01).
3. Download the filing documents, convert them to text and score them for
   product-announcement cue phrases (used only to order the queue).
4. Send each filing to the local LLM, validate the answer (grounding checks plus
   a second LLM verification call per product), and append a JSON record to
   ``data/processed/extractions.jsonl`` (one line per filing). This file is the
   checkpoint: a re-run skips filings that were processed successfully and
   retries filings whose LLM call failed.
5. Regenerate the outputs from the checkpoint file:
   ``output/new_products.csv`` (one row per product), ``output/filings_processed.csv``
   (one row per filing), ``output/new_products_table.txt`` (pipe-separated table)
   and ``output/summary.json`` (statistics for the report).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import edgar, llm

log = logging.getLogger(__name__)

CSV_COLUMNS = ("company_name", "stock_name", "filing_time", "new_product", "product_description")

#: Stop the LLM loop after this many consecutive failures (Ollama died, machine slept).
MAX_CONSECUTIVE_ERRORS = 3

#: Rough tokens-per-character ratio of dense financial text, used to size num_ctx.
CHARS_PER_TOKEN = 3


@dataclass
class PipelineConfig:
    """All knobs of a run; defaults reproduce the submitted results."""

    user_agent: str
    model: str = "llama3"
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = 8192
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    target_rows: int = 100          # stop once this many (unique) product rows exist
    max_filings: int = 400          # filings sent to the LLM in one run, at most
    per_company: int = 2            # qualifying filings per company to consider
    feed_count: int = 20            # 8-K entries requested from the Atom feed
    items: tuple[str, ...] = edgar.PRESS_RELEASE_ITEMS
    include_amendments: bool = False
    max_chars: int = 10_000         # characters of filing text sent to the LLM
    cue_sort: bool = True           # process filings with more cue phrases first
    verify: bool = True             # second LLM call to confirm each product
    tickers: Optional[list[str]] = None      # restrict the universe (testing)
    limit_companies: Optional[int] = None    # take only the first N companies
    refresh_sp500: bool = False

    @property
    def sp500_path(self) -> Path:
        return self.data_dir / "sp500_constituents.csv"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "raw" / "http_cache"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "processed" / "extractions.jsonl"

    def required_num_ctx(self) -> int:
        """Tokens needed for the filing text, both prompts and the answer."""
        return self.max_chars // CHARS_PER_TOKEN + 1_500 + llm.DEFAULT_NUM_PREDICT


@dataclass
class RunStats:
    """Counters reported at the end of a run."""

    companies_in_universe: int = 0
    companies_resolved: int = 0
    companies_with_candidates: int = 0
    candidates: int = 0
    texts_downloaded: int = 0
    download_errors: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    verification_calls: int = 0
    llm_seconds: float = 0.0
    product_rows: int = 0
    target_reached: bool = False
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


# --------------------------------------------------------------------------- #
# Step 1-2: company universe and candidate filings
# --------------------------------------------------------------------------- #


def load_universe(
    client: edgar.EdgarClient, cfg: PipelineConfig
) -> tuple[list[edgar.Company], list[edgar.Company]]:
    """
    Load the S&P 500 list, apply ticker/limit filters and resolve CIKs.
    Returns ``(all_companies_in_file, resolved_companies)``.
    """
    universe = edgar.load_sp500(cfg.sp500_path, client=client, refresh=cfg.refresh_sp500)
    companies = universe
    if cfg.tickers:
        wanted = {t.upper() for t in cfg.tickers}
        companies = [c for c in companies if c.symbol.upper() in wanted
                     or c.symbol.upper().replace(".", "-") in wanted]
    if cfg.limit_companies:
        companies = companies[: cfg.limit_companies]
    return universe, edgar.resolve_companies(client, companies)


def build_candidates(
    client: edgar.EdgarClient, companies: list[edgar.Company], cfg: PipelineConfig
) -> list[edgar.Filing]:
    """
    For each company, list recent 8-Ks and keep up to ``cfg.per_company`` filings
    (newest first) that are original 8-Ks reporting one of the wanted items.
    The returned list is ordered round-robin by recency rank so that the first
    pass covers every company once. Accession numbers are unique in the result.
    """
    by_rank: dict[int, list[edgar.Filing]] = defaultdict(list)
    seen_accessions: set[str] = set()
    for index, company in enumerate(companies):
        try:
            filings = edgar.list_8k_filings(
                client, company.cik, company.edgar_ticker, company.edgar_title, cfg.feed_count
            )
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad company
            log.warning("feed failed for %s: %s", company.symbol, exc)
            continue
        kept = 0
        for filing in filings:
            if filing.is_amendment and not cfg.include_amendments:
                continue
            if not filing.has_any_item(cfg.items):
                continue
            if filing.accession in seen_accessions:
                continue
            seen_accessions.add(filing.accession)
            filing.rank = kept
            filing.sp500_name = company.name
            filing.sector = company.sector
            by_rank[kept].append(filing)
            kept += 1
            if kept >= cfg.per_company:
                break
        if (index + 1) % 50 == 0:
            log.info("listed filings for %d/%d companies", index + 1, len(companies))

    ordered: list[edgar.Filing] = []
    for rank in sorted(by_rank):
        ordered.extend(by_rank[rank])
    return ordered


# --------------------------------------------------------------------------- #
# Step 3: text download and ordering
# --------------------------------------------------------------------------- #


def download_texts(
    client: edgar.EdgarClient,
    candidates: list[edgar.Filing],
    limit: int,
    cfg: PipelineConfig,
    stats: RunStats,
) -> list[edgar.FilingText]:
    """Download text for up to ``limit`` candidate filings."""
    texts: list[edgar.FilingText] = []
    for filing in candidates:
        if len(texts) >= limit:
            break
        try:
            filing_text = edgar.download_filing_text(client, filing, max_chars=cfg.max_chars)
        except Exception as exc:  # noqa: BLE001
            stats.download_errors += 1
            log.warning("download failed for %s %s: %s", filing.ticker, filing.accession, exc)
            continue
        if not filing_text.text.strip():
            log.warning("no text for %s %s; skipping", filing.ticker, filing.accession)
            continue
        texts.append(filing_text)
        stats.texts_downloaded += 1
        if len(texts) % 50 == 0:
            log.info("downloaded %d filing texts", len(texts))
    return texts


def order_for_llm(texts: list[edgar.FilingText], cue_sort: bool = True) -> list[edgar.FilingText]:
    """
    Order filings for LLM processing. With ``cue_sort`` the queue is bucketed by
    cue-phrase score (3+, 1-2, 0) and, within a bucket, by recency rank and the
    original round-robin position, so results stay diverse across companies.
    Every filing in the queue is still classified by the LLM.
    """
    def bucket(score: int) -> int:
        return 2 if score >= 3 else (1 if score >= 1 else 0)

    indexed = list(enumerate(texts))
    if cue_sort:
        indexed.sort(key=lambda pair: (-bucket(pair[1].cue_score), pair[1].filing.rank, pair[0]))
    else:
        indexed.sort(key=lambda pair: (pair[1].filing.rank, pair[0]))
    return [text for _, text in indexed]


# --------------------------------------------------------------------------- #
# Step 4: LLM extraction with checkpointing
# --------------------------------------------------------------------------- #


def load_checkpoint(path: Path) -> dict[str, dict]:
    """
    Read the JSONL checkpoint into ``{accession: record}``. A later line for the
    same accession (a retried filing) replaces the earlier one; a corrupt line
    (e.g. a partial write after a crash) is skipped with a warning.
    """
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[record["accession"]] = record
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("skipping corrupt checkpoint line %d in %s: %s", number, path, exc)
    return records


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def successful_accessions(records: dict[str, dict]) -> set[str]:
    """Accessions whose LLM call succeeded; errored filings are retried on resume."""
    return {accession for accession, record in records.items() if not record.get("error")}


def process_filing(
    ollama: llm.OllamaClient, filing_text: edgar.FilingText, cfg: PipelineConfig, stats: RunStats
) -> dict:
    """Run the LLM on one filing (plus verification calls) and build the checkpoint record."""
    filing = filing_text.filing
    record = {
        "accession": filing.accession,
        "ticker": filing.ticker,
        "cik": filing.cik,
        "edgar_company": filing.company,
        "sp500_name": filing.sp500_name,
        "sector": filing.sector,
        "filing_type": filing.filing_type,
        "filing_date": filing.filing_date,
        "filing_time": filing.filed_at,
        "accepted_at": filing_text.accepted_at,
        "items": filing.items,
        "index_url": filing.index_url,
        "documents": [d.doc_type for d in filing_text.documents if d.is_primary or d.is_press_release],
        "text_chars": len(filing_text.text),
        "truncated": filing_text.truncated,
        "cue_score": filing_text.cue_score,
        "model": cfg.model,
        "prompt_version": llm.PROMPT_VERSION,
        "validation_version": llm.VALIDATION_VERSION,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "company_name": filing.sp500_name or filing.company,
        "llm_company_name": None,
        "has_new_product": False,
        "products": [],
        "rejected": [],
        "raw_response": None,
        "llm_seconds": None,
        "verify_seconds": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }
    try:
        raw, response = llm.extract_entities(
            ollama,
            filing_text.text,
            ticker=filing.ticker,
            edgar_name=filing.company,
            filing_date=filing.filing_date,
            items=filing.items,
        )
        cleaned = llm.postprocess(
            raw, filing_text.text, ticker=filing.ticker,
            fallback_company=filing.sp500_name, edgar_title=filing.company,
        )
        products: list[dict] = []
        verify_seconds = 0.0
        for product in cleaned.products:
            if cfg.verify:
                context = llm.context_window(
                    filing_text.text, product.get("evidence_span") or product.get("span")
                )
                verdict, reason, seconds = llm.verify_product(
                    ollama, cleaned.company_name, product["new_product"],
                    product["product_description"], product["evidence"], context,
                )
                stats.verification_calls += 1
                verify_seconds += seconds
                product["verified"] = verdict
                product["verification_reason"] = reason
                if not verdict:
                    product["reason"] = f"LLM verification: {reason}"
                    cleaned.rejected.append(product)
                    continue
            products.append(product)
        record.update(
            company_name=cleaned.company_name,
            llm_company_name=cleaned.llm_company_name,
            has_new_product=bool(products),
            products=products,
            rejected=cleaned.rejected,
            raw_response=response.content,
            llm_seconds=round(response.seconds, 2),
            verify_seconds=round(verify_seconds, 2),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
    except llm.ExtractionError as exc:
        record["error"] = str(exc)
    return record


def run(cfg: PipelineConfig, fetch_only: bool = False) -> RunStats:
    """Execute the whole pipeline; returns run statistics."""
    stats = RunStats()
    client = edgar.EdgarClient(cfg.user_agent, cache_dir=cfg.cache_dir)

    num_ctx = max(cfg.num_ctx, cfg.required_num_ctx())
    if num_ctx != cfg.num_ctx:
        log.warning("raising num_ctx from %d to %d so %d chars of text fit", cfg.num_ctx, num_ctx, cfg.max_chars)
    ollama = llm.OllamaClient(model=cfg.model, host=cfg.ollama_host, num_ctx=num_ctx)

    checkpoint = load_checkpoint(cfg.checkpoint_path)
    done = successful_accessions(checkpoint)
    rows = len(product_rows(checkpoint))
    log.info("checkpoint: %d filings processed (%d with errors), %d product rows",
             len(checkpoint), len(checkpoint) - len(done), rows)

    if rows >= cfg.target_rows and not fetch_only:
        log.info("target of %d product rows already reached; regenerating outputs only", cfg.target_rows)
        stats.product_rows, stats.target_reached = rows, True
        write_outputs(checkpoint, cfg, stats)
        return stats
    if not fetch_only:
        ollama.check_available()

    # 1-2. universe and candidates
    universe, companies = load_universe(client, cfg)
    stats.companies_in_universe = len(universe)
    stats.companies_resolved = len(companies)
    log.info("resolved CIKs for %d of %d companies", len(companies), len(universe))

    candidates = build_candidates(client, companies, cfg)
    stats.candidates = len(candidates)
    stats.companies_with_candidates = len({f.cik for f in candidates})
    log.info("%d candidate filings from %d companies", len(candidates), stats.companies_with_candidates)

    # 3. texts (errored filings are retried, successful ones skipped)
    pending = [f for f in candidates if f.accession not in done]
    texts = download_texts(client, pending, cfg.max_filings, cfg, stats)
    queue = order_for_llm(texts, cfg.cue_sort)
    log.info("%d filing texts ready (%d SEC requests, %d cache hits)",
             len(queue), client.requests_made, client.cache_hits)
    if fetch_only:
        return stats

    # 4. LLM
    consecutive_errors = 0
    try:
        for position, filing_text in enumerate(queue, start=1):
            if rows >= cfg.target_rows:
                log.info("target of %d product rows reached", cfg.target_rows)
                break
            filing = filing_text.filing
            if filing.accession in done:
                continue
            record = process_filing(ollama, filing_text, cfg, stats)
            append_checkpoint(cfg.checkpoint_path, record)
            checkpoint[record["accession"]] = record
            stats.llm_calls += 1
            if record["error"]:
                stats.llm_errors += 1
                consecutive_errors += 1
                log.error("[%d/%d] %-5s %s failed: %s", position, len(queue), filing.ticker,
                          filing.accession, record["error"])
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error("%d consecutive LLM failures; is Ollama still running? Stopping this run.",
                              consecutive_errors)
                    break
                continue
            consecutive_errors = 0
            done.add(record["accession"])
            stats.llm_seconds += (record["llm_seconds"] or 0.0) + (record["verify_seconds"] or 0.0)
            rows = len(product_rows(checkpoint))
            log.info(
                "[%d/%d] %-5s %s items=%s cue=%d -> %d product(s)%s (%.1fs) | rows %d/%d",
                position, len(queue), filing.ticker, filing.accession, ",".join(filing.items),
                filing_text.cue_score, len(record["products"]),
                f", {len(record['rejected'])} rejected" if record["rejected"] else "",
                (record["llm_seconds"] or 0.0) + (record["verify_seconds"] or 0.0), rows, cfg.target_rows,
            )
    except KeyboardInterrupt:
        log.warning("interrupted; writing outputs for %d processed filings", len(checkpoint))

    # 5. outputs
    stats.product_rows = rows
    stats.target_reached = rows >= cfg.target_rows
    write_outputs(checkpoint, cfg, stats)
    if not stats.target_reached:
        log.warning(
            "only %d of %d product rows so far. Re-run to continue (the checkpoint resumes), "
            "or raise --max-filings / --per-company to consider more filings.",
            rows, cfg.target_rows,
        )
    return stats


# --------------------------------------------------------------------------- #
# Maintenance: re-apply validation rules, forget filings
# --------------------------------------------------------------------------- #


def rewrite_checkpoint(path: Path, records: dict[str, dict]) -> None:
    """Replace the checkpoint file atomically with ``records`` (insertion order kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def forget(cfg: PipelineConfig, tickers: list[str]) -> int:
    """
    Drop the checkpoint records of the given tickers so those filings are
    extracted again on the next run (used after a prompt fix). Returns the count.
    """
    wanted = {t.upper() for t in tickers}
    checkpoint = load_checkpoint(cfg.checkpoint_path)
    kept = {a: r for a, r in checkpoint.items() if r.get("ticker", "").upper() not in wanted}
    dropped = len(checkpoint) - len(kept)
    rewrite_checkpoint(cfg.checkpoint_path, kept)
    log.info("forgot %d filings of %s", dropped, ",".join(sorted(wanted)))
    return dropped


def revalidate(cfg: PipelineConfig) -> RunStats:
    """
    Re-run :func:`llm.postprocess` with the current validation rules on the model
    answers stored in the checkpoint, using the cached filing texts. No extraction
    call is repeated; earlier verification verdicts are carried over, and only a
    product that was never verified before triggers a new verification call.
    """
    stats = RunStats()
    client = edgar.EdgarClient(cfg.user_agent, cache_dir=cfg.cache_dir)
    ollama = llm.OllamaClient(model=cfg.model, host=cfg.ollama_host, num_ctx=cfg.num_ctx)
    checkpoint = load_checkpoint(cfg.checkpoint_path)
    changed = 0
    for accession, record in checkpoint.items():
        if record.get("error") or not record.get("raw_response"):
            continue
        filing = edgar.Filing(
            accession=record["accession"], cik=record["cik"], ticker=record["ticker"],
            company=record["edgar_company"], filing_type=record["filing_type"],
            filing_date=record["filing_date"], filed_at=record["filing_time"],
            items=record["items"], index_url=record["index_url"],
            sp500_name=record.get("sp500_name", ""), sector=record.get("sector", ""),
        )
        text = edgar.download_filing_text(client, filing, max_chars=cfg.max_chars).text
        try:
            raw = llm.parse_json_answer(record["raw_response"])
        except llm.ExtractionError as exc:
            log.warning("cannot re-parse stored answer for %s: %s", accession, exc)
            continue
        cleaned = llm.postprocess(
            raw, text, ticker=filing.ticker, fallback_company=filing.sp500_name, edgar_title=filing.company
        )
        previous = {
            llm._normalise(p["new_product"]): p
            for p in (record.get("products") or []) + (record.get("rejected") or [])
            if "verified" in p
        }
        products: list[dict] = []
        for product in cleaned.products:
            earlier = previous.get(llm._normalise(product["new_product"]))
            if earlier is not None:
                product["verified"] = earlier["verified"]
                product["verification_reason"] = earlier.get("verification_reason", "")
            elif cfg.verify:
                context = llm.context_window(text, product.get("evidence_span") or product.get("span"))
                verdict, reason, _ = llm.verify_product(
                    ollama, cleaned.company_name, product["new_product"],
                    product["product_description"], product["evidence"], context,
                )
                stats.verification_calls += 1
                product["verified"], product["verification_reason"] = verdict, reason
            if product.get("verified") is False:
                product["reason"] = f"LLM verification: {product['verification_reason']}"
                cleaned.rejected.append(product)
                continue
            products.append(product)
        before = [p["new_product"] for p in record.get("products") or []]
        after = [p["new_product"] for p in products]
        if before != after or record.get("validation_version") != llm.VALIDATION_VERSION:
            changed += 1
            if before != after:
                log.info("%-5s %s: %s -> %s", filing.ticker, accession, before, after)
        record.update(
            company_name=cleaned.company_name,
            llm_company_name=cleaned.llm_company_name,
            has_new_product=bool(products),
            products=products,
            rejected=cleaned.rejected,
            validation_version=llm.VALIDATION_VERSION,
            revalidated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    rewrite_checkpoint(cfg.checkpoint_path, checkpoint)
    stats.product_rows = len(product_rows(checkpoint))
    stats.target_reached = stats.product_rows >= cfg.target_rows
    log.info("revalidated %d filings (%d changed): %d product rows now", len(checkpoint), changed, stats.product_rows)
    write_outputs(checkpoint, cfg, stats)
    return stats


# --------------------------------------------------------------------------- #
# Step 5: outputs
# --------------------------------------------------------------------------- #


def product_rows(records: dict[str, dict]) -> list[dict]:
    """
    Flatten checkpoint records to one dict per product, newest filing first.
    The same product announced by the same company in two filings (e.g. a launch
    press release and the following earnings release) is reported once.
    """
    rows: list[dict] = []
    seen_by_ticker: dict[str, set[str]] = defaultdict(set)
    for record in sorted(records.values(), key=lambda r: r.get("filing_time", ""), reverse=True):
        for product in record.get("products") or []:
            key = llm._normalise(product["new_product"])
            if llm.is_duplicate_name(key, seen_by_ticker[record["ticker"]]):
                continue
            seen_by_ticker[record["ticker"]].add(key)
            rows.append(
                {
                    "company_name": record["company_name"],
                    "stock_name": record["ticker"],
                    "filing_time": record["filing_time"],
                    "new_product": product["new_product"],
                    "product_description": product["product_description"],
                }
            )
    return rows


def write_outputs(records: dict[str, dict], cfg: PipelineConfig, stats: Optional[RunStats] = None) -> dict:
    """Write CSVs, the pipe-separated table and the summary; returns the summary."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = product_rows(records)

    # Main deliverable: one row per newly announced product.
    with (cfg.output_dir / "new_products.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    # The same table in the pipe-separated layout from the assignment text.
    with (cfg.output_dir / "new_products_table.txt").open("w", encoding="utf-8") as handle:
        handle.write(" | ".join(["Company Name", "Stock Name", "Filing Time", "New Product", "Product Description"]) + "\n")
        handle.write("-" * 120 + "\n")
        for row in rows:
            handle.write(" | ".join(str(row[c]).replace("|", "/") for c in CSV_COLUMNS) + "\n")

    # One row per filing analysed, including those without a product.
    filing_columns = ("accession", "ticker", "company_name", "sector", "filing_type", "filing_date",
                      "filing_time", "items", "cue_score", "text_chars", "has_new_product",
                      "n_products", "n_rejected", "llm_seconds", "verify_seconds", "error", "index_url")
    computed = {"items", "n_products", "n_rejected"}
    with (cfg.output_dir / "filings_processed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=filing_columns)
        writer.writeheader()
        for record in sorted(records.values(), key=lambda r: r.get("filing_time", ""), reverse=True):
            row = {column: record.get(column) for column in filing_columns if column not in computed}
            row["items"] = ";".join(record.get("items") or [])
            row["n_products"] = len(record.get("products") or [])
            row["n_rejected"] = len(record.get("rejected") or [])
            writer.writerow(row)

    summary = build_summary(records, cfg, stats)
    with (cfg.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    log.info("wrote %d product rows from %d filings to %s", len(rows), len(records), cfg.output_dir)
    return summary


def build_summary(records: dict[str, dict], cfg: PipelineConfig, stats: Optional[RunStats]) -> dict:
    """Aggregate statistics used in the report."""
    processed = list(records.values())
    ok = [r for r in processed if not r.get("error")]
    with_product = [r for r in ok if r.get("products")]
    seconds = [r["llm_seconds"] for r in ok if r.get("llm_seconds")]
    verify_seconds = [r.get("verify_seconds") or 0.0 for r in ok]
    rows = product_rows(records)
    kept_products = [p for r in ok for p in (r.get("products") or [])]
    rejected = [rej for r in ok for rej in (r.get("rejected") or [])]
    descriptions = [r["product_description"] for r in rows]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.model,
        "items_filter": list(cfg.items),
        "max_chars": cfg.max_chars,
        "verification_enabled": cfg.verify,
        "filings_processed": len(processed),
        "filings_with_llm_error": len(processed) - len(ok),
        "filings_with_new_product": len(with_product),
        "product_rows": len(rows),
        "products_kept_before_cross_filing_dedupe": len(kept_products),
        "companies_processed": len({r["cik"] for r in processed}),
        "companies_with_product": len({r["cik"] for r in with_product}),
        "sectors_with_product": dict(Counter(r.get("sector", "") for r in with_product).most_common()),
        "items_of_filings_with_product": dict(Counter(
            item for r in with_product for item in (r.get("items") or []) if item != "9.01"
        ).most_common()),
        "announcement_types": dict(Counter(p.get("announcement_type") or "unknown" for p in kept_products).most_common()),
        "rejected_products": len(rejected),
        "rejection_reasons": dict(Counter(
            (rej.get("reason") or "").split(":")[0] for rej in rejected
        ).most_common()),
        "products_with_span": sum(1 for p in kept_products if p.get("span")),
        "llm_seconds_mean": round(sum(seconds) / len(seconds), 1) if seconds else None,
        "verify_seconds_mean_per_filing": round(sum(verify_seconds) / len(verify_seconds), 1) if verify_seconds else None,
        "llm_seconds_total": round(sum(seconds) + sum(verify_seconds), 1),
        "description_chars_max": max((len(d) for d in descriptions), default=0),
        "filing_date_min": min((r.get("filing_date", "") for r in processed), default=None),
        "filing_date_max": max((r.get("filing_date", "") for r in processed), default=None),
        "run_stats": {
            "companies_in_universe": stats.companies_in_universe if stats else None,
            "companies_resolved": stats.companies_resolved if stats else None,
            "candidates": stats.candidates if stats else None,
            "download_errors": stats.download_errors if stats else None,
            "llm_calls_this_run": stats.llm_calls if stats else None,
            "verification_calls_this_run": stats.verification_calls if stats else None,
            "elapsed_seconds": round(stats.elapsed, 1) if stats else None,
            "target_reached": stats.target_reached if stats else None,
        },
    }
