#!/usr/bin/env python3
"""
Extract newly announced products from SEC Form 8-K filings with a local LLM.

Typical use (defaults reproduce the submitted results):

    export SEC_USER_AGENT="Your Name your.email@example.com"   # required by SEC
    ollama pull llama3                                         # once (server must be running)
    python run_extraction.py                                   # stops at 100 product rows

Smoke test on a few companies:

    python run_extraction.py --tickers AAPL,MSFT,NVDA --per-company 1 --target-rows 3

Outputs are written to ./output (new_products.csv is the main deliverable) and the
per-filing checkpoint to ./data/processed/extractions.jsonl; re-running resumes
(successful filings are skipped, failed ones retried).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sec8k import edgar, llm
from sec8k.pipeline import PipelineConfig, run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"),
                        help="identifying User-Agent for SEC ('Name email'); default: $SEC_USER_AGENT")
    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="model context window in tokens; raised automatically if --max-chars needs more")
    parser.add_argument("--target-rows", type=int, default=100, help="stop after this many product rows (default: 100)")
    parser.add_argument("--max-filings", type=int, default=400,
                        help="maximum filings sent to the LLM in this run (default: 400)")
    parser.add_argument("--per-company", type=int, default=2, help="qualifying 8-Ks per company to consider (default: 2)")
    parser.add_argument("--feed-count", type=int, default=20, help="8-K entries requested per company feed (default: 20)")
    parser.add_argument("--items", default=",".join(edgar.PRESS_RELEASE_ITEMS),
                        help="comma-separated 8-K items to keep (default: 2.02,7.01,8.01 = press-release items)")
    parser.add_argument("--include-amendments", action="store_true", help="also process 8-K/A filings")
    parser.add_argument("--max-chars", type=int, default=10_000,
                        help="characters of filing text sent to the LLM (default: 10000)")
    parser.add_argument("--no-cue-sort", action="store_true",
                        help="process filings in recency order instead of cue-phrase order")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the second LLM call that verifies each extracted product")
    parser.add_argument("--tickers", help="comma-separated tickers to restrict the universe (testing)")
    parser.add_argument("--limit-companies", type=int, help="only use the first N companies of the list")
    parser.add_argument("--refresh-sp500", action="store_true", help="re-download the S&P 500 list")
    parser.add_argument("--fetch-only", action="store_true", help="download filings but skip the LLM")
    parser.add_argument("--data-dir", default="data", help="where the S&P list, HTTP cache and checkpoint live (default: data)")
    parser.add_argument("--output-dir", default="output", help="where the CSV files are written (default: output)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.user_agent or "@" not in args.user_agent:
        print(
            "SEC EDGAR requires an identifying User-Agent with your name and e-mail.\n"
            "  export SEC_USER_AGENT=\"Your Name your.email@example.com\"\n"
            "or pass --user-agent \"Your Name your.email@example.com\".",
            file=sys.stderr,
        )
        return 2

    cfg = PipelineConfig(
        user_agent=args.user_agent,
        model=args.model,
        ollama_host=args.ollama_host,
        num_ctx=args.num_ctx,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        target_rows=args.target_rows,
        max_filings=args.max_filings,
        per_company=args.per_company,
        feed_count=args.feed_count,
        items=tuple(i.strip() for i in args.items.split(",") if i.strip()),
        include_amendments=args.include_amendments,
        max_chars=args.max_chars,
        cue_sort=not args.no_cue_sort,
        verify=not args.no_verify,
        tickers=[t.strip() for t in args.tickers.split(",")] if args.tickers else None,
        limit_companies=args.limit_companies,
        refresh_sp500=args.refresh_sp500,
    )
    try:
        stats = run(cfg, fetch_only=args.fetch_only)
    except (llm.ExtractionError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    logging.getLogger(__name__).info(
        "done in %.0fs: %d product rows, %d LLM calls (%d errors), %d verification calls, "
        "%d texts downloaded, %d download errors",
        stats.elapsed, stats.product_rows, stats.llm_calls, stats.llm_errors,
        stats.verification_calls, stats.texts_downloaded, stats.download_errors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
