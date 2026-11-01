"""
sec8k.edgar
===========

A small, polite client for SEC EDGAR that gives the pipeline everything it needs
from the two public endpoints used in the course's Week-7 data-source notebook:

1. ``https://www.sec.gov/files/company_tickers.json``
   maps a stock ticker to the company's Central Index Key (CIK),
   e.g. ``AAPL -> 0000320193``.
2. ``https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=8-K&count=<n>&output=atom``
   an Atom feed listing a company's most recent Form 8-K filings, including the
   filing timestamp, accession number, the 8-K *item* numbers and a link to the
   filing index page.

From each feed entry the client follows ``filing-href`` to the filing index page,
selects the primary 8-K document plus every press-release exhibit (``EX-99.*``),
downloads them and converts the HTML to plain text with BeautifulSoup.

SEC fair-access rules (https://www.sec.gov/os/accessing-edgar-data) require a
``User-Agent`` header that identifies the requester (name and e-mail) and allow at
most 10 requests per second. :class:`EdgarClient` enforces both and caches every
response on disk, so re-running the pipeline never downloads the same filing twice
(feeds and the ticker map expire after 24 hours so new filings are picked up).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SEC_BASE_URL = "https://www.sec.gov"

#: Ticker -> CIK lookup table (one JSON document for every registrant with a ticker).
COMPANY_TICKERS_URL = f"{SEC_BASE_URL}/files/company_tickers.json"

#: Atom feed of a company's most recent 8-K filings.
BROWSE_EDGAR_ATOM_URL = (
    f"{SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany"
    "&CIK={cik}&type=8-K&count={count}&output=atom"
)

#: Public, regularly refreshed list of S&P 500 constituents (CSV).
SP500_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)

#: Feeds and the ticker map change daily; filing documents never change.
FEED_MAX_AGE_SECONDS = 24 * 3600

#: 8-K items under which companies attach press releases, which is where new-product
#: announcements appear: 2.02 Results of Operations, 7.01 Regulation FD Disclosure,
#: 8.01 Other Events. Items such as 5.02 (officer changes) or 5.07 (shareholder
#: votes) essentially never announce products, so they are skipped by default.
PRESS_RELEASE_ITEMS: tuple[str, ...] = ("2.02", "7.01", "8.01")

#: File extensions that contain readable text (as opposed to images, PDFs, XBRL).
TEXTUAL_EXTENSIONS: tuple[str, ...] = (".htm", ".html", ".txt")

#: HTTP statuses worth retrying: SEC throttling (403/429) and transient front-end errors.
RETRY_STATUSES: tuple[int, ...] = (403, 429, 500, 502, 503, 504)

#: Regular expressions for phrases that typically accompany a new-product
#: announcement. They are only used to *prioritise* which filings reach the LLM
#: first; every filing that is processed is still classified by the LLM.
PRODUCT_CUE_PATTERNS: tuple[str, ...] = (
    r"\blaunch(?:es|ed|ing)?\b",
    r"\bintroduc(?:es|ed|ing|tion)\b",
    r"\bunveil(?:s|ed|ing)?\b",
    r"\bdebut(?:s|ed|ing)?\b",
    r"\bnew (?:product|service|offering|platform|device|model)s?\b",
    r"\bnow available\b",
    r"\bcommercial(?:ly)? availab",
    r"\broll(?:s|ed|ing)?[- ]?out\b",
    r"\bnext[- ]generation\b",
    r"\bfirst[- ]of[- ]its[- ]kind\b",
    r"\bFDA (?:approv|clear)\w*",
    r"\b(?:regulatory|marketing) (?:approval|authori[sz]ation|clearance)\b",
    r"\bCE mark\b",
    r"\brelease of (?:its|the|a|an) new\b",
    r"\bexpand(?:s|ed|ing)? (?:its |the )?(?:product )?(?:portfolio|line-?up|offering)s?\b",
    r"\bannounc\w+ (?:the )?(?:availability|launch|introduction|release)\b",
)
_PRODUCT_CUE_RES = tuple(re.compile(p, re.IGNORECASE) for p in PRODUCT_CUE_PATTERNS)

_ATOM_FIELDS = (
    "accession-number",
    "filing-date",
    "filing-href",
    "filing-type",
    "form-name",
    "items-desc",
    "updated",
)

#: SGML wrapper text that EDGAR prepends to each document ("EX-99.1 2 file.htm EX-99.1 Document").
_SGML_HEADER_RE = re.compile(
    r"^(?:EX-[\w.]+|8-K(?:/A)?)\s+\d+\s+\S+\.(?:htm|html|txt)\s+(?:(?:EX-[\w.]+|8-K(?:/A)?)\s+)?(?:Document\s+)?",
    re.IGNORECASE,
)
_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_SEC_BLOCK_MARKERS = ("Request Rate Threshold Exceeded", "Undeclared Automated Tool")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Company:
    """A company from the S&P 500 list, enriched with its EDGAR identifiers."""

    symbol: str          # ticker as listed in the S&P 500 file, e.g. "BRK.B"
    name: str            # security name from the S&P 500 file, e.g. "Berkshire Hathaway"
    sector: str = ""     # GICS sector
    cik: str = ""        # 10-digit zero-padded CIK, filled from company_tickers.json
    edgar_ticker: str = ""  # ticker spelling used by EDGAR, e.g. "BRK-B"
    edgar_title: str = ""   # company title as registered with the SEC


@dataclass
class Filing:
    """One Form 8-K filing as described by an entry of the browse-edgar Atom feed."""

    accession: str       # e.g. "0000320193-26-000018"
    cik: str
    ticker: str          # EDGAR ticker spelling
    company: str         # EDGAR registrant title
    filing_type: str     # "8-K" or "8-K/A"
    filing_date: str     # "YYYY-MM-DD"
    filed_at: str        # ISO-8601 timestamp from the feed's <updated> element
    items: list[str]     # e.g. ["2.02", "9.01"]
    index_url: str       # filing index page (…-index.htm)
    rank: int = 0        # 0 = most recent qualifying filing of this company
    sp500_name: str = ""
    sector: str = ""

    @property
    def is_amendment(self) -> bool:
        """True for 8-K/A filings, which only amend an earlier report."""
        return self.filing_type.upper().endswith("/A")

    def has_any_item(self, items: Iterable[str]) -> bool:
        """True if the filing reports at least one of the given 8-K item numbers."""
        wanted = set(items)
        return any(item in wanted for item in self.items)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilingDocument:
    """One row of the filing index page's document table."""

    seq: str
    description: str
    filename: str
    doc_type: str        # "8-K", "EX-99.1", "GRAPHIC", "EX-101.SCH", ...
    url: str

    @property
    def is_textual(self) -> bool:
        return self.filename.lower().endswith(TEXTUAL_EXTENSIONS)

    @property
    def is_primary(self) -> bool:
        """The main 8-K (or 8-K/A) document."""
        return self.doc_type.upper().startswith("8-K") and self.is_textual

    @property
    def is_press_release(self) -> bool:
        """Exhibit 99.x documents are the press releases attached to the 8-K."""
        return self.doc_type.upper().startswith("EX-99") and self.is_textual


@dataclass
class FilingText:
    """Plain-text content of a filing, ready to be sent to the LLM."""

    filing: Filing
    text: str                          # composed text (8-K body + exhibits), truncated
    body_text: str = ""                # full text of the primary 8-K document
    exhibit_texts: list[tuple[str, str]] = field(default_factory=list)  # (type, text)
    accepted_at: Optional[str] = None  # "YYYY-MM-DD HH:MM:SS" acceptance time from index
    documents: list[FilingDocument] = field(default_factory=list)
    truncated: bool = False
    cue_score: int = 0                 # number of product-announcement cue phrases


# --------------------------------------------------------------------------- #
# HTTP client with SEC fair-access behaviour
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Blocks so that successive calls are at least ``1/max_per_second`` apart."""

    def __init__(self, max_per_second: float = 8.0) -> None:
        self.min_interval = 1.0 / max_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class EdgarClient:
    """
    HTTP client for SEC EDGAR that (1) sends the required identifying User-Agent,
    (2) stays under the 10 requests/second limit, (3) retries on throttling and
    transient server errors and (4) caches every successful response on disk.

    Parameters
    ----------
    user_agent:
        Must identify you, e.g. ``"Jane Doe jane.doe@example.edu"``. SEC blocks
        anonymous or browser-like user agents.
    cache_dir:
        Directory for the on-disk response cache (``None`` disables caching).
    max_per_second:
        Request rate ceiling; 8/s keeps a safe margin below SEC's limit of 10/s.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: Optional[Path | str] = None,
        max_per_second: float = 8.0,
        timeout: int = 30,
        max_retries: int = 4,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with your name and e-mail address, "
                "e.g. 'Jane Doe jane.doe@example.edu'. Set the SEC_USER_AGENT "
                "environment variable or pass --user-agent."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(max_per_second)
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests_made = 0
        self.cache_hits = 0

    # -- caching ----------------------------------------------------------- #

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.bin"

    @staticmethod
    def _cache_is_fresh(path: Path, max_age: Optional[float]) -> bool:
        if not path.exists():
            return False
        if max_age is None:
            return True
        return (time.time() - path.stat().st_mtime) <= max_age

    # -- requests ---------------------------------------------------------- #

    def get(self, url: str, use_cache: bool = True, max_age: Optional[float] = None) -> bytes:
        """
        GET ``url`` and return the response body, honouring cache and rate limit.

        ``max_age`` (seconds) makes a cached copy older than that expire; ``None``
        means the cached copy never expires (right for immutable filing documents).
        """
        cache_path = self._cache_path(url)
        if use_cache and cache_path and self._cache_is_fresh(cache_path, max_age):
            self.cache_hits += 1
            return cache_path.read_bytes()

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                response = self.session.get(url, timeout=self.timeout)
                self.requests_made += 1
            except requests.RequestException as exc:  # network hiccup -> retry
                last_error = exc
                log.warning("request error for %s (%s); retrying", url, exc)
                time.sleep(2 * (attempt + 1))
                continue

            if response.status_code == 200:
                if cache_path:
                    cache_path.write_bytes(response.content)
                return response.content

            if response.status_code in RETRY_STATUSES:
                body = response.text[:2000]
                if any(marker in body for marker in _SEC_BLOCK_MARKERS):
                    # SEC blocks a client for about ten minutes after it exceeds the
                    # rate limit or uses an undeclared User-Agent; wait it out.
                    wait = 60 * (attempt + 1)
                    log.error(
                        "SEC rejected the request (%s). Check SEC_USER_AGENT; waiting %ss",
                        response.status_code, wait,
                    )
                else:
                    wait = 5 * (attempt + 1)
                    log.warning("HTTP %s for %s; backing off %ss", response.status_code, url, wait)
                time.sleep(wait)
                last_error = requests.HTTPError(f"HTTP {response.status_code}")
                continue

            response.raise_for_status()

        raise RuntimeError(f"giving up on {url}: {last_error}")

    def get_text(self, url: str, use_cache: bool = True, max_age: Optional[float] = None) -> str:
        """GET ``url`` and decode it as UTF-8 (replacing undecodable bytes)."""
        return self.get(url, use_cache=use_cache, max_age=max_age).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Endpoint 1: ticker -> CIK
# --------------------------------------------------------------------------- #


def load_ticker_map(client: EdgarClient) -> dict[str, dict]:
    """
    Download ``company_tickers.json`` and return ``{TICKER: {cik, ticker, title}}``.

    The CIK is zero-padded to 10 digits, which is the form EDGAR URLs expect.
    """
    data = json.loads(client.get(COMPANY_TICKERS_URL, max_age=FEED_MAX_AGE_SECONDS))
    ticker_map: dict[str, dict] = {}
    for record in data.values():
        ticker = str(record["ticker"]).upper()
        ticker_map[ticker] = {
            "cik": f"{int(record['cik_str']):010d}",
            "ticker": ticker,
            "title": record["title"],
        }
    return ticker_map


def lookup_cik(ticker_map: dict[str, dict], symbol: str) -> Optional[dict]:
    """
    Find a ticker in the map, tolerating the two common spellings of share
    classes (S&P files use ``BRK.B`` while EDGAR uses ``BRK-B``).
    """
    symbol = symbol.upper().strip()
    for candidate in (symbol, symbol.replace(".", "-"), symbol.replace("-", ".")):
        if candidate in ticker_map:
            return ticker_map[candidate]
    return None


# --------------------------------------------------------------------------- #
# Endpoint 2: 8-K Atom feed
# --------------------------------------------------------------------------- #


def _local_name(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def list_8k_filings(
    client: EdgarClient,
    cik: str,
    ticker: str = "",
    company: str = "",
    count: int = 20,
) -> list[Filing]:
    """
    Return the ``count`` most recent 8-K filings for ``cik`` from the browse-edgar
    Atom feed, newest first. Returns an empty list if the feed cannot be parsed
    (for example when EDGAR answers with an HTML error page).
    """
    url = BROWSE_EDGAR_ATOM_URL.format(cik=cik, count=count)
    raw = client.get(url, max_age=FEED_MAX_AGE_SECONDS)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        log.warning("could not parse Atom feed for CIK %s: %s", cik, exc)
        return []

    filings: list[Filing] = []
    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue
        fields: dict[str, str] = {}
        for element in entry.iter():
            name = _local_name(element.tag)
            if name in _ATOM_FIELDS and element.text:
                fields[name] = element.text.strip()
        if "accession-number" not in fields or "filing-href" not in fields:
            continue
        filings.append(
            Filing(
                accession=fields["accession-number"],
                cik=cik,
                ticker=ticker,
                company=company,
                filing_type=fields.get("filing-type", "8-K"),
                filing_date=fields.get("filing-date", ""),
                filed_at=fields.get("updated", fields.get("filing-date", "")),
                items=re.findall(r"\d+\.\d+", fields.get("items-desc", "")),
                index_url=fields["filing-href"],
            )
        )
    return filings


# --------------------------------------------------------------------------- #
# Filing index page -> documents -> plain text
# --------------------------------------------------------------------------- #


def fetch_filing_documents(
    client: EdgarClient, filing: Filing
) -> tuple[list[FilingDocument], Optional[str]]:
    """
    Parse the filing index page and return its document table plus the
    acceptance timestamp shown on the page.

    Links to the primary document go through EDGAR's inline-XBRL viewer
    (``/ix?doc=/Archives/...``), which needs JavaScript; the prefix is removed so
    the raw HTML document is downloaded instead.
    """
    html = client.get_text(filing.index_url)
    soup = BeautifulSoup(html, "html.parser")

    documents: list[FilingDocument] = []
    for table in soup.find_all("table", class_="tableFile"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            anchor = cells[2].find("a")
            if not anchor or not anchor.get("href"):
                continue
            href = anchor["href"]
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            documents.append(
                FilingDocument(
                    seq=cells[0].get_text(strip=True),
                    description=cells[1].get_text(" ", strip=True),
                    filename=anchor.get_text(strip=True),
                    doc_type=cells[3].get_text(strip=True),
                    url=href if href.startswith("http") else SEC_BASE_URL + href,
                )
            )

    accepted = None
    match = re.search(r"Accepted</div>\s*<div class=\"info\">\s*([^<]+?)\s*<", html)
    if match:
        accepted = match.group(1)
    return documents, accepted


def html_to_text(html: str) -> str:
    """
    Convert an EDGAR HTML document to plain text.

    Inline-XBRL documents start with a hidden ``<ix:header>`` block of tagging
    metadata (contexts, note identifiers, ...) that would otherwise pollute the
    text, so that block, every ``display:none`` element, scripts and styles are
    removed first. The remaining text is joined with spaces, whitespace runs are
    collapsed and EDGAR's SGML document header ("EX-99.1 2 file.htm ...") is cut.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "title", "ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE_RE):
        tag.decompose()
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return _SGML_HEADER_RE.sub("", text, count=1).strip()


def product_cue_score(text: str) -> int:
    """Count product-announcement cue phrases in ``text`` (used only for ordering)."""
    return sum(len(pattern.findall(text)) for pattern in _PRODUCT_CUE_RES)


def excerpt_body(body_text: str, budget: int) -> str:
    """
    Shorten the 8-K body to ``budget`` characters while keeping the two parts that
    matter: the cover page up to the registrant's name (for the ORGANIZATION
    entity) and the narrative that starts at the first "Item X.XX" heading, which
    says what the filing is about. Everything in between (check boxes, listed
    securities table) is dropped.
    """
    if len(body_text) <= budget:
        return body_text
    name_match = re.search(r"\(Exact name of registrant", body_text, re.IGNORECASE)
    head_end = min(len(body_text), name_match.end() + 60 if name_match else 600)
    head = body_text[:head_end]
    item_match = re.search(r"\bItem\s+[1-9]\.\d\d\b", body_text)
    if item_match and item_match.start() > head_end:
        rest_budget = max(0, budget - len(head) - 7)
        return head + " [...] " + body_text[item_match.start(): item_match.start() + rest_budget]
    return body_text[:budget]


def compose_filing_text(
    body_text: str,
    exhibit_texts: list[tuple[str, str]],
    max_chars: int = 10_000,
    body_chars: int = 2_000,
) -> tuple[str, bool]:
    """
    Assemble the text sent to the LLM within a character budget.

    Press releases (exhibits) carry the actual announcement, so they get most of
    the budget; the 8-K body is reduced with :func:`excerpt_body` but always
    included because it names the registrant and the 8-K item. When a filing has
    no exhibit the whole body is used.

    Returns ``(text, truncated)``.
    """
    truncated = False
    parts: list[str] = []
    body_budget = body_chars if exhibit_texts else max_chars
    if body_text:
        excerpt = excerpt_body(body_text, body_budget)
        truncated = len(excerpt) < len(body_text)
        parts.append("FORM 8-K BODY:\n" + excerpt)

    remaining = max_chars - sum(len(p) for p in parts)
    for doc_type, text in exhibit_texts:
        if remaining <= 200:
            truncated = True
            break
        header = f"\n\n{doc_type} (press release exhibit):\n"
        chunk = text[: max(0, remaining - len(header))]
        if len(chunk) < len(text):
            truncated = True
        parts.append(header + chunk)
        remaining -= len(header) + len(chunk)
    return "".join(parts), truncated


def download_filing_text(
    client: EdgarClient,
    filing: Filing,
    max_chars: int = 10_000,
    body_chars: int = 2_000,
) -> FilingText:
    """
    Download the primary 8-K document and all EX-99 press-release exhibits of a
    filing, convert them to plain text and compose the LLM input.
    """
    documents, accepted = fetch_filing_documents(client, filing)

    primary = next((d for d in documents if d.is_primary), None)
    exhibits = [d for d in documents if d.is_press_release]

    body_text = html_to_text(client.get_text(primary.url)) if primary else ""
    exhibit_texts = [(d.doc_type, html_to_text(client.get_text(d.url))) for d in exhibits]

    text, truncated = compose_filing_text(body_text, exhibit_texts, max_chars, body_chars)
    return FilingText(
        filing=filing,
        text=text,
        body_text=body_text,
        exhibit_texts=exhibit_texts,
        accepted_at=accepted,
        documents=documents,
        truncated=truncated,
        cue_score=product_cue_score(text),
    )


# --------------------------------------------------------------------------- #
# Company universe: S&P 500
# --------------------------------------------------------------------------- #


def load_sp500(
    path: Path | str,
    client: Optional[EdgarClient] = None,
    refresh: bool = False,
) -> list[Company]:
    """
    Load the S&P 500 constituent list from ``path`` (a CSV with ``Symbol`` and
    ``Security`` columns). If the file is missing or ``refresh`` is True, download
    the public dataset first.
    """
    path = Path(path)
    if refresh or not path.exists():
        if client is None:
            raise FileNotFoundError(f"{path} not found and no client given to download it")
        log.info("downloading S&P 500 constituents to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.get(SP500_CONSTITUENTS_URL, use_cache=False))

    companies: list[Company] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            companies.append(
                Company(
                    symbol=row["Symbol"].strip(),
                    name=row["Security"].strip(),
                    sector=(row.get("GICS Sector") or "").strip(),
                )
            )
    return companies


def resolve_companies(client: EdgarClient, companies: list[Company]) -> list[Company]:
    """
    Fill ``cik``, ``edgar_ticker`` and ``edgar_title`` using company_tickers.json.

    Companies listed under two share classes (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) share
    one CIK and therefore one set of filings; only the first ticker is kept so a
    filing is never processed twice.
    """
    ticker_map = load_ticker_map(client)
    resolved: list[Company] = []
    seen_ciks: set[str] = set()
    for company in companies:
        record = lookup_cik(ticker_map, company.symbol)
        if record is None:
            log.warning("no CIK found for %s (%s); skipping", company.symbol, company.name)
            continue
        if record["cik"] in seen_ciks:
            log.info("%s shares CIK %s with an earlier ticker; skipping duplicate share class",
                     company.symbol, record["cik"])
            continue
        seen_ciks.add(record["cik"])
        company.cik = record["cik"]
        company.edgar_ticker = record["ticker"]
        company.edgar_title = record["title"]
        resolved.append(company)
    return resolved
