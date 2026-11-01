"""
sec8k.llm
=========

Named-entity extraction from 8-K text with a local LLM served by Ollama.

The model is asked to behave as an NER system with two entity types:

* ``ORGANIZATION`` – the registrant filing the 8-K (company name), and
* ``PRODUCT``      – every product or service the filing announces as *new*.

Ollama's structured-output feature (``format`` = JSON schema) guarantees that the
answer is valid JSON with the expected keys. The schema deliberately asks for the
``evidence`` sentence *before* the product name so the model commits to a quote
first and extracts the entity from it. Because small local models still
hallucinate, :func:`postprocess` then applies deterministic checks:

* a product must actually be mentioned in the filing text (grounding), with the
  character span of the mention recorded,
* the evidence sentence must occur in the text, mention the product and contain a
  launch / approval / availability cue,
* a product must not be the company itself or a financial instrument,
* descriptions are clamped to the 180-character limit and duplicates removed.

:func:`verify_product` adds a second, cheap LLM call per surviving product that
re-asks the model whether the evidence really announces a new product of the filer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

#: The assignment caps descriptions at "less than 180 characters".
MAX_DESCRIPTION_CHARS = 179

#: Generation cap; multi-product answers need ~130 tokens per product.
DEFAULT_NUM_PREDICT = 2500

#: JSON schema handed to Ollama; the model's output is constrained to match it.
#: Property order matters: Ollama's grammar follows it, so evidence comes first.
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "ORGANIZATION entity: name of the company filing the 8-K, as written in the press release",
        },
        "has_new_product": {
            "type": "boolean",
            "description": "true only if the filing announces at least one new product or service of the filer",
        },
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Verbatim sentence from the text that announces the product as new",
                    },
                    "announcement_type": {
                        "type": "string",
                        "enum": ["launch", "regulatory_approval", "availability", "unveiling"],
                    },
                    "new_product": {
                        "type": "string",
                        "description": "PRODUCT entity: the product's name exactly as written in the evidence",
                    },
                    "product_description": {
                        "type": "string",
                        "description": "Self-contained sentence saying what the product is and does, under 150 characters",
                    },
                },
                "required": ["evidence", "announcement_type", "new_product", "product_description"],
            },
        },
    },
    "required": ["company_name", "has_new_product", "products"],
}

SYSTEM_PROMPT = """You are a named-entity recognition (NER) system specialised in SEC Form 8-K filings.

You label two entity types:
- ORGANIZATION: the registrant, i.e. the company that filed this 8-K. Use the name as written in the press release (for example "Apple Inc." or "Eli Lilly and Company").
- PRODUCT: a product, service, device, drug, software, platform, vehicle or model that THIS filing announces as NEW and that belongs to the registrant: launched, introduced, unveiled, released, made commercially available, or newly approved for sale (for example an FDA approval).

The following are NOT products and must never be returned:
- financial results, revenue, sales growth, earnings, guidance, dividends, share repurchases
- debt or equity offerings, notes, credit facilities, financing
- mergers, acquisitions, divestitures, investments, partnerships or contracts by themselves
- executive appointments, board changes, shareholder votes, legal matters
- existing products that are merely mentioned, sold, or reported on (for example "iPhone revenue grew 5%")
- drug candidates still in clinical trials, trial results, regulatory submissions or anything not yet approved
- products described as planned, upcoming, or expected to launch at a future date
- products of other companies (partners, customers, suppliers, acquisition targets)
- facilities, factories, store openings, events, conferences, awards

Rules:
1. Only return products explicitly announced as new in the text. If there is none, set has_new_product to false and products to [].
2. evidence must be a verbatim sentence copied from the text; it must name the product and state the launch, approval or availability.
3. new_product must be the product name exactly as it appears in the evidence (no paraphrasing, no trademark symbols).
4. product_description is ONE self-contained sentence in plain English that says what the product is and does, at most 150 characters (for example "Once-weekly basal insulin for adults with type 2 diabetes."). Never copy a bullet fragment and never exceed 180 characters.
5. Respond with JSON only, matching the requested schema."""

USER_PROMPT_TEMPLATE = """Filer hints from EDGAR (may help you recognise the ORGANIZATION):
- ticker: {ticker}
- registered name: {edgar_name}
- filing date: {filing_date}
- 8-K items: {items}

<filing_text>
{text}
</filing_text>

Perform NER on the filing text above and return the ORGANIZATION (company_name) and every PRODUCT that the registrant announces as new, following the rules."""

#: Second-pass check per product (small prompt, fast). The model first writes what
#: the passage says and then picks one of several categories: a multi-way choice is
#: more reliable for a small model than a yes/no answer, which flips with the framing.
VERIFY_CATEGORIES: tuple[str, ...] = (
    "new_launch_or_approval",     # the only category that keeps the product
    "existing_product_reported",
    "pipeline_or_future_product",
    "third_party_product",
    "not_a_product",
)

VERIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "what_the_passage_says": {
            "type": "string",
            "description": "one sentence: what the passage states about the candidate",
        },
        "category": {"type": "string", "enum": list(VERIFY_CATEGORIES)},
    },
    "required": ["what_the_passage_says", "category"],
}

VERIFY_SYSTEM_PROMPT = (
    "You are a careful analyst of SEC Form 8-K press releases. You classify how a "
    "passage refers to a candidate product. Answer with JSON."
)

VERIFY_USER_TEMPLATE = """Company filing the 8-K: {company}
Candidate: "{product}"
Proposed description: {description}
Sentence quoted by the extractor: "{evidence}"

Passage from the filing around that sentence:
\"\"\"{context}\"\"\"

Pick the ONE category that best describes how the passage refers to "{product}":
- new_launch_or_approval: {company} launched, introduced, unveiled, released, rolled out, made available, began shipping, or obtained regulatory approval (or a new indication) for it. New versions, generations and models count.
- existing_product_reported: it is an existing product that is only reported on (revenue, sales, demand, volume, pricing) with no launch or approval statement.
- pipeline_or_future_product: it is a drug candidate in clinical trials, awaiting approval, or a product described as planned, upcoming or expected later.
- third_party_product: it belongs to another company (partner, customer, supplier, acquisition target).
- not_a_product: it is not a product or service at all: a financial instrument, cost or productivity programme, initiative, strategy, event, facility, partnership, award, or a generic category of offerings rather than a named product.

First state in one sentence what the passage says about "{product}", then give the category."""


class ExtractionError(RuntimeError):
    """Raised when the LLM cannot be queried or returns unusable output."""


@dataclass
class LLMResponse:
    """Raw answer of the model plus timing/token statistics reported by Ollama."""

    content: str
    seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model: str = ""
    done_reason: Optional[str] = None   # "stop" normally, "length" when num_predict was hit


class OllamaClient:
    """
    Minimal client for Ollama's ``/api/chat`` endpoint.

    Parameters
    ----------
    model:
        Name of a model that has been pulled, e.g. ``"llama3"``.
    host:
        Base URL of the Ollama server.
    num_ctx:
        Context window in tokens; must be large enough for the filing text plus
        the generated answer.
    temperature:
        0 makes extraction deterministic.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.0,
        timeout: int = 600,
        keep_alive: str = "30m",
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_predict = num_predict

    def check_available(self) -> None:
        """Raise :class:`ExtractionError` unless the server runs and the model is pulled."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ExtractionError(
                f"cannot reach Ollama at {self.host} ({exc}). Start it with `ollama serve`."
            ) from exc
        names = {m.get("name", "") for m in response.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            raise ExtractionError(
                f"model '{self.model}' is not pulled (have: {sorted(names)}). "
                f"Run `ollama pull {self.model}`."
            )

    def chat(
        self,
        system: str,
        user: str,
        schema: Optional[dict] = None,
        num_predict: Optional[int] = None,
    ) -> LLMResponse:
        """Send one chat turn and return the model's answer."""
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict or self.num_predict,
            },
        }
        if schema is not None:
            payload["format"] = schema

        started = time.monotonic()
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise ExtractionError(f"Ollama request failed: {exc}") from exc
        result = LLMResponse(
            content=data.get("message", {}).get("content", ""),
            seconds=time.monotonic() - started,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            model=data.get("model", self.model),
            done_reason=data.get("done_reason"),
        )
        if result.done_reason == "length":
            raise ExtractionError(
                f"model output truncated at num_predict={num_predict or self.num_predict} tokens"
            )
        return result


def build_user_prompt(
    text: str, ticker: str, edgar_name: str, filing_date: str, items: list[str]
) -> str:
    """Fill the user prompt template."""
    return USER_PROMPT_TEMPLATE.format(
        ticker=ticker or "unknown",
        edgar_name=edgar_name or "unknown",
        filing_date=filing_date or "unknown",
        items=", ".join(items) if items else "unknown",
        text=text,
    )


def parse_json_answer(content: str) -> dict:
    """
    Parse the model's JSON. With schema-constrained decoding this is a plain
    ``json.loads``; the fallback extracts the first ``{...}`` block in case the
    model wrapped the JSON in prose or code fences.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ExtractionError(f"model did not return JSON: {content[:200]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionError("model JSON is not an object")
    return data


def extract_entities(
    client: OllamaClient,
    text: str,
    ticker: str = "",
    edgar_name: str = "",
    filing_date: str = "",
    items: Optional[list[str]] = None,
    retries: int = 1,
) -> tuple[dict, LLMResponse]:
    """
    Run the NER prompt on ``text`` and return ``(parsed_json, raw_response)``.

    A failed or unparsable call is retried ``retries`` times (after a short pause,
    which covers transient server errors) before raising :class:`ExtractionError`.
    """
    user_prompt = build_user_prompt(text, ticker, edgar_name, filing_date, items or [])
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat(SYSTEM_PROMPT, user_prompt, schema=EXTRACTION_SCHEMA)
            return parse_json_answer(response.content), response
        except ExtractionError as exc:
            last_error = exc
            log.warning("extraction attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(3)
    raise ExtractionError(str(last_error))


def verify_product(
    client: OllamaClient,
    company: str,
    product: str,
    description: str,
    evidence: str,
    context: str,
) -> tuple[bool, str, float]:
    """
    Second-pass check: ask the model how ``context`` refers to ``product``
    (see :data:`VERIFY_CATEGORIES`). Returns ``(keep, "category: summary", seconds)``;
    ``keep`` is True only for ``new_launch_or_approval``.
    """
    user_prompt = VERIFY_USER_TEMPLATE.format(
        company=company, product=product, description=description, evidence=evidence, context=context
    )
    response = client.chat(VERIFY_SYSTEM_PROMPT, user_prompt, schema=VERIFY_SCHEMA, num_predict=250)
    data = parse_json_answer(response.content)
    category = str(data.get("category") or "unknown")
    verdict = category == "new_launch_or_approval"
    return verdict, f"{category}: {data.get('what_the_passage_says', '')}", response.seconds


# --------------------------------------------------------------------------- #
# Validation of the model's answer
# --------------------------------------------------------------------------- #

_STOPWORDS = {"the", "and", "for", "with", "its", "new", "of", "a", "an", "in", "to", "by"}

#: Whole product names that denote financial events/instruments rather than products.
_NON_PRODUCT_NAME_PATTERNS = (
    re.compile(
        r"^(?:(?:q[1-4]|first|second|third|fourth|fiscal|full|record|quarterly|annual|monthly|special|cash|\d{4}|fy\d*)[ -]?(?:quarter|year)?[ ]?)*"
        r"(?:financial )?(?:results|earnings|guidance|outlook|dividends?|revenues?|net income|eps)$"
    ),
    re.compile(r"\b(?:senior|convertible|unsecured|subordinated|exchangeable) notes?\b"),
    re.compile(r"\b(?:debentures|credit facility|term loan|share repurchase|stock repurchase|buyback)\b"),
    re.compile(r"^(?:the )?(?:merger|acquisition|divestiture|offering|transaction|partnership|agreement)$"),
)

_PLACEHOLDER_NAMES = {"the company", "company", "registrant", "the registrant", "unknown", "n a", "none", ""}

#: A sentence announcing a new product normally contains one of these cues.
_LAUNCH_CUE_RE = re.compile(
    r"\b(?:launch\w*|introduc\w*|unveil\w*|debut\w*|releas\w*|roll(?:s|ed|ing)?[- ]?out|rollout|"
    r"available|availability|approv\w*|clearance|cleared|authori[sz]\w*|new|all-new|"
    r"next[- ]generation|commercial\w*|expand\w* (?:its |the )?(?:portfolio|line-?up|offering)s?|"
    r"began shipping|begins shipping|started shipping)\b",
    re.IGNORECASE,
)


@dataclass
class CleanedExtraction:
    """Validated extraction for one filing."""

    company_name: str
    llm_company_name: str = ""
    products: list[dict] = field(default_factory=list)   # kept products
    rejected: list[dict] = field(default_factory=list)   # dropped products with reason

    @property
    def has_new_product(self) -> bool:
        return bool(self.products)


def _normalise(value: str) -> str:
    """Lower-case and keep only alphanumerics so that spelling variants match."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _bounded(token: str) -> str:
    """Regex for ``token`` delimited by non-alphanumerics (works on normalised text)."""
    return rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"


def is_grounded(name: str, normalised_text: str, min_token_share: float = 0.75) -> bool:
    """
    True if ``name`` occurs in the (normalised) text: either verbatim with word
    boundaries or, for multi-word names the model slightly rephrased, if at least
    ``min_token_share`` of its meaningful tokens co-occur within an 80-character
    window. Single-token names must match verbatim.
    """
    key = _normalise(name)
    if not key:
        return False
    if re.search(_bounded(key), normalised_text):
        return True
    tokens = [t for t in key.split() if len(t) > 1 and t not in _STOPWORDS]
    if len(tokens) < 2:
        return False
    for match in re.finditer(_bounded(tokens[0]), normalised_text):
        window = normalised_text[max(0, match.start() - 80): match.end() + 80]
        found = 1 + sum(1 for t in tokens[1:] if re.search(_bounded(t), window))
        if found / len(tokens) >= min_token_share:
            return True
    return False


def find_span(name: str, text: str) -> Optional[tuple[int, int]]:
    """
    Locate the first mention of ``name`` in the original ``text`` and return its
    character span (NER-style offsets). Tokens may be separated by up to three
    non-alphanumeric characters so that "MacBook Pro" also matches "MacBook® Pro".
    """
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    if not tokens:
        return None
    pattern = r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]{0,3}".join(re.escape(t) for t in tokens) + r"(?![A-Za-z0-9])"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def locate_evidence(evidence: str, text: str) -> Optional[tuple[int, int]]:
    """
    Find where the quoted evidence sits in the text. The full quote is tried first;
    if the model altered a few words, its first or last eight tokens are used.
    """
    span = find_span(evidence, text)
    if span:
        return span
    tokens = re.findall(r"[a-z0-9]+", evidence.lower())
    if len(tokens) < 5:
        return None
    for size in (8, 5):
        for probe in (" ".join(tokens[:size]), " ".join(tokens[-size:])):
            span = find_span(probe, text)
            if span:
                return span
    return None


#: End of a sentence: terminal punctuation followed by whitespace and a capital (or a
#: bullet). Single-letter abbreviations ("U.S.") and "Inc."/"Corp."/"Ltd."/"Co."/"No."
#: are not sentence ends.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!Inc\.)(?<!Corp\.)(?<!Ltd\.)(?<!\bCo\.)(?<!\bNo\.)"
    r"\s+(?=[A-Z\u2022\"\u201c])|\u2022|\s\|\s"
)


def sentence_bounds(text: str, span: tuple[int, int], max_reach: int = 400) -> tuple[int, int]:
    """
    Extend ``span`` to the boundaries of the sentence (or bullet) that contains it,
    looking at most ``max_reach`` characters to each side. Semicolons do not end a
    sentence here because press releases list several announcements with them.
    """
    start, end = span
    left = text[max(0, start - max_reach): start]
    right = text[end: end + max_reach]
    left_matches = list(_SENTENCE_BOUNDARY_RE.finditer(left))
    new_start = max(0, start - max_reach) + (left_matches[-1].end() if left_matches else 0)
    right_match = _SENTENCE_BOUNDARY_RE.search(right)
    new_end = end + (right_match.start() if right_match else len(right))
    return new_start, new_end


def evidence_grounded(evidence: str, normalised_text: str, min_token_share: float = 0.85) -> bool:
    """
    True if the evidence sentence really comes from the text: verbatim after
    normalisation, or with at least ``min_token_share`` of its tokens present.
    """
    key = _normalise(evidence)
    if len(key) < 20:
        return False
    if key in normalised_text:
        return True
    tokens = [t for t in key.split() if len(t) > 2]
    if len(tokens) < 4:
        return False
    hits = sum(1 for t in tokens if re.search(_bounded(t), normalised_text))
    return hits / len(tokens) >= min_token_share


def clamp_description(description: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """
    Tidy a description: collapse whitespace, drop trailing bullet punctuation,
    capitalise the first letter and cut at a word boundary so ``len <= limit``.
    """
    description = " ".join(str(description).split()).strip(" ;:,-•")
    if description:
        description = description[0].upper() + description[1:]
    if len(description) <= limit:
        return description
    cut = description[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def resolve_company_name(
    model_name: str,
    ticker: str,
    sp500_name: str,
    edgar_title: str,
    normalised_text: str,
) -> str:
    """
    Decide the company name written to the CSV. The model's ORGANIZATION entity is
    used when it is a real name found in the text; when the model echoes the
    ticker, a placeholder, the all-caps EDGAR/cover-page form, or something not in
    the text, the S&P 500 security name (proper case) is used instead.
    """
    name = " ".join(str(model_name or "").split())
    key = _normalise(name)
    unusable = (
        key in _PLACEHOLDER_NAMES
        or key == _normalise(ticker)
        or len(key) < 2
        or not is_grounded(name, normalised_text)
    )
    all_caps = name.isupper() and len(re.findall(r"[A-Za-z]", name)) > 3
    if unusable or all_caps:
        return sp500_name or edgar_title or name
    return name


def postprocess(
    raw: dict,
    text: str,
    ticker: str = "",
    fallback_company: str = "",
    edgar_title: str = "",
) -> CleanedExtraction:
    """
    Validate the model's JSON against the filing text.

    Parameters
    ----------
    raw:
        Parsed JSON from :func:`extract_entities`.
    text:
        The filing text that was sent to the model (used for grounding checks).
    ticker:
        EDGAR ticker (a model answer equal to it is not a company name).
    fallback_company:
        S&P 500 security name, used when the model's company name is unusable.
    edgar_title:
        Registrant title from EDGAR; products equal to the company are rejected.
    """
    normalised_text = _normalise(text)
    llm_company = " ".join(str(raw.get("company_name") or "").split())
    company_name = resolve_company_name(llm_company, ticker, fallback_company, edgar_title, normalised_text)
    company_keys = {_normalise(company_name), _normalise(llm_company), _normalise(fallback_company), _normalise(edgar_title)}
    company_keys.discard("")
    model_says_none = raw.get("has_new_product") is False

    cleaned = CleanedExtraction(company_name=company_name, llm_company_name=llm_company)
    seen: set[str] = set()
    for item in raw.get("products") or []:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("new_product") or "").split()).strip("™®© ")
        description = clamp_description(item.get("product_description") or "")
        evidence = " ".join(str(item.get("evidence") or "").split())
        key = _normalise(name)

        # Where does the quoted evidence sit? Press releases often pack several
        # announcements into one semicolon-separated sentence, so the product name
        # and the launch cue are checked in the whole sentence that contains the
        # quote rather than in the quote alone.
        evidence_span = locate_evidence(evidence, text) if evidence else None
        if evidence_span:
            window_start, window_end = sentence_bounds(text, evidence_span)
            window = text[window_start:window_end]
        else:
            window_start, window = 0, evidence
        normalised_window = _normalise(window)

        reason = None
        if not key:
            reason = "empty product name"
        elif key in seen:
            reason = "duplicate"
        elif key in company_keys:
            reason = "product name is the company itself"
        elif any(p.search(key) for p in _NON_PRODUCT_NAME_PATTERNS):
            reason = "financial instrument/event, not a product"
        elif model_says_none:
            reason = "model flagged has_new_product=false"
        elif not is_grounded(name, normalised_text):
            reason = "product name not found in filing text"
        elif evidence_span is None and not evidence_grounded(evidence, normalised_text):
            reason = "evidence sentence not found in filing text"
        elif not is_grounded(name, normalised_window):
            reason = "evidence does not mention the product"
        elif not _LAUNCH_CUE_RE.search(window):
            reason = "evidence has no launch/approval/availability cue"
        elif not description:
            reason = "empty description"

        # NER span: the mention closest to the evidence, else the first mention.
        local = find_span(name, window) if evidence_span else None
        span = (window_start + local[0], window_start + local[1]) if local else find_span(name, text)
        record = {
            "entity_type": "PRODUCT",
            "new_product": name,
            "product_description": description,
            "announcement_type": item.get("announcement_type"),
            "evidence": evidence,
            "evidence_span": evidence_span,
            "span": span,
        }
        if reason:
            record["reason"] = reason
            cleaned.rejected.append(record)
        else:
            seen.add(key)
            cleaned.products.append(record)
    return cleaned


def context_window(text: str, span: Optional[tuple[int, int]], radius: int = 400) -> str:
    """Return the text around ``span`` (or the beginning of the text) for verification."""
    if span is None:
        return text[: 2 * radius]
    start, end = span
    return text[max(0, start - radius): min(len(text), end + radius)]
