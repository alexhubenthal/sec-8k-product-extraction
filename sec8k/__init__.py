"""
sec8k
=====

Extract newly announced products from SEC Form 8-K filings with a local large
language model and write the results to a CSV file.

Modules
-------
- :mod:`sec8k.edgar`    – polite SEC EDGAR client (ticker -> CIK, 8-K Atom feed,
                          filing documents -> plain text).
- :mod:`sec8k.llm`      – NER-style extraction with a local Ollama model
                          (schema-constrained JSON) plus validation / grounding checks.
- :mod:`sec8k.pipeline` – orchestration: candidate selection, checkpointing,
                          CSV / pipe-table / summary outputs.
"""

__version__ = "1.0.0"
