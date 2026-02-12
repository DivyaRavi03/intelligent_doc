"""Prompt templates for LLM-powered extraction and generation.

All prompts are stored as module-level constants to enable easy testing,
versioning, and potential A/B testing of prompt variants.  Placeholders
use Python ``str.format()`` syntax (e.g. ``{text}``, ``{query}``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

EXTRACT_KEY_FINDINGS = """\
You are an expert research paper analyst. Extract the key findings from \
the following research paper text.

For each finding, provide:
- "claim": a concise statement of the finding
- "supporting_quote": an exact quote from the text that supports this claim
- "confidence": your confidence (0.0-1.0) that this is a genuine key finding

Return ONLY valid JSON in this format:
{{"findings": [{{"claim": "...", "supporting_quote": "...", "confidence": 0.9}}]}}

---
{text}
---"""

EXTRACT_KEY_FINDINGS_ALT = """\
You are a systematic reviewer analyzing a research paper. Identify all \
significant findings, results, and conclusions stated in the text below.

For each finding:
- "claim": state the finding clearly in one sentence
- "supporting_quote": copy the exact text that supports this claim
- "confidence": rate 0.0-1.0 how certain you are this is a key finding

Return ONLY valid JSON:
{{"findings": [{{"claim": "...", "supporting_quote": "...", "confidence": 0.9}}]}}

---
{text}
---"""

EXTRACT_METHODOLOGY = """\
You are a research methodology expert. Extract the methodology details from \
the following research paper text.

Extract:
- "approach": the main methodological approach or technique
- "datasets": list of datasets used
- "tools": list of tools, frameworks, or libraries mentioned
- "eval_metrics": list of evaluation metrics used

Return ONLY valid JSON:
{{"approach": "...", "datasets": ["..."], "tools": ["..."], "eval_metrics": ["..."]}}

---
{text}
---"""

EXTRACT_RESULTS = """\
You are a quantitative research analyst. Extract all quantitative results \
from the following research paper text.

For each result, provide:
- "metric_name": the evaluation metric (e.g., F1, accuracy, BLEU)
- "value": the reported value (include units/percentage)
- "baseline": the baseline or comparison value if mentioned, or null
- "improvement": the improvement over baseline if mentioned, or null
- "table_reference": reference to the table containing this result, or null

Return ONLY valid JSON:
{{"results": [{{"metric_name": "...", "value": "...", "baseline": null, \
"improvement": null, "table_reference": null}}]}}

---
{text}
---"""

EXTRACT_METADATA = """\
You are a metadata extraction system for academic research papers.
Given the following text from a paper, extract:
- "title": the paper title
- "authors": list of author full names
- "abstract": the abstract text
- "doi": the DOI if present
- "journal": the journal or conference name if present
- "publication_date": publication date/year if present
- "keywords": list of keywords if present

Return ONLY valid JSON. Use null for missing strings, empty arrays for missing lists.

---
{text}
---"""

# ---------------------------------------------------------------------------
# QA prompts
# ---------------------------------------------------------------------------

QA_ANSWER = """\
You are a research paper question-answering system. Answer the question \
based ONLY on the provided context passages. Cite your sources using [N] \
notation where N is the source number.

Rules:
1. Only use information from the provided context
2. Cite every factual claim with [N] referring to the source passage
3. If the context does not contain enough information, say so explicitly
4. Do not make up information not present in the context

Context:
{context}

Question: {query}

Answer (cite sources with [N]):"""

QA_VERIFY_CLAIMS = """\
You are a fact-checking system. For each claim in the answer, verify \
whether it is supported by the cited source passage.

Answer to verify:
{answer}

Source passages:
{sources}

For each claim, determine:
- "claim": the specific factual claim
- "cited_source_index": which source [N] was cited (integer or null if uncited)
- "status": one of "SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"
- "explanation": brief explanation of your verdict

Return ONLY valid JSON:
{{"verifications": [{{"claim": "...", "cited_source_index": 1, \
"status": "SUPPORTED", "explanation": "..."}}]}}"""

# ---------------------------------------------------------------------------
# Summarization prompts
# ---------------------------------------------------------------------------

SUMMARIZE_ONE_LINE = """\
Summarize the following research paper in exactly one sentence \
(maximum 30 words). Capture the core contribution or finding.

Return ONLY valid JSON: {{"summary": "..."}}

---
{text}
---"""

SUMMARIZE_ABSTRACT = """\
Write a comprehensive summary of the following research paper in \
100-150 words. Cover the main problem, approach, key results, and \
significance.

Return ONLY valid JSON: {{"summary": "..."}}

---
{text}
---"""

SUMMARIZE_SECTION = """\
Summarize the following section of a research paper in 2-4 sentences. \
Preserve key details and quantitative results.

Section title: {section_title}
Section type: {section_type}

Return ONLY valid JSON: {{"summary": "..."}}

---
{text}
---"""

SUMMARIZE_SYNTHESIZE = """\
You are given summaries of individual sections from a research paper. \
Synthesize them into a single coherent detailed summary of 300-500 words.

The summary should:
1. Open with the paper's main contribution
2. Cover methodology and approach
3. Present key results with specific numbers
4. Discuss significance and limitations

Return ONLY valid JSON: {{"summary": "..."}}

Section summaries:
{section_summaries}"""

# ---------------------------------------------------------------------------
# Hallucination / verification prompts
# ---------------------------------------------------------------------------

HALLUCINATION_CHECK = """\
You are a verification system. Given a claim extracted from a paper and \
the original text, determine if the claim is grounded in the source text.

Claim: {claim}
Source text: {source_text}

Is this claim supported by the source text?
Return ONLY valid JSON: {{"supported": true, "explanation": "..."}}"""
