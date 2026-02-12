# Full Test Summary — Phases 1 through 4

**Run date:** 2026-02-12
**Platform:** Linux 6.14.0-37-generic, Python 3.12.3, pytest 9.0.2
**Result:** 300 passed, 0 failed, 1 warning
**Duration:** ~4 seconds

---

## Overview

| Metric | Value |
|--------|-------|
| Total tests | 300 |
| Passed | 300 |
| Failed | 0 |
| Skipped | 0 |
| Warnings | 1 (FutureWarning: `google.generativeai` deprecated in favor of `google.genai`) |
| Test files | 15 |
| Test classes | 28 |

---

## Tests by Phase

| Phase | Description | Test files | Tests | Status |
|-------|-------------|-----------|-------|--------|
| Phase 1 | Document Ingestion | 4 files | 78 | All passing |
| Phase 2 | Chunking & Embedding | 3 files | 64 | All passing |
| Phase 3 | Hybrid Retrieval | 4 files | 81 | All passing |
| Phase 4 | LLM-Powered Processing | 4 files | 77 | All passing |

---

## Phase 1: Document Ingestion (78 tests)

Phase 1 covers PDF parsing, layout analysis, metadata extraction, and table
extraction. These are the foundation — they turn raw PDFs into structured
pages, sections, metadata, and tables that all later phases build on.

### test_pdf_parser.py — 14 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_extract_returns_all_pages` | Parser returns one `PageContent` per PDF page |
| 2 | `test_extract_contains_expected_text` | Extracted text matches known content in the test PDF |
| 3 | `test_confidence_is_valid_range` | Extraction confidence is between 0.0 and 1.0 |
| 4 | `test_native_extraction_method` | Method is set to "native" for text-based PDFs |
| 5 | `test_char_count_matches_text_length` | Reported char_count equals `len(text)` |
| 6 | `test_file_not_found_raises` | `FileNotFoundError` on missing path |
| 7 | `test_invalid_pdf_raises_runtime_error` | `RuntimeError` on corrupt/non-PDF file |
| 8 | `test_clean_text_collapses_blank_lines` | Multiple blank lines collapsed to one |
| 9 | `test_clean_text_strips_trailing_whitespace` | Trailing spaces/tabs removed per line |
| 10 | `test_clean_text_empty_input` | Empty string returns empty string |
| 11 | `test_needs_ocr_with_rich_text` | Rich text pages do NOT trigger OCR |
| 12 | `test_needs_ocr_with_sparse_text` | Sparse text pages DO trigger OCR |
| 13 | `test_ocr_fallback_called` | OCR path is invoked when native extraction is sparse |
| 14 | `test_avg_confidence_calculation` | Average confidence computed correctly across pages |

**Source module:** `src/ingestion/pdf_parser.py`
**Test patterns:** Real PDF fixture (`tests/fixtures/sample.pdf`), `@patch` for OCR fallback

### test_layout_analyzer.py — 25 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_analyze_empty_pages` | Empty input returns empty section list |
| 2 | `test_analyze_detects_sections` | Finds introduction, methodology, etc. in multi-section text |
| 3 | `test_analyze_detects_methodology` | Methodology section correctly identified |
| 4 | `test_analyze_detects_experiments` | Experiments section correctly identified |
| 5 | `test_analyze_detects_results` | Results section correctly identified |
| 6 | `test_analyze_detects_conclusion` | Conclusion section correctly identified |
| 7 | `test_sections_have_correct_order` | `order_index` is sequential |
| 8 | `test_sections_have_text_content` | Every detected section has non-empty text |
| 9 | `test_extract_references` | Reference entries extracted from bibliography section |
| 10 | `test_page_ranges_are_valid` | `page_start <= page_end` for every section |
| 11 | `test_detect_headers_numbered` | "1. Introduction" recognized as header |
| 12 | `test_detect_headers_uppercase` | "METHODOLOGY" recognized as header |
| 13 | `test_detect_headers_rejects_long_lines` | Long lines (>100 chars) rejected as headers |
| 14 | `test_detect_headers_rejects_empty` | Empty/whitespace lines rejected |
| 15 | `test_detect_headers_plain_text` | Normal sentences rejected as headers |
| 16 | `test_classify_abstract` | "Abstract" header classified as ABSTRACT |
| 17 | `test_classify_introduction` | "Introduction" header classified as INTRODUCTION |
| 18 | `test_classify_related_work` | "Related Work" classified as RELATED_WORK |
| 19 | `test_classify_methodology` | "Methodology" classified as METHODOLOGY |
| 20 | `test_classify_experiments` | "Experiments" classified as EXPERIMENTS |
| 21 | `test_classify_results` | "Results" classified as RESULTS |
| 22 | `test_classify_conclusion` | "Conclusion" classified as CONCLUSION |
| 23 | `test_classify_references` | "References" classified as REFERENCES |
| 24 | `test_classify_appendix` | "Appendix" classified as APPENDIX |
| 25 | `test_classify_unknown_returns_none` | Unrecognized header returns None |

**Source module:** `src/ingestion/layout_analyzer.py`
**Test patterns:** In-memory page text, no external dependencies

### test_metadata_extractor.py — 22 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_extract_empty_pages` | Empty pages return empty metadata |
| 2 | `test_heuristic_extracts_title` | Title extracted from first page text heuristically |
| 3 | `test_heuristic_extracts_doi` | DOI extracted via regex |
| 4 | `test_heuristic_extracts_keywords` | Keywords parsed from "Keywords:" line |
| 5 | `test_heuristic_extracts_abstract` | Abstract text extracted between markers |
| 6 | `test_heuristic_confidence` | Heuristic confidence is < 1.0 |
| 7 | `test_extract_with_llm_disabled` | LLM disabled returns heuristic-only results |
| 8 | `test_extract_skips_llm_without_api_key` | Missing API key skips LLM gracefully |
| 9 | `test_merge_prefers_llm_title` | LLM title preferred over heuristic |
| 10 | `test_merge_falls_back_to_heuristic` | Missing LLM field falls back to heuristic |
| 11 | `test_merge_prefers_llm_authors` | LLM authors preferred over heuristic |
| 12 | `test_merge_max_confidence` | Merged confidence = max(llm, heuristic) |
| 13 | `test_merge_both_empty` | Both empty results produce empty metadata |
| 14 | `test_extract_title_skips_boilerplate` | Boilerplate lines (journal names, etc.) skipped |
| 15 | `test_extract_title_empty_lines` | Blank lines do not become the title |
| 16 | `test_extract_doi_bare` | Bare DOI format `10.xxxx/yyyy` extracted |
| 17 | `test_extract_doi_with_prefix` | `doi:10.xxxx/yyyy` format extracted |
| 18 | `test_extract_doi_none` | Text without DOI returns None |
| 19 | `test_extract_keywords_semicolons` | Semicolon-separated keywords parsed |
| 20 | `test_extract_keywords_commas` | Comma-separated keywords parsed |
| 21 | `test_extract_keywords_empty` | No keyword line returns empty list |
| 22 | `test_extract_authors_from_sample_pages` | Author names extracted from page text |

**Source module:** `src/ingestion/metadata_extractor.py`
**Test patterns:** `@patch("src.ingestion.metadata_extractor.settings")`, mock Gemini responses

### test_table_extractor.py — 17 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_extract_tables_file_not_found` | `FileNotFoundError` on missing PDF |
| 2 | `test_extract_tables_no_tables` | PDF without tables returns empty list |
| 3 | `test_needs_vision_fallback_empty_table` | Empty DataFrame triggers vision fallback |
| 4 | `test_needs_vision_fallback_all_none` | All-None cells trigger vision fallback |
| 5 | `test_needs_vision_fallback_well_filled` | Well-filled table does NOT trigger fallback |
| 6 | `test_needs_vision_fallback_sparse` | Sparse table triggers vision fallback |
| 7 | `test_parse_raw_table_normal` | Normal camelot table parsed to headers + rows |
| 8 | `test_parse_raw_table_with_none_cells` | None cells replaced with empty strings |
| 9 | `test_parse_raw_table_empty` | Empty table returns empty headers and rows |
| 10 | `test_score_table_perfect` | Perfect table scores 1.0 |
| 11 | `test_score_table_empty` | Empty table scores 0.0 |
| 12 | `test_score_table_partial` | Partially filled table scores between 0 and 1 |
| 13 | `test_find_caption_with_matching_table` | "Table 1:" line matched to table on same page |
| 14 | `test_find_caption_no_match` | No matching caption returns None |
| 15 | `test_find_caption_fallback` | Fallback to nearest "Table" mention |
| 16 | `test_gemini_vision_skipped_without_api_key` | No API key skips Gemini vision gracefully |
| 17 | `test_extract_tables_with_table_pdf` | End-to-end extraction from real PDF with tables |

**Source module:** `src/ingestion/table_extractor.py`
**Test patterns:** Real fixtures, `@patch` for Gemini vision, camelot mock

---

## Phase 2: Chunking & Embedding (64 tests)

Phase 2 splits sections into semantically coherent chunks and converts them
into vector embeddings. These chunks become the atoms that retrieval searches
over.

### test_chunker.py — 29 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_chunk_produces_enriched_chunks` | Returns list of `EnrichedChunk` objects |
| 2 | `test_chunk_indices_are_sequential` | `chunk_index` is 0, 1, 2, ... |
| 3 | `test_chunk_ids_are_unique` | Every `chunk_id` is globally unique |
| 4 | `test_chunk_ids_are_deterministic` | Same input produces same `chunk_id` |
| 5 | `test_every_chunk_has_paper_id` | All chunks carry the source `paper_id` |
| 6 | `test_every_chunk_has_paper_title` | All chunks carry the paper title |
| 7 | `test_every_chunk_has_token_count` | `token_count > 0` for non-empty chunks |
| 8 | `test_empty_paper_returns_no_chunks` | Paper with no sections returns empty list |
| 9 | `test_chunks_never_mix_sections` | Each chunk belongs to exactly one section |
| 10 | `test_short_section_is_single_chunk` | Short section not split |
| 11 | `test_long_section_splits_into_multiple_chunks` | Long section split at token boundary |
| 12 | `test_table_is_single_chunk` | Table kept as one indivisible chunk |
| 13 | `test_table_chunk_contains_headers_and_rows` | Table chunk includes header and data rows |
| 14 | `test_table_chunk_includes_caption` | Table caption prepended to chunk text |
| 15 | `test_references_are_grouped` | Reference entries grouped, not split mid-entry |
| 16 | `test_reference_chunks_contain_multiple_entries` | Multiple refs in one chunk when small enough |
| 17 | `test_equations_stay_with_context` | Equations not split from surrounding text |
| 18 | `test_overlap_adds_preceding_context` | Overlap tokens from previous chunk prepended |
| 19 | `test_zero_overlap_no_duplication` | `overlap=0` means no duplicated text |
| 20 | `test_page_numbers_populated` | `page_numbers` list set correctly |
| 21 | `test_multi_page_section_has_range` | Section spanning pages has correct range |
| 22 | `test_count_tokens_approximation` | Token count within expected range |
| 23 | `test_count_tokens_minimum_one` | Minimum token count is 1 for any non-empty text |
| 24 | `test_make_id_deterministic` | `_make_id()` returns same hash for same input |
| 25 | `test_make_id_different_for_different_input` | Different inputs produce different hashes |
| 26 | `test_recursive_split_short_text` | Short text not split further |
| 27 | `test_recursive_split_paragraphs` | Splits on paragraph boundaries first |
| 28 | `test_recursive_split_sentences` | Falls back to sentence boundaries |
| 29 | `test_recursive_split_preserves_equations` | Equations kept intact during splitting |

**Source module:** `src/chunking/chunker.py`
**Test patterns:** In-memory `DetectedSection` + `ExtractedTable` objects

### test_embedding_service.py — 14 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_embed_chunks_raises_without_api_key` | `RuntimeError` when API key missing |
| 2 | `test_embed_query_raises_without_api_key` | `RuntimeError` when API key missing |
| 3 | `test_embed_chunks_raises_on_empty_list` | `ValueError` on empty chunk list |
| 4 | `test_embed_query_raises_on_empty_string` | `ValueError` on empty query |
| 5 | `test_embed_query_raises_on_whitespace` | `ValueError` on whitespace-only query |
| 6 | `test_embed_chunks_returns_vectors` | Returns list of float lists matching input count |
| 7 | `test_embed_query_returns_single_vector` | Returns one vector of correct dimension |
| 8 | `test_batch_splitting` | Large chunk lists split into batches of 100 |
| 9 | `test_embed_chunks_uses_document_task_type` | `task_type="RETRIEVAL_DOCUMENT"` for chunks |
| 10 | `test_embed_query_uses_query_task_type` | `task_type="RETRIEVAL_QUERY"` for queries |
| 11 | `test_retry_on_transient_failure` | Retries on 429/500/503 errors |
| 12 | `test_raises_after_max_retries` | Raises after exhausting retry budget |
| 13 | `test_default_model_from_settings` | Uses model name from settings by default |
| 14 | `test_custom_parameters` | Accepts custom model, dimension, batch_size |

**Source module:** `src/embedding/embedding_service.py`
**Test patterns:** `@patch("src.embedding.embedding_service.settings")`, mock `genai.embed_content`

### test_vector_store.py — 21 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_store_returns_count` | `store()` returns number of chunks stored |
| 2 | `test_store_multiple_chunks` | Can store and count multiple chunks |
| 3 | `test_store_empty_returns_zero` | Empty list returns 0 |
| 4 | `test_store_mismatched_lengths_raises` | Mismatched chunks/vectors raises `ValueError` |
| 5 | `test_store_is_idempotent` | Storing same chunk twice does not duplicate |
| 6 | `test_search_returns_results` | `search()` returns non-empty results |
| 7 | `test_search_returns_similarity_score` | Results include similarity scores in [0, 1] |
| 8 | `test_search_respects_n_results` | Returns at most `n_results` items |
| 9 | `test_search_empty_store_returns_empty` | Empty collection returns empty results |
| 10 | `test_search_metadata_populated` | Result metadata contains paper_id, section_type |
| 11 | `test_search_filter_by_paper_id` | `paper_id` filter restricts results |
| 12 | `test_search_filter_by_section_type` | `section_type` filter restricts results |
| 13 | `test_search_combined_filters` | Both filters applied together |
| 14 | `test_delete_paper_removes_chunks` | `delete_paper()` removes all chunks for a paper |
| 15 | `test_delete_paper_nonexistent_returns_zero` | Deleting nonexistent paper returns 0 |
| 16 | `test_delete_then_search_returns_empty` | Deleted chunks not returned by search |
| 17 | `test_build_where_no_filters` | No filters returns None where clause |
| 18 | `test_build_where_paper_only` | Paper-only filter builds correct where clause |
| 19 | `test_build_where_section_only` | Section-only filter builds correct where clause |
| 20 | `test_build_where_combined` | Combined filter uses `$and` operator |
| 21 | `test_chunk_to_metadata_flattens_correctly` | `EnrichedChunk` flattened to ChromaDB metadata dict |

**Source module:** `src/embedding/vector_store.py`
**Test patterns:** In-memory ChromaDB (`chromadb.Client()`), real store/search operations

---

## Phase 3: Hybrid Retrieval (81 tests)

Phase 3 builds the retrieval pipeline: BM25 sparse search, hybrid
dense+sparse fusion with RRF, LLM-based reranking, and query preprocessing
with classification and expansion.

### test_bm25_index.py — 21 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_tokenize_lowercases` | Tokenizer converts to lowercase |
| 2 | `test_tokenize_removes_punctuation` | Punctuation stripped from tokens |
| 3 | `test_tokenize_removes_stopwords` | Common stopwords removed |
| 4 | `test_tokenize_without_stopword_removal` | `remove_stopwords=False` keeps all tokens |
| 5 | `test_tokenize_empty_string` | Empty string returns empty list |
| 6 | `test_tokenize_all_stopwords` | All-stopword input returns empty list |
| 7 | `test_build_index_stores_chunks` | Indexed chunks accessible after build |
| 8 | `test_build_index_sets_is_built` | `is_built` flag set to True |
| 9 | `test_build_index_empty_raises` | Empty chunk list raises `ValueError` |
| 10 | `test_build_index_replaces_previous` | Rebuilding replaces old index |
| 11 | `test_search_returns_search_results` | Returns `SearchResult` objects |
| 12 | `test_search_before_build_raises` | Search before build raises `RuntimeError` |
| 13 | `test_search_exact_match_ranked_high` | Exact term match is top result |
| 14 | `test_search_respects_n_results` | Returns at most `n_results` items |
| 15 | `test_search_scores_normalized_0_to_1` | All scores in [0.0, 1.0] |
| 16 | `test_search_top_result_score_is_1` | Top result has score 1.0 |
| 17 | `test_search_no_match_returns_empty` | Unmatched query returns empty list |
| 18 | `test_search_empty_query_returns_empty` | Empty query returns empty list |
| 19 | `test_search_filter_by_paper_id` | `paper_id` filter restricts results |
| 20 | `test_search_filter_by_section_type` | `section_type` filter restricts results |
| 21 | `test_search_combined_filters` | Both filters applied together |

**Source module:** `src/retrieval/bm25_index.py`
**Test patterns:** In-memory `EnrichedChunk` objects, real BM25 scoring

### test_hybrid_retriever.py — 18 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_default_scores` | `RankedResult` defaults: rrf_score=0, final_score=0 |
| 2 | `test_fields_stored` | `RankedResult` stores all provided fields |
| 3 | `test_retrieve_returns_ranked_results` | Returns list of `RankedResult` objects |
| 4 | `test_retrieve_combines_dense_and_sparse` | Both dense and sparse sources queried |
| 5 | `test_rrf_formula_correct` | RRF = alpha/(k+dense_rank) + (1-alpha)/(k+sparse_rank) |
| 6 | `test_missing_from_dense_gets_penalty_rank` | Missing from dense gets penalty rank (n+1) |
| 7 | `test_missing_from_sparse_gets_penalty_rank` | Missing from sparse gets penalty rank (n+1) |
| 8 | `test_duplicate_chunk_ids_handled` | Same chunk from both sources merged, not duplicated |
| 9 | `test_retrieve_sorted_by_rrf_score` | Results sorted descending by RRF score |
| 10 | `test_retrieve_respects_top_k` | Returns at most `top_k` results |
| 11 | `test_alpha_1_pure_dense` | alpha=1.0 uses only dense scores |
| 12 | `test_alpha_0_pure_sparse` | alpha=0.0 uses only sparse scores |
| 13 | `test_default_alpha_used_when_none` | None alpha uses default from settings |
| 14 | `test_dense_failure_falls_back_to_sparse` | Dense error falls back to sparse-only |
| 15 | `test_sparse_failure_falls_back_to_dense` | Sparse error falls back to dense-only |
| 16 | `test_both_empty_returns_empty` | Both sources empty returns empty list |
| 17 | `test_paper_id_passed_to_both_sources` | paper_id filter forwarded to both sources |
| 18 | `test_section_type_passed_to_both_sources` | section_type filter forwarded to both sources |

**Source module:** `src/retrieval/hybrid_retriever.py`
**Test patterns:** Mock `VectorStore.search()` and `BM25Index.search()`, verify RRF math

### test_reranker.py — 20 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_parse_scores_valid_json` | `{"scores": [8, 6, 9]}` parsed correctly |
| 2 | `test_parse_scores_bare_list` | `[8, 6, 9]` parsed correctly |
| 3 | `test_parse_scores_with_markdown_fences` | Markdown-fenced JSON parsed correctly |
| 4 | `test_parse_scores_invalid_json_returns_empty` | Invalid JSON returns empty list |
| 5 | `test_parse_scores_wrong_length_returns_empty` | Mismatched length returns empty list |
| 6 | `test_parse_scores_clamps_out_of_range` | Scores clamped to [1, 10] |
| 7 | `test_normalize_score_1_gives_0` | Score 1 normalizes to 0.0 |
| 8 | `test_normalize_score_10_gives_1` | Score 10 normalizes to 1.0 |
| 9 | `test_normalize_score_5_point_5` | Score 5.5 normalizes to 0.5 |
| 10 | `test_rerank_updates_scores` | `final_score` updated after reranking |
| 11 | `test_rerank_combines_with_rrf_score` | final = 0.6*rrf + 0.4*rerank |
| 12 | `test_rerank_respects_top_k` | Returns at most `top_k` results |
| 13 | `test_rerank_reorders_candidates` | Candidates reordered by final score |
| 14 | `test_rerank_no_api_key_returns_original` | Missing API key returns original order |
| 15 | `test_rerank_api_failure_returns_original` | LLM failure returns original order |
| 16 | `test_rerank_parse_failure_returns_original` | Unparseable LLM output returns original |
| 17 | `test_rerank_empty_candidates_returns_empty` | Empty input returns empty list |
| 18 | `test_rerank_empty_query_returns_original` | Empty query returns original order |
| 19 | `test_strip_markdown_fences` | Code fences removed from LLM output |
| 20 | `test_strip_markdown_fences_no_fences` | Plain text returned as-is |

**Source module:** `src/retrieval/reranker.py`
**Test patterns:** Mock Gemini API via `@patch("src.retrieval.reranker.settings")`, math verification

### test_query_processor.py — 22 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_query_type_values` | `QueryType` enum has FACTUAL, CONCEPTUAL, COMPARISON, METADATA |
| 2 | `test_query_type_from_string` | String-to-enum conversion works |
| 3 | `test_classify_factual` | Factual query classified correctly |
| 4 | `test_classify_conceptual` | Conceptual query classified correctly |
| 5 | `test_classify_comparison` | Comparison query classified correctly |
| 6 | `test_classify_metadata` | Metadata query classified correctly |
| 7 | `test_classify_fallback_on_failure` | LLM failure falls back to FACTUAL |
| 8 | `test_classify_no_api_key_returns_factual` | Missing API key defaults to FACTUAL |
| 9 | `test_expand_includes_original` | Expanded queries include the original |
| 10 | `test_expand_returns_reformulations` | Returns 2-3 reformulated queries |
| 11 | `test_expand_fallback_on_failure` | LLM failure returns only original |
| 12 | `test_expand_no_api_key_returns_original` | Missing API key returns only original |
| 13 | `test_process_returns_query_result` | Full pipeline returns `ProcessedQuery` |
| 14 | `test_process_factual_sets_correct_params` | Factual type sets top_k=5, alpha=0.5 |
| 15 | `test_process_conceptual_sets_correct_params` | Conceptual type sets top_k=8, alpha=0.7 |
| 16 | `test_process_metadata_skips_retrieval` | Metadata type sets skip_retrieval=True |
| 17 | `test_process_calls_reranker` | Reranker invoked when available |
| 18 | `test_process_passes_filters` | paper_id/section_type filters forwarded |
| 19 | `test_process_empty_query_returns_early` | Empty query returns early with defaults |
| 20 | `test_retrieve_and_merge_deduplicates` | Multiple expanded queries deduplicated |
| 21 | `test_retrieve_and_merge_boosts_overlapping` | Chunks found by multiple queries get score boost |
| 22 | `test_retrieve_and_merge_single_query_no_merge` | Single query skips merge logic |

**Source module:** `src/retrieval/query_processor.py`
**Test patterns:** Mock `HybridRetriever`, `GeminiReranker`, verify strategy selection

---

## Phase 4: LLM-Powered Processing (77 tests)

Phase 4 adds the intelligence layer: a centralized Gemini client with
retry/caching/cost, structured paper extraction with dual-prompt confidence
scoring, citation-tracked QA with hallucination detection, and multi-level
summarization.

### test_gemini_client.py — 16 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_generate_returns_llm_response` | Returns well-formed `LLMResponse` |
| 2 | `test_generate_missing_api_key_raises` | `RuntimeError` when GEMINI_API_KEY not set |
| 3 | `test_generate_with_system_instruction` | System instruction passed through correctly |
| 4 | `test_generate_includes_latency` | `latency_ms >= 0` and model name populated |
| 5 | `test_retry_succeeds_after_transient_failure` | Succeeds after 2 transient failures (429) |
| 6 | `test_retry_raises_after_max_attempts` | `RuntimeError` after exhausting retry budget |
| 7 | `test_retry_does_not_retry_non_transient` | 400 errors propagate immediately |
| 8 | `test_retry_exponential_backoff_delays` | Sleep delays are 1.0s, 2.0s, 4.0s |
| 9 | `test_cache_hit_returns_cached_content` | Cache hit returns content with `cached=True`, `cost_usd=0.0` |
| 10 | `test_cache_miss_calls_api` | Cache miss calls API and stores result |
| 11 | `test_cache_key_deterministic` | Same input produces same cache key |
| 12 | `test_cache_key_varies_with_model` | Different models produce different keys |
| 13 | `test_redis_unavailable_degrades_gracefully` | Missing Redis does not raise errors |
| 14 | `test_compute_cost_zero_tokens` | Zero tokens costs $0.00 |
| 15 | `test_compute_cost_known_values` | 1M input + 1M output = $0.375 |
| 16 | `test_compute_cost_small_tokens` | 100 input + 50 output computed correctly |

**Source module:** `src/llm/gemini_client.py`
**Test patterns:** `patch.object(client, "_retry_with_backoff")` to avoid mocking namespace package imports, `patch.dict(sys.modules)` for Redis

### test_extractor.py — 23 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_extract_findings_returns_list` | Returns `list[Finding]` with correct fields |
| 2 | `test_extract_findings_confidence_bounded` | All confidence values in [0.0, 1.0] |
| 3 | `test_extract_findings_empty_text_returns_empty` | Empty text returns empty list, LLM not called |
| 4 | `test_extract_findings_api_failure_returns_empty` | LLM error returns empty list gracefully |
| 5 | `test_dual_prompt_computes_consistency` | Two identical responses yield consistency=1.0 |
| 6 | `test_extract_methodology_returns_object` | Returns `MethodologyExtraction` with fields |
| 7 | `test_extract_methodology_empty_text_returns_none` | Empty text returns None |
| 8 | `test_extract_methodology_api_failure_returns_none` | LLM error returns None |
| 9 | `test_extract_results_returns_list` | Returns `list[ResultExtraction]` |
| 10 | `test_extract_results_empty_text_returns_empty` | Empty text returns empty list |
| 11 | `test_extract_results_api_failure_returns_empty` | LLM error returns empty list |
| 12 | `test_extract_combines_all_components` | Full pipeline runs findings+methodology+results |
| 13 | `test_extract_uses_metadata_title_as_paper_id` | Paper ID from metadata.title |
| 14 | `test_extract_low_confidence_flags_review` | Low confidence sets `needs_review=True` |
| 15 | `test_source_grounding_exact_match` | Exact substring match returns 1.0 |
| 16 | `test_source_grounding_empty_quote` | Empty quote returns 0.0 |
| 17 | `test_compute_confidence_formula` | 0.4*consistency + 0.4*grounding + 0.2*completeness |
| 18 | `test_compute_confidence_partial_scores` | Partial scores computed correctly |
| 19 | `test_parse_findings_dict_format` | `{"findings": [...]}` format parsed |
| 20 | `test_parse_findings_list_format` | Bare list format parsed |
| 21 | `test_parse_findings_invalid_json` | Invalid JSON returns empty list |
| 22 | `test_strip_markdown_fences` | Code fences removed |
| 23 | `test_strip_markdown_fences_no_fences` | Plain text returned stripped |

**Source module:** `src/llm/extractor.py`
**Test patterns:** Mock `GeminiClient`, JSON responses with `_llm_response()` helper

### test_qa_engine.py — 22 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_build_context_formats_sources` | Context string has `Source [1]:`, `Source [2]:` markers |
| 2 | `test_build_context_creates_citations` | Returns `Citation` objects with correct indices |
| 3 | `test_build_context_truncates_snippets` | Citation `text_snippet` truncated to 200 chars |
| 4 | `test_answer_empty_query_returns_empty` | Empty query returns empty `QAResponse` |
| 5 | `test_answer_no_retriever_returns_no_context` | No retriever returns "No relevant context" |
| 6 | `test_answer_with_retriever_returns_response` | Full pipeline: retrieve + generate + verify |
| 7 | `test_generate_extracts_answer_from_json` | Extracts `answer` field from JSON response |
| 8 | `test_generate_returns_raw_on_non_json` | Non-JSON response returned as-is |
| 9 | `test_generate_api_failure_returns_fallback` | LLM error returns "Unable to generate" |
| 10 | `test_verify_all_supported` | All SUPPORTED claims yield faithfulness=1.0 |
| 11 | `test_verify_mixed_support` | Mixed support yields faithfulness=0.5 |
| 12 | `test_verify_empty_answer` | Empty answer yields empty verifications |
| 13 | `test_verify_api_failure` | LLM error yields empty verifications, score=0.0 |
| 14 | `test_parse_verifications_valid_json` | Standard verification JSON parsed |
| 15 | `test_parse_verifications_invalid_json` | Invalid JSON returns empty list |
| 16 | `test_split_claims_extracts_citations` | `[1]`, `[2]` citation markers extracted |
| 17 | `test_split_claims_uncited_sentence` | Uncited sentence has `None` source index |
| 18 | `test_split_claims_empty_string` | Empty string returns empty list |
| 19 | `test_split_claims_multiple_citations_takes_first` | Multiple `[1][2]` takes first |
| 20 | `test_retrieve_chunks_no_retriever` | No retriever returns empty list |
| 21 | `test_retrieve_chunks_no_paper_ids` | Calls `retriever.retrieve()` directly |
| 22 | `test_retrieve_chunks_deduplicates` | Same chunk from multiple papers deduplicated |

**Source module:** `src/llm/qa_engine.py`
**Test patterns:** Mock `GeminiClient` and `HybridRetriever`, `_ranked()` helper for `RankedResult`

### test_summarizer.py — 16 tests

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_one_line_returns_summary` | Returns `SummaryResult` with `ONE_LINE` level |
| 2 | `test_one_line_truncates_to_30_words` | Output truncated to max 30 words |
| 3 | `test_one_line_api_failure_returns_empty` | LLM error returns empty summary |
| 4 | `test_abstract_returns_summary` | Returns `SummaryResult` with `ABSTRACT` level |
| 5 | `test_abstract_is_default_level` | Default level is `abstract` |
| 6 | `test_abstract_api_failure_returns_empty` | LLM error returns empty summary |
| 7 | `test_detailed_uses_sections` | Map-reduce: summarizes each section, then synthesizes |
| 8 | `test_detailed_skips_references_and_title` | REFERENCES, TITLE, APPENDIX sections skipped |
| 9 | `test_detailed_fallback_when_no_sections` | No sections falls back to abstract-level |
| 10 | `test_detailed_empty_sections_list` | Empty sections list falls back to abstract |
| 11 | `test_detailed_section_failure_skips_section` | Failed section skipped, rest proceeds |
| 12 | `test_invalid_level_raises_value_error` | Invalid level string raises `ValueError` |
| 13 | `test_word_count_tracked` | `word_count` matches actual word count |
| 14 | `test_extract_summary_from_json` | Extracts `summary` field from JSON |
| 15 | `test_extract_summary_non_json_returns_stripped` | Non-JSON returned stripped |
| 16 | `test_strip_markdown_fences` | Markdown code fences removed |

**Source module:** `src/llm/summarizer.py`
**Test patterns:** Mock `GeminiClient`, `_section()` helper for `DetectedSection`, sequential `side_effect` for map-reduce calls

---

## Test-to-Source Coverage Map

| Source module | Test file | Tests |
|---------------|-----------|-------|
| `src/ingestion/pdf_parser.py` | `test_pdf_parser.py` | 14 |
| `src/ingestion/layout_analyzer.py` | `test_layout_analyzer.py` | 25 |
| `src/ingestion/metadata_extractor.py` | `test_metadata_extractor.py` | 22 |
| `src/ingestion/table_extractor.py` | `test_table_extractor.py` | 17 |
| `src/chunking/chunker.py` | `test_chunker.py` | 29 |
| `src/embedding/embedding_service.py` | `test_embedding_service.py` | 14 |
| `src/embedding/vector_store.py` | `test_vector_store.py` | 21 |
| `src/retrieval/bm25_index.py` | `test_bm25_index.py` | 21 |
| `src/retrieval/hybrid_retriever.py` | `test_hybrid_retriever.py` | 18 |
| `src/retrieval/reranker.py` | `test_reranker.py` | 20 |
| `src/retrieval/query_processor.py` | `test_query_processor.py` | 22 |
| `src/llm/gemini_client.py` | `test_gemini_client.py` | 16 |
| `src/llm/extractor.py` | `test_extractor.py` | 23 |
| `src/llm/qa_engine.py` | `test_qa_engine.py` | 22 |
| `src/llm/summarizer.py` | `test_summarizer.py` | 16 |
| **Total** | **15 files** | **300** |

---

## What the Tests Cover

Each test category and what aspect of the system it validates:

| Category | Count | Examples |
|----------|-------|---------|
| **Happy path** | ~90 | Correct output on valid input |
| **Edge cases** | ~60 | Empty input, whitespace, missing fields |
| **Error handling** | ~45 | API failures, missing keys, corrupt files |
| **Graceful degradation** | ~30 | Redis down, LLM errors, missing retriever |
| **Math verification** | ~25 | RRF formula, confidence formula, cost computation |
| **Filter/query logic** | ~25 | paper_id filter, section_type filter, dedup |
| **Data integrity** | ~25 | Deterministic IDs, sequential indices, unique keys |

---

## Warnings

```
tests/unit/test_embedding_service.py::TestEmbeddingService::test_embed_chunks_returns_vectors
  FutureWarning: All support for the `google.generativeai` package has ended.
  Please switch to the `google.genai` package.
```

This is a deprecation warning from the `google-generativeai` SDK. The package
still works but Google recommends migrating to `google-genai`. No functional
impact on tests.

---

## How to Run

```bash
# All tests
python3 -m pytest tests/ -v

# Single phase
python3 -m pytest tests/unit/test_pdf_parser.py tests/unit/test_layout_analyzer.py tests/unit/test_metadata_extractor.py tests/unit/test_table_extractor.py -v  # Phase 1
python3 -m pytest tests/unit/test_chunker.py tests/unit/test_embedding_service.py tests/unit/test_vector_store.py -v  # Phase 2
python3 -m pytest tests/unit/test_bm25_index.py tests/unit/test_hybrid_retriever.py tests/unit/test_reranker.py tests/unit/test_query_processor.py -v  # Phase 3
python3 -m pytest tests/unit/test_gemini_client.py tests/unit/test_extractor.py tests/unit/test_qa_engine.py tests/unit/test_summarizer.py -v  # Phase 4

# Single file
python3 -m pytest tests/unit/test_gemini_client.py -v

# With coverage (if pytest-cov installed)
python3 -m pytest tests/ --cov=src --cov-report=term-missing
```
