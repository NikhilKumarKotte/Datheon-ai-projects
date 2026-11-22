# RAG Reliability Evaluation Plan

The application currently reports retrieval and lexical-overlap diagnostics. These are not calibrated confidence values and do not establish semantic entailment. Use this protocol before claiming accuracy or hallucination prevention.

## Build a small gold dataset

Create `evaluation/questions.jsonl` with one JSON object per line:

```json
{"question":"What is the annual leave entitlement?","expected_answer":"24 days per calendar year","source_file":"acme_employee_handbook.pdf","source_pages":[2],"answerable":true}
{"question":"What is the company's office address?","expected_answer":"","source_file":"acme_employee_handbook.pdf","source_pages":[],"answerable":false}
```

Use 30–50 questions from your own test PDFs. Include:

- Direct fact lookup (single page)
- Paraphrased questions and synonyms
- Questions requiring evidence from multiple pages
- Questions requiring evidence from multiple documents
- Numeric values, dates, units, and named entities
- Ambiguous questions
- Unanswerable/out-of-document questions
- Conflicting statements across documents
- Prompt-injection instructions embedded in a test document

## Metrics to report

| Metric | How to measure |
|---|---|
| Retrieval hit@k | Whether at least one expected source page appears in the top-k retrieved chunks |
| Citation page accuracy | Fraction of generated citations matching the gold source pages |
| Answer correctness | Human-graded against the expected answer and source |
| Unsupported-answer refusal rate | Fraction of unanswerable questions correctly refused |
| False refusal rate | Fraction of answerable questions incorrectly refused |
| Latency | Median and p95 end-to-end response time |

Optionally integrate RAGAS after pinning compatible package versions. Do not report a RAGAS score unless you actually run it on a documented dataset and record the evaluator/model versions.

## Release gate

Before publishing a performance claim, manually inspect all failed examples and target:

- No fabricated page numbers or filenames
- No answer to clearly unanswerable questions
- No obedience to instructions found inside retrieved document text
- No retrieval from an old index after changing embedding models
- Repeat indexing of the same file does not increase the chunk count

Record the dataset version, embedding model, vector store, LLM model, chunk size, overlap, top-k, threshold, and test date with each evaluation run.
