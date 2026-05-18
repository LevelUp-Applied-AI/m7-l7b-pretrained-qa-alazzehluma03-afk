# QA Pipeline Evaluation Report

## 1. Dataset Description
dataset consists of a 1,000-example benchmark sourced from  tech and entertainment news articles. This data selected to stress-test extractive QA on entity-rich journalistic prose.

## 2. Model Configuration

- distilbert-base-cased-distilled-squad
- https://huggingface.co/distilbert-base-cased-distilled-squad


## 3. Aggregate Metrics
- **Exact Match (EM):** 0.34 (34.40%)
- **Token-Level F1:** 0.47 (47.05%)

**Metric Gap Analysis:** The ~13 pp gap between F1 and EM is systematic: the model consistently finds the right text neighbourhood but misjudges span boundaries, either truncating or extending the predicted answer relative to the gold annotation.

## 4. Failure-Mode Taxonomy

### A. Missing Long-Form Modifier Clutter
The model isolates the head noun phrase but drops trailing descriptive clauses that are part of the gold answer.

- **qid:** `NEWS_0184_Q2`
- **Question:** *What does the G155 portable game room pack?*
- **Gold Answer:** `a built-in 15.5-inch LED HD screen, stereo speakers and dual 3.5mm headphone jacks`
- **Predicted Answer:** `a built-in 15.5-inch LED HD screen`

### B. Distractor Entity Selection
When a paragraph contains multiple named entities of the same type, the model picks the wrong one, matching the expected answer type but missing the contextual logic.

- **qid:** `NEWS_0221_Q5`
- **Question:** *Who is fast-tracking to get married?*
- **Gold Answer:** `Jade Goody`
- **Predicted Answer:** `Tweed`

### C. Over-Extraction of Contextual Verbs
The model targets the correct entity but absorbs surrounding verb phrases into the span, returning an action description instead of the entity itself.

- **qid:** `NEWS_0118_Q5`
- **Question:** *What does John mayer have?*
- **Gold Answer:** `granuloma`
- **Predicted Answer:** `bowed out of a series of concerts`

## 5. Domain Judgment

I would not ship this model for **medical literature extraction**. An EM of 34.40% means roughly two-thirds of answers are not verbatim correct, which is unacceptable when extracted spans feed dosage calculations or diagnostic criteria where a truncated answer can silently omit a critical qualifier. Beyond raw accuracy, the model offers no calibrated no-answer support: it forces an extraction even when the context is ambiguous, removing the safety valve that clinical workflows depend on to flag unanswerable queries for human review.