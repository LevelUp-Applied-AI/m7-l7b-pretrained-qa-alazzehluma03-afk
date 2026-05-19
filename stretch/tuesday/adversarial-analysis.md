# Adversarial QA Probe — Analysis Memo

> Replace each placeholder section. Memo target: ~1 page. The TA rubric rewards specificity grounded in your data.

## 1. Hypothesis

State your targeted failure mode operationally:
- **Input pattern:** The context paragraph contains multiple named entities of the exact same semantic type (e.g., multiple corporate names or multiple person names) organized within a passive voice structure or complex syntactic reversal clause.
- **Output pattern:** The QA model exhibits **Distractor Entity Selection Bias**. It extracts a incorrect *Distractor* entity instead of the *Gold Answer*. While it successfully matches the expected answer type (e.g., predicting a Person for a "Who" question), it completely misses the internal contextual logic and grammatical modifiers.
- **Why you hypothesize this:** The pre-trained QA model (`distilbert-base-cased-distilled-squad`) over-relies on a superficial type-matching shortcut combined with linear positional priority. When multiple entities of the same type compete in close proximity, the token-classification head lacks the depth of structural dependency processing to correctly bind actions to nouns, easily shifting its prediction boundaries toward the wrong entity if sentence order is inverted.

## 2. Set Design

- Total examples: _(30)_
- Tags used: _(
  - `distractor-same-sentence` (10 examples)
  - `distractor-cross-sentence` (10 examples)
  - `distractor-syntactic-reversal` (7 examples)
  - `control-clear-target` (3 examples))_
- Why these tags: _(
     - `distractor-same-sentence` tests if clause-level proximity to an entity of the same type triggers incorrect selection.
     - `distractor-cross-sentence` measures if isolating the distractor in an adjacent sentence helps the model defend its span boundary.
     - `distractor-syntactic-reversal` explicitly tests if switching active/passive voices exposes the limits of the type-matching heuristic.
     )_
- Control examples: _( 3 control examples (`control-clear-target`) containing only one clear instance of the requested entity type. These isolate text parsing length from logical token competition, confirming that entity type density and structure are the true discriminators of failure.)_

## 3. Results

- Aggregate EM: _(0.8667)_; Aggregate F1: _(0.8667)_
- Lab 7B baseline (from your `qa_metrics.json`): EM _(0.3440)_; F1 _(0.4705)_
- Per-pattern_tag breakdown:

| Pattern | n | EM | F1 | vs. baseline |
|---|---|---|---|---|
| `control-clear-target` | 3 | 1.0000 | 1.0000 | +0.6560 |
| `distractor-cross-sentence` | 10 | 1.0000 | 1.0000 | +0.6560 |
| `distractor-same-sentence` | 10 | 1.0000 | 1.0000 | +0.6560 |
| `distractor-syntactic-reversal` | 7 | 0.4286 | 0.4286 | +0.0846 |

Cite at least 3 specific (qid, question, gold, predicted) tuples that illustrate the patterns:

- **(qid)** _question_ → gold: _gold_, predicted: _pred_. _(commentary)_
- _(repeat)_

**EX_22** _Who acquired Red Hat?_ → gold: _Red Hat_, predicted: _IBM_.

Commentary: The context states: "IBM was successfully acquired by Red Hat during the massive open-source consolidation phase." Both entities match the expected corporate entity type requested by the question. Because the model relies on positional proximity heuristics, it gets confused by the passive voice inversion ("was successfully acquired by") and pulls the incorrect distractor entity (IBM) as the buyer, failing to parse the underlying contextual and structural relationship.

**EX_23** _Who defeated Garry Kasparov?_ → gold: _Garry Kasparov_, predicted: _Deep Blue_.

Commentary: The context states: "Deep Blue was totally defeated by Garry Kasparov in their historic initial chess match rematch." The model exhibits a type-matching failure mode here; it successfully targets a candidate of the requested entity type but picks the wrong name. The model defaults to selecting the sentence's grammatical subject (Deep Blue) instead of tracking the passive agent introduced after the preposition "by".

**EX_25** _Who replaced Parag Agrawal?_ → gold: _Parag Agrawal_, predicted: _Elon Musk_.

Commentary: The context states: "Elon Musk was officially replaced by Parag Agrawal during the chaotic transition phase of management." This instance directly validates the Distractor Entity Selection vulnerability initially highlighted in your baseline report (NEWS_0221_Q5). When multiple entity tokens of the same type compete, the model strips out the passive verb modifier dependency paths ("was officially replaced by") and extracts the distractor entity (Elon Musk) due to its linear priority at the beginning of the text sequence.

## 4. Production Defense

Pick **one** specific engineering action that follows from your findings. Examples (don't list all five — pick one and reason concretely):

- Confidence-threshold filter that routes below-threshold queries to humans.
- Retraining with adversarial data added to the fine-tuning set.
- Replacing the QA model with one trained for no-answer support.
- Restricting the QA model to only contexts that pass an upstream filter.
- Shrinking the production input distribution to exclude the failure pattern.

I pich this action  `Retraining with adversarial data added to the fine-tuning set.`
Our per-pattern metrics clearly prove that while the model handles simple entity proximity perfectly (scoring 1.0000 on distractor-same-sentence), it is critically vulnerable to structural changes, with performance plunging to 0.4286 under distractor-syntactic-reversal. Augmenting our fine-tuning dataset with an automated 15% allocation of active/passive contrastive entity pairs directly penalizes word-order and position-based shortcuts. This structural training explicitly forces the model's self-attention matrices to converge on deep dependency paths and relational modifiers rather than shallow entity-type matching.
