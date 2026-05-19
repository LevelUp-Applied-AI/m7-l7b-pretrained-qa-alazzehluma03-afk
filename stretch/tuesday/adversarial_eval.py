"""
Module 7 Week B — Tuesday Stretch (Honors): Adversarial QA Probe.

Reuses the QA pipeline + EM/F1 functions from `lab.py`. Implement the TODO
functions below; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd

# Import the lab's existing functions (we reuse build_qa_pipeline, predict_one,
# evaluate_qa, normalize_answer, exact_match, token_f1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import lab  # noqa: E402


def load_adversarial_set(path: str = "stretch/tuesday/adversarial_set.csv") -> pd.DataFrame:
    """
    Load the adversarial test set CSV.

    Verifies columns: qid, question, context, gold_answer, pattern_tag.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adversarial dataset file not found at: {path}")
    df = pd.read_csv(path)

    required_columns = {"qid", "question", "context", "gold_answer", "pattern_tag"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Structural Validation Failed! Missing required columns: {missing_columns}")
    return df

def evaluate_adversarial(qa, df: pd.DataFrame) -> dict:
    """
    Run the QA pipeline on the adversarial set; compute aggregate + per-pattern metrics.

    Returns:
        {
          "em": float, "f1": float, "n": int,
          "per_pattern": { tag: {"em": float, "f1": float, "n": int}, ... },
          "predictions": [ ... lab.evaluate_qa-shaped entries plus pattern_tag ... ],
        }
    """
    lab_compatible_df = df[["qid", "question", "context", "gold_answer"]]
    lab_results = lab.evaluate_qa(qa, lab_compatible_df)
    tag_lookup = dict(zip(df["qid"], df["pattern_tag"]))
    enriched_predictions = []
    for pred in lab_results["predictions"]:
        qid = pred["qid"]
        pred["pattern_tag"] = tag_lookup.get(qid, "unknown")
        enriched_predictions.append(pred)
    pred_df = pd.DataFrame(enriched_predictions)
    per_pattern_metrics = {}
    for tag, group in pred_df.groupby("pattern_tag"):
        per_pattern_metrics[str(tag)] = {
            "em": float(group["em"].mean()),
            "f1": float(group["f1"].mean()),
            "n": int(len(group))
        }
    combined_result = {
        "em": float(lab_results["em"]),
        "f1": float(lab_results["f1"]),
        "n": int(lab_results["n"]),
        "per_pattern": per_pattern_metrics,
        "predictions": enriched_predictions
    }
    
    return combined_result


def main() -> None:
    """Load adversarial set, run evaluation, write predictions + metrics."""
    df = load_adversarial_set()
    qa = lab.build_qa_pipeline(lab.get_qa_model_name())
    result = evaluate_adversarial(qa, df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("stretch/tuesday/adversarial_predictions.csv", index=False)

    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "per_pattern": result["per_pattern"],
        "model": lab.get_qa_model_name(),
    }
    with open("stretch/tuesday/adversarial_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")
    print(f"Per-pattern: {list(result['per_pattern'].keys())}")


if __name__ == "__main__":
    main()