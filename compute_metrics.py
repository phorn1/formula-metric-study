#!/usr/bin/env python3
"""Compute BLEU, Levenshtein, and CDM metrics for all formula pairs in all_formulas.json."""

import json
import os
from pathlib import Path

from scorers.text_similarity import bleu_score, levenshtein_similarity
from scorers.cdm import cdm_score

DATA_PATH = Path(__file__).parent / "all_formulas.json"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    enable_cdm = bool(os.getenv("CDM_SERVICE_URL"))
    print(f"Loaded {len(data)} formula pairs from {DATA_PATH}")
    print(
        f"CDM evaluation: {'enabled' if enable_cdm else 'disabled (set CDM_SERVICE_URL to enable)'}"
    )

    for i, row in enumerate(data, 1):
        gt = row["gt_formula"]
        pred = row["extracted_formula"]

        metrics = {
            "bleu_score": bleu_score(gt, pred),
            "levenshtein_similarity": levenshtein_similarity(gt, pred),
        }

        if enable_cdm:
            try:
                metrics["cdm_score"] = cdm_score(gt, pred)
            except Exception as e:
                print(f"  [{i}/{len(data)}] {row['gt_id']}: CDM failed: {e}")
                metrics["cdm_score"] = 0.0
        elif "metrics" in row and "cdm_score" in row["metrics"]:
            metrics["cdm_score"] = row["metrics"]["cdm_score"]

        row["metrics"] = metrics

        m = row["metrics"]
        cdm_str = f", cdm={m['cdm_score']:.4f}" if "cdm_score" in m else ""
        print(
            f"  [{i}/{len(data)}] {row['gt_id']}: "
            f"bleu={m['bleu_score']:.4f}, lev={m['levenshtein_similarity']:.4f}{cdm_str}"
        )

    # Write back
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(data)} formula pairs back to {DATA_PATH}")


if __name__ == "__main__":
    main()

