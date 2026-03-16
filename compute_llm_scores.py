#!/usr/bin/env python3
"""Compute LLM-as-a-judge scores for all formula pairs in all_formulas.json."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_PATH = Path(__file__).parent / "all_formulas.json"
MODEL = "openai/gpt-5.4"
MAX_WORKERS = 8
MAX_RETRIES = 10

PROMPT = """\
You are a mathematical formula evaluator. Your task is to determine if the extracted formula correctly represents the ground truth formula, focusing on both semantic meaning AND proper mathematical notation.

Ground Truth Formula:
{gt_formula}

Extracted Formula:
{extracted_formula}

Evaluate the extracted formula using the following criteria:
1. Correctness: Are the mathematical symbols, variables, and operations accurately preserved?
2. Completeness: Are all parts of the formula present without omissions?
3. Semantic equivalence: Does the extracted formula convey the same mathematical meaning?

Assign a score from 0 to 10, where 10 is a perfect match."""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}


def evaluate_formula(
    client: OpenAI,
    model: str,
    gt_formula: str,
    extracted_formula: str,
) -> int:
    prompt = PROMPT.format(gt_formula=gt_formula, extracted_formula=extracted_formula)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Score",
                        "strict": True,
                        "schema": SCORE_SCHEMA,
                    },
                },
            )
            return json.loads(response.choices[0].message.content)["score"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                raise


def main():
    judge_label = MODEL

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} formula pairs from {DATA_PATH}")

    for row in data:
        row["llm_scores"] = [
            s for s in row["llm_scores"] if s["judge_model"] != judge_label
        ]

    print(f"Evaluating {len(data)} pairs with {judge_label}")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    lock = threading.Lock()
    done = 0

    def process(row: dict) -> None:
        nonlocal done
        score = evaluate_formula(
            client,
            MODEL,
            row["gt_formula"],
            row["extracted_formula"],
        )
        score_entry = {
            "judge_model": judge_label,
            "score": max(0, min(10, score)),
        }
        with lock:
            row["llm_scores"].append(score_entry)
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            done += 1
            print(
                f"  [{done}/{len(data)}] {row['gt_id']}: score={score_entry['score']}"
            )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, row): row for row in data}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                row = futures[future]
                print(f"  FAILED {row['gt_id']}: {exc}")

    print(f"Done. {done}/{len(data)} pairs scored.")


if __name__ == "__main__":
    main()

