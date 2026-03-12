"""CDM (Character Detection Matching) metric for formula evaluation.

Wraps the CDM evaluation from https://github.com/opendatalab/UniMERNet/tree/main/cdm.
Requires a running CDM service (set CDM_SERVICE_URL env var).
"""

import os

import requests


def cdm_score(gt_formula: str, pred_formula: str) -> float:
    """Compute CDM F1 score between two LaTeX formulas via the CDM service."""
    cdm_service_url = os.getenv("CDM_SERVICE_URL")
    if not cdm_service_url:
        raise ValueError(
            "CDM_SERVICE_URL environment variable is required.\n"
            "CDM evaluation requires a separate local service installation.\n"
            "See https://github.com/opendatalab/UniMERNet/tree/main/cdm"
        )

    response = requests.post(cdm_service_url, json={'gt': gt_formula, 'pred': pred_formula})
    response.raise_for_status()
    return response.json()['cdm_f1']