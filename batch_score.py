"""
batch_score.py
──────────────
Programmatic API for scoring many resumes at once.

Example
───────
    from batch_score import batch_score_directory

    results = batch_score_directory(
        jd="We are looking for a Python ML engineer with PyTorch, CNNs, and NLP experience.",
        resume_dir="./parsed_resumes/",
        weights={"skills": 0.30, "experience": 0.45, "projects": 0.25},
        top_n=10,
    )
    for r in results:
        print(r.as_dict())
"""

from __future__ import annotations

import json
from pathlib import Path

from scorer import ResumeData, ResumeScorer, ScoredResume, DEFAULT_WEIGHTS, DEFAULT_MODEL


def batch_score_directory(
    jd:           str | dict,
    resume_dir:   str | Path,
    *,
    glob:         str  = "*.json",
    model_name:   str  = DEFAULT_MODEL,
    weights:      dict = DEFAULT_WEIGHTS,
    top_n:        int | None = None,
    device:       str | None = None,
    use_fp16:     bool | None = None,
    scorer:       ResumeScorer | None = None,
) -> list[ScoredResume]:
    """
    Load all JSON resume files from a directory and score them.

    Parameters
    ----------
    jd          : JD as plain text or structured dict
    resume_dir  : directory containing parsed resume JSON files
    glob        : file pattern (default *.json)
    model_name  : sentence-transformers model
    weights     : section weights dict
    top_n       : return only top-N results (None = all)
    scorer      : reuse an existing ResumeScorer instance

    Returns
    -------
    Ranked list of ScoredResume objects
    """
    resume_dir = Path(resume_dir)
    files = list(resume_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"No files matching '{glob}' in {resume_dir}")

    resumes = []
    for f in files:
        try:
            with open(f) as fh:
                resumes.append(ResumeData.from_dict(json.load(fh)))
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

    if scorer is None:
        scorer = ResumeScorer(model_name=model_name, weights=weights,
                              device=device, use_fp16=use_fp16)

    results = scorer.score(jd, resumes)
    return results[:top_n] if top_n else results


def batch_score_list(
    jd:           str | dict,
    resume_dicts: list[dict],
    *,
    model_name:   str  = DEFAULT_MODEL,
    weights:      dict = DEFAULT_WEIGHTS,
    top_n:        int | None = None,
    device:       str | None = None,
    use_fp16:     bool | None = None,
    scorer:       ResumeScorer | None = None,
) -> list[ScoredResume]:
    """
    Score a list of resume dicts (e.g. straight from your parser pipeline).

    Parameters
    ----------
    jd            : JD as plain text or structured dict
    resume_dicts  : list of dicts as produced by extractor.py
    """
    if scorer is None:
        scorer = ResumeScorer(model_name=model_name, weights=weights,
                              device=device, use_fp16=use_fp16)
    resumes = [ResumeData.from_dict(d) for d in resume_dicts]
    results = scorer.score(jd, resumes)
    return results[:top_n] if top_n else results
