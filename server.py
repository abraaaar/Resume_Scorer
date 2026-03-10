from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

from scorer import ResumeScorer, ResumeData, DEFAULT_WEIGHTS, DEFAULT_MODEL


class ScoreRequest(BaseModel):
    jd: Union[str, dict]
    resumes: List[dict]
    weights: dict | None = None
    model: str | None = None
    top: int | None = None  # return only top-N results


app = FastAPI(
    title="Resume Scoring API",
    description="Expose the existing Python resume scorer to the frontend via HTTP",
    version="0.1",
)

# allow all origins so that the React dev server can reach us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/score")
async def score(request: ScoreRequest):
    """Score a job description against one or more resume dicts.

    The request body should be JSON with the following keys::

        {
            "jd": "job description text or structured dict",
            "resumes": [ {resume-dict}, ... ],
            "weights": {"skills":0.3, "experience":0.45, "projects":0.25},  # optional
            "model": "all-MiniLM-L6-v2"  # optional
        }

    Returns a JSON object with a `results` list mirroring the
    `ScoredResume.as_dict()` format from the CLI.

    """
    weights = request.weights or DEFAULT_WEIGHTS
    model_name = request.model or DEFAULT_MODEL

    try:
        scorer = ResumeScorer(model_name=model_name, weights=weights)
        results = scorer.score_from_dicts(request.jd, request.resumes)
        if request.top:
            results = results[: request.top]
    except Exception as exc:  # pragma: no cover - we want the error message
        raise HTTPException(status_code=500, detail=str(exc))

    return {"results": [r.as_dict() for r in results]}


@app.get("/")
def root():
    return {"message": "Resume scoring server is running"}


@app.get("/sample")
def sample():
    """Return the example JD and a single sample resume from the backend folder.

    This lets the frontend preload data for demonstration purposes.
    """
    import json
    from pathlib import Path

    jd_path = Path(__file__).parent / "sample_jd.txt"
    resume_path = Path(__file__).parent / "sample_resume.json"
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else ""
    resume_json = {}
    if resume_path.exists():
        with open(resume_path) as f:
            resume_json = json.load(f)
    return {"jd": jd_text, "resume": resume_json}
