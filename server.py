from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

import json
from pathlib import Path

# pdf/docx support
from PyPDF2 import PdfReader
import docx

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
    jd_path = Path(__file__).parent / "sample_jd.txt"
    resume_path = Path(__file__).parent / "sample_resume.json"
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else ""
    resume_json = {}
    if resume_path.exists():
        with open(resume_path) as f:
            resume_json = json.load(f)
    return {"jd": jd_text, "resume": resume_json}


# ---- feedback helpers and endpoint ----

def _extract_text_file(upload: UploadFile) -> str | dict:
    """Read an uploaded file and return either plain text or a dict (if JSON)."""
    suffix = Path(upload.filename).suffix.lower()
    data = upload.file.read()
    # JSON case
    if suffix == ".json":
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}
    # plain text types
    if suffix in {".txt", ".md", ".html"}:
        return data.decode("utf-8", errors="ignore")
    # PDF
    if suffix == ".pdf":
        try:
            reader = PdfReader(upload.file)
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return "\n".join(texts)
        except Exception:
            return ""
    # DOCX
    if suffix in {".docx", ".doc"}:
        try:
            doc = docx.Document(upload.file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    # fallback
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


@app.post("/feedback")
async def feedback(
    jd_file: UploadFile = File(...),
    resume_file: UploadFile = File(...),
):
    """Generate simple feedback given a JD file and a resume file.

    The files may be text, JSON, PDF or DOCX.  The response contains
    a short message and the raw scores used to build it.
    """
    jd = _extract_text_file(jd_file)
    resume_data = _extract_text_file(resume_file)

    # convert resume_data to ResumeData if it's a dict, otherwise treat as
    # skills-only text.
    if isinstance(resume_data, dict):
        resume_obj = ResumeData.from_dict(resume_data)
    else:
        resume_obj = ResumeData(
            resume_id="uploaded",
            source_file=resume_file.filename,
            skills=[resume_data],
            experience=[],
            projects=[],
        )

    try:
        scorer = ResumeScorer()
        scored = scorer.score(jd, [resume_obj])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # build feedback text
    notes = []
    if scored.skills_score < 0.5:
        notes.append("Consider adding more relevant skills to your resume.")
    else:
        notes.append("Skills section matches the JD well.")
    if scored.experience_score < 0.5:
        notes.append("Experience details could be more closely aligned with the job.")
    else:
        notes.append("Experience section looks good.")
    if scored.projects_score < 0.5:
        notes.append("Including more project descriptions may help.")
    else:
        notes.append("Projects seem relevant.")

    return {"feedback": " ".join(notes), "scores": scored.as_dict()}
