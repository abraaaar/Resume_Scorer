"""
Resume Scoring System
─────────────────────
Embeds JD sections + resume sections (skills, experience, projects)
locally using sentence-transformers, then ranks resumes by weighted
cosine similarity against the JD.

GPU/CUDA support
────────────────
CUDA is used automatically when available. The scorer will:
  - Detect and use CUDA (or MPS on Apple Silicon) automatically
  - Run embeddings in float16 on CUDA for ~2× throughput
  - Batch all resume section texts in a single GPU forward pass
  - Fall back gracefully to CPU if no GPU is present
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEFAULT_MODEL = "all-MiniLM-L6-v2"   # fast, ~80 MB, good quality

DEFAULT_WEIGHTS = {
    "skills":     0.30,
    "experience": 0.45,
    "projects":   0.25,
}

# Batch size for GPU encoding — tune upward if VRAM allows
GPU_BATCH_SIZE = 64
CPU_BATCH_SIZE = 32


def detect_device() -> torch.device:
    """
    Return the best available device:
      CUDA  → Nvidia GPU
      MPS   → Apple Silicon GPU
      CPU   → fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        idx  = device.index or 0
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"CUDA — {name} ({vram:.1f} GB VRAM)"
    if device.type == "mps":
        return "MPS — Apple Silicon GPU"
    return "CPU"


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ResumeData:
    """Parsed resume as produced by extractor.py / run.py."""
    resume_id:   str
    source_file: str
    skills:      list[str]       = field(default_factory=list)
    experience:  list[dict]      = field(default_factory=list)
    projects:    list[dict]      = field(default_factory=list)
    raw:         dict            = field(default_factory=dict)   # full original dict

    @classmethod
    def from_dict(cls, d: dict) -> "ResumeData":
        return cls(
            resume_id=d.get("resume_id", "unknown"),
            source_file=d.get("source_file", "unknown"),
            skills=d.get("skills", []),
            experience=d.get("experience", []),
            projects=d.get("projects", []),
            raw=d,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ResumeData":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class ScoredResume:
    resume_id:        str
    source_file:      str
    skills_score:     float
    experience_score: float
    projects_score:   float
    weighted_score:   float
    rank:             int = 0

    def as_dict(self) -> dict:
        return {
            "rank":             self.rank,
            "resume_id":        self.resume_id,
            "source_file":      self.source_file,
            "weighted_score":   round(self.weighted_score, 4),
            "skills_score":     round(self.skills_score, 4),
            "experience_score": round(self.experience_score, 4),
            "projects_score":   round(self.projects_score, 4),
        }


# ──────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────

def _skills_to_text(skills: list[str]) -> str:
    return ", ".join(skills) if skills else ""


def _experience_to_text(experience: list[dict]) -> str:
    parts = []
    for exp in experience:
        title   = exp.get("title", "")
        company = exp.get("company", "")
        resp    = " ".join(exp.get("responsibilities", []))
        tech    = ", ".join(exp.get("tech_stack", []))
        parts.append(f"{title} at {company}. {resp}. Technologies: {tech}".strip())
    return " | ".join(parts)


def _projects_to_text(projects: list[dict]) -> str:
    parts = []
    for proj in projects:
        name  = proj.get("name", "")
        desc  = " ".join(proj.get("description", []))
        tech  = ", ".join(proj.get("tech_stack", []))
        parts.append(f"{name}: {desc}. Tech: {tech}".strip())
    return " | ".join(parts)


def _jd_to_text(jd: str | dict) -> str:
    """Accept either a plain string JD or a structured dict."""
    if isinstance(jd, str):
        return jd
    # structured dict: flatten all values
    parts = []
    for v in jd.values():
        if isinstance(v, list):
            parts.append(" ".join(str(x) for x in v))
        else:
            parts.append(str(v))
    return " ".join(parts)



# ──────────────────────────────────────────────
# Main scorer class
# ──────────────────────────────────────────────

class ResumeScorer:
    """
    Scores resumes against a Job Description using local embeddings.

    GPU / CUDA behaviour
    ────────────────────
    By default the scorer auto-detects the best device (CUDA > MPS > CPU).
    On CUDA it also uses float16 precision, roughly doubling throughput.
    All resume section texts are embedded in a single batched GPU call
    rather than one-by-one, so scoring 100 resumes costs ~3 forward passes
    instead of 300.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model name or local path.
        Recommended:
          - "all-MiniLM-L6-v2"      → fast, good quality  (~80 MB)
          - "BAAI/bge-small-en-v1.5" → very fast, great quality (~130 MB)
          - "all-mpnet-base-v2"      → best quality, slower (~420 MB)
    weights : dict
        Keys: "skills", "experience", "projects". Must sum to 1.0.
    device : str | torch.device | None
        Force a specific device, e.g. "cuda:1", "cpu".
        None (default) = auto-detect.
    use_fp16 : bool
        Use float16 on CUDA for ~2× throughput. Ignored on CPU/MPS.
        Default: True when CUDA is available.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        weights:    dict = DEFAULT_WEIGHTS,
        device:     str | torch.device | None = None,
        use_fp16:   bool | None = None,
    ):
        self.weights = self._validate_weights(weights)

        # ── device setup ─────────────────────────────
        self.device = torch.device(device) if device else detect_device()
        # fp16 only makes sense on CUDA; MPS has partial support, skip it
        if use_fp16 is None:
            self.use_fp16 = (self.device.type == "cuda")
        else:
            self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        self.batch_size = GPU_BATCH_SIZE if self.device.type != "cpu" else CPU_BATCH_SIZE

        # ── load model ───────────────────────────────
        print(f"[ResumeScorer] Device   : {device_info(self.device)}")
        print(f"[ResumeScorer] Precision: {'float16 (fp16)' if self.use_fp16 else 'float32'}")
        print(f"[ResumeScorer] Loading  : {model_name} …")
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name, device=str(self.device))
        if self.use_fp16:
            self.model.half()   # cast all model weights to fp16
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[ResumeScorer] Ready in {elapsed:.1f} ms\n")

    # ── public API ──────────────────────────────────

    def score(
        self,
        jd:      str | dict,
        resumes: list[ResumeData],
    ) -> list[ScoredResume]:
        """
        Score all resumes against the JD and return ranked results.

        All section texts are embedded in one batched GPU call per section
        type (skills / experience / projects), making GPU utilisation
        proportional to the number of resumes rather than the number of
        individual encode() calls.

        Parameters
        ----------
        jd      : plain-text JD string **or** structured dict
        resumes : list of ResumeData objects

        Returns
        -------
        List of ScoredResume sorted by weighted_score descending.
        """
        if not resumes:
            return []

        t0 = time.perf_counter()

        jd_text = _jd_to_text(jd)
        jd_emb  = self._encode([jd_text])[0]   # shape: (dim,)

        # ── build per-section text lists ─────────────
        skills_texts = [_skills_to_text(r.skills)     for r in resumes]
        exp_texts    = [_experience_to_text(r.experience) for r in resumes]
        proj_texts   = [_projects_to_text(r.projects)    for r in resumes]

        # ── single batched GPU call per section ──────
        skills_embs = self._encode(skills_texts)   # (N, dim)
        exp_embs    = self._encode(exp_texts)
        proj_embs   = self._encode(proj_texts)

        # ── cosine similarity: (N,) each ─────────────
        skills_scores = _batch_cosine(jd_emb, skills_embs, skills_texts)
        exp_scores    = _batch_cosine(jd_emb, exp_embs,    exp_texts)
        proj_scores   = _batch_cosine(jd_emb, proj_embs,   proj_texts)

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[ResumeScorer] Scored {len(resumes)} resume(s) in {elapsed:.1f} ms "
              f"on {self.device.type.upper()}")

        # ── assemble results ─────────────────────────
        scored = []
        w = self.weights
        for i, resume in enumerate(resumes):
            ws = (w["skills"] * skills_scores[i] +
                  w["experience"] * exp_scores[i] +
                  w["projects"] * proj_scores[i])
            scored.append(ScoredResume(
                resume_id=resume.resume_id,
                source_file=resume.source_file,
                skills_score=skills_scores[i],
                experience_score=exp_scores[i],
                projects_score=proj_scores[i],
                weighted_score=ws,
            ))

        scored.sort(key=lambda x: x.weighted_score, reverse=True)
        for rank, s in enumerate(scored, start=1):
            s.rank = rank

        return scored

    def score_from_dicts(
        self,
        jd:      str | dict,
        resumes: list[dict],
    ) -> list[ScoredResume]:
        """Convenience wrapper: accepts raw dicts from the parser."""
        return self.score(jd, [ResumeData.from_dict(r) for r in resumes])

    # ── private helpers ──────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings in batches.
        Returns float32 numpy array of shape (N, dim).
        """
        # sentence-transformers handles batching internally;
        # we pass batch_size so it fills VRAM efficiently
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → dot product == cosine
        )

    @staticmethod
    def _validate_weights(w: dict) -> dict:
        required = {"skills", "experience", "projects"}
        if not required.issubset(w):
            raise ValueError(f"weights must contain keys: {required}")
        total = sum(w[k] for k in required)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")
        return w


# ──────────────────────────────────────────────
# Vectorised cosine (replaces scalar _cosine)
# ──────────────────────────────────────────────

def _batch_cosine(
    query:    np.ndarray,   # (dim,)
    matrix:   np.ndarray,   # (N, dim)
    texts:    list[str],    # used to zero-out empty sections
) -> list[float]:
    """
    Dot product of unit-normalised vectors == cosine similarity.
    Returns a plain list of floats, one per resume.
    Empty sections get score 0.0 without hitting the model.
    """
    scores = matrix @ query          # (N,) — fast BLAS matmul
    return [
        float(scores[i]) if texts[i] else 0.0
        for i in range(len(texts))
    ]
