# Resume Scoring System

Ranks resumes against a Job Description using **local embeddings** (no API calls).  
Each resume section — **skills, experience, projects** — is embedded separately and compared to the JD via cosine similarity, then combined with configurable weights.

---

## Architecture

```
JD text ──► embed() ──► jd_vec
                            │
resume.skills ──► embed() ──► cosine ──┐
resume.experience ──► embed() ──► cosine ──┤──► weighted_score
resume.projects ──► embed() ──► cosine ──┘
```

### Default weights

| Section    | Weight |
|------------|--------|
| Experience | 0.45   |
| Skills     | 0.30   |
| Projects   | 0.25   |

---

## Setup

```bash
pip install -r requirements.txt
```

Model is downloaded automatically on first run (~80 MB for default model).

---

## Usage

### CLI — single or multiple resumes

```bash
# Basic
python score.py --jd sample_jd.txt --resumes sample_resume.json

# Multiple resumes
python score.py --jd sample_jd.txt --resumes r1.json r2.json r3.json

# Custom weights
python score.py --jd jd.txt --resumes *.json \
    --weight-skills 0.25 --weight-experience 0.50 --weight-projects 0.25

# Save results to JSON
python score.py --jd jd.txt --resumes *.json --output-json results.json

# Show only top 5
python score.py --jd jd.txt --resumes *.json --top 5

# Use a higher quality model
python score.py --jd jd.txt --resumes *.json --model all-mpnet-base-v2
```

### Programmatic API

```python
from scorer import ResumeScorer, ResumeData

scorer = ResumeScorer(
    model_name="all-MiniLM-L6-v2",
    weights={"skills": 0.30, "experience": 0.45, "projects": 0.25},
)

results = scorer.score_from_dicts(jd="...", resumes=[dict1, dict2, dict3])

for r in results:
    print(r.as_dict())
```

### Batch — score all JSONs in a folder

```python
from batch_score import batch_score_directory

results = batch_score_directory(
    jd="Senior Python ML Engineer with PyTorch and NLP experience.",
    resume_dir="./parsed_resumes/",
    top_n=10,
)
```

---

## Recommended models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` *(default)* | ~80 MB | ⚡ fast | ★★★☆ |
| `BAAI/bge-small-en-v1.5` | ~130 MB | ⚡ fast | ★★★★ |
| `all-mpnet-base-v2` | ~420 MB | 🐢 slower | ★★★★★ |

All models run fully locally — no internet required after first download.

---

## Files

```
resume_scorer/
├── scorer.py          # Core: ResumeScorer class, text helpers, cosine scoring
├── score.py           # CLI entrypoint
├── batch_score.py     # Programmatic batch API
├── requirements.txt
├── sample_jd.txt      # Example JD
└── sample_resume.json # Example parsed resume (your parser's output format)
```
