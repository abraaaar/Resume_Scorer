"""
CLI entrypoint for the Resume Scoring System.

Usage examples
──────────────
# Score a single resume JSON against a plain-text JD:
    python score.py --jd jd.txt --resumes resume1.json

# Score multiple resumes, custom weights:
    python score.py --jd jd.txt --resumes r1.json r2.json r3.json \
        --weight-skills 0.25 --weight-experience 0.50 --weight-projects 0.25

# Use a better (slower) model:
    python score.py --jd jd.txt --resumes *.json --model all-mpnet-base-v2

# Pass the raw parser output directly as JSON (inline):
    python score.py --jd jd.txt --resumes-json '[{...}, {...}]'
"""

import argparse
import json
import sys
from pathlib import Path

from scorer import ResumeScorer, ResumeData


# ──────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────

BAR_WIDTH = 30

def _bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)

def _pct(score: float) -> str:
    return f"{score * 100:5.1f}%"

def print_results(results, weights):
    print()
    print("═" * 68)
    print(f"  RESUME RANKING  (weights: skills={weights['skills']:.0%}  "
          f"exp={weights['experience']:.0%}  proj={weights['projects']:.0%})")
    print("═" * 68)

    for r in results:
        print(f"\n  #{r.rank}  {r.source_file}  [{r.resume_id}]")
        print(f"  {'Overall':12s}  {_bar(r.weighted_score)}  {_pct(r.weighted_score)}")
        print(f"  {'Skills':12s}  {_bar(r.skills_score)}  {_pct(r.skills_score)}")
        print(f"  {'Experience':12s}  {_bar(r.experience_score)}  {_pct(r.experience_score)}")
        print(f"  {'Projects':12s}  {_bar(r.projects_score)}  {_pct(r.projects_score)}")
        print(f"  {'─'*64}")

    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Score resumes against a Job Description using local embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    p.add_argument("--jd", required=True,
                   help="Path to plain-text (.txt) or JSON (.json) Job Description file.")
    
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--resumes", nargs="+", metavar="FILE",
                             help="One or more resume JSON files (parser output).")
    input_group.add_argument("--resumes-json", metavar="JSON_STRING",
                             help="Inline JSON array of resume dicts.")

    # Model
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="Sentence-transformers model name or local path. "
                        "(default: all-MiniLM-L6-v2)")
    p.add_argument("--device", default=None,
                   help="Force device: 'cuda', 'cuda:1', 'cpu', 'mps'. "
                        "Default: auto-detect (CUDA > MPS > CPU).")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable float16 on CUDA (use float32 instead).")

    # Weights
    p.add_argument("--weight-skills",     type=float, default=0.30)
    p.add_argument("--weight-experience", type=float, default=0.45)
    p.add_argument("--weight-projects",   type=float, default=0.25)

    # Output
    p.add_argument("--output-json", metavar="FILE",
                   help="Optional: save full results as JSON to this path.")
    p.add_argument("--top", type=int, default=None,
                   help="Only show top-N results.")

    return p


def main():
    args = build_parser().parse_args()

    # ── Load JD ──────────────────────────────
    jd_path = Path(args.jd)
    if not jd_path.exists():
        sys.exit(f"[ERROR] JD file not found: {jd_path}")

    if jd_path.suffix.lower() == ".json":
        with open(jd_path) as f:
            jd = json.load(f)
    else:
        jd = jd_path.read_text(encoding="utf-8")

    # ── Load resumes ─────────────────────────
    if args.resumes:
        resumes = []
        for path in args.resumes:
            p = Path(path)
            if not p.exists():
                print(f"[WARN] Resume file not found, skipping: {p}")
                continue
            with open(p) as f:
                resumes.append(ResumeData.from_dict(json.load(f)))
    else:
        raw_list = json.loads(args.resumes_json)
        if isinstance(raw_list, dict):
            raw_list = [raw_list]
        resumes = [ResumeData.from_dict(d) for d in raw_list]

    if not resumes:
        sys.exit("[ERROR] No valid resumes to score.")

    # ── Build weights ────────────────────────
    weights = {
        "skills":     args.weight_skills,
        "experience": args.weight_experience,
        "projects":   args.weight_projects,
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        sys.exit(f"[ERROR] Weights must sum to 1.0, got {total:.4f}")

    # ── Score ────────────────────────────────
    scorer  = ResumeScorer(
        model_name=args.model,
        weights=weights,
        device=args.device,
        use_fp16=not args.no_fp16,
    )
    results = scorer.score(jd, resumes)

    if args.top:
        results = results[: args.top]

    # ── Display ──────────────────────────────
    print_results(results, weights)

    # ── Optional JSON output ─────────────────
    if args.output_json:
        out = [r.as_dict() for r in results]
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ResumeScorer] Results saved → {args.output_json}")


if __name__ == "__main__":
    main()
