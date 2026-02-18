#!/usr/bin/env python3
"""
git-commit-gen — Generate git commit messages from diffs.

Usage:
  git diff | git-commit-gen
  git diff --cached | git-commit-gen
  git-commit-gen --diff "paste diff here"
"""

import sys
import argparse
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "git-commit-gen"

def find_model(override: str = None) -> Path:
    if override:
        return Path(override)
    candidates = list(MODEL_DIR.glob("*.gguf"))
    if not candidates:
        return None
    for candidate in candidates:
        if "Q4_K_M" in candidate.name:
            return candidate
    for candidate in candidates:
        if "Q4" in candidate.name:
            return candidate
    return candidates[0]

SYSTEM_PROMPT = (
    "You are an expert developer assistant that generates clear, concise git commit messages. "
    "Analyze the provided git diff and generate a meaningful commit message following "
    "conventional commit format (e.g. feat:, fix:, chore:, refactor:). "
    "Output only the commit message — subject line first, then an optional body separated by a blank line. "
    "Do not include any explanation or preamble."
)


def load_model(model_path: Path):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python not installed. Run install.sh first.", file=sys.stderr)
        sys.exit(1)

    import os
    import contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=max(4, os.cpu_count() or 4),
            verbose=False,
        )
    return llm


def generate(llm, diff: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a commit message for this diff:\n\n{diff}"},
        ],
        max_tokens=256,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a git commit message from a diff.")
    parser.add_argument("--diff", type=str, help="Diff string (otherwise reads from stdin)")
    parser.add_argument("--model", type=str, help=f"Path to GGUF model file (default: auto-detect from {MODEL_DIR})")
    args = parser.parse_args()

    model_path = find_model(args.model)

    if not model_path or not model_path.exists():
        print(f"❌ No model found in: {MODEL_DIR}", file=sys.stderr)
        print("   Run install.sh to download it.", file=sys.stderr)
        sys.exit(1)

    if args.diff:
        diff = args.diff
    elif not sys.stdin.isatty():
        diff = sys.stdin.read().strip()
    else:
        print("❌ No diff provided.", file=sys.stderr)
        print("   Usage: git diff --cached | git-commit-gen", file=sys.stderr)
        sys.exit(1)

    if not diff:
        print("❌ Diff is empty — nothing to generate from.", file=sys.stderr)
        sys.exit(1)

    # Truncate large diffs to fit TinyLlama's context window
    max_diff_chars = 4000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n... (diff truncated)"
        print("⚠️  Large diff truncated to fit model context.", file=sys.stderr)

    print("⏳ Loading model...", file=sys.stderr)
    llm = load_model(model_path)

    print("✍️  Generating commit message...\n", file=sys.stderr)
    message = generate(llm, diff)
    print(message)


if __name__ == "__main__":
    main()