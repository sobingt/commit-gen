#!/usr/bin/env python3
"""
git-commit-gen — Generate git commit messages from diffs.

Usage:
  git-commit-gen                  # auto-detects staged or unstaged changes
  git-commit-gen --hook           # git hook mode (used by prepare-commit-msg)
  git-commit-gen --install-hook   # install hook in current repo
  git-commit-gen --version        # show version
  git-commit-gen --model /path    # use a specific model file
"""

import sys
import os
import re
import argparse
import subprocess
import urllib.request
import json
from pathlib import Path

VERSION = "1.3.0"
GITHUB_REPO = "sobingt/commit-gen"
MODEL_DIR = Path.home() / ".cache" / "git-commit-gen"
VERSION_CACHE = MODEL_DIR / ".last_update_check"

# Context budget — leave room for system prompt + output
MAX_CONTEXT_CHARS = 3500

# ── ANSI colors ───────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# Auto-update check
# ─────────────────────────────────────────────────────────────────────────────

def check_for_update():
    import time
    if VERSION_CACHE.exists():
        last_check = float(VERSION_CACHE.read_text().strip() or 0)
        if time.time() - last_check < 86400:
            return
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(url, headers={"User-Agent": "git-commit-gen"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        latest = data.get("tag_name", "").lstrip("v")
        VERSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VERSION_CACHE.write_text(str(time.time()))
        if latest and latest != VERSION:
            print(f"{YELLOW}💡 Update available: v{VERSION} → v{latest}{RESET}")
            print(f"{DIM}   Run: curl -fsSL https://raw.githubusercontent.com/{GITHUB_REPO}/main/install.sh | bash{RESET}\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def find_model(override: str = None) -> Path:
    if override:
        return Path(override)
    candidates = list(MODEL_DIR.glob("*.gguf"))
    if not candidates:
        return None
    for c in candidates:
        if "Q4_K_M" in c.name:
            return c
    for c in candidates:
        if "Q4" in c.name:
            return c
    return candidates[0]


SYSTEM_PROMPT = (
    "You are an expert developer assistant that generates clear, concise git commit messages. "
    "You will receive a structured diff summary containing:\n"
    "  - A stat block showing all changed files and line counts\n"
    "  - Full diffs for small files\n"
    "  - One-line summaries for large files\n"
    "Generate a meaningful commit message following conventional commit format "
    "(e.g. feat:, fix:, chore:, refactor:). "
    "Output only the commit message — subject line first, then an optional body separated by a blank line. "
    "Do not include any explanation, preamble, or markdown formatting."
)


def load_model(model_path: Path):
    try:
        from llama_cpp import Llama
    except ImportError:
        print(f"{RED}❌ llama-cpp-python not installed. Run install.sh first.{RESET}", file=sys.stderr)
        sys.exit(1)

    import contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=max(4, os.cpu_count() or 4),
            verbose=False,
        )
    return llm


def generate(llm, context: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a commit message for these changes:\n\n{context}"},
        ],
        max_tokens=256,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Smart diff preparation  (Strategy 1 + 2 + 4)
# ─────────────────────────────────────────────────────────────────────────────

# Strategy 2 — files to drop entirely (never useful for commit messages)
SKIP_PATTERNS = [
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "Gemfile.lock", "poetry.lock", "Cargo.lock",
    "logs/", ".log",
    "dist/", "build/", ".next/", "__pycache__/",
    ".min.js", ".min.css",
]

def should_skip(filename: str) -> bool:
    return any(p in filename for p in SKIP_PATTERNS)


def parse_files_from_diff(diff: str) -> list[dict]:
    """
    Split a raw diff into a list of per-file dicts:
      { filename, diff_text, additions, deletions }
    """
    files = []
    current = None

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current:
                files.append(current)
            # Extract filename: diff --git a/foo.ts b/foo.ts
            match = re.search(r"diff --git a/(.+?) b/(.+)", line)
            filename = match.group(2) if match else line
            current = {"filename": filename, "lines": [line], "additions": 0, "deletions": 0}
        elif current:
            current["lines"].append(line)
            if line.startswith("+") and not line.startswith("+++"):
                current["additions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                current["deletions"] += 1

    if current:
        files.append(current)

    for f in files:
        f["diff_text"] = "\n".join(f.pop("lines"))

    return files


def file_summary(f: dict) -> str:
    """One-line summary for a file that's too large to include in full."""
    total = f["additions"] + f["deletions"]
    parts = []
    if f["additions"]:
        parts.append(f"+{f['additions']} lines")
    if f["deletions"]:
        parts.append(f"-{f['deletions']} lines")
    change_str = ", ".join(parts) if parts else f"{total} lines changed"
    return f"  {f['filename']}  ({change_str})"


def get_stat(staged: bool) -> str:
    """Get git diff --stat for a compact overview of all changes."""
    cmd = ["git", "diff", "--cached", "--stat"] if staged else ["git", "diff", "--stat"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def prepare_diff(raw_diff: str, staged: bool) -> str:
    """
    Combines strategy 1 + 2 + 4:
      1. Always include the --stat block (tiny, gives full picture)
      2. Filter out noisy/generated files entirely
      3. For remaining files: include full diff if it fits the budget,
         otherwise replace with a one-line summary
    """
    # Strategy 4 — stat block is always first, costs very little
    stat = get_stat(staged)
    stat_block = f"=== Changes overview ===\n{stat}\n\n=== Detailed changes ==="
    budget = MAX_CONTEXT_CHARS - len(stat_block)

    # Parse into per-file chunks
    files = parse_files_from_diff(raw_diff)

    included = []   # full diffs that fit
    summarized = [] # files summarized to one line

    for f in files:
        # Strategy 2 — skip noisy files
        if should_skip(f["filename"]):
            print(f"{DIM}   skipping {f['filename']}{RESET}", file=sys.stderr)
            continue

        if len(f["diff_text"]) <= budget:
            # Fits — include full diff
            included.append(f["diff_text"])
            budget -= len(f["diff_text"])
        else:
            # Too large — summarize to one line (Strategy 1)
            summarized.append(file_summary(f))

    sections = [stat_block]

    if included:
        sections.append("\n\n".join(included))

    if summarized:
        sections.append("=== Large files (summarized) ===\n" + "\n".join(summarized))
        skipped_count = len(summarized)
        print(
            f"{DIM}   {skipped_count} large file(s) summarized to save context.{RESET}",
            file=sys.stderr
        )

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Diff retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_diff() -> tuple[str, bool]:
    """Returns (raw_diff, is_staged)."""
    staged = subprocess.run(
        ["git", "diff", "--cached"], capture_output=True, text=True
    ).stdout.strip()

    if staged:
        return staged, True

    unstaged = subprocess.run(
        ["git", "diff"], capture_output=True, text=True
    ).stdout.strip()

    if not unstaged:
        print(f"{RED}❌ No staged or unstaged changes found.{RESET}")
        sys.exit(1)

    print(f"{YELLOW}⚠️  Nothing is staged.{RESET}")
    print(f"{DIM}   Found unstaged changes — use those instead?{RESET}")
    answer = input("   [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        return unstaged, False
    print("Aborted.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def copy_to_clipboard(text: str) -> bool:
    for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"], ["wl-copy"]):
        try:
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True)
            if proc.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


def interactive_loop(llm, context: str) -> str:
    while True:
        message = generate(llm, context)

        print(f"\n{BOLD}{CYAN}┌─ Generated commit message {'─' * 40}{RESET}")
        for line in message.splitlines():
            print(f"{CYAN}│{RESET} {line}")
        print(f"{BOLD}{CYAN}└{'─' * 50}{RESET}\n")

        print(f"  {GREEN}[Y]{RESET} Use this message")
        print(f"  {YELLOW}[r]{RESET} Regenerate")
        print(f"  {YELLOW}[e]{RESET} Edit then use")
        print(f"  {RED}[n]{RESET} Abort")
        answer = input(f"\n{BOLD}Your choice [Y/r/e/n]:{RESET} ").strip().lower()

        if answer in ("", "y", "yes"):
            return message
        elif answer == "r":
            print(f"\n{DIM}Regenerating...{RESET}")
        elif answer == "e":
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(message)
                tmp = f.name
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, tmp])
            edited = Path(tmp).read_text().strip()
            os.unlink(tmp)
            return edited
        elif answer in ("n", "no", "q"):
            print("Aborted.")
            sys.exit(0)
        else:
            print(f"{DIM}Please enter Y, r, e, or n.{RESET}")


def run_commit(message: str):
    result = subprocess.run(["git", "commit", "-m", message])
    sys.exit(result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# Hook
# ─────────────────────────────────────────────────────────────────────────────

def hook_mode(commit_msg_file: str, llm):
    raw_diff, staged = get_diff()
    context = prepare_diff(raw_diff, staged)
    message = generate(llm, context)
    Path(commit_msg_file).write_text(message + "\n")
    print(f"{GREEN}✅ Commit message generated by git-commit-gen{RESET}", file=sys.stderr)


def install_hook():
    try:
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        print(f"{RED}❌ Not inside a git repository.{RESET}")
        sys.exit(1)

    hook_path = Path(git_dir) / "hooks" / "prepare-commit-msg"

    if hook_path.exists():
        answer = input(f"{YELLOW}⚠️  Hook already exists. Overwrite? [y/N]:{RESET} ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    hook_path.write_text("""#!/bin/bash
COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"
# Only run when no message source (not amend, merge, squash, etc.)
if [ -z "$COMMIT_SOURCE" ]; then
    git-commit-gen "$COMMIT_MSG_FILE"
fi
""")
    hook_path.chmod(0o755)
    print(f"{GREEN}✅ Hook installed: {hook_path}{RESET}")
    print(f"{DIM}   'git commit' will now auto-generate messages in this repo.{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a git commit message from a diff.")
    parser.add_argument("commit_msg_file", nargs="?",    help="Used by prepare-commit-msg hook")
    parser.add_argument("--hook",          action="store_true", help="Run in git hook mode")
    parser.add_argument("--install-hook",  action="store_true", help="Install prepare-commit-msg hook in current repo")
    parser.add_argument("--version",       action="store_true", help="Show version and exit")
    parser.add_argument("--model",         type=str,            help="Path to GGUF model file")
    args = parser.parse_args()

    if args.version:
        print(f"git-commit-gen v{VERSION}")
        return

    if args.install_hook:
        install_hook()
        return

    check_for_update()

    model_path = find_model(args.model)
    if not model_path or not model_path.exists():
        print(f"{RED}❌ No model found in: {MODEL_DIR}{RESET}", file=sys.stderr)
        print("   Run install.sh to download it.", file=sys.stderr)
        sys.exit(1)

    print(f"{DIM}⏳ Loading model...{RESET}", file=sys.stderr)
    llm = load_model(model_path)

    # Hook mode
    if args.hook or args.commit_msg_file:
        commit_file = args.commit_msg_file or sys.argv[1]
        hook_mode(commit_file, llm)
        return

    # Interactive mode
    raw_diff, staged = get_diff()
    label = "staged" if staged else "unstaged"
    print(f"{DIM}✍️  Preparing {label} diff...{RESET}", file=sys.stderr)

    context = prepare_diff(raw_diff, staged)

    print(f"{DIM}🤖 Generating commit message...{RESET}", file=sys.stderr)
    message = interactive_loop(llm, context)

    copied = copy_to_clipboard(message)
    if copied:
        print(f"\n{DIM}📋 Copied to clipboard.{RESET}")

    print(f"\n{BOLD}Run `git commit` with this message? [Y/n]:{RESET} ", end="")
    answer = input().strip().lower()
    if answer in ("", "y", "yes"):
        run_commit(message)
    else:
        print(f"\n{DIM}Commit message:{RESET}\n{message}")


if __name__ == "__main__":
    main()