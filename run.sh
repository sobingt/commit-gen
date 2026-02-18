#!/usr/bin/env python3
"""
git-commit-gen — Generate git commit messages from diffs.

Usage:
  git-commit-gen                  # auto-detects staged or unstaged changes
  git-commit-gen --hook           # git hook mode (used by prepare-commit-msg)
  git-commit-gen --model /path    # use a specific model file
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "git-commit-gen"

# ── ANSI colors ──────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"

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
    "Analyze the provided git diff and generate a meaningful commit message following "
    "conventional commit format (e.g. feat:, fix:, chore:, refactor:). "
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


def get_diff() -> tuple[str, str]:
    """
    Returns (diff, source) where source is 'staged', 'unstaged', or raises.
    If nothing is staged, asks the user whether to use unstaged changes.
    """
    # Check staged
    staged = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True, text=True
    ).stdout.strip()

    if staged:
        return staged, "staged"

    # Nothing staged — check unstaged
    unstaged = subprocess.run(
        ["git", "diff"],
        capture_output=True, text=True
    ).stdout.strip()

    if not unstaged:
        print(f"{RED}❌ No staged or unstaged changes found.{RESET}")
        sys.exit(1)

    # Ask user to confirm using unstaged changes
    print(f"{YELLOW}⚠️  Nothing is staged.{RESET}")
    print(f"{DIM}   Found unstaged changes — use those instead?{RESET}")
    answer = input("   [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        return unstaged, "unstaged"
    else:
        print("Aborted.")
        sys.exit(0)


def copy_to_clipboard(text: str) -> bool:
    """Try to copy text to clipboard. Returns True on success."""
    for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"], ["xdotool", "type"]):
        try:
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True)
            if proc.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


def interactive_loop(llm, diff: str) -> str:
    """Show generated message and let user accept, regenerate, or edit."""
    while True:
        message = generate(llm, diff)

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
            continue
        elif answer == "e":
            # Write to temp file and open in $EDITOR
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
    """Run git commit with the given message."""
    result = subprocess.run(["git", "commit", "-m", message])
    sys.exit(result.returncode)


def hook_mode(commit_msg_file: str, llm):
    """
    Used as a prepare-commit-msg hook.
    Writes the generated message to the commit message file.
    """
    diff, _ = get_diff()
    diff = truncate_diff(diff)
    message = generate(llm, diff)
    Path(commit_msg_file).write_text(message + "\n")
    print(f"{GREEN}✅ Commit message generated by git-commit-gen{RESET}", file=sys.stderr)


def truncate_diff(diff: str) -> str:
    max_chars = 4000
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n... (diff truncated)"
        print(f"{YELLOW}⚠️  Large diff truncated to fit model context.{RESET}", file=sys.stderr)
    return diff


def main():
    parser = argparse.ArgumentParser(description="Generate a git commit message from a diff.")
    parser.add_argument("commit_msg_file", nargs="?", help="Used by prepare-commit-msg hook")
    parser.add_argument("--hook", action="store_true", help="Run in git hook mode")
    parser.add_argument("--install-hook", action="store_true", help="Install prepare-commit-msg hook in current repo")
    parser.add_argument("--model", type=str, help=f"Path to GGUF model file")
    args = parser.parse_args()

    if args.install_hook:
        install_hook()
        return

    model_path = find_model(args.model)
    if not model_path or not model_path.exists():
        print(f"{RED}❌ No model found in: {MODEL_DIR}{RESET}", file=sys.stderr)
        print("   Run install.sh to download it.", file=sys.stderr)
        sys.exit(1)

    print(f"{DIM}⏳ Loading model...{RESET}", file=sys.stderr)
    llm = load_model(model_path)

    # ── Git hook mode ────────────────────────────────────────────────────────
    if args.hook or args.commit_msg_file:
        commit_file = args.commit_msg_file or sys.argv[1]
        hook_mode(commit_file, llm)
        return

    # ── Interactive mode ─────────────────────────────────────────────────────
    diff, source = get_diff()
    label = "staged" if source == "staged" else "unstaged"
    print(f"{DIM}✍️  Generating from {label} changes...{RESET}", file=sys.stderr)

    diff = truncate_diff(diff)
    message = interactive_loop(llm, diff)

    # Copy to clipboard
    copied = copy_to_clipboard(message)
    if copied:
        print(f"\n{DIM}📋 Copied to clipboard.{RESET}")

    # Ask to commit
    print(f"\n{BOLD}Run `git commit` with this message? [Y/n]:{RESET} ", end="")
    answer = input().strip().lower()
    if answer in ("", "y", "yes"):
        run_commit(message)
    else:
        print(f"\n{DIM}Commit message (copy manually if needed):{RESET}")
        print(message)


if __name__ == "__main__":
    main()


def install_hook():
    """Install prepare-commit-msg hook in the current git repo."""
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

    hook_content = """#!/bin/bash
COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"
if [ -z "$COMMIT_SOURCE" ]; then
    git-commit-gen "$COMMIT_MSG_FILE"
fi
"""
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print(f"{GREEN}✅ Hook installed in: {hook_path}{RESET}")
    print(f"{DIM}   'git commit' will now auto-generate messages in this repo.{RESET}")