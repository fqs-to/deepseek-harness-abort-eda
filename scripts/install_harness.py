"""Install the team Harness into a fresh copy of the official contest source."""

import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile


TEAM_DIR = Path(__file__).resolve().parents[1] / "harness/claude/AgenticBench/agent/loss_decoder"
TEAM_FILES = (
    "agent.py", "decoder_probe.py", "opening_gradients.py", "opening_review_prompt.md",
    "paired_feedback.py", "prompt.md", "search_policy.py",
)
START_HASH = "f18b6e303cad3e67b7dd5be6a63a2ca7399e43997fd0c900334adae1a31bc87b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contest_source", type=Path, help="freshly extracted contest-source directory")
    args = parser.parse_args()
    bench = args.contest_source.resolve() / "AgenticBench"
    start = bench / "local_tasks/tp_gnn_loss_decoder_timing/initial_program.py"
    target = bench / "agent/loss_decoder"
    if not target.is_dir() or not start.is_file():
        parser.error("expected a complete official contest-source directory")
    if sha256(start.read_bytes()).hexdigest() != START_HASH:
        parser.error("official starting program hash changed; check the contest package version")
    files = [(TEAM_DIR / name, target / name) for name in TEAM_FILES]
    files.append((start, target / "program.py"))
    if any(not src.is_file() or dest.is_symlink() for src, dest in files) or target.resolve() != target:
        parser.error("missing input or unexpected symlink; use a fresh official source tree")
    for src, dest in files:
        copyfile(src, dest)
    print(f"Installed {len(TEAM_FILES)} Harness files and the official start in {target}")


if __name__ == "__main__":
    main()
