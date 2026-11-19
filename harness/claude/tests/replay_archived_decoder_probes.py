"""Compare decoder admission on archived nine-cell rounds without running training.

From the repo root, pass the released loss_decoder initial_program.py:
    python -B harness/claude/tests/replay_archived_decoder_probes.py /path/to/initial_program.py
The Git history must include the experiments archive and the reviewed probe snapshot.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
PROBE = Path("harness/claude/AgenticBench/agent/loss_decoder/decoder_probe.py")
ARCHIVE = "results/experiments-2026-11-18/"


def blob(revision, name):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", revision + ":" + name]).decode("utf-8")


def decoder_body(source):
    decoder = ast.parse(source).body[1]
    if ast.get_docstring(decoder) is not None:
        decoder.body.pop(0)
    return ast.dump(decoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("--archive", default="1e7c625d68f4e0679bc325eb37b257423aa0a235")
    parser.add_argument("--before", help="override the reviewed probe with the file at this Git revision")
    args = parser.parse_args()
    initial = args.initial.read_text(encoding="utf-8")
    if hashlib.sha256(initial.encode()).hexdigest() != "f18b6e303cad3e67b7dd5be6a63a2ca7399e43997fd0c900334adae1a31bc87b":
        parser.error("supply the released loss_decoder initial program")

    sys.path.insert(0, str(ROOT / PROBE.parent))
    import decoder_probe as current
    if current.np is None:
        parser.error("the CPU-only decoder replay requires NumPy")
    previous = types.ModuleType("previous_decoder_probe")
    old_probe = (blob(args.before, PROBE.as_posix()) if args.before else
                 subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob",
                                          "9f2202222ae6302b3e370e6731e9f9711ad87114"]).decode("utf-8"))
    exec(compile(old_probe, "<old probe>", "exec"), previous.__dict__)

    totals = dict(complete_rounds=0, changed_decoders=0, old_rejected=0, new_rejected=0)
    for run in json.loads(blob(args.archive, ARCHIVE + "index.json")):
        records = {}
        for row in run["rounds"]:
            if row["expected_cells"] != 9 or row["captured_cells"] != 9 or row["score"] is None:
                continue
            path = ARCHIVE + run["run"] + "/round-{:02d}/".format(row["round"])
            source = blob(args.archive, path + "candidate.py")
            report = json.loads(blob(args.archive, path + "round.json"))
            scores = {(cell["case"], cell["seed"]): cell["score"] for cell in report["cells"]}
            assert len(scores) == 9 and abs(sum(scores.values()) / 9 - row["score"]) < 1e-8
            assert hashlib.sha256(source.encode()).hexdigest() == row["candidate_sha256"]
            parent = records.get(row["parent_id"])
            parent_source = initial if parent is None else parent[0]
            if parent is None and row["parent_id"] != run["rounds"][0]["parent_id"]:
                raise ValueError("archive lacks the recorded parent of " + path)
            records[row["candidate_id"]] = (source, scores)
            totals["complete_rounds"] += 1
            if decoder_body(source) == decoder_body(parent_source):
                continue
            totals["changed_decoders"] += 1
            old = previous.probe_decoder(source, parent_source)
            new = current.probe_decoder(source, parent_source)
            totals["old_rejected"] += int(bool(old))
            totals["new_rejected"] += int(bool(new))
            result = dict(run=run["run"], round=row["round"], old_issue=old, new_issue=new)
            if parent is not None:
                result["unchanged_cell_scores"] = sum(v == parent[1][k] for k, v in scores.items())
                result["delta_points"] = row["score"] - sum(parent[1].values()) / 9
            print(json.dumps(result))
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
