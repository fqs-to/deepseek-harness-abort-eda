"""Compare archived decoders on saved embeddings, before native feasibility repair.

Requires the released Place-3D source, NumPy and PyTorch. The saved training
directory provides embeddings.npy, contracted_supernodes.csv, training_config.json
and program.json. The parent directory supplies native-report.json, program.json,
contracted_supernodes.csv and decoder_tpgnn_assignment.csv. No training is run.
Example from the repository root, with DATA pointing to one saved CPU run:

  python scripts/compare_saved_decoder_labels.py --place-dir /path/to/Place-3D \
    --embeddings-dir "$DATA/r2" --parent-dir "$DATA/r4" --parent-round 4 --other-round 6
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


INTERFACE = "tp-gnn-loss-decoder-v1"
ARCHIVE = "results/experiments-2026-11-18/15332-claude-original"
DEFAULT_COMMIT = "1e7c625d68f4e0679bc325eb37b257423aa0a235"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def loss_structure(source):
    loss = ast.parse(source).body[0]
    if loss.name != "compute_tp_gnn_loss":
        raise ValueError("Expected a loss_decoder program")
    if ast.get_docstring(loss) is not None:
        loss.body.pop(0)
    return ast.dump(loss, include_attributes=False)


def archive_candidate(repo, commit, number):
    prefix = f"{commit}:{ARCHIVE}/round-{number:02d}/"
    def blob(name):
        return subprocess.check_output(["git", "-C", str(repo), "show", prefix + name])
    raw = blob("candidate.py")
    receipt = json.loads(blob("round.json"))
    if (sha(raw) != receipt["candidate_sha256"] or receipt["round"] != number
            or receipt["ledger_status"] != "succeeded" or receipt["captured_cells"] != 9):
        raise ValueError(f"Round {number} source or completed training receipt differs")
    return raw.decode("utf-8"), {"round": number, "source_sha256": sha(raw), "candidate_id": receipt["candidate_id"]}


def groups_from_csv(path, n):
    areas = np.zeros(n, dtype=np.float64)
    owners = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            group, name, area = int(row["supernode_id"]), row["inst_name"], float(row["area"])
            if not 0 <= group < n or not name or name in owners or not np.isfinite(area) or area < 0:
                raise ValueError("Invalid or repeated instance in contracted supernodes")
            areas[group] += area
            owners[name] = group
    if not (areas > 0).all():
        raise ValueError("Each embedding row must correspond to a positive-area supernode")
    return areas, owners


def compare(args):
    sys.path.insert(0, str(args.place_dir.resolve(strict=True)))
    from src.partition.lol.methods import tp_gnn, tp_gnn_evolution
    saved, parent_dir = args.embeddings_dir, args.parent_dir
    def read(path):
        return json.loads(path.read_bytes())

    parent, parent_id = archive_candidate(args.repo, args.commit, args.parent_round)
    alternative, alternative_id = archive_candidate(args.repo, args.commit, args.other_round)
    report = read(parent_dir / "native-report.json")
    cfg, training = report["config"], read(saved / "training_config.json")
    train_program = read(saved / "program.json")
    parent_program = read(parent_dir / "program.json")
    trained = sha(train_program["source"].encode())
    if (parent_program != {"interface_version": INTERFACE, "source": parent}
            or report["evolution_decoder"]["source_sha256"] != parent_id["source_sha256"]
            or train_program["interface_version"] != INTERFACE or trained != training["evolution"]["source_sha256"]
            or trained != report["training"]["evolution"]["source_sha256"] or training["seed"] != cfg["seed"]
            or report["training"]["seed"] != cfg["seed"]):
        raise ValueError("Native report, saved training or candidate source provenance differs")
    if len({loss_structure(s) for s in (parent, alternative, train_program["source"])}) != 1:
        raise ValueError("Candidates and saved training must use the same loss")
    groups = saved / "contracted_supernodes.csv"
    if filehash(groups) != filehash(parent_dir / groups.name):
        raise ValueError("Parent and saved training group membership differ")
    path = saved / "embeddings.npy"
    points = np.load(path, allow_pickle=False)
    if points.shape != (report["supernode_count"], training["embedding_dim"]):
        raise ValueError("Saved embedding shape differs from its native training record")
    areas, owners = groups_from_csv(groups, len(points))

    def decode(program_path):
        functions, metadata = tp_gnn_evolution.load_evolution_program(
            program_path, INTERFACE, reference_decoder=tp_gnn.area_balanced_weighted_kmeans)
        labels, _ = tp_gnn_evolution.decode_evolution_embeddings(
            functions, points, areas, cfg["seed"], cfg["kmeans_iterations"], cfg["balance_target"], metadata)
        return np.asarray(labels)

    parent_labels = decode(parent_dir / "program.json")
    labels_path = parent_dir / "decoder_tpgnn_assignment.csv"
    checked = set()
    with labels_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["inst_name"]
            if (name in checked or name not in owners
                    or row["die"] != ("top" if parent_labels[owners[name]] else "bottom")):
                raise ValueError("Parent replay disagrees with native preclosure assignment for " + name)
            checked.add(name)
    if checked != owners.keys():
        raise ValueError("Native parent assignment omits instances")
    with tempfile.TemporaryDirectory() as folder:
        other_path = Path(folder) / "other.json"
        other_path.write_text(json.dumps({"interface_version": INTERFACE, "source": alternative}), encoding="utf-8")
        different = decode(other_path) != parent_labels
    count = int(different.sum())
    return {"scope": "saved-embedding/preclosure", "seed": cfg["seed"], "training_device": training["device"],
            "archive_commit": args.commit, "parent": parent_id, "alternative": alternative_id,
            "training_source_sha256": trained, "embeddings_sha256": filehash(path), "groups_sha256": filehash(groups),
            "native_parent_assignment_sha256": filehash(labels_path), "parent_verified_instances": len(checked),
            "groups": len(points), "changed_groups": count, "instances": len(owners),
            "changed_instances": int(different[np.fromiter(owners.values(), dtype=np.intp)].sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("place-dir", "embeddings-dir", "parent-dir"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--parent-round", type=int, default=4)
    parser.add_argument("--other-round", type=int, default=6)
    print(json.dumps(compare(parser.parse_args()), indent=2))
