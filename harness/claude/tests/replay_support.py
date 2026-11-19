"""Offline stand-ins for the organizer runtime.

Nothing here is part of the Agent package. It provides
  * stub modules for the framework imports the Agent needs,
  * a mocked bridge that returns official-looking replies from a scripted world,
  * a scripted model persona, and
  * fixture programs. The fixtures are development material and stay out of the
    Agent directory; the Agent only ever sees them as model output.
"""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types


# Published official measurements of the initial program (reference/published_training_reference.json).
PUBLISHED = {
    "ariane133-ng45": {"wns_ref": 0.5290353246723486, "tns_ref": 96.42498143235612, "seeds": {
        "1001": (-0.46473759489492794, -22.69812846368777),
        "1002": (-0.511323442504157, -101.00025635050363),
        "1003": (-0.5738236714876622, -29.767709140211068)}},
    "bp-ng45": {"wns_ref": 8.663351750761597, "tns_ref": 6622.78575944895, "seeds": {
        "1001": (-8.392474508445966, -9059.339646875793),
        "1002": (-7.841612026113778, -10349.946316496395),
        "1003": (-7.870566643414895, -8622.276498609612)}},
    "bp_fe-ng45": {"wns_ref": 0.9002498352590634, "tns_ref": 716.9190029865956, "seeds": {
        "1001": (-0.6611195111821111, -174.72074415142822),
        "1002": (-0.7280568576313236, -211.7487329856929),
        "1003": (-0.639466275808939, -188.35428327940795)}},
}
CELLS = [{"case": case, "seed": int(seed)} for case, entry in PUBLISHED.items() for seed in entry["seeds"]]
DECODER_TASK = "tp_gnn_loss_decoder"
FUNCTION_NAMES = ["compute_tp_gnn_loss", "decode_tp_gnn_embeddings"]

LOSS_SIGNATURE = '''def compute_tp_gnn_loss(
    positive_scores,
    negative_scores,
    embeddings,
    positive_sources,
    positive_targets,
    negative_sources,
    negative_targets,
    source_batch_size,
    epoch,
    max_epochs,
    graph_scores,
    graph_sources,
    graph_targets,
    graph_net_ids,
    graph_weights,
    net_weights,
    node_features,
    node_areas,
):
'''
DECODER_SIGNATURE = "def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):\n"
SAMPLED = '''    sampled = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
'''
LOSSES = {
    "initial": '    """Reproduce the maintained random-walk negative-sampling objective."""\n' + SAMPLED
               + "    return sampled\n",
    # experiments/candidates/criticality_excess_affinity*.py with the coefficient as a parameter
    "excess": '    """Use excess timing weight to pull together the ends of important nets."""\n' + SAMPLED
              + "    edge_net_weight = net_weights[graph_net_ids]\n"
              + "    criticality = graph_weights * (edge_net_weight - 1.0) / edge_net_weight\n"
              + "    pull = (criticality * (1.0 - graph_scores)).sum()\n"
              + "    pull = pull / torch.clamp(criticality.sum(), min=1e-12)\n"
              + "    return sampled + {c} * pull\n",
    "all_edges": '    """Encourage embedding agreement on nets with higher timing weight."""\n' + SAMPLED
                 + "    affinity = (graph_weights * (1.0 - graph_scores)).sum()\n"
                 + "    affinity = affinity / torch.clamp(graph_weights.sum(), min=1e-12)\n"
                 + "    return sampled + {c} * affinity\n",
    "hard_negative": '    """Give highly similar negative pairs an additional bounded penalty."""\n' + SAMPLED
                     + "    hard = F.relu(negative_scores - 0.25).square().sum() / max(1, source_batch_size)\n"
                     + "    return sampled + {c} * hard\n",
    "slack_axis": '    """Correlate embedding axis 0 with slack so the decoder can orient dies by content."""\n' + SAMPLED
                  + "    slack = node_features[:, 0]\n"
                  + "    anchor = -((slack.mean() - slack) * embeddings[:, 0]).sum() / max(1, source_batch_size)\n"
                  + "    return sampled + {c} * anchor\n",
}
REFERENCE = "    labels = reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)\n"
CENTERS = (REFERENCE
           + "    weights = np.maximum(node_areas, 1.0)\n"
           + "    top = labels == 1\n"
           + "    center_top = (embeddings * (weights * top)[:, None]).sum(axis=0) / (weights * top).sum()\n"
           + "    center_bottom = (embeddings * (weights * ~top)[:, None]).sum(axis=0) / (weights * ~top).sum()\n")
GAP = (CENTERS
       + "    projection = (embeddings * (center_top - center_bottom)).sum(axis=1)\n"
       + "    order = np.argsort(-projection, kind='stable')\n"
       + "    share = np.cumsum(weights[order]) / weights.sum()\n"
       + "    gaps = projection[order][:-1] - projection[order][1:]\n"
       + "    legal = np.abs(share[:-1] - balance_target) <= {c}\n"
       + "    cut = np.argmax(np.where(legal, gaps, -1.0))\n"
       + "    rank = np.argsort(order, kind='stable')\n")
LOOP = ("    weights = np.maximum(node_areas, 1.0)\n"
        "    total = weights.sum()\n"
        "    center = (embeddings * weights[:, None]).sum(axis=0) / total\n"
        "    spread = (weights[:, None] * (embeddings - center) ** 2).sum(axis=0)\n"
        "    direction = (np.arange(embeddings.shape[1]) == np.argmax(spread)).astype(np.float64)\n"
        "    labels = np.zeros(embeddings.shape[0], dtype=np.int64)\n"
        "    for _ in range(max_iterations):\n"
        "        order = np.argsort(-(embeddings * direction).sum(axis=1), kind='stable')\n"
        "        chosen = np.cumsum(weights[order]) <= total * balance_target\n"
        "        labels = chosen[np.argsort(order, kind='stable')].astype(np.int64)\n"
        "        top = labels == 1\n"
        "        direction = ((embeddings * (weights * top)[:, None]).sum(axis=0) / (weights * top).sum()\n"
        "                     - (embeddings * (weights * ~top)[:, None]).sum(axis=0) / (weights * ~top).sum())\n")
DECODERS = {
    "initial": '    """Reproduce the frozen area-balanced global two-way decoder."""\n' + REFERENCE + "    return labels\n",
    "flip": '    """Evaluate the same grouping with the two die assignments exchanged."""\n' + REFERENCE
            + "    return 1 - labels\n",
    "orient_axis": '    """Keep the grouping; put the cluster with the higher axis-0 mean on the bottom die."""\n' + CENTERS
                   + "    return np.where(center_top[0] > center_bottom[0], 1 - labels, labels)\n",
    "orient_count": '    """Keep the grouping; put the cluster with more groups on the top die."""\n' + REFERENCE
                    + "    return np.where(labels.sum() * 2 < labels.shape[0], 1 - labels, labels)\n",
    "gap": '    """Re-cut the reference ordering at the widest margin inside the legal area window."""\n' + GAP
           + "    return (rank <= cut).astype(np.int64)\n",
    "gap_flip": '    """Widest-margin cut inside the legal area window, with the dies exchanged."""\n' + GAP
                + "    return (rank > cut).astype(np.int64)\n",
    "loop": '    """Seed-free split: start on the highest-variance axis and iterate area-median cuts."""\n' + LOOP
            + "    return labels\n",
    "loop_flip": '    """Seed-free area-median split with the dies exchanged."""\n' + LOOP + "    return 1 - labels\n",
    # Contract and equivalence fixtures for the decoder probe.
    "noop": '    """Rescale the labels."""\n' + REFERENCE + "    return labels * 1\n",
    "float_labels": '    """Return soft labels."""\n' + REFERENCE + "    return labels * 1.0\n",
    "empty": '    """Return an empty top side."""\n    return np.zeros_like(node_areas).astype(np.int64)\n',
    "skewed": '    """Produce a two-sided, roughly 55% top-area proposal before closure."""\n'
              + "    order = np.argsort(embeddings[:, 0])\n"
              + "    top = np.cumsum(node_areas[order]) <= 0.55 * node_areas.sum()\n"
              + "    return top[np.argsort(order)].astype(np.int64)\n",
}


def program(loss="initial", decoder="initial", c=1.0, d=0.02):
    """A complete two-function fixture program."""
    return (LOSS_SIGNATURE + LOSSES[loss].replace("{c}", repr(float(c))) + "\n\n"
            + DECODER_SIGNATURE + DECODERS[decoder].replace("{c}", repr(float(d))))


INITIAL_PROGRAM = program()


def score_metrics(case, raw):
    """The published objective: 40% WNS and 60% TNS deficit improvement, floored at -100."""
    entry = PUBLISHED[case]
    wns = 1 - max(0.0, -raw["final_global_wns_setup_ns"]) / entry["wns_ref"]
    tns = 1 - max(0.0, -raw["final_global_tns_setup_ns"]) / entry["tns_ref"]
    return {**raw, "wns_deficit_improvement": wns, "tns_deficit_improvement": tns,
            "combined_score": max(-100.0, 100 * (0.4 * wns + 0.6 * tns))}


def validate_program(source, task):
    import ast
    tree = ast.parse(source)
    if [getattr(node, "name", None) for node in tree.body] != FUNCTION_NAMES:
        raise ValueError("Return exactly the two editable functions in their original order")
    return source.strip() + "\n"


class AgentAPIError(RuntimeError):
    pass


def install_framework_stubs():
    def module(name, **attributes):
        created = types.ModuleType(name)
        created.__dict__.update(attributes)
        sys.modules[name] = created
        return created

    module("contest2", __path__=[])
    module("contest2.agent_api", AgentAPIError=AgentAPIError)
    module("contest2.model_pricing", MODEL_ID="deepseek-v4.1-flash")
    module("contest2.scoring", score_metrics=score_metrics, reference_table=lambda: {"cases": {
        case: {"split": "train", "measurements": {"loss_decoder": {
            seed: {"wns_ns": wns, "tns_ns": tns} for seed, (wns, tns) in entry["seeds"].items()}}}
        for case, entry in PUBLISHED.items()}})
    module("local_tasks", __path__=[])
    module("local_tasks.tp_gnn_local_task_definitions", DECODER_TASK=DECODER_TASK, validate_program=validate_program)


def load_agent(agent_directory, workdir):
    """Import a private copy of an Agent directory so its log and program.py stay out of the source tree."""
    install_framework_stubs()
    copy = Path(workdir) / "agent"
    shutil.copytree(agent_directory, copy, ignore=shutil.ignore_patterns("__pycache__", "search.jsonl", "opening.txt"))
    for name in ("agent", "paired_feedback", "search_policy", "decoder_probe", "opening_gradients"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(copy))
    try:
        loaded = importlib.import_module("agent")
        # The organizer keeps the participant path active throughout Agent.run.
        # Cache this lazily imported helper from the same private copy before restoring sys.path.
        if (copy / "opening_gradients.py").is_file():
            importlib.import_module("opening_gradients")
    finally:
        sys.path.remove(str(copy))
    loaded.time.sleep = lambda seconds: None
    return loaded, copy


def mean(values):
    values = list(values)
    return sum(values) / len(values)


class MockBridge:
    """An AgentContext look-alike. `world(source, attempt)` scripts what the official flow would report."""

    task = "loss_decoder"
    baseline_candidate = "official-initial"

    def __init__(self, workdir, world, persona, evaluation_limit=10, model_limit=80):
        root = Path(workdir)
        task = root / "framework" / "local_tasks" / DECODER_TASK
        task.mkdir(parents=True)
        (task / "initial_program.py").write_text(INITIAL_PROGRAM, encoding="utf-8")
        (task / "prompt.md").write_text("# Official numerical interface (placeholder for offline tests)\n", encoding="utf-8")
        (root / "workspaces").mkdir()
        (root / "evaluations").mkdir()
        self.files = {"framework": str(root / "framework"), "workspaces": str(root / "workspaces"),
                      "evaluations": str(root / "evaluations")}
        self.evaluation_cells = [dict(cell) for cell in CELLS]
        self.world, self.persona = world, persona
        self.evaluation_limit, self.model_limit = evaluation_limit, model_limit
        self.records, self.order, self.model_calls, self.submissions = {}, [], 0, {}
        self.model = types.SimpleNamespace(responses_create=self._model)

    def _model(self, payload):
        if self.model_calls >= self.model_limit:
            raise AgentAPIError("model budget exhausted")
        self.model_calls += 1
        system, user = (message["content"] for message in payload["input"])
        text = self.persona(system, json.loads(user))
        return {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": text}]}]}

    def status(self):
        remaining = self.evaluation_limit - len(self.order)
        return {"history": [{"request_id": request_id} for request_id in self.order],
                "evaluations_remaining": remaining, "evaluation_dispatch_allowed": remaining > 0,
                "model_dispatch_allowed": self.model_calls < self.model_limit, "model_stop_reasons": []}

    def wait_evaluation(self, request_id, poll_seconds=5):
        return self.records[request_id]

    evaluation_status = wait_evaluation

    def submit_evaluation(self, source, request_id=None):
        if request_id in self.records:
            return self.records[request_id]
        if len(self.order) >= self.evaluation_limit:
            raise AgentAPIError("evaluation budget exhausted")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        attempt = self.submissions.get(digest, 0)
        self.submissions[digest] = attempt + 1
        spec = self.world(source, attempt)
        artifacts = Path(self.files["evaluations"]) / request_id
        cells = []
        for cell in CELLS:
            case, seed = cell["case"], str(cell["seed"])
            wns, tns = PUBLISHED[case]["seeds"][seed]
            if (case, seed) in spec.get("not_completed", {}):
                cells.append({"case": case, "seed": cell["seed"], "status": "not_completed", "valid": False,
                              **spec["not_completed"][case, seed]})
                continue
            if spec["kind"] == "contract":
                cells.append({"case": case, "seed": cell["seed"], "status": "failed", "valid": False,
                              "score": -100.0, "error_type": "CandidateContractError",
                              "reason": "20_partition: decoder must return one binary integer label per group"})
                continue
            points = spec.get("points", {}).get((case, seed), 0.0)
            metrics = score_metrics(case, {
                "final_global_wns_setup_ns": wns,
                "final_global_tns_setup_ns": tns + points / 60 * PUBLISHED[case]["tns_ref"],
                "cut_net_count": 3102 + spec.get("cut", 0), "timing_weighted_cut": 5576.0 + spec.get("cut", 0)})
            cells.append({"case": case, "seed": cell["seed"], "status": "succeeded", "valid": True,
                          "score": metrics["combined_score"], "metrics": metrics})
            changed = spec.get("changed_fraction", 0.0)
            native = artifacts / "cells" / case / seed / "result"
            native.mkdir(parents=True, exist_ok=True)
            (native / "evaluation_result.json").write_text(json.dumps({"metrics": {
                "partition_assignment_identical": int(changed == 0),
                "partition_assignment_changed_count": int(round(changed * 40000)),
                "partition_assignment_changed_fraction": changed,
                "tp_gnn_repair_moved_area_fraction": 0.0}}), encoding="utf-8")
        scored = [cell for cell in cells if "score" in cell]
        result = {"request_id": request_id, "candidate_id": "candidate-" + request_id[2:10],
                  "status": "failed" if spec["kind"] == "contract" else "succeeded",
                  "reason": "candidate contract violation" if spec["kind"] == "contract" else "",
                  "source": source, "artifacts_dir": str(artifacts), "spec": spec,
                  "feedback": {"cells": cells, "score": mean(
                      mean(cell["score"] for cell in scored if cell["case"] == case)
                      for case in sorted({cell["case"] for cell in scored})) if scored else None}}
        self.records[request_id] = result
        self.order.append(request_id)
        return result

    # Views used by the assertions.
    def evaluations(self):
        return [self.records[request_id] for request_id in self.order]

    def complete(self, result):
        cells = result["feedback"]["cells"]
        return result["status"] == "succeeded" and len(cells) == len(CELLS) and all(
            cell["status"] != "not_completed" and "score" in cell for cell in cells)


def spread(name, mean_points, width=6.0):
    """Deterministic per-cell score movement in points: a mean shift plus design-and-seed scatter."""
    points = {}
    for cell in CELLS:
        digest = hashlib.sha256(f"{name}/{cell['case']}/{cell['seed']}".encode()).digest()
        points[cell["case"], str(cell["seed"])] = mean_points + width * (digest[0] / 127.5 - 1)
    return points


def make_workdir():
    return tempfile.mkdtemp(prefix="task2-replay-")
