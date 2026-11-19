"""Mocked-bridge replays of a complete ten-evaluation run.

Run against the revised Agent directory (default) or any other one:

    python3 -m unittest discover -s tests -v
    AGENT_DIR=/path/to/AgenticBench/agent/loss_decoder python3 -m unittest discover -s tests -v

The scripted world encodes what the development experiments measured: adding the
excess-weight timing pull at 1x and at 4x changed the embeddings but left the
official assignment, the crossing nets and the timing identical to the start.
The scripted model is a stubborn proposer whose habit is to retune that
coefficient. Everything else about the world is invented and only has to be
self-consistent: the tests check how the harness spends evaluations, not which
algorithm is physically better.
"""

import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_support as support

AGENT_DIR = Path(os.environ.get("AGENT_DIR") or Path(__file__).resolve().parents[1]
                 / "AgenticBench" / "agent" / "loss_decoder")
LOSS_EFFECT = {  # name -> (mean points, scatter, fraction of instances reassigned)
    "hard_negative": (-2.0, 9.0, 0.21),
    "all_edges": (-1.5, 6.0, 0.00015),
}
DECODER_EFFECT = {
    "flip": (2.5, 5.0, 1.0), "orient_axis": (7.0, 5.0, 0.5), "orient_count": (-3.0, 6.0, 0.5),
    "gap": (-4.0, 6.0, 0.03), "gap_flip": (-6.0, 6.0, 1.0), "loop": (1.0, 5.0, 0.35), "loop_flip": (0.5, 5.0, 0.65),
}


def identify(source):
    """Name the fixture loss and decoder inside a submitted program."""
    loss_text, decoder_text = source.split(support.DECODER_SIGNATURE)
    loss = next(name for name, body in support.LOSSES.items()
                if body.split("\n")[0] in loss_text)
    coefficient = re.search(r"return sampled \+ ([0-9.e+-]+) \*", loss_text)
    decoder = next(name for name, body in support.DECODERS.items() if body.split("\n")[0] in decoder_text)
    return loss, float(coefficient[1]) if coefficient else 0.0, decoder


def world(source, attempt):
    """What the official flow reports for a fixture program."""
    loss, coefficient, decoder = identify(source)
    if decoder in ("float_labels",):
        return {"kind": "contract"}
    points = {key: 0.0 for key in support.spread("zero", 0.0, 0.0)}
    changed = 0.0
    # A normalized timing pull below 10x never reorders groups at the area-balanced boundary.
    loss_active = loss in LOSS_EFFECT or (loss in ("excess", "slack_axis") and coefficient >= 10)
    for active, name, table in ((loss_active, loss, LOSS_EFFECT), (decoder in DECODER_EFFECT, decoder, DECODER_EFFECT)):
        if active:
            mean_points, scatter, fraction = table.get(name, (-1.0, 6.0, 0.1))
            label = name if table is DECODER_EFFECT else f"{name}:{coefficient:g}"
            for key, value in support.spread(label, mean_points, scatter).items():
                points[key] += value
            changed = max(changed, fraction)
    if loss_active and decoder in DECODER_EFFECT:  # combined changes interfere
        for key, value in support.spread(loss + "+" + decoder, -3.0, 6.0).items():
            points[key] += value
    spec = {"kind": "effective" if changed else "inert", "points": points, "changed_fraction": changed,
            "cut": int(changed * 100)}
    if (loss, decoder, attempt) == ("initial", "orient_axis", 0):  # transient infrastructure failure
        spec["not_completed"] = {("bp-ng45", "1002"): {
            "error_type": "ResourceAdmissionError",
            "reason": "insufficient available memory for candidate admission"}}
    return spec


class StubbornProposer:
    """Scripted model. Its habit is to retune the coefficient of its first loss idea."""

    def __init__(self, losses, decoders):
        self.losses, self.decoders, self.calls = iter(losses), iter(decoders), []

    def __call__(self, system, request):
        if system.startswith("Review these opening proposals"):
            return "PICK A"
        intent = request.get("search_intent")
        component = (request.get("decision") or {}).get("component")
        wants_decoder = (intent == "decoder_only" or (intent == "repair" and component == "decoder")
                         or (request.get("required_component") == "decoder"
                             and request.get("previous_generation_issue")))
        parent_loss, parent_decoder = request["parent_program"].split(support.DECODER_SIGNATURE)
        name = next(self.decoders if wants_decoder else self.losses, None)
        if name is None:
            reply = "I have no further ideas."
        elif wants_decoder:
            reply = parent_loss + support.DECODER_SIGNATURE + support.DECODERS[name].replace("{c}", "0.02")
        else:
            reply = (support.LOSS_SIGNATURE + support.LOSSES[name[0]].replace("{c}", repr(float(name[1])))
                     + "\n\n" + support.DECODER_SIGNATURE + parent_decoder)
        self.calls.append((intent, name))
        return reply


def run(losses, decoders, evaluation_limit=10, model_limit=80, world=world):
    workdir = support.make_workdir()
    try:
        agent, copy = support.load_agent(AGENT_DIR, workdir)
        bridge = support.MockBridge(workdir, world, StubbornProposer(losses, decoders), evaluation_limit, model_limit)
        returned = agent.Agent().run(bridge)
        events = [json.loads(line) for line in (copy / "search.jsonl").read_text(encoding="utf-8").splitlines()]
        return bridge, returned, events
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def timing(result):
    return tuple((cell["case"], cell["seed"], cell["metrics"]["final_global_wns_setup_ns"],
                  cell["metrics"]["final_global_tns_setup_ns"]) for cell in result["feedback"]["cells"]
                 if "metrics" in cell)


def repeats(bridge):
    """Evaluations whose complete matrix reproduces the start or an earlier complete matrix exactly."""
    known = {tuple((cell["case"], cell["seed"], *support.PUBLISHED[cell["case"]]["seeds"][str(cell["seed"])])
                   for cell in support.CELLS)}
    repeated = []
    for result in bridge.evaluations():
        if bridge.complete(result):
            if timing(result) in known:
                repeated.append(result)
            known.add(timing(result))
    return repeated


def trace(bridge):
    lines = []
    repeated = repeats(bridge)
    for index, result in enumerate(bridge.evaluations(), 1):
        loss, coefficient, decoder = identify(result["source"])
        score = result["feedback"]["score"]
        note = "REPEATS A KNOWN RESULT" if result in repeated else "" if bridge.complete(result) else "incomplete"
        lines.append(f"  evaluation {index:2d}: loss={loss + ':' + format(coefficient, 'g'):18s} decoder={decoder:12s} "
                     f"score={'%6.2f' % score if score is not None else '  none'} {note}")
    return "\n".join(lines)


START_SCORE = support.mean(
    support.mean(support.score_metrics(case, {"final_global_wns_setup_ns": wns, "final_global_tns_setup_ns": tns})
                 ["combined_score"] for wns, tns in entry["seeds"].values())
    for case, entry in sorted(support.PUBLISHED.items()))


class ReplayTenEvaluations(unittest.TestCase):
    """One complete formal-sized run against a stubborn proposer."""

    @classmethod
    def setUpClass(cls):
        cls.bridge, cls.returned, cls.events = run(
            losses=[("excess", 1), ("excess", 4), ("excess", 2), ("excess", 8), ("hard_negative", 0.25),
                    ("all_edges", 4), ("excess", 3), ("excess", 6), ("slack_axis", 1), ("excess", 5),
                    ("excess", 7), ("hard_negative", 1.0), ("excess", 9), ("all_edges", 8)],
            decoders=["flip", "orient_axis", "gap", "loop", "orient_count", "loop_flip", "gap_flip"])
        cls.trace = "\n" + trace(cls.bridge)
        if os.environ.get("REPLAY_TRACE"):
            print(f"\n{AGENT_DIR}\n  published start score {START_SCORE:.2f}{cls.trace}\n  returned {cls.returned}, "
                  f"model calls {cls.bridge.model_calls}")

    def test_evaluations_are_not_spent_on_candidates_that_reproduce_the_parent(self):
        self.assertLessEqual(len(repeats(self.bridge)), 2, self.trace)

    def test_a_reviewed_out_or_infrastructure_stopped_mechanism_is_still_measured(self):
        # The proposer wrote the strongest mechanism of this world as its second decoder idea.
        measured = [result for result in self.bridge.evaluations()
                    if identify(result["source"])[2] == "orient_axis" and self.bridge.complete(result)]
        self.assertTrue(measured, self.trace)

    def test_returned_candidate_is_the_best_complete_verified_matrix(self):
        complete = [result for result in self.bridge.evaluations() if self.bridge.complete(result)]
        best = max(complete, key=lambda result: result["feedback"]["score"])
        self.assertEqual(self.returned, best["candidate_id"], self.trace)
        self.assertGreater(best["feedback"]["score"], START_SCORE + 5, self.trace)

    def test_budgets_are_respected(self):
        self.assertEqual(len(self.bridge.evaluations()), 10, self.trace)
        self.assertLess(self.bridge.model_calls, self.bridge.model_limit, self.trace)


class InitialPrompt(unittest.TestCase):
    def test_experience_only_in_first_request_even_after_interruption(self):
        with tempfile.TemporaryDirectory() as workdir:
            agent, folder = support.load_agent(AGENT_DIR, workdir)
            prompt = (folder / "prompt.md").read_text(encoding="utf-8")
            self.assertEqual(prompt.count(agent.EXPERIENCE_HEADING), 1)
            declaration = prompt[prompt.index(agent.EXPERIENCE_HEADING):].strip()
            units = sum(2 if word.isascii() and word.isalpha() else 1
                        for word in re.findall(r"[A-Za-z]+|[0-9]+|[\u3400-\u9fff]", declaration))
            self.assertLessEqual(units, 200)
            prompts = []
            proposer = StubbornProposer([], ["flip", "orient_axis"])

            def observe(system, request):
                prompts.append(system + "\n" + json.dumps(request, ensure_ascii=False))
                if len(prompts) == 1:
                    raise InterruptedError("stop before first evaluation")
                return proposer(system, request)

            bridge = support.MockBridge(workdir, world, observe, evaluation_limit=1)
            with self.assertRaises(InterruptedError):
                agent.Agent().run(bridge)
            self.assertEqual(bridge.evaluations(), [])
            agent.Agent().run(bridge)
            self.assertIn(declaration, prompts[0])
            self.assertGreaterEqual(len(prompts), 4)
            self.assertTrue(any(text.startswith(agent.OPENING_ALTERNATIVE) for text in prompts))
            self.assertTrue(any(text.startswith(agent.OPENING_REVIEW) for text in prompts))
            for text in prompts[1:]:
                self.assertNotIn(agent.EXPERIENCE_HEADING, text)
                self.assertNotIn(declaration.splitlines()[-1], text)


class IncompleteMatrix(unittest.TestCase):
    def test_a_candidate_with_an_unfinished_cell_is_never_the_final_answer(self):
        def scripted(source, attempt):
            spec = world(source, attempt)
            if identify(source)[0] == "hard_negative":  # its weakest cell never finishes, for an unknown reason
                spec["points"] = support.spread("lucky", 9.0, 2.0)
                spec["not_completed"] = {("bp-ng45", "1002"): {"reason": "cell did not finish"}}
            return spec

        bridge, returned, _ = run([("hard_negative", 0.25)], ["flip"], evaluation_limit=2, world=scripted)
        by_id = {result["candidate_id"]: result for result in bridge.evaluations()}
        self.assertTrue(bridge.complete(by_id[returned]), "\n" + trace(bridge))


class PartitionFeedback(unittest.TestCase):
    def test_parent_assignment_pairing_requires_complete_unique_instance_coverage(self):
        with tempfile.TemporaryDirectory() as workdir:
            agent, _ = support.load_agent(AGENT_DIR, workdir)
            root = Path(workdir) / "evaluations"
            schedule = [{"case": "chip", "seed": "1"}]
            cell = {**schedule[0], "status": "succeeded", "score": 0.0}

            def receipt(tag):
                return {"request_id": tag, "candidate_id": "id-" + tag, "status": "succeeded",
                        "artifacts_dir": str(root / tag), "feedback": {"cells": [cell], "score": 0.0}}

            parent, child = receipt("parent"), receipt("child")
            parents = {parent["candidate_id"]: parent}
            paths = []
            for record, tree, name in ((parent, "run", "partition_assignment.csv"),
                                       (child, "retained_artifacts/algorithm", "partition_assignment.csv.gz")):
                folder = Path(record["artifacts_dir"]) / "cells/chip/1/result"
                path = folder / tree / "20_partition" / name
                path.parent.mkdir(parents=True)
                (folder / "evaluation_result.json").write_text('{"metrics":{"instance_count":4}}', encoding="utf-8")
                paths.append(path)

            header = "inst_name,die\n"
            paths[0].write_text(header + "a,top\nb,top\nc,bottom\nd,bottom\n", encoding="utf-8")

            def movement(child_rows, cache):
                with gzip.open(paths[1], "wt", encoding="utf-8") as out:
                    out.write(header + child_rows)
                result = agent.comparison_for_prompt(child, parent["candidate_id"], parents, schedule, root, cache)
                return result["cells"][0].get("assignment_vs_parent")

            cache = {}
            valid = movement("d,bottom\nc,top\nb,bottom\na,bottom\n", cache)
            self.assertEqual(valid, {"instances": 4, "changed_fraction": 0.75,
                                     "changed_fraction_allowing_global_swap": 0.25})
            self.assertEqual(movement("", cache), valid)  # published pairs are immutable and read once
            self.assertIsNone(movement("a,top\na,top\nc,top\nd,bottom\n", {}))
            self.assertIsNone(movement("a,top\nb,top\nx,top\nd,bottom\n", {}))
            paths[0].write_text(header + "a,top\nb,top\n", encoding="utf-8")
            self.assertIsNone(movement("a,top\nb,bottom\n", {}))  # same two missing on both sides

    def test_next_proposal_receives_native_cut_measurements_without_an_assignment_reference(self):
        with tempfile.TemporaryDirectory() as workdir:
            agent, _ = support.load_agent(AGENT_DIR, workdir)
            requests = []
            proposer = StubbornProposer([("hard_negative", 0.25)], ["flip"])

            def observe(system, request):
                requests.append(request)
                return proposer(system, request)

            bridge = support.MockBridge(workdir, world, observe, evaluation_limit=1)
            agent.Agent().run(bridge)
            receipt = bridge.evaluations()[0]
            cells = receipt["feedback"]["cells"]
            stage_fields = {"tp_gnn_balanced_timing_weighted_cut": 120.0,
                            "tp_gnn_feasible_timing_weighted_cut": 112.0,
                            "tp_gnn_repair_moved_area_fraction": 0.0}
            for index, cell in enumerate(cells[:2]):
                path = (Path(receipt["artifacts_dir"]) / "cells" / cell["case"] / str(cell["seed"])
                        / "result/evaluation_result.json")
                if index == 0:
                    with gzip.open(path.with_suffix(".json.gz"), "wt", encoding="utf-8") as stream:
                        json.dump({"metrics": stage_fields}, stream)
                    path.unlink()
                else:
                    native = json.loads(path.read_text(encoding="utf-8"))
                    native["metrics"].update(tp_gnn_balanced_timing_weighted_cut=False,
                                             tp_gnn_feasible_timing_weighted_cut=102.0)
                    path.write_text(json.dumps(native), encoding="utf-8")

            requests.clear()
            bridge.evaluation_limit = 2
            agent.Agent().run(bridge)
            previous = next(r["incumbent_feedback"] for r in requests if r.get("incumbent_feedback"))
            first, second = previous["cells"][:2]
            self.assertEqual(first["partition_stages"], stage_fields)
            self.assertNotIn("partition_change_vs_initial", first)
            self.assertNotIn("tp_gnn_balanced_timing_weighted_cut", second["partition_stages"])
            self.assertEqual(second["partition_stages"]["tp_gnn_feasible_timing_weighted_cut"], 102.0)
            self.assertIn("partition_change_vs_initial", second)
            self.assertEqual(previous["partition_changes_vs_initial"]["reported_cells"], len(cells) - 1)


@unittest.skipUnless(support.importlib.util.find_spec("numpy"), "the decoder probe needs NumPy")
class DecoderContract(unittest.TestCase):
    def test_equivalence_requires_unequal_areas_and_clustered_embeddings(self):
        with patch.object(sys, "path", [str(AGENT_DIR), *sys.path]):
            import decoder_probe as probe
        parent = support.INITIAL_PROGRAM
        before_return = parent.removesuffix("    return labels\n")
        alternative = """        order = np.argsort(embeddings[:, 0], kind='stable')
        chosen = np.cumsum(node_areas[order]) <= balance_target * np.sum(node_areas)
        return chosen[np.argsort(order)].astype(np.int64)
    return labels
"""
        for profile, condition in (
            ("unequal_areas", "np.any(node_areas > 2 * np.mean(node_areas))"),
            ("clustered", "np.any(np.mean(np.square(embeddings), axis=0) > 2 / embeddings.shape[1])"),
        ):
            with self.subTest(profile=profile):
                source = before_return + "    if " + condition + ":\n" + alternative
                candidate, original = probe._decoder(source), probe._decoder(parent)
                for seed in probe.PROBE_SEEDS:
                    args = (*probe._probe_inputs(seed), seed, 50, 0.5)
                    self.assertTrue(probe.np.array_equal(candidate(*args), original(*args)))
                embeddings, areas = probe._probe_inputs(0, profile)
                args = embeddings, areas, 0, 50, 0.5
                labels = candidate(*args)
                self.assertIsNone(probe._contract_issue(labels, areas))
                self.assertAlmostEqual(areas[labels == 1].sum() / areas.sum(), 0.5, delta=0.03)
                changed = (labels != original(*args)).sum()
                self.assertGreater(min(changed, len(labels) - changed), 0)
                self.assertIsNone(probe.probe_decoder(source, parent))
        self.assertIn("matched the parent's labels", probe.probe_decoder(support.program(decoder="noop"), parent))

    def test_extra_probes_are_conditional_and_both_check_the_contract(self):
        with patch.object(sys, "path", [str(AGENT_DIR), *sys.path]):
            import decoder_probe as probe
        parent = support.INITIAL_PROGRAM
        with patch.object(probe, "_probe_inputs", wraps=probe._probe_inputs) as inputs:
            self.assertIsNone(probe.probe_decoder(support.program(decoder="flip"), parent))
            self.assertEqual([call.args for call in inputs.call_args_list], [(seed, "random") for seed in range(6)])
        source = parent.removesuffix("    return labels\n") + """    if np.any(node_areas > 2 * np.mean(node_areas)):
        return 1 - labels
    if np.any(np.mean(np.square(embeddings), axis=0) > 2 / embeddings.shape[1]):
        return labels * 1.0
    return labels
"""
        self.assertIn("labels must be integers", probe.probe_decoder(source, parent))

    def test_archived_decoder_runs_on_cold_numpy(self):
        archive = Path(__file__).resolve().parents[3] / "candidates"
        script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import decoder_probe as probe
archive = Path(sys.argv[2])
source = (archive / 'claude-round-04/program.py').read_text(encoding='utf-8')
parent = (archive / 'claude-round-02/program.py').read_text(encoding='utf-8')
embeddings, areas = probe._probe_inputs(0)
labels = probe._decoder(source)(embeddings, areas, 0, 50, 0.5)
assert probe._contract_issue(labels, areas) is None
issue = probe.probe_decoder(source, parent)
assert issue is None, issue
"""
        subprocess.run([sys.executable, "-B", "-c", script, str(AGENT_DIR), str(archive)], check=True)

    def test_nonempty_skewed_decoder_reaches_evaluation_after_invalid_proposals(self):
        bridge, _, events = run([("hard_negative", 0.25)], ["float_labels", "noop", "empty", "skewed"],
                                evaluation_limit=2)
        submitted = {identify(result["source"])[2] for result in bridge.evaluations()}
        self.assertFalse(submitted & {"float_labels", "noop", "empty"}, "\n" + trace(bridge))
        self.assertIn("skewed", submitted, "\n" + trace(bridge))


if __name__ == "__main__":
    unittest.main()
