"""Unit tests for the revised decision rules and the paths the ten-evaluation replay does not reach."""

import json
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_support as support
import test_replay as replay

AGENT_DIR = replay.AGENT_DIR
sys.path.insert(0, str(AGENT_DIR))
import search_policy as policy

START = 20.0
CELL_KEYS = [(cell["case"], str(cell["seed"])) for cell in support.CELLS]


def trial(name, outcome="effective", score=START, intent="diversify", changed=("decoder",), **fields):
    row = {"request_id": "e-" + name, "candidate_id": name, "outcome": outcome,
           "complete": outcome in policy.VERIFIED, "failure_component": None, "score": score, "tns": 0.0,
           "intent": intent, "parent_id": "official-initial", "retry_of": None, "from_shelf": False,
           "edited": list(changed), "key": ("loss:" + name, "decoder:" + name), "family": name,
           "changed_vs_start": list(changed), "cell_scores": {key: score for key in CELL_KEYS},
           "comparison_complete": True, "official_score_delta": None if score is None else score - START,
           "improved_pairs": 0,
           "regressed_pairs": 0, "scored_pairs": 9, "largest_regression": None, "invalid_cells": 0,
           "largest_assignment_change_fraction": 0.3}
    row.update(fields)
    return row


def step(trials, **options):
    return policy.next_step(trials, START, {key: START for key in CELL_KEYS}, evaluations_remaining=5, **options)


OPENED = [trial("d1", intent="decoder_only", score=19.0), trial("l1", intent="loss_only", changed=("loss",), score=19.5)]


class Structure(unittest.TestCase):
    def test_coefficient_changes_are_twins_but_subscript_positions_are_not(self):
        base, stronger = support.program("excess", c=1), support.program("excess", c=4)
        self.assertNotEqual(policy.function_keys(base), policy.function_keys(stronger))
        self.assertEqual(policy.function_keys(base, True), policy.function_keys(stronger, True))
        self.assertEqual(policy.rescale_ratio(base, stronger), 4.0)
        other_column = support.program("slack_axis").replace("node_features[:, 0]", "node_features[:, 3]")
        self.assertNotEqual(policy.function_keys(support.program("slack_axis"), True),
                            policy.function_keys(other_column, True))

    def test_family_names_the_inputs_and_decisions_an_edit_adds(self):
        initial = support.INITIAL_PROGRAM
        self.assertEqual(policy.mechanism_family(initial, initial), "initial")
        self.assertEqual(policy.mechanism_family(support.program("excess", c=4), initial),
                         "loss[graph_net_ids+graph_scores+graph_weights+net_weights]")
        self.assertEqual(policy.mechanism_family(support.program("hard_negative"), initial), "loss[sampled_pairs]")
        self.assertEqual(policy.mechanism_family(support.program(decoder="flip"), initial), "decoder[reference]")
        self.assertTrue(policy.mechanism_family(support.program(decoder="loop"), initial).startswith("decoder[custom+"))


class Evidence(unittest.TestCase):
    def summary(self, **cell_changes):
        cells = [{"case": case, "seed": seed, "status": "succeeded", "score": 1.0} for case, seed in CELL_KEYS]
        cells[4].update(cell_changes)
        return {"status": "succeeded", "score": 1.0, "cells": cells}

    def test_one_identical_cell_does_not_classify_nine_as_inert(self):
        summary = self.summary()
        summary["parent_comparison"] = {
            "parent_source": "published_initial_reference",
            "cells": [{"wns_slack_delta_ns": 0.0, "tns_slack_delta_ns": 0.0}],
        }
        summary["partition_changes_vs_initial"] = {"reported_cells": 1, "identical_assignments": 1}
        self.assertNotEqual(policy.classify(summary, support.CELLS)[0], "inert")

    def test_only_a_full_verified_matrix_is_complete(self):
        self.assertTrue(policy.matrix_complete(self.summary(), support.CELLS))
        self.assertTrue(policy.matrix_complete(self.summary(status="failed", score=-100.0), support.CELLS))
        self.assertFalse(policy.matrix_complete(self.summary(status="not_completed"), support.CELLS))
        self.assertFalse(policy.matrix_complete(self.summary(score=None), support.CELLS))
        self.assertFalse(policy.matrix_complete(self.summary(failure={"kind": "runtime_environment"}), support.CELLS))
        short = self.summary()
        short["cells"].pop()
        self.assertFalse(policy.matrix_complete(short, support.CELLS))

    def test_official_scoring_includes_published_penalties_and_requires_all_cells(self):
        penalty = {"status": "penalized", "valid": False, "score": -100,
                   "failure": {"kind": "runtime_environment"}}
        self.assertTrue(policy.officially_scored(self.summary(**penalty), support.CELLS))
        for change in ({"status": "not_completed"}, {"status": "failed"}, {"score": None},
                       {"score": -90}, {"valid": True}):
            self.assertFalse(policy.officially_scored(self.summary(**{**penalty, **change}), support.CELLS))
        summary = self.summary()
        summary["cells"][-1] = summary["cells"][0]
        self.assertFalse(policy.officially_scored(summary, support.CELLS))
        summary = self.summary()
        summary["status"] = "failed"
        self.assertFalse(policy.officially_scored(summary, support.CELLS))

    def test_same_timing_as_the_parent_is_inert_and_infrastructure_is_not_evidence(self):
        same = self.summary()
        same["parent_comparison"] = {"cells": [{"wns_slack_delta_ns": 0.0, "tns_slack_delta_ns": 0.0}] * 9}
        self.assertEqual(policy.classify(same, support.CELLS)[0], "inert")
        moved = self.summary()
        moved["parent_comparison"] = {"cells": [{"wns_slack_delta_ns": 0.01, "tns_slack_delta_ns": -3.0}] * 9}
        self.assertEqual(policy.classify(moved, support.CELLS)[0], "effective")
        stopped = self.summary(status="not_completed", score=None, failure={"kind": "runtime_environment"})
        self.assertEqual(policy.classify(stopped, support.CELLS)[0], "environment_failure")
        broken = self.summary(status="failed", failure={"kind": "candidate_contract", "component": "decoder"})
        self.assertEqual(policy.classify(broken, support.CELLS), ("contract_failure", "decoder"))


class NextStep(unittest.TestCase):
    def test_opening_tests_each_function_once_and_ignores_infrastructure_failures(self):
        self.assertEqual(step([])["intent"], "decoder_only")
        stopped = trial("d0", "environment_failure", None, "decoder_only", cell_scores={}, retry_of="e-earlier")
        self.assertEqual(step([stopped])["intent"], "decoder_only")
        self.assertEqual(step([OPENED[0]])["intent"], "loss_only")

    def test_an_infrastructure_failure_is_retried_once_when_untested_or_promising(self):
        untested = trial("d0", "environment_failure", None, "decoder_only", cell_scores={})
        self.assertEqual(step([untested])["action"], "resubmit")
        lagging = trial("x", "environment_failure", None, cell_scores={CELL_KEYS[0]: START - 5})
        self.assertEqual(step(OPENED + [lagging])["action"], "propose")
        leading = trial("y", "environment_failure", None, cell_scores={CELL_KEYS[0]: START + 5})
        self.assertEqual(step(OPENED + [leading])["target"], "e-y")
        retried = trial("y2", "environment_failure", None, cell_scores={CELL_KEYS[0]: START + 5}, retry_of="e-y")
        self.assertEqual(step(OPENED + [leading, retried])["action"], "propose")

    def test_a_contract_failure_repairs_the_same_candidate_and_function(self):
        broken = trial("d0", "contract_failure", None, "decoder_only", failure_component="decoder")
        chosen = step([broken])
        self.assertEqual((chosen["intent"], chosen["parent"], chosen["component"]), ("repair", "e-d0", "decoder"))

    def test_an_inert_loss_edit_sends_the_search_to_the_decoder(self):
        inert = trial("l1", "inert", START, "loss_only", ("loss",))
        chosen = step([OPENED[0], inert])
        self.assertEqual((chosen["intent"], chosen["required_component"], chosen["decision"]["reason"]),
                         ("diversify", "decoder", "assignment_unchanged"))
        self.assertIsNone(chosen["parent"])

    def test_set_aside_opening_alternatives_are_measured_decoder_first(self):
        shelved = [{"key": "a", "changed": ["loss"]}, {"key": "b", "changed": ["decoder"]}]
        self.assertEqual(step(OPENED, shelved=shelved)["index"], 1)
        self.assertEqual(step(OPENED, shelved=shelved, considered={"b"})["index"], 0)
        used = [trial("s1", from_shelf=True, score=19.0), trial("s2", from_shelf=True, score=19.0)]
        self.assertEqual(step(OPENED + used, shelved=shelved)["action"], "propose")

    def test_two_verified_single_function_gains_are_combined_exactly(self):
        decoder = trial("d1", intent="decoder_only", score=24.0)
        loss = trial("l1", intent="loss_only", changed=("loss",), score=23.0)
        chosen = step([decoder, loss])
        self.assertEqual((chosen["action"], chosen["loss"], chosen["decoder"], chosen["parent"]),
                         ("recombine", "e-l1", "e-d1", "e-d1"))
        self.assertEqual(step([decoder, loss], considered={chosen["key"]})["action"], "propose")
        self.assertEqual(step([decoder, trial("l1", "inert", START, "loss_only", ("loss",))])["action"], "propose")

    def test_an_incumbent_is_refined_once_and_then_built_on_with_other_families(self):
        leader = trial("d1", intent="decoder_only", score=25.0)
        loss = trial("l1", intent="loss_only", changed=("loss",), score=19.0)
        self.assertEqual(step([loss, leader])["intent"], "refine")
        later = step([leader, loss, trial("r1", score=24.0, parent_id="d1", changed=("loss", "decoder"))])
        self.assertEqual((later["intent"], later["parent"], later["decision"]["reason"]),
                         ("diversify", "e-d1", "explore_from_incumbent"))
        self.assertIn("d1", later["avoid"])

    def test_with_nothing_ahead_the_search_restarts_from_the_start_on_the_less_tested_function(self):
        chosen = step(OPENED + [trial("d2", score=18.0)])
        self.assertEqual((chosen["parent"], chosen["required_component"], chosen["decision"]["reason"]),
                         (None, "loss", "published_start_still_ahead"))

    def test_final_choice_needs_a_complete_matrix(self):
        lucky = trial("partial", "unverified", 40.0)
        best, selection = policy.final_choice([OPENED[0], lucky, trial("d2", score=22.0)], START)
        self.assertEqual(best["candidate_id"], "d2")
        self.assertEqual((selection["eligible_candidates"], selection["beats_published_start"]), (2, True))
        self.assertIsNone(policy.final_choice([lucky], START)[0])


class AgentPaths(unittest.TestCase):
    def test_terminal_choice_can_differ_from_search_incumbent_after_a_scored_penalty(self):
        workdir = support.make_workdir()
        self.addCleanup(shutil.rmtree, workdir)
        agent, folder = support.load_agent(AGENT_DIR, workdir)
        bridge = support.MockBridge(workdir, replay.world, None, evaluation_limit=2)
        clean = bridge.submit_evaluation(support.program(decoder="flip"), "e-first")
        leader = bridge.submit_evaluation(support.program(decoder="gap"), "e-second")
        cells = leader["feedback"]["cells"]
        for cell in cells:
            cell["score"] = 60.0
        cells[0].update(status="penalized", valid=False, score=-100.0, metrics={},
                        reason="30_placement: insufficient GPU memory")
        leader["feedback"]["score"] = (8 * 60.0 - 100.0) / 9
        self.assertGreater(leader["feedback"]["score"], clean["feedback"]["score"])
        self.assertEqual(agent.Agent().run(bridge), leader["candidate_id"])
        events = [json.loads(line) for line in (folder / "search.jsonl").read_text().splitlines()]
        self.assertEqual([e["candidate_id"] for e in events if e["event"] == "incumbent"], [clean["candidate_id"]])
        self.assertEqual(events[-1]["selection"]["eligible_candidates"], 2)
        self.assertEqual(events[-1]["score"], leader["feedback"]["score"])
        self.assertEqual((folder / "program.py").read_text(), leader["source"])

        # A legacy cached partial mean can survive even when every evaluation lacks a finished cell.
        cells[0]["status"] = clean["feedback"]["cells"][0]["status"] = "not_completed"
        with self.assertRaisesRegex(RuntimeError, "No candidate completed"):
            agent.Agent().run(bridge)

    def test_recombination_is_an_exact_splice_that_costs_no_model_call(self):
        def world(source, attempt):
            spec = replay.world(source, attempt)
            loss, _, decoder = replay.identify(source)
            spec["points"] = {key: 4.0 * (loss == "hard_negative") + 3.0 * (decoder == "flip") for key in CELL_KEYS}
            return spec

        bridge, returned, events = replay.run([("hard_negative", 0.25)], ["flip"], evaluation_limit=3, world=world)
        programs = [replay.identify(result["source"]) for result in bridge.evaluations()]
        self.assertEqual(programs[2], ("hard_negative", 0.25, "flip"), "\n" + replay.trace(bridge))
        first, second, third = (result["source"] for result in bridge.evaluations())
        self.assertEqual(third.split(support.DECODER_SIGNATURE)[1], first.split(support.DECODER_SIGNATURE)[1])
        self.assertEqual(third.split(support.DECODER_SIGNATURE)[0].rstrip(),
                         second.split(support.DECODER_SIGNATURE)[0].rstrip())
        # Each opening costs a proposal, an alternative and a review. The third evaluation costs nothing.
        self.assertEqual(bridge.model_calls, 6)
        self.assertEqual(returned, bridge.evaluations()[2]["candidate_id"])

    def test_a_restarted_process_continues_the_same_run(self):
        losses = [("excess", 1), ("excess", 4), ("hard_negative", 0.25), ("all_edges", 4), ("slack_axis", 1)]
        decoders = ["flip", "orient_axis", "gap", "loop", "orient_count", "loop_flip", "gap_flip"]
        workdir = support.make_workdir()
        try:
            agent, copy = support.load_agent(AGENT_DIR, workdir)
            bridge = support.MockBridge(workdir, replay.world, replay.StubbornProposer(losses, decoders), 3)
            agent.Agent().run(bridge)                      # stops after the infrastructure failure
            self.assertFalse(bridge.complete(bridge.evaluations()[-1]))
            bridge.evaluation_limit = 6
            returned = agent.Agent().run(bridge)           # a new Agent rebuilds its state from the bridge and log
            programs = [replay.identify(result["source"]) for result in bridge.evaluations()]
            self.assertEqual(programs[3], programs[2], "\n" + replay.trace(bridge))   # the retry
            self.assertEqual(len(set(bridge.order)), 6)
            self.assertEqual(returned, bridge.evaluations()[3]["candidate_id"])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_model_output_parsing(self):
        workdir = support.make_workdir()
        try:
            agent, _ = support.load_agent(AGENT_DIR, workdir)
            reply = lambda text: {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": text}]}]}
            fenced = "Here is the program.\n```python\n" + support.program(decoder="flip") + "```\nIt swaps the dies."
            self.assertEqual(agent.generated_program(reply(fenced)), support.program(decoder="flip"))
            self.assertEqual(agent.generated_program(reply(support.program("excess"))), support.program("excess"))
            with self.assertRaises((ValueError, SyntaxError)):
                agent.generated_program(reply("I have no further ideas."))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
