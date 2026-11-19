"""Synthetic opening-review regressions. No new EDA score is measured here."""
import json
import shutil
import sys
import unittest
from unittest.mock import patch

import replay_support as support
import test_replay as replay


class ReviewedShelf(unittest.TestCase):
    def setUp(self):
        self.root = support.make_workdir()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.agent, self.folder = support.load_agent(replay.AGENT_DIR, self.root)
        (self.folder / "opening.txt").write_text("loss_only", encoding="utf-8")
        self.path = self.folder / "search.jsonl"
        self.offered = [support.program(loss="all_edges", c=3.125), support.program(loss="slack_axis", c=7.125)]
        self.delegate = replay.StubbornProposer(
            [("excess", 1), ("excess", 4), ("excess", 2), ("excess", 8), ("hard_negative", 0.25),
             ("all_edges", 4), ("excess", 3), ("excess", 6), ("slack_axis", 1), ("excess", 5),
             ("excess", 7), ("hard_negative", 1), ("excess", 9), ("all_edges", 8)],
            ["flip", "orient_axis", "gap", "loop", "orient_count", "loop_flip", "gap_flip"])
        self.calls = 0

    def persona(self, system, request):
        self.calls += 1
        if self.calls <= 2:
            return self.offered[self.calls - 1]
        if self.calls == 3:
            self.assertEqual(list(request["proposals"]), ["A", "B"])
            return "REVISE: Neither proposed change addresses the target behavior."
        return self.delegate(system, request)

    def launch(self, bridge):
        with patch.object(sys, "path", [str(self.folder), *sys.path]):
            return self.agent.Agent().run(bridge)

    def test_ten_submissions_leave_both_revised_proposals_unused(self):
        bridge = support.MockBridge(self.root, replay.world, self.persona, evaluation_limit=10)
        selected = self.launch(bridge)
        rows = bridge.evaluations()
        self.assertEqual(len(rows), 10)
        self.assertIn(selected, [row["candidate_id"] for row in rows])
        removed = {self.agent.function_keys(s) for s in self.offered}
        self.assertTrue(removed.isdisjoint(self.agent.function_keys(row["source"]) for row in rows))
        self.assertTrue(removed.isdisjoint(row["key"] for row in self.agent.load_shelf(self.path)))

    def test_exhausted_model_cannot_submit_revised_proposals_even_after_old_log_recovery(self):
        bridge = support.MockBridge(self.root, replay.world, self.persona, evaluation_limit=10, model_limit=3)
        with self.assertRaisesRegex(RuntimeError, "No candidate completed"):
            self.launch(bridge)
        self.assertEqual(bridge.evaluations(), [])
        for source in self.offered:
            self.agent.write_event(self.folder, "shelved", source=source, parent_id=bridge.baseline_candidate,
                                   search_intent="loss_only", changed_components={"loss": "fixture"})
        # In the old writer these two rows immediately followed that review.
        events = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        review = next(e for e in events if e.get("verdict") == "revise")
        self.path.write_text("\n".join(json.dumps(e) for e in [review, *events[-2:]]) + "\n", encoding="utf-8")
        self.assertEqual(self.agent.load_shelf(self.path), [])
        with self.assertRaisesRegex(RuntimeError, "No candidate completed"):
            self.launch(bridge)
        self.assertEqual(bridge.evaluations(), [])
        self.assertEqual(bridge.model_calls, 3)

    def test_a_different_review_or_a_missing_review_preserves_the_shelved_program(self):
        def shelve(source, parent):
            self.agent.write_event(self.folder, "shelved", source=source, parent_id=parent,
                                   search_intent="loss_only", changed_components={"loss": "fixture"})
        self.agent.write_event(self.folder, "opening_review", verdict="revise", proposals=["A", "B"],
                               parent_id="old", search_intent="loss_only")
        shelve(self.offered[0], "other")
        self.agent.write_event(self.folder, "opening_review", verdict="A", proposals=["A", "B"],
                               parent_id="old", search_intent="loss_only")
        shelve(self.offered[1], "old")
        self.assertEqual([r["source"] for r in self.agent.load_shelf(self.path)], self.offered)
