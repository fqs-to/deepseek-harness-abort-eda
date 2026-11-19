import unittest
from copy import deepcopy
from statistics import mean

from test_search_policy import policy, support
from paired_feedback import comparison_for_prompt


def paired_summary(initial=True, movement_index=0, movement=0.25, gain=4.0, timing=0.05):
    cells = [{**entry, "status": "succeeded", "score": 20.0,
              "metrics": {"final_global_wns_setup_ns": -0.8, "final_global_tns_setup_ns": -2.0}}
             for entry in support.CELLS]
    parent = {"candidate_id": "p", "request_id": "before",
              "status": "published_reference" if initial else "succeeded",
              "feedback": {"score": 20.0, "cells": cells}}
    child = deepcopy(parent)
    child.update(candidate_id="c", request_id="after", status="succeeded")
    measured = child["feedback"]["cells"]
    measured[0]["score"] += gain
    measured[0]["metrics"]["final_global_wns_setup_ns"] += timing
    child["feedback"]["score"] = mean(cell["score"] for cell in measured)

    moved = policy.cell_key(measured[movement_index])
    cache = {("before", "after"): {moved: {"instances": 100, "changed_fraction": movement}}}
    comparison = comparison_for_prompt(child, "p", {"p": parent}, support.CELLS, assignment_cache=cache)
    if initial:
        for entry in measured:
            count = round(100 * movement) if policy.cell_key(entry) == moved else 0
            entry["partition_change_vs_initial"] = {"partition_assignment_changed_count": count,
                                                    "partition_assignment_changed_fraction": count / 100}
    return {**child["feedback"], "request_id": "after", "candidate_id": "c", "status": "succeeded",
            "parent_comparison": comparison}


class HeterogeneousEffect(unittest.TestCase):
    def test_one_measured_seed_can_make_a_single_function_gain_eligible_for_combination(self):
        summary = paired_summary()
        start = support.INITIAL_PROGRAM
        trials = []
        for intent, new in (("loss_only", support.program("hard_negative")),
                            ("decoder_only", support.program(decoder="flip"))):
            trial = policy.describe_trial({**summary, "request_id": intent,
                                          "attempt": {"search_intent": intent}},
                                         support.CELLS, start, new)
            self.assertEqual(trial["outcome"], "effective")
            trials.append(trial)
        self.assertEqual(policy.next_step(trials, 20.0, {})["action"], "recombine")

    def test_a_later_parent_uses_its_own_assignment_comparison(self):
        summary = paired_summary(initial=False)
        self.assertEqual(policy.classify(summary, support.CELLS)[0], "effective")
        summary["parent_comparison"]["cells"][0].pop("assignment_vs_parent")
        summary["cells"][0]["partition_change_vs_initial"] = {"partition_assignment_changed_count": 25}
        self.assertEqual(policy.classify(summary, support.CELLS)[0], "inert")

    def test_one_losing_seed_with_measured_movement_is_effective(self):
        for initial in (True, False):
            summary = paired_summary(initial=initial, gain=-35.0, timing=-0.5)
            with self.subTest(initial=initial):
                self.assertEqual(policy.classify(summary, support.CELLS)[0], "effective")

    def test_score_and_timing_regression_need_direct_concordant_evidence(self):
        noise = paired_summary(gain=-1.0, timing=-0.05)
        noise["parent_comparison"]["official_score_delta"] = -policy.SAME_SCORE_POINTS / 2
        for summary in (noise, paired_summary(gain=-1, timing=0.05),
                        paired_summary(gain=-1, timing=-policy.SAME_TIMING_NS / 2),
                        paired_summary(gain=-1, timing=-0.05, movement_index=1)):
            with self.subTest(comparison=summary["parent_comparison"]):
                self.assertEqual(policy.classify(summary, support.CELLS)[0], "inert")

    def test_incomplete_comparison_or_unconfirmed_coincident_gain_stays_inert(self):
        unpaired = paired_summary()
        unpaired["parent_comparison"]["complete"] = False
        no_net_gain = paired_summary()
        no_net_gain["parent_comparison"]["official_score_delta"] = -1
        for summary in (unpaired, no_net_gain, paired_summary(movement=0), paired_summary(movement_index=1),
                        paired_summary(gain=0), paired_summary(timing=policy.SAME_TIMING_NS / 2)):
            with self.subTest(comparison=summary["parent_comparison"]):
                self.assertEqual(policy.classify(summary, support.CELLS)[0], "inert")
