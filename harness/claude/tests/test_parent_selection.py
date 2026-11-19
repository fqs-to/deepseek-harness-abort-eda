import unittest

import replay_support as support
from test_search_policy import CELL_KEYS, OPENED, START, policy, trial


def choose(trials, remaining, **kwargs):
    return policy.next_step(OPENED + trials, START, dict.fromkeys(CELL_KEYS, START),
                            evaluations_remaining=remaining, **kwargs)


class ParentSelection(unittest.TestCase):
    def test_alternative_shows_an_exact_measured_function_and_cross_case_cost(self):
        focus = CELL_KEYS[0][0]
        original = support.program("hard_negative", "gap")
        changed = support.program("hard_negative", "loop")
        parent_scores = {key: (8 if key[0] == focus else 45) for key in CELL_KEYS}
        other_scores = {key: (15 if key[0] == focus else 20) for key in CELL_KEYS}
        parent = trial("parent", key=policy.function_keys(original), cell_scores=parent_scores)
        other = trial("other", key=policy.function_keys(changed), cell_scores=other_scores)
        sources = {other["request_id"]: changed}
        result = policy.alternative_on_weakest_case(parent, [parent, other], sources)
        self.assertEqual(result["changed_function"], "decoder")
        self.assertEqual(result["focus_case"], focus)
        self.assertTrue(result["function_source"].startswith("def decode_tp_gnn_embeddings("))
        self.assertNotIn("def compute_tp_gnn_loss", result["function_source"])
        self.assertEqual(set(result["paired_score_deltas_on_focus_case"].values()), {7})
        self.assertIn(-25, result["mean_score_delta_by_case"].values())
        self.assertIsNone(policy.alternative_on_weakest_case(parent, [other], sources, component="loss"))
        self.assertIsNone(policy.alternative_on_weakest_case(parent, [dict(other, complete=False)], sources))
        one_winner = dict(parent_scores)
        one_winner[CELL_KEYS[0]] += 60
        self.assertIsNone(policy.alternative_on_weakest_case(
            parent, [dict(other, cell_scores=one_winner)], sources))

    def test_pair_of_measured_gains_precedes_the_review_shelf(self):
        loss = trial("loss", score=24, changed=("loss",))
        decoder = trial("decoder", score=25)
        shelf = [{"key": "unmeasured", "changed": ["loss"]}]
        self.assertEqual(choose([loss, decoder], 5, shelved=shelf)["action"], "recombine")

    def test_broader_gains_get_one_parent_slot_and_preserve_the_official_winner(self):
        leader_scores = dict.fromkeys(CELL_KEYS, 25)
        leader_scores[CELL_KEYS[0]] = 88
        leader = trial("leader", score=32, cell_scores=leader_scores, changed=("loss", "decoder"))
        other = trial("other", score=30, changed=("loss", "decoder"))
        history = [other, leader]
        self.assertEqual(choose(history, 2)["parent"], other["request_id"])
        self.assertEqual(choose(history, 1)["parent"], leader["request_id"])
        self.assertEqual(policy.final_choice(history, START)[0]["candidate_id"], "leader")

    def test_one_outlying_case_does_not_earn_an_alternative_parent_slot(self):
        leader = trial("leader", score=30, changed=("loss", "decoder"))
        first_case = CELL_KEYS[0][0]
        scores = {key: 50 if key[0] == first_case else 18 for key in CELL_KEYS}
        other = trial("other", score=sum(scores.values()) / len(scores), cell_scores=scores,
                      changed=("loss", "decoder"))
        self.assertEqual(choose([other, leader], 2)["parent"], leader["request_id"])
