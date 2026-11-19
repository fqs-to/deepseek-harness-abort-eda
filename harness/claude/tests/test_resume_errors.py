import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock

import replay_support as support
import test_replay as replay


AGENT_DIR = Path(__file__).resolve().parents[1] / "AgenticBench/agent/loss_decoder"
LOST = "Agent bridge communication failed; request outcome is unknown"


class ResumeErrors(unittest.TestCase):
    def setUp(self):
        self.workdir = support.make_workdir()
        self.addCleanup(shutil.rmtree, self.workdir)
        self.agent, self.folder = support.load_agent(AGENT_DIR, self.workdir)
        self.log = self.folder / "search.jsonl"

    def test_metered_model_calls_record_visible_content_and_gateway_errors_once(self):
        bridge = Mock()
        request = {"hypothesis": "分层"}
        private = "fixture private reasoning"
        partial = {"status": "incomplete", "output": [
            {"type": "reasoning", "summary": private},
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]},
        ], "usage": {"input_tokens": 8, "output_tokens": 2, "other": private}}
        complete = {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": " final "}]}
        ]}
        bridge.model.responses_create.side_effect = [partial, support.AgentAPIError(LOST), complete]
        self.assertIs(self.agent.ask_model(bridge, "first", request), partial)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.agent.model_text(partial)
        with self.assertRaisesRegex(support.AgentAPIError, "outcome is unknown"):
            self.agent.ask_model(bridge, "second", request, purpose="opening_alternative")
        self.assertEqual(self.agent.model_text(self.agent.ask_model(bridge, "third", request)), "final")
        events = list(self.agent.search_events(self.log))
        self.assertEqual([e["event"] for e in events], ["model_request", "model_response"] * 3)
        self.assertEqual(bridge.model.responses_create.call_count, 3)
        self.assertEqual(len({e["call_id"] for e in events}), 3)
        for sent, received, actual in zip(events[::2], events[1::2], bridge.model.responses_create.call_args_list):
            self.assertEqual(received["call_id"], sent["call_id"])
            self.assertEqual(received["purpose"], sent["purpose"])
            self.assertEqual(sent["payload"], actual.args[0])
        self.assertEqual(events[1]["output_text"], "partial")
        self.assertEqual(events[1]["usage"], {"input_tokens": 8, "output_tokens": 2})
        self.assertEqual(events[3]["status"], "gateway_error")
        self.assertEqual(events[5]["output_text"], " final ")
        self.assertNotIn(private, self.log.read_text(encoding="utf-8"))

    def test_restart_recovers_failed_source_and_retries_once_without_a_model_call(self):
        losses = [("excess", 1), ("excess", 4), ("hard_negative", 0.25)]
        decoders = ["flip", "orient_axis", "gap", "loop"]
        bridge = support.MockBridge(self.workdir, replay.world, replay.StubbornProposer(losses, decoders), 3)
        self.agent.Agent().run(bridge)
        failed = bridge.evaluations()[-1]
        old_source = failed["source"]
        failed.update(status="failed", reason="insufficient available memory for candidate admission")
        failed.pop("source")  # The real bridge exposes the source only on a successful receipt.
        original_wait, original_status = bridge.wait_evaluation, bridge.status
        remaining_blips = {"status": 1, "wait": 1}

        def read(method, *args):
            key = "wait" if args else "status"
            if remaining_blips[key]:
                remaining_blips[key] -= 1
                raise support.AgentAPIError(LOST)
            return method(*args)

        bridge.wait_evaluation = lambda rid: read(original_wait, rid)
        bridge.status = lambda: read(original_status)
        calls = bridge.model_calls
        bridge.evaluation_limit = 4
        self.agent.Agent().run(bridge)
        key = self.agent.function_keys(old_source)
        self.assertEqual(bridge.order[-1], self.agent.evaluation_id(key, failed["request_id"]))
        self.assertEqual(bridge.evaluations()[-1]["source"], old_source)
        self.assertEqual(bridge.model_calls, calls)
        self.assertEqual(len(set(bridge.order)), 4)

    def test_old_log_and_mismatched_source_still_block_duplicate_evaluation_id(self):
        source = support.program("hard_negative")
        key = self.agent.function_keys(source)
        rid = self.agent.evaluation_id(key)
        old_event = {"event": "submitted", "request_id": rid}
        wrong_event = {**old_event, "source": support.program("excess")}
        self.log.write_text("\n".join(map(json.dumps, [old_event, wrong_event])) + '\n{"incomplete')
        self.assertEqual(self.agent.submitted_sources(self.log, {rid}), {})
        with self.assertRaisesRegex(ValueError, "already has an official evaluation"):
            self.agent.check_proposal(source, support.INITIAL_PROGRAM, support.INITIAL_PROGRAM,
                                      {"intent": "loss_only"}, True, set(), [], recorded_ids={rid})

    def test_accepted_after_lost_ack_only_polls_same_request(self):
        bridge = Mock()
        bridge.submit_evaluation.side_effect = support.AgentAPIError(LOST)
        bridge.status.side_effect = [support.AgentAPIError(LOST), {"history": [{"request_id": "e-one"}]}]
        bridge.wait_evaluation.side_effect = [support.AgentAPIError(LOST), {"status": "failed"}]
        self.assertEqual(self.agent.submit_and_wait(bridge, "program", "e-one"), {"status": "failed"})
        bridge.submit_evaluation.assert_called_once_with("program", request_id="e-one")
        self.assertEqual([c.args for c in bridge.wait_evaluation.call_args_list], [("e-one",)] * 2)
        bridge.model.responses_create.assert_not_called()

    def test_missing_after_lost_ack_resends_identical_request(self):
        bridge = Mock()
        bridge.status.return_value = {"history": []}
        bridge.submit_evaluation.side_effect = [support.AgentAPIError(LOST), {"status": "running"}]
        bridge.wait_evaluation.return_value = {"status": "succeeded"}
        self.assertEqual(self.agent.submit_and_wait(bridge, "program", "e-one"), {"status": "succeeded"})
        self.assertEqual([(c.args, c.kwargs) for c in bridge.submit_evaluation.call_args_list],
                         [(("program",), {"request_id": "e-one"})] * 2)
        bridge.model.responses_create.assert_not_called()

    def test_definitive_failure_and_exhausted_bridge_do_not_spin(self):
        bridge = Mock()
        bridge.status.return_value = {"history": []}
        bridge.submit_evaluation.side_effect = support.AgentAPIError("different candidate for this request")
        with self.assertRaisesRegex(support.AgentAPIError, "different candidate"):
            self.agent.submit_and_wait(bridge, "program", "e-one")
        self.assertEqual(bridge.submit_evaluation.call_count, 1)
        bridge.wait_evaluation.side_effect = support.AgentAPIError(LOST)
        with self.assertRaisesRegex(support.AgentAPIError, "communication failed"):
            self.agent.bridge_read(bridge.wait_evaluation, "e-one")
        self.assertEqual(bridge.wait_evaluation.call_count, 5)

    def test_twelve_rejected_opening_proposals_recover_and_finish_ten_official_evaluations(self):
        losses = [(name, n) for n in range(1, 15) for name in ("hard_negative", "all_edges", "slack_axis")]
        decoders = ["flip", "orient_axis", "gap", "loop", "orient_count", "loop_flip", "gap_flip"]
        regular = replay.StubbornProposer(losses, decoders)
        requests = []

        def persona(system, request):
            requests.append(request)
            if len(requests) <= 12:
                return support.INITIAL_PROGRAM  # already measured, so all twelve are declined locally
            if len(requests) == 13:
                self.assertEqual(request["decision"]["reason"], "recover_after_rejected_proposals")
                self.assertIsNone(request["required_component"])
                return support.program("hard_negative", c=0.125)  # loss-only despite the original decoder opening
            return regular(system, request)

        bridge = support.MockBridge(self.workdir, replay.world, persona)
        chosen = self.agent.Agent().run(bridge)
        events = [json.loads(l) for l in self.log.read_text(encoding="utf-8").splitlines()]
        first = next(i for i, event in enumerate(events) if event["event"] == "submitted")
        self.assertEqual(sum(event["event"] == "invalid_proposal" for event in events[:first]), 12)
        self.assertEqual(events[first]["decision"]["reason"], "recover_after_rejected_proposals")
        self.assertEqual(len(set(bridge.order)), 10)
        self.assertIn(chosen, [row["candidate_id"] for row in bridge.evaluations()])

    def test_recovery_obeys_actual_model_budget(self):
        bridge = support.MockBridge(self.workdir, replay.world, lambda *_: support.INITIAL_PROGRAM, model_limit=12)
        with self.assertRaisesRegex(RuntimeError, "No candidate completed"):
            self.agent.Agent().run(bridge)
        self.assertEqual(bridge.model_calls, 12)
        self.assertEqual(bridge.order, [])
        stops = [e for e in self.agent.search_events(self.log) if e["event"] == "budget_stop"]
        self.assertEqual(stops[-1]["reason"], "model_budget")

    def test_unchanged_unusable_recovery_reply_eventually_stops(self):
        bridge = support.MockBridge(self.workdir, replay.world, lambda *_: support.INITIAL_PROGRAM)
        with self.assertRaisesRegex(RuntimeError, "No candidate completed"):
            self.agent.Agent().run(bridge)
        self.assertEqual(bridge.model_calls, 12 + self.agent.REPEATED_RECOVERY_LIMIT)
        self.assertEqual(bridge.order, [])
        self.assertIn("same_unusable_recovery_reply", self.log.read_text(encoding="utf-8"))
