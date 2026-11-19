import gzip
from pathlib import Path
import sys
import tempfile
import unittest

from test_replay import AGENT_DIR
sys.path.insert(0, str(AGENT_DIR))
from paired_feedback import published_critical_paths


def path_block(start, end, group, slack, *, path_type="max", status="VIOLATED"):
    return (f"Startpoint: {start}\nEndpoint: {end}\nPath Group: {group}\nPath Type: {path_type}\n"
            f"  Delay    Time    Description\n   0.00    0.00    clock clk (rise edge)\n"
            f"   {slack}   slack ({status})\n\n")


def register_trace(middle="u_bottom/comb/Z", slack="-0.39"):
    return f"""Startpoint: u_top/launch (rising edge-triggered flip-flop clocked by clk)
Endpoint: u_top/capture
          (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
Path Type: max

  Delay    Time   Description
---------------------------------------------------------
   0.00    0.00   clock clk (rise edge)
   0.00    0.00   clock network delay (ideal)
   0.00    0.00 ^ u_top/launch/CK (DFF_X1)
   0.05    0.05 v u_top/launch/Q (DFF_X1)
   0.10    0.15 ^ {middle} (BUF_X1)
   0.00    0.15 ^ u_top/capture/D (DFF_X1)
           0.15   data arrival time
   0.00    4.00 ^ u_bottom/clockbuf/Z (CLKBUF)
           4.00 ^ u_top/capture/CK (DFF_X1)
           3.96   data required time
---------------------------------------------------------
           3.96   data required time
          -0.15   data arrival time
           {slack}   slack (VIOLATED)
"""


class CriticalPaths(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "request/cells/example/1/result"
        self.cell = {"case": "example", "seed": "1", "status": "succeeded",
                     "metrics": {"final_global_wns_setup_ns": -0.3941}}
        self.result = {"request_id": "request", "status": "succeeded", "artifacts_dir": str(self.root / "request"),
                       "feedback": {"cells": [self.cell]}}
        self.reg = "r1 (rising edge-triggered flip-flop clocked by clk)"
        self.port = "din (input port clocked by in_clk)"
        self.out = "dout (output port clocked by out_clk)"

    def write(self, tree, text, compressed=False):
        file = self.folder / tree / "60_opt/assembly/passive/setup.rpt"
        file.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            with gzip.open(file.with_suffix(".rpt.gz"), "wt", encoding="utf-8") as out:
                out.write(text)
        else:
            file.write_text(text, encoding="utf-8")

    def paths(self):
        return published_critical_paths(self.result, self.root).get(("example", "1"))

    def test_compressed_report_lists_every_tied_type_across_path_groups(self):
        report = path_block(self.reg, self.reg, "clk", "-0.21")
        report += path_block(self.port, self.reg, "in_clk", "-0.39")
        report += path_block(self.reg, "dout\n          (output port clocked by clk)", "clk", "-0.39")
        report += path_block("macro (internal pin)", self.out, "other", "-0.39")
        report += path_block(self.port, self.reg, "other", "-1.02", path_type="min")
        self.write("retained_artifacts/algorithm", report, compressed=True)
        self.assertEqual(self.paths(), {"reported_slack_ns": "-0.39", "matched_paths": 3,
                                        "path_types": ["input_to_register", "register_to_output", "unknown_to_output"]})

    def test_original_report_precedes_retained_and_requires_agreement_at_printed_precision(self):
        self.write("run", path_block(self.reg, self.reg, "clk", "-0.394"))
        self.write("retained_artifacts/algorithm", path_block(self.port, self.out, "old", "-0.394"))
        self.assertEqual(self.paths()["path_types"], ["register_to_register"])
        self.cell["metrics"]["final_global_wns_setup_ns"] = -0.3946
        self.assertIsNone(self.paths())

    def test_report_and_receipt_must_both_show_a_violation(self):
        self.assertIsNone(self.paths())
        self.write("run", path_block(self.reg, self.reg, "clk", "-0.00", status="MET"))
        self.cell["metrics"]["final_global_wns_setup_ns"] = -0.002
        self.assertIsNone(self.paths())
        self.write("run", path_block(self.reg, self.reg, "clk", "-0.00"))
        self.assertEqual(self.paths()["reported_slack_ns"], "-0.00")
        self.cell["metrics"]["final_global_wns_setup_ns"] = 0.0
        self.assertIsNone(self.paths())

    def test_partly_written_report_cannot_attach_a_category(self):
        report = path_block(self.port, self.reg, "in_clk", "-0.39")
        report += "Startpoint: a\nEndpoint: b\nPath Type: max\n"
        self.write("run", report)
        self.assertIsNone(self.paths())

    def test_switch_range_covers_all_ties_and_ends_at_first_arrival(self):
        report = register_trace() + register_trace(middle="u_top/comb/Z")
        report += register_trace(middle="unqualified/comb/Z", slack="-0.21")
        self.write("retained_artifacts/algorithm", report, compressed=True)
        self.assertEqual(self.paths(), {"reported_slack_ns": "-0.39", "matched_paths": 2,
                                        "path_types": ["register_to_register"], "visible_layer_switch_range": [0, 2]})
        self.cell["metrics"]["final_global_wns_setup_ns"] = -0.41
        self.assertIsNone(self.paths())

    def test_unresolved_tied_trace_withholds_range_for_entire_cell(self):
        trace = register_trace()
        for original, replacement in (
            ("u_bottom/comb/Z", "wrapper/comb/Z"),
            ("^ u_bottom/comb/Z (BUF_X1)", "input external delay"),
            ("u_top/launch/CK", "u_top/another/CK"),
            ("   0.05    0.05 v u_top/launch/Q (DFF_X1)\n", ""),
            ("u_top/capture/D", "u_top/another/D"),
            ("0.15   data arrival time", "0.15   something missing"),
            ("clock network delay (ideal)", "clock network delay (propagated)"),
            ("0.10    0.15 ^", "...\n   0.10    0.15 ^"),
            ("u_top/launch (rising edge-triggered flip-flop", "u_top/launch (input port"),
            ("u_top/capture\n          (rising edge-triggered flip-flop", "u_top/capture (output port"),
        ):
            with self.subTest(original=original, replacement=replacement):
                self.write("run", trace + trace.replace(original, replacement, 1))
                summary = self.paths()
                self.assertEqual(summary["matched_paths"], 2)
                self.assertNotIn("visible_layer_switch_range", summary)
