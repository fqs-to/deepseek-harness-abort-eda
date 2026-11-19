import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AgenticBench/agent/loss_decoder"))
from opening_gradients import opening_gradients


def context(source, framework, case, seed):
    return SimpleNamespace(files={"source": str(source), "framework": str(framework)},
                           evaluation_cells=[{"case": case, "seed": str(seed)}])


class OpeningGradients(unittest.TestCase):
    def test_only_loss_with_two_choices_launches(self):
        with patch("opening_gradients.subprocess.run") as launch:
            self.assertIsNone(opening_gradients("decoder_only", None, "parent", {"A": "a", "B": "b"}))
            self.assertIsNone(opening_gradients("loss_only", None, "parent", {"A": "a"}))
            launch.assert_not_called()

    def test_missing_input_and_timeout_leave_review_available(self):
        with tempfile.TemporaryDirectory() as folder, patch("opening_gradients.subprocess.run") as launch:
            root = Path(folder)
            env = context(root, root, "training-design", 17)
            choices = {"A": "a", "B": "b"}
            self.assertIsNone(opening_gradients("loss_only", env, "p", choices, inputs=root))
            launch.assert_not_called()
            fp = root / "training-design/10_floorplan"
            fp.mkdir(parents=True)
            (fp / "partition.input").touch()
            launch.side_effect = subprocess.TimeoutExpired("python", 60)
            self.assertIsNone(opening_gradients("loss_only", env, "p", choices, inputs=root))
            request = json.loads(launch.call_args.kwargs["input"])
            self.assertEqual(request["programs"], {"parent": "p", **choices})
            self.assertEqual(request["seed"], 17)

    @unittest.skipUnless(os.environ.get("GRADIENT_TEST_TRAIN_CELL"), "needs a published training cell")
    def test_real_native_batch(self):
        # Provide the first training case/seed and the same read-only paths as the Agent.
        config = json.loads(os.environ["GRADIENT_TEST_TRAIN_CELL"])
        env = context(config["source"], config["framework"], config["case"], config["seed"])
        start = Path(config["framework"]) / "local_tasks/tp_gnn_loss_decoder_timing/initial_program.py"
        parent = start.read_text(encoding="utf-8")
        choices = {"A": parent.replace("return -(", "return -2.0 * (", 1),
                   "B": parent.replace("F.logsigmoid(-negative_scores).sum()",
                                       "2.0 * F.logsigmoid(-negative_scores).sum()", 1)}
        self.assertNotIn(parent, choices.values())
        report = opening_gradients("loss_only", env, parent, choices, inputs=Path(config["inputs"]))
        self.assertIsNotNone(report)
        self.assertEqual((report["case"], report["seed"]), (config["case"], config["seed"]))
        self.assertEqual((report["device"], report["epoch"], report["optimizer_steps"]), ("cpu", 0, 0))
        self.assertGreater(report["source_batch_size"], 0)
        self.assertRegex(report["sample_sha256"], "^[a-f0-9]{64}$")
        self.assertRegex(report["encoder_sha256"], "^[a-f0-9]{64}$")
        self.assertGreater(report["parent_gradient_norm"], 0)
        self.assertAlmostEqual(report["proposals"]["A"]["norm_ratio"], 2.0, places=5)
        self.assertAlmostEqual(report["proposals"]["A"]["delta_ratio"], 1.0, places=5)
        self.assertAlmostEqual(report["proposals"]["A"]["cosine"], 1.0, places=7)
        self.assertLess(report["proposals"]["B"]["cosine"], 1.0)
        print(json.dumps(report))


if __name__ == "__main__":
    unittest.main()
