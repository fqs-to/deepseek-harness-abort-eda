"""Measure three opening loss programs on one shared native CPU training batch."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def opening_gradients(intent, context, parent, choices, *, inputs=Path("/inputs")):
    if intent != "loss_only" or set(choices) != {"A", "B"}:
        return None
    try:
        cell = context.evaluation_cells[0]
        case, seed = cell["case"], int(cell["seed"])
        if not isinstance(case, str) or not case or Path(case).name != case or case in {".", ".."}:
            return None
        data = inputs / case
        if not (data / "10_floorplan/partition.input").is_file():
            return None
        request = {"case": case, "seed": seed, "data": str(data), "source": context.files["source"],
                   "framework": context.files["framework"], "programs": {"parent": parent, **choices}}
        env = dict(os.environ, PYTHONHASHSEED="0", OPEN3DBENCH_THREADS="2", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
        completed = subprocess.run([sys.executable, "-B", __file__], input=json.dumps(request),
                                   text=True, capture_output=True, timeout=60, env=env, check=True)
        return json.loads(completed.stdout)
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return None


def measure(request, workspace):
    import yaml
    import torch
    from unittest.mock import patch

    place = Path(request["source"]) / "Place-3D"
    sys.path.insert(0, str(place))
    from src.partition.lol.core import parse_lol_input, load_name_map
    from src.partition.lol.methods import tp_gnn, tp_gnn_encoder, tp_gnn_evolution

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    task = Path(request["framework"]) / "local_tasks/tp_gnn_loss_decoder_timing/task.yaml"
    recipe = yaml.safe_load(task.read_text())["definition"]["training"]["recipe"]
    settings = json.loads((place / recipe).read_text())["partition"]
    prefix = "tp_gnn_"
    settings = {(k[len(prefix):] if k.startswith(prefix) else k): value
                for k, value in settings.items() if k != "method"}
    data = Path(request["data"])
    floorplan, timing = data / "10_floorplan", data / "20_partition/timing_input"
    prototype = floorplan / "design.def"
    manifest = json.loads((timing / "timing_manifest.json").read_text())
    if hashlib.sha256(prototype.read_bytes()).hexdigest() != manifest["prototype_def_sha256"]:
        raise ValueError("Partition prototype differs from the training timing manifest")
    names = load_name_map(floorplan / "partition.names.csv")
    _, instances, nets = parse_lol_input(floorplan / "partition.input")
    if set(names) != {inst.name for inst in instances}:
        raise ValueError("Training instance names differ from the published name map")

    losses = {}
    interface = "tp-gnn-loss-decoder-v1"
    for label, source in request["programs"].items():
        path = workspace / (label + ".json")
        path.write_text(json.dumps({"interface_version": interface, "source": source}))
        functions, _ = tp_gnn_evolution.load_evolution_program(
            path, interface, reference_decoder=tp_gnn.area_balanced_weighted_kmeans)
        losses[label] = functions["compute_tp_gnn_loss"]

    params = []
    build = tp_gnn_encoder.build_encoder
    train = tp_gnn.train_random_walk_embeddings
    report = {}

    def remember_model(*args, **kwargs):
        model = build(*args, **kwargs)
        params.extend(p for p in model.parameters() if p.requires_grad)
        return model

    class FirstBatchMeasured(Exception):
        pass

    def observe(*batch):
        if sys.platform == "linux":
            import resource
            # Limit new virtual allocations after PyTorch has loaded on the shared 4-GiB host.
            pages = int(Path("/proc/self/statm").read_text().split()[0])
            target = pages * os.sysconf("SC_PAGE_SIZE") + 1024**3
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limits = [v for v in (soft, hard, target) if v != resource.RLIM_INFINITY]
            resource.setrlimit(resource.RLIMIT_AS, (min(limits), hard))

        def gradient(loss):
            parts = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
            result = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                                for g, p in zip(parts, params)]).double()
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite encoder gradient")
            return result

        reference = gradient(losses["parent"](*batch))
        norm = reference.norm().item()
        sample = hashlib.sha256()
        for indices in batch[3:7]:
            sample.update(indices.numel().to_bytes(8, "little"))
            sample.update(indices.detach().contiguous().numpy().tobytes())
        initial_weights = hashlib.sha256()
        for p in params:
            initial_weights.update(p.detach().contiguous().numpy().tobytes())
        report.update(case=request["case"], seed=request["seed"], device="cpu", epoch=int(batch[8]),
                      optimizer_steps=0, source_batch_size=int(batch[7]), sample_sha256=sample.hexdigest(),
                      encoder_sha256=initial_weights.hexdigest(), parent_gradient_norm=norm, proposals={},
                      meaning="One shared initial batch of CPU encoder gradients. "
                              "norm_ratio = ||g||/||parent||, delta_ratio = ||g-parent||/||parent||, "
                              "cosine = cos(g,parent). Physical timing is measured in the official evaluation.")
        for label in ("A", "B"):
            g = gradient(losses[label](*batch))
            g_norm = g.norm().item()
            cosine = g.dot(reference).item() / (norm * g_norm) if norm and g_norm else None
            report["proposals"][label] = {
                "norm_ratio": g_norm / norm if norm else None,
                "delta_ratio": (g - reference).norm().item() / norm if norm else None,
                "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None}
        raise FirstBatchMeasured

    def train_once(*args, evolution, **kwargs):
        functions, metadata = evolution
        return train(*args, evolution=({**functions, "compute_tp_gnn_loss": observe}, metadata), **kwargs)

    settings.update(seed=request["seed"], device="cpu", prototype_gpdef=str(prototype),
                    timing_features=str(timing / "timing_features.csv"),
                    timing_net_weights=str(timing / "timing_net_weights.csv"),
                    evolution_program=str(workspace / "parent.json"), output_dir=str(workspace))
    try:
        # Published canonical names map the partition instance IDs back to the DEF.
        with patch.object(tp_gnn, "load_c_to_original_name", return_value=names), \
                patch.object(tp_gnn_encoder, "build_encoder", remember_model), \
                patch.object(tp_gnn, "train_random_walk_embeddings", train_once):
            tp_gnn.tp_gnn_partition(instances, nets, tp_gnn.TPGNNConfig(**settings))
    except FirstBatchMeasured:
        return report
    raise ValueError("The native partitioner did not produce a training batch")


if __name__ == "__main__":
    from contextlib import redirect_stdout

    with tempfile.TemporaryDirectory() as scratch, redirect_stdout(sys.stderr):
        result = measure(json.load(sys.stdin), Path(scratch))
    print(json.dumps(result, allow_nan=False))
