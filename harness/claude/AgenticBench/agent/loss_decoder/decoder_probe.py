"""Pre-submission contract probe for candidate decoders.

A decoder is deterministic NumPy code, so its contract can be checked before an
official evaluation is spent on it. The probe runs a validated candidate on
small synthetic inputs and reports violations of the published decoder
contract. Matching labels on these inputs do not establish equivalence on
the actual training designs. Official evaluations measure physical effects.
The probe is skipped when NumPy is unavailable in the agent container.

The stand-in for the frozen helper mirrors the published excerpt of the
official area-balanced decoder. The organizer's helper stays authoritative.
"""

import builtins
import random
import signal
import threading

try:
    import numpy as np
except Exception:  # The probe is optional; the search runs without it.
    np = None


PROBE_SEEDS = range(6)
PROBE_ITERATIONS = 50
PROBE_BALANCE = 0.5
PROBE_SECONDS = 20
SAFE_BUILTINS = {name: getattr(builtins, name)
                 for name in ("abs", "bool", "float", "int", "len", "max", "min", "range", "round", "sum")}


def _area_constrained_labels(preference, weights, target_area):
    order = sorted(range(len(preference)), key=lambda index: (-preference[index], index))
    labels = [0] * len(preference)
    selected_area = 0.0
    for index in order:
        next_area = selected_area + max(float(weights[index]), 0.0)
        if abs(next_area - target_area) <= abs(selected_area - target_area):
            labels[index] = 1
            selected_area = next_area
    return labels


def reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target):
    array = np.asarray(embeddings.tolist(), dtype=np.float64)
    weights = np.maximum(np.asarray(node_areas.tolist(), dtype=np.float64), 1.0)
    target_area = float(weights.sum()) * balance_target
    first, second = random.Random(seed).sample(range(array.shape[0]), 2)
    centers = np.stack([array[first].copy(), array[second].copy()])
    labels = None
    for _ in range(max(1, max_iterations)):
        distance_0 = np.sum((array - centers[0]) ** 2, axis=1)
        distance_1 = np.sum((array - centers[1]) ** 2, axis=1)
        following = np.asarray(_area_constrained_labels((distance_0 - distance_1).tolist(), weights.tolist(),
                                                        target_area), dtype=np.int8)
        if labels is not None and np.array_equal(labels, following):
            break
        labels = following
        for cluster in (0, 1):
            mask = labels == cluster
            if not np.any(mask):
                raise RuntimeError("area-balanced TP-GNN clustering produced an empty cluster")
            centers[cluster] = np.average(array[mask], axis=0, weights=weights[mask])
    return np.asarray(labels.tolist(), dtype=np.int64)


def _probe_inputs(seed, profile="random"):
    """Generate unit-length embeddings and positive group areas for an interface check."""
    generator = np.random.default_rng(seed)
    count, width = 256, 16
    embeddings = generator.standard_normal((count, width))
    if profile == "clustered":
        # Three overlapping directions in a plane, with independent noise in every dimension.
        angles = generator.integers(3, size=count) * (2 * np.pi / 3)
        embeddings *= 0.16
        embeddings[:, 0] += np.cos(angles)
        embeddings[:, 1] += np.sin(angles)
    # Use the official decoder's float64 input. Running the ndarray reduction here also
    # loads NumPy's reduction helpers before validated code enters its restricted scope.
    embeddings = embeddings / np.sqrt(np.square(embeddings).sum(axis=1, keepdims=True))
    if profile == "random":
        areas = generator.uniform(1, 2, size=count)
    else:
        areas = np.ones(count)
        if profile == "unequal_areas":
            areas[generator.choice(count, size=count // 16, replace=False)] = 9.0
    embeddings.setflags(write=False)
    areas.setflags(write=False)
    return embeddings, areas


def _decoder(source):
    scope = {"__builtins__": SAFE_BUILTINS, "np": np, "reference_global_two_way": reference_global_two_way}
    exec(compile(source, "<candidate>", "exec"), scope)
    return scope["decode_tp_gnn_embeddings"]


def _contract_issue(labels, areas):
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    if labels.shape != areas.shape:
        return f"it returned shape {labels.shape} for {areas.shape[0]} groups"
    if not np.issubdtype(labels.dtype, np.integer):
        return f"it returned dtype {labels.dtype}; labels must be integers 0 or 1"
    if not np.all((labels == 0) | (labels == 1)):
        return "it returned values other than 0 and 1"
    # The official wrapper applies area and macro constraints after decoding.
    if not np.any(labels == 0) or not np.any(labels == 1):
        return "it returned an empty side"
    return None


class _Timeout(Exception):
    pass


def _expired(signum, frame):
    raise _Timeout()


def probe_decoder(source, parent_source):
    """Return a contract issue for the model. Keep the caller's two-source interface."""
    if np is None:
        return None
    timed = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGALRM, _expired) if timed else None
    try:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, PROBE_SECONDS)
        candidate = _decoder(source)
        profiles = [(seed, "random") for seed in PROBE_SEEDS] + [(0, "unequal_areas"), (0, "clustered")]
        for seed, profile in profiles:
            embeddings, areas = _probe_inputs(seed, profile)
            try:
                labels = candidate(embeddings, areas, seed, PROBE_ITERATIONS, PROBE_BALANCE)
            except _Timeout:
                return None
            except NameError:
                continue  # The probe's sandbox lacks some names from the official evaluator.
            except KeyError as error:
                # Some NumPy versions lazily load internal helpers from the caller's scope.
                if error.args == ("__import__",):
                    continue
                return f"The decoder raised KeyError on a synthetic probe: {str(error)[:160]}"
            except Exception as error:
                return (f"The decoder raised {type(error).__name__} on a synthetic probe with read-only "
                        f"inputs: {str(error)[:160]}")
            issue = _contract_issue(labels, areas)
            if issue:
                return "The decoder breaks the published contract on a synthetic probe: " + issue
        return None
    except (_Timeout, SyntaxError, KeyError):
        return None
    finally:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
