"""Evidence classification and next-hypothesis policy for the task-2 search.

Everything here is a pure function of official bridge replies. Nothing in this
module calls the bridge, calls the model or computes a score: every score it
reads is the official number returned for a submitted candidate.
"""

import ast
from math import isfinite
from statistics import mean, median


COMPONENTS = ("loss", "decoder")
OTHER_COMPONENT = {"loss": "decoder", "decoder": "loss"}
# Default order for the two single-component trials.
OPENING = ("decoder_only", "loss_only")
UNCHANGED_FUNCTION = {"loss_only": 1, "decoder_only": 0}
REFERENCE_HELPER = "reference_global_two_way"
VERIFIED = ("inert", "marginal", "effective")

SAME_TIMING_NS = 1e-6       # WNS and TNS both within this of the parent: same physical result
SAME_SCORE_POINTS = 1e-9    # Ignore floating-point residue in cell scores and their aggregate.
MARGINAL_FRACTION = 1e-3    # threshold for the search policy's marginal category
INERT_ESCALATION = 10.0     # an inert mechanism may return only if a coefficient grows this much
FAMILY_LIMIT = 2            # official evaluations spent on one mechanism family before it is exhausted
SHELF_EVALUATIONS = 2       # reviewed-but-unmeasured opening alternatives that get their own evaluation


def _functions(source):
    functions = ast.parse(source).body
    for function in functions:
        if ast.get_docstring(function, clean=False) is not None:
            function.body.pop(0)
    return functions


def _coefficients(function):
    """Numeric literals that tune strength. Subscript positions select data and are kept."""
    positions = {id(node) for holder in ast.walk(function) if isinstance(holder, ast.Subscript)
                 for node in ast.walk(holder.slice)}
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float) and id(node) not in positions]


def function_keys(source, ignore_numbers=False):
    """Docstring-free AST of each function; with ignore_numbers, coefficients are blanked."""
    functions = _functions(source)
    if ignore_numbers:
        for function in functions:
            for node in _coefficients(function):
                node.value = 0
    return tuple(ast.dump(function, include_attributes=False) for function in functions)


def rescale_ratio(source, other, index=None):
    """Largest ratio between corresponding coefficients of two programs with the same structure.

    With index, only that function (0 loss, 1 decoder) is compared.
    """
    ratio = 1.0
    pairs = list(zip(_functions(source), _functions(other)))
    for first, second in pairs if index is None else pairs[index:index + 1]:
        for a, b in zip(_coefficients(first), _coefficients(second)):
            low, high = sorted((abs(a.value), abs(b.value)))
            if high != low:
                ratio = max(ratio, high / low if low else float("inf"))
    return ratio


def mechanism_family(source, initial_source):
    """Name the information a candidate adds to the official initial program.

    A loss family lists the published inputs the edit starts to read. A decoder
    family says whether the frozen helper is kept and which NumPy operations
    make the new decision. Coefficients and docstrings do not change a family.
    """
    (loss, decoder), (loss0, decoder0) = _functions(source), _functions(initial_source)
    parts = []
    if ast.dump(loss) != ast.dump(loss0):
        arguments = {argument.arg for argument in loss.args.args}
        names = lambda function: {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
        added = sorted((names(loss) - names(loss0)) & arguments)
        parts.append("loss[" + ("+".join(added) or "sampled_pairs") + "]")
    if ast.dump(decoder) != ast.dump(decoder0):
        calls = [node.func for node in ast.walk(decoder) if isinstance(node, ast.Call)]
        helper = any(isinstance(call, ast.Name) and call.id == REFERENCE_HELPER for call in calls)
        operations = sorted({call.attr for call in calls if isinstance(call, ast.Attribute)
                             and isinstance(call.value, ast.Name) and call.value.id == "np"})
        parts.append("decoder[" + "+".join(["reference" if helper else "custom", *operations]) + "]")
    return " ".join(parts) or "initial"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def cell_key(cell):
    return cell["case"], str(cell["seed"])


def _has_scheduled_scores(summary, schedule):
    if summary.get("status") != "succeeded" or not _finite(summary.get("score")):
        return False
    cells = summary.get("cells") or []
    keys = [cell_key(cell) for cell in cells]
    if not schedule or len(set(keys)) != len(keys) or set(keys) != {cell_key(cell) for cell in schedule}:
        return False
    return all(_finite(cell.get("score")) for cell in cells)


def matrix_complete(summary, schedule):
    """For search, an infrastructure failure leaves the mechanism unverified."""
    return _has_scheduled_scores(summary, schedule) and all(
        cell.get("status") != "not_completed" and (cell.get("failure") or {}).get("kind") != "runtime_environment"
        for cell in summary["cells"])


def officially_scored(summary, schedule):
    """The successful bridge receipt carries the organizer's score, including physical penalties."""
    return _has_scheduled_scores(summary, schedule) and all(
        (cell.get("status") == "succeeded" and cell["score"] >= -100)
        or (cell.get("status") == "penalized" and cell.get("valid") is False and cell["score"] == -100)
        for cell in summary["cells"])


def _has_measured_partition_effect(comparison, cells):
    """Check for a changed assignment and timing in the direction of the official net score change."""
    net_delta = comparison.get("official_score_delta")
    if not comparison.get("complete") or not _finite(net_delta) or abs(net_delta) <= SAME_SCORE_POINTS:
        return False
    direction = 1 if net_delta > 0 else -1
    by_key = {cell_key(cell): cell for cell in cells}
    for pair in comparison["cells"]:
        score_delta = pair["score_delta"]
        slacks = (pair["wns_slack_delta_ns"], pair["tns_slack_delta_ns"])
        if (not _finite(score_delta) or direction * score_delta <= SAME_SCORE_POINTS
                or not all(_finite(delta) for delta in slacks)
                or not any(direction * delta >= SAME_TIMING_NS for delta in slacks)):
            continue
        cell = by_key[cell_key(pair)]
        if comparison["parent_source"] == "published_initial_reference":
            changed = (cell.get("partition_change_vs_initial") or {}).get("partition_assignment_changed_count")
        else:
            changed = (pair.get("assignment_vs_parent") or {}).get("changed_fraction")
        if _finite(changed) and changed > 0:
            return True
    return False


def classify(summary, schedule):
    """Sort one official reply into the evidence class that decides the next step."""
    cells = summary.get("cells") or []
    failures = [cell["failure"] for cell in cells if cell.get("failure")]
    if summary.get("failure"):
        failures.append(summary["failure"])
    contract = next((failure for failure in failures if failure.get("kind") == "candidate_contract"), None)
    if contract:
        return "contract_failure", contract.get("component")
    if not matrix_complete(summary, schedule):
        environment = failures and all(failure.get("kind") == "runtime_environment" for failure in failures)
        return ("environment_failure" if environment else "unverified"), None

    comparison = summary.get("parent_comparison") or {}
    if _has_measured_partition_effect(comparison, cells):
        return "effective", None
    pairs = comparison.get("cells") or []
    same_timing = sum(all(pair.get(metric) is not None and abs(pair[metric]) < SAME_TIMING_NS
                          for metric in ("wns_slack_delta_ns", "tns_slack_delta_ns")) for pair in pairs)
    if same_timing > len(cells) // 2:
        return "inert", None
    if comparison.get("parent_source") == "published_initial_reference":
        partition = summary.get("partition_changes_vs_initial") or {}
        if partition.get("identical_assignments", 0) > len(cells) // 2:
            return "inert", None
        fractions = [fraction for fraction in (
            (cell.get("partition_change_vs_initial") or {}).get("partition_assignment_changed_fraction")
            for cell in cells) if _finite(fraction)]
        if len(fractions) > len(cells) // 2 and max(fractions) < MARGINAL_FRACTION:
            return "marginal", None
    return "effective", None


def describe_trial(summary, schedule, initial_source, source):
    """Reduce one official reply and its lineage to the facts the policy reads."""
    attempt = summary.get("attempt") or {}
    comparison = summary.get("parent_comparison") or {}
    cells = summary.get("cells") or []
    outcome, failure_component = classify(summary, schedule)
    key = family = None
    changed = []
    if source:
        try:
            key, family = function_keys(source), mechanism_family(source, initial_source)
            changed = [name for name, new, old in zip(COMPONENTS, key, function_keys(initial_source)) if new != old]
        except (SyntaxError, ValueError, TypeError):
            key = None
    tns = [(cell.get("metrics") or {}).get("tns_deficit_improvement") for cell in cells]
    fractions = [fraction for fraction in (
        (cell.get("partition_change_vs_initial") or {}).get("partition_assignment_changed_fraction")
        for cell in cells) if _finite(fraction)]
    return {
        "request_id": summary["request_id"], "candidate_id": summary["candidate_id"],
        "outcome": outcome, "complete": outcome in VERIFIED, "failure_component": failure_component,
        "score": summary.get("score"),
        "tns": mean(tns) if tns and all(_finite(value) for value in tns) else None,
        "intent": attempt.get("search_intent"), "parent_id": attempt.get("parent_id"),
        "retry_of": attempt.get("retry_of"), "from_shelf": bool(attempt.get("from_shelf")),
        "edited": sorted(attempt.get("changed_components") or {}),
        "key": key, "family": family, "changed_vs_start": changed,
        "cell_scores": {cell_key(cell): cell["score"] for cell in cells if _finite(cell.get("score"))},
        "comparison_complete": bool(comparison.get("complete")),
        "official_score_delta": comparison.get("official_score_delta"),
        "improved_pairs": comparison.get("improved_pairs", 0),
        "regressed_pairs": comparison.get("regressed_pairs", 0),
        "scored_pairs": comparison.get("scored_pairs", 0),
        "largest_regression": comparison.get("largest_regression"),
        "invalid_cells": comparison.get("candidate_invalid_cells", 0),
        "largest_assignment_change_fraction": max(fractions) if fractions else None,
    }


def rank(trial):
    """Official score first, mean TNS improvement second."""
    return trial["score"], trial["tns"] if trial["tns"] is not None else float("-inf")


def alternative_on_weakest_case(parent, trials, sources, component=None):
    """Show one measured single-function alternative for a parent's lowest-scoring case."""
    if not parent or not parent["complete"] or not parent["key"]:
        return None
    scores = parent["cell_scores"]
    by_case = {}
    for key in scores:
        by_case.setdefault(key[0], []).append(key)
    weakest = min(by_case, key=lambda case: mean(scores[k] for k in by_case[case]))
    keys = by_case[weakest]
    best = None
    for trial in trials:
        if not trial["complete"] or not trial["key"] or trial["request_id"] not in sources:
            continue
        changed = [i for i, (old, new) in enumerate(zip(parent["key"], trial["key"])) if old != new]
        if len(changed) != 1 or (component and COMPONENTS[changed[0]] != component):
            continue
        other = trial["cell_scores"]
        if scores.keys() != other.keys():
            continue
        deltas = [other[k] - scores[k] for k in keys]
        if mean(deltas) <= 0 or sum(delta > 0 for delta in deltas) <= len(keys) // 2:
            continue
        priority = (mean(deltas), rank(trial))
        if best is None or priority > best[0]:
            best = (priority, trial, changed[0])
    if best is None:
        return None
    _, trial, index = best
    source = sources[trial["request_id"]]
    other = trial["cell_scores"]
    return {
        "candidate_id": trial["candidate_id"], "official_training_score": trial["score"],
        "focus_case": weakest, "changed_function": COMPONENTS[index],
        "paired_score_deltas_on_focus_case": {k[1]: other[k] - scores[k] for k in keys},
        "mean_score_delta_by_case": {case: mean(other[k] - scores[k] for k in group)
                                     for case, group in by_case.items()},
        "function_source": ast.get_source_segment(source, ast.parse(source).body[index]),
    }


def final_choice(trials, start_score, *, scored_requests=None):
    """Rank verified search trials, or the officially scored requests at the end of the run."""
    if scored_requests is None:
        eligible = [trial for trial in trials if trial["complete"]]
        rule = "highest official score among complete verified training matrices"
    else:
        eligible = [trial for trial in trials if trial["request_id"] in scored_requests]
        rule = "highest official score among complete scored training matrices"
    best = None
    for trial in eligible:  # the earliest candidate wins an exact tie
        if best is None or rank(trial) > rank(best):
            best = trial
    return best, {
        "rule": rule,
        "eligible_candidates": len(eligible), "evaluated_candidates": len(trials),
        "published_start_score": start_score,
        "beats_published_start": bool(best and (start_score is None or best["score"] > start_score)),
    }


def _retry_allowed(latest, trials, reference_scores, evaluations_remaining):
    """One resubmission for a candidate that only infrastructure stopped, when the evidence is worth it."""
    if latest["key"] is None or latest["retry_of"] or any(
            trial["retry_of"] == latest["request_id"] for trial in trials):
        return False
    shared = [key for key in latest["cell_scores"] if key in reference_scores]
    if not shared:  # nothing was measured: the hypothesis is still untested
        return evaluations_remaining >= 2
    lead = mean(latest["cell_scores"][key] - reference_scores[key] for key in shared)
    return lead > 0 and evaluations_remaining >= 1


def _recombination(trials, start_score, considered):
    """The best verified loss-only gain and decoder-only gain over the start, not yet combined."""
    if start_score is None:
        return None
    gains = [trial for trial in trials if trial["outcome"] == "effective" and trial["key"]
             and trial["score"] > start_score and len(trial["changed_vs_start"]) == 1]
    losses = [trial for trial in gains if trial["changed_vs_start"] == ["loss"]]
    decoders = [trial for trial in gains if trial["changed_vs_start"] == ["decoder"]]
    if not losses or not decoders:
        return None
    loss, decoder = max(losses, key=rank), max(decoders, key=rank)
    key = (loss["key"][0], decoder["key"][1])
    return None if key in considered else (loss, decoder, key)


def _exhausted_families(trials):
    counts = {}
    for trial in trials:
        if trial["family"] and trial["outcome"] != "environment_failure" and not trial["retry_of"]:
            counts[trial["family"]] = counts.get(trial["family"], 0) + 1
    return sorted(family for family, count in counts.items() if count >= FAMILY_LIMIT)


def _alternative_parent(verified, incumbent, start_score):
    """Find a parent that beats the leader on a majority of seeds in a majority of cases."""
    leader = incumbent["cell_scores"]
    choices = []
    for trial in verified:
        scores = trial["cell_scores"]
        if (trial["outcome"] != "effective" or not trial["key"] or scores.keys() != leader.keys()
                or trial["candidate_id"] == incumbent["candidate_id"] or trial["score"] <= start_score):
            continue
        differences = {}
        for (case, seed), score in scores.items():
            differences.setdefault(case, []).append(score - leader[case, seed])
        winning_cases = [median(deltas) for deltas in differences.values()
                         if sum(delta > 0 for delta in deltas) > len(deltas) // 2]
        if len(winning_cases) > len(differences) // 2:
            choices.append(((len(winning_cases), median(winning_cases), rank(trial)), trial))
    return max(choices, key=lambda pair: pair[0])[1] if choices else None


def _concentrated_regression(trial):
    paired = trial["scored_pairs"]
    return (trial["comparison_complete"] and trial["official_score_delta"] is not None
            and trial["official_score_delta"] < 0 and trial["largest_regression"]
            and trial["improved_pairs"] > paired // 2 and 0 < trial["regressed_pairs"] <= max(1, paired // 3)
            and not trial["invalid_cells"] and trial["intent"] != "repair")


def next_step(trials, start_score, start_cell_scores, opening=OPENING, evaluations_remaining=1,
              considered=frozenset(), shelved=()):
    """Choose the next action from every official reply of this run.

    `considered` holds the function keys that already have an official record.
    `shelved` lists validated single-function edits of the start that an opening
    review set aside: {"key": ..., "changed": [...]} in the order they were written.

    Returns one of
      {"action": "resubmit", "target": request_id, ...}      same source, no model call
      {"action": "shelved", "index": position, ...}           set-aside proposal, no model call
      {"action": "recombine", "loss": id, "decoder": id, ...} exact splice, no model call
      {"action": "propose", "parent": request_id or None, "intent": ..., ...}
    """
    def propose(parent, intent, reason, **fields):
        decision = {"reason": reason, **{name: value for name, value in fields.items()
                                         if name in ("candidate_id", "component", "largest_regression")}}
        return {"action": "propose", "parent": parent["request_id"] if parent else None, "intent": intent,
                "decision": decision, "component": fields.get("component"),
                "required_component": fields.get("required_component"), "avoid": fields.get("avoid", [])}

    verified = [trial for trial in trials if trial["complete"]]
    incumbent = max(verified, key=rank) if verified else None
    ahead = incumbent is not None and (start_score is None or incumbent["score"] > start_score)

    # 1. Official failure evidence about the latest submission.
    latest = trials[-1] if trials else None
    if latest and latest["outcome"] == "environment_failure":
        reference = incumbent["cell_scores"] if ahead else start_cell_scores
        if _retry_allowed(latest, trials, reference, evaluations_remaining):
            return {"action": "resubmit", "target": latest["request_id"], "intent": "retry",
                    "decision": {"reason": "environment_failure_retry", "candidate_id": latest["candidate_id"]}}
    if (latest and latest["outcome"] == "contract_failure" and latest["failure_component"]
            and latest["key"] and latest["intent"] != "repair"):
        return propose(latest, "repair", "candidate_contract_failure", candidate_id=latest["candidate_id"],
                       component=latest["failure_component"])

    # Infrastructure failures say nothing about a mechanism and do not advance the schedule.
    informative = [trial for trial in trials if trial["outcome"] != "environment_failure"]

    # 2. Opening controls: one single-function edit of the official start per function.
    opened = {trial["intent"] for trial in informative}
    for intent in opening:
        if intent not in opened:
            return propose(None, intent, "opening_control")

    # Combine two measured gains before investing another evaluation in an unmeasured opening idea.
    pair = _recombination(trials, start_score, considered)
    if pair:
        loss, decoder, key = pair
        return {"action": "recombine", "loss": loss["request_id"], "decoder": decoder["request_id"], "key": key,
                "parent": max((loss, decoder), key=rank)["request_id"], "intent": "recombine",
                "decision": {"reason": "combine_verified_gains", "loss_candidate_id": loss["candidate_id"],
                             "decoder_candidate_id": decoder["candidate_id"]}}

    # Measure the opening alternatives too. A text review ranked them without evidence, and each is
    #    one more single-function comparison against the same published start. Decoder edits go first
    #    because they always reach the assignment.
    if sum(trial["from_shelf"] for trial in informative) < SHELF_EVALUATIONS:
        waiting = [(entry["changed"] != ["decoder"], index) for index, entry in enumerate(shelved)
                   if entry["key"] not in considered]
        if waiting:
            return {"action": "shelved", "index": min(waiting)[1],
                    "decision": {"reason": "opening_alternative_control"}}

    last = informative[-1]
    base = incumbent if ahead else None
    exhausted = _exhausted_families(trials)

    # Try a second lineage with one subsequent evaluation still available to build on a new winner.
    if evaluations_remaining == 2 and ahead and start_score is not None:
        alternative = _alternative_parent(verified, incumbent, start_score)
        if alternative:
            return propose(alternative, "refine", "refine_parent_winning_more_paired_cases",
                           candidate_id=alternative["candidate_id"])

    # 5. The last edit never reached the assignment: change the mechanism, not its coefficient.
    if last["outcome"] in ("inert", "marginal"):
        edited = last["edited"] or last["changed_vs_start"]
        return propose(base, "diversify",
                       "assignment_unchanged" if last["outcome"] == "inert" else "assignment_barely_changed",
                       candidate_id=last["candidate_id"], avoid=exhausted,
                       required_component=OTHER_COMPONENT[edited[0]] if len(edited) == 1 else None)

    # 6. Most pairs improved and a few regressed: one targeted repair of that candidate.
    if last["outcome"] == "effective" and last["key"] and _concentrated_regression(last):
        return propose(last, "repair", "concentrated_regression", candidate_id=last["candidate_id"],
                       largest_regression=last["largest_regression"],
                       component=last["edited"][0] if len(last["edited"]) == 1 else None)

    # 7. Nothing verified beats the published start: explore from the start, least-tested function first.
    if not ahead:
        tested = {name: sum(name in trial["changed_vs_start"] for trial in informative
                            if trial["outcome"] == "effective") for name in COMPONENTS}
        return propose(None, "diversify", "published_start_still_ahead", avoid=exhausted,
                       required_component="loss" if tested["loss"] < tested["decoder"] else "decoder")

    # 8. A verified incumbent leads: exploit it once, then widen the mechanisms built on it.
    if last["request_id"] == incumbent["request_id"]:
        return propose(incumbent, "refine", "refine_incumbent", candidate_id=incumbent["candidate_id"])
    avoid = sorted(set(exhausted) | {incumbent["family"]} - {None})
    return propose(incumbent, "diversify", "explore_from_incumbent", candidate_id=incumbent["candidate_id"],
                   avoid=avoid)
