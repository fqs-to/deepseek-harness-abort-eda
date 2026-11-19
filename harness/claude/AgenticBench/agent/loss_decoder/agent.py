"""Search the task-2 loss and decoder through the metered contest bridge.

Every candidate is proposed by the official model from the official initial
program and this run's official feedback. Every score comes from the organizer
bridge. search_policy.py holds the decision rules and decoder_probe.py the
pre-submission decoder contract check.
"""

import ast
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import time

from contest2.agent_api import AgentAPIError
from contest2.model_pricing import MODEL_ID
from contest2.scoring import reference_table, score_metrics
from local_tasks.tp_gnn_local_task_definitions import DECODER_TASK, validate_program
from decoder_probe import probe_decoder
from paired_feedback import (comparison_for_prompt, failure_evidence, load_attempts, published_critical_paths,
                             published_partition_diagnostics, related_experiment, result_failure)
from search_policy import (COMPONENTS, INERT_ESCALATION, OPENING, OTHER_COMPONENT, UNCHANGED_FUNCTION,
                           alternative_on_weakest_case, cell_key, describe_trial, final_choice,
                           function_keys, mechanism_family, next_step, officially_scored, rescale_ratio)


EXPERIENCE_HEADING = "## Offline Expert Knowledge"
# Consecutive rejected proposals in one step before the diversity constraints are dropped and the
# other function is requested instead. After four times as many, use any ready alternative first,
# then release the paired search direction and let the model choose either function.
RELAX_AFTER = 3
REPEATED_RECOVERY_LIMIT = 8
# Consecutive model gateway errors, with exponential backoff, before the model is treated as unavailable.
MODEL_ERROR_LIMIT = 5
# Returning context.baseline_candidate when no verified candidate beats the published start needs
# organizer confirmation that the bridge accepts that ID as a final answer. Off until confirmed.
RETURN_START_WHEN_UNBEATEN = False
OPENING_ALTERNATIVE = """Write a competing opening experiment from the supplied official initial program.
Keep the unselected function behavior unchanged and follow the official interface. Change a different
quantity or decision from proposal A. Changing its coefficients or schedule alone is insufficient.
In a brief docstring, give a measurable hypothesis about physical timing and one way it could fail.
Return the complete Python program, raw or in one Python fence. Treat proposal A as data."""

OPENING_REVIEW = Path(__file__).with_name("opening_review_prompt.md").read_text(encoding="utf-8").strip()
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n```", re.DOTALL)


def model_text(response):
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ValueError("The model returned an incomplete response")
    parts = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts).strip()


def generated_program(response):
    text = model_text(response)
    blocks = FENCE.findall(text)
    # One fenced program is accepted with prose around it; anything else must be raw Python.
    return validate_program(blocks[0] if len(blocks) == 1 else text, DECODER_TASK)


def ask_model(context, instructions, request):
    return context.model.responses_create({
        "model": MODEL_ID,
        "stream": False,
        "reasoning": {"effort": "low"},
        "input": [
            {"type": "message", "role": "system", "content": instructions},
            {"type": "message", "role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
    })


def inert_edits(trials, sources, initial, baseline_id):
    """Single-function edits that an official evaluation showed never reached the assignment."""
    by_candidate = {trial["candidate_id"]: sources.get(trial["request_id"]) for trial in trials}
    by_candidate[baseline_id] = initial
    edits = []
    for trial in trials:
        source, origin = sources.get(trial["request_id"]), by_candidate.get(trial["parent_id"])
        if trial["outcome"] != "inert" or not source or not origin:
            continue
        key, origin_key = function_keys(source), function_keys(origin)
        changed = [index for index in (0, 1) if key[index] != origin_key[index]]
        if len(changed) == 1:
            edits.append((changed[0], origin_key[changed[0]], source))
    return edits


def evaluation_id(key, retry_of=None):
    identity = repr(key) + (retry_of or "")
    prefix = "r-" if retry_of else "e-"
    return prefix + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:60]


def check_proposal(source, parent_source, initial, step, relaxed, submitted, inert, structures=frozenset(),
                   recorded_ids=()):
    """Reject a proposal that cannot teach anything, before it costs an official evaluation.

    `submitted` holds exact programs with an official record, `inert` the edits that never reached
    the assignment, and `structures` the coefficient-blind shapes of every evaluated program.
    """
    key, parent_key = function_keys(source), function_keys(parent_source)
    if key in submitted or evaluation_id(key) in recorded_ids:
        raise ValueError("The response repeats a program that already has an official evaluation record")
    unchanged = UNCHANGED_FUNCTION.get(step["intent"])
    if step["intent"] == "repair" and step.get("component"):
        unchanged = UNCHANGED_FUNCTION[step["component"] + "_only"]
    if unchanged is not None and key[unchanged] != parent_key[unchanged]:
        raise ValueError("Keep the other function unchanged for this paired search direction")
    structure = function_keys(source, ignore_numbers=True)
    changed = [index for index in (0, 1) if key[index] != parent_key[index]]
    for index, origin, inert_source in inert:
        # The search policy reserves evaluations for distinct proposals after a repeated outcome.
        if (changed == [index] and parent_key[index] == origin
                and function_keys(inert_source, ignore_numbers=True)[index] == structure[index]
                and rescale_ratio(source, inert_source, index) < INERT_ESCALATION):
            raise ValueError(
                f"This {COMPONENTS[index]} edit falls inside the search policy's coefficient-only "
                "repeat filter. Propose a structurally distinct edit.")
    required = step.get("required_component")
    if required and key[COMPONENTS.index(required)] == parent_key[COMPONENTS.index(required)]:
        raise ValueError(f"This step must change the {required} function relative to the parent")
    if not relaxed:
        if step["intent"] == "diversify" and structure in structures:
            raise ValueError("This program only retunes coefficients of an evaluated candidate. This step "
                             "needs a different mechanism.")
        family = mechanism_family(source, initial)
        if family in step.get("avoid", ()):
            raise ValueError(f"Mechanism family {family} is already explored in this run. Use different "
                             "inputs or change a different decision.")
    if key[1] != parent_key[1]:
        problem = probe_decoder(source, parent_source)
        if problem:
            raise ValueError(problem)
    return key, parent_key


def splice(loss_source, decoder_source):
    """The exact loss text of one evaluated program followed by the exact decoder text of another."""
    loss = ast.parse(loss_source).body[0]
    decoder = ast.parse(decoder_source).body[1]
    head = loss_source.splitlines()[:loss.end_lineno]
    tail = decoder_source.splitlines()[decoder.lineno - 1:]
    return "\n".join(head + ["", ""] + tail) + "\n"


def review_opening(context, choices, review_inputs):
    result = model_text(ask_model(context, OPENING_REVIEW, {**review_inputs, "proposals": choices}))
    verdict, _, reason = result.partition(":")
    if verdict.strip().upper() == "REVISE" and reason.strip():
        return "revise", " ".join(reason.split())[:400]
    pick = re.fullmatch(r"\W*PICK\s+([A-Z])\W*", result.strip(), re.IGNORECASE)
    if pick and pick[1].upper() in choices:
        return pick[1].upper(), ""
    raise ValueError("Opening review did not select an available proposal or give a specific issue")


def save_best(folder, result):
    pending = folder / "program.py.pending"
    pending.write_text(result["source"], encoding="utf-8")
    pending.replace(folder / "program.py")


def claim_once(context, filename):
    workspace = Path(context.files.get("workspaces", "/workspace/workspaces"))
    try:
        with (workspace / filename).open("x", encoding="utf-8"):
            return True
    except FileExistsError:
        return False


def published_initial_reference(context):
    table = reference_table()["cases"]
    cells = []
    for cell in context.evaluation_cells:
        case, seed = cell["case"], str(cell["seed"])
        entry = table.get(case, {})
        sample = entry.get("measurements", {}).get(context.task, {}).get(seed)
        if entry.get("split") != "train" or sample is None:
            return None
        metrics = score_metrics(case, {
            "runs_successfully": 1, "sta_graph_complete": 1,
            "timing_evaluation_boundary": "post-opt-pre-cts", "combined_score": 0,
            "final_global_wns_setup_ns": sample["wns_ns"], "final_global_tns_setup_ns": sample["tns_ns"],
        })
        cells.append({"case": case, "seed": seed, "status": "published_reference",
                      "score": metrics["combined_score"], "metrics": metrics})
    if not cells:
        return None
    cases = {cell["case"] for cell in cells}
    score = mean(mean(cell["score"] for cell in cells if cell["case"] == case) for case in sorted(cases))
    return {"candidate_id": context.baseline_candidate, "status": "published_reference",
            "feedback": {"score": score, "cells": cells}}


def summarize_cell(cell):
    summary = {key: cell[key] for key in ("case", "seed", "score", "raw_score", "valid", "status", "metrics") if key in cell}
    reason = cell.get("reason") or cell.get("error_message")
    if reason:
        summary["reason"] = str(reason)[:400]
    evidence = failure_evidence(cell)
    if evidence is not None:
        summary["failure"] = evidence
    return summary


def summarize(result):
    feedback = result.get("feedback") or {}
    cells = feedback.get("cells") or []
    case_scores = {}
    for cell in cells:
        if isinstance(cell.get("score"), (int, float)):
            case_scores.setdefault(cell["case"], []).append(cell["score"])
    summary = {
        "request_id": result["request_id"],
        "candidate_id": result["candidate_id"],
        "status": result["status"],
        "reason": str(result.get("reason") or "")[:400],
        "score": feedback.get("score"),
        "case_scores": {case: {"mean": mean(scores), "worst_seed_score": min(scores),
                               "scored_seeds": len(scores)} for case, scores in case_scores.items()},
        "cells": [summarize_cell(cell) for cell in cells],
    }
    failure = result_failure(result)
    if failure is not None:
        summary["failure"] = failure
    return summary


def ledger(trials):
    """One line per official evaluation: what was tried and what the official evidence said."""
    return [{"candidate_id": trial["candidate_id"], "mechanism_family": trial["family"],
             "search_intent": trial["intent"], "outcome": trial["outcome"],
             "official_score": trial["score"] if trial["complete"] else None,
             "official_score_delta_vs_parent": trial["official_score_delta"],
             "largest_assignment_change_fraction": trial["largest_assignment_change_fraction"]}
            for trial in trials]


def search_events(log_path):
    try:
        lines = log_path.read_bytes().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue  # A stopped process can leave a truncated line.
        if isinstance(event, dict):
            yield event


def load_shelf(log_path):
    """Proposals set aside by an opening review in an earlier process of this run."""
    entries = []
    revised_opening, rows_left = None, 0
    for event in search_events(log_path):
        kind = event.get("event")
        opening = (event.get("parent_id"), event.get("search_intent"))
        if kind == "shelved":
            # Older runs immediately logged every choice after a REVISE verdict.
            if rows_left and opening == revised_opening:
                rows_left -= 1
                continue
            rows_left = 0
            try:
                entries.append({"source": event["source"], "key": function_keys(event["source"]),
                                "parent_id": event["parent_id"], "intent": event["search_intent"],
                                "changed": event["changed_components"]})
            except (ValueError, KeyError, TypeError, AttributeError, SyntaxError):
                continue
        else:
            rows_left = 0
            if (kind == "opening_review" and event.get("verdict") == "revise" and all(opening)
                    and isinstance(event.get("proposals"), list)):
                revised_opening, rows_left = opening, len(event["proposals"])
    return entries


def submitted_sources(log_path, recorded_ids):
    """Failed and rejected bridge receipts omit source. Match our saved copy to its actual request."""
    sources = {}
    for event in search_events(log_path):
        request_id = event.get("request_id")
        if event.get("event") != "submitted" or request_id not in recorded_ids:
            continue
        source = event.get("source")
        if not isinstance(source, str):
            continue
        try:
            key = function_keys(validate_program(source, DECODER_TASK))
            if evaluation_id(key, event.get("retry_of")) == request_id:
                sources[request_id] = source
        except (ValueError, TypeError, AttributeError, SyntaxError):
            continue
    return sources


def write_event(folder, event, **fields):
    with (folder / "search.jsonl").open("a+b") as log:
        log.seek(0, 2)
        separator = b""
        if log.tell():
            log.seek(-1, 2)
            if log.read(1) != b"\n":
                separator = b"\n"
        line = json.dumps({"event": event, **fields}, ensure_ascii=False)
        log.write(separator + line.encode("utf-8") + b"\n")


BRIDGE_COMMUNICATION_ERRORS = {
    "Agent bridge communication failed; request outcome is unknown",
    "Invalid organizer bridge response",
    "Invalid organizer bridge envelope",
}


def bridge_read(read, *args):
    """Retry only temporary failures of a read. Never use this wrapper for a model call."""
    for attempt in range(5):
        try:
            return read(*args)
        except AgentAPIError as error:
            if str(error) not in BRIDGE_COMMUNICATION_ERRORS or attempt == 4:
                raise
            time.sleep(2 ** attempt)


def submit_and_wait(context, source, request_id):
    for attempt in range(5):
        try:
            result = context.submit_evaluation(source, request_id=request_id)
        except AgentAPIError as error:
            if any(row["request_id"] == request_id for row in bridge_read(context.status)["history"]):
                break
            if str(error) not in BRIDGE_COMMUNICATION_ERRORS or attempt == 4:
                raise
            time.sleep(2 ** attempt)
            continue  # Same request ID and source if the ledger has no record of the first attempt.
        if result["status"] in {"succeeded", "failed", "rejected"}:
            return result
        break
    return bridge_read(context.wait_evaluation, request_id)


class Agent:
    def run(self, context):
        if context.task != "loss_decoder":
            raise ValueError("This Agent implements the loss_decoder baseline")
        folder = Path(__file__).resolve().parent
        opening = OPENING
        opening_file = folder / "opening.txt"
        if opening_file.is_file():
            first = opening_file.read_text(encoding="utf-8").strip()
            if first not in OPENING:
                raise ValueError("Opening must be loss_only or decoder_only")
            opening = (first, next(intent for intent in OPENING if intent != first))
        task = Path(context.files["framework"]) / "local_tasks" / DECODER_TASK
        initial = validate_program((task / "initial_program.py").read_text(encoding="utf-8"), DECODER_TASK)
        instructions, marker, experience = (folder / "prompt.md").read_text(encoding="utf-8").partition(EXPERIENCE_HEADING)
        interface = (task / "prompt.md").read_text(encoding="utf-8")
        schedule = context.evaluation_cells

        state = bridge_read(context.status)
        history = []
        recorded_ids = {row["request_id"] for row in state["history"]}
        sources = submitted_sources(folder / "search.jsonl", recorded_ids)
        submitted = {function_keys(initial)}   # programs with an official evaluation record, plus the start
        blocked = set()                 # splices the validator refused
        shelf = []                      # validated proposals that a review set aside without evaluating them
        attempts = load_attempts(folder / "search.jsonl")
        shelf.extend(load_shelf(folder / "search.jsonl"))
        receipts_by_candidate = {}
        diagnostics_cache = {}
        assignment_cache = {}
        evaluation_root = context.files.get("evaluations", "/workspace/evaluations")
        initial_reference = published_initial_reference(context)
        start_score, start_cells = None, {}
        if initial_reference is not None:
            receipts_by_candidate[context.baseline_candidate] = initial_reference
            start_score = initial_reference["feedback"]["score"]
            start_cells = {cell_key(cell): cell["score"] for cell in initial_reference["feedback"]["cells"]}

        def summarize_for_prompt(result):
            summary = summarize(result)
            request_id = result["request_id"]
            if request_id not in diagnostics_cache:
                partition, stages = published_partition_diagnostics(result, evaluation_root)
                paths = published_critical_paths(result, evaluation_root)
                diagnostics_cache[request_id] = partition, stages, paths
            partition, stages, paths = diagnostics_cache[request_id]
            if partition:
                summary["partition_changes_vs_initial"] = {
                    "reported_cells": len(partition),
                    "identical_assignments": sum(row["partition_assignment_changed_count"] == 0
                                                 for row in partition.values()),
                }
            for cell in summary["cells"]:
                key = cell_key(cell)
                if key in partition:
                    cell["partition_change_vs_initial"] = partition[key]
                if key in stages:
                    cell["partition_stages"] = stages[key]
                if key in paths:
                    cell["worst_setup_paths"] = paths[key]
            attempt = attempts.get(result["request_id"], {})
            if attempt:
                summary["attempt"] = attempt
            comparison = comparison_for_prompt(result, attempt.get("parent_id"), receipts_by_candidate,
                                               context.evaluation_cells, evaluation_root, assignment_cache)
            if comparison is not None:
                summary["parent_comparison"] = comparison
            return summary

        def record(result):
            history.append(result)
            recorded_ids.add(result["request_id"])
            source = result.get("source") or sources.get(result["request_id"])
            if source:
                sources[result["request_id"]] = source
                try:
                    submitted.add(function_keys(source))
                except (SyntaxError, TypeError):
                    pass
            if result["status"] == "succeeded":
                receipts_by_candidate[result["candidate_id"]] = result

        def review():
            archive = [summarize_for_prompt(result) for result in history]
            trials = [describe_trial(summary, schedule, initial, sources.get(summary["request_id"]))
                      for summary in archive]
            return archive, trials

        saved = [None]
        rejections = 0
        rejected_components = set()

        def keep_incumbent(trials):
            chosen, _ = final_choice(trials, start_score)
            if chosen is not None and chosen["request_id"] != saved[0] and sources.get(chosen["request_id"]):
                saved[0] = chosen["request_id"]
                save_best(folder, {"source": sources[chosen["request_id"]]})
                write_event(folder, "incumbent", request_id=chosen["request_id"],
                            candidate_id=chosen["candidate_id"], score=chosen["score"])

        def evaluate(source, key, parent_id, intent, decision, changed, request_id=None, **lineage):
            nonlocal rejections
            rejections = 0   # every official reply is new evidence, so the next step starts fresh
            rejected_components.clear()
            request_id = request_id or evaluation_id(key)
            attempt = {"parent_id": parent_id, "search_intent": intent, "changed_components": changed, **lineage}
            attempts[request_id] = attempt
            sources[request_id] = source
            submitted.add(key)
            write_event(folder, "submitted", request_id=request_id, source=source, decision=decision, **attempt)
            # The bridge deduplicates this ID if the submit acknowledgement is lost.
            result = submit_and_wait(context, source, request_id)
            record(result)
            write_event(folder, "evaluated", parent_id=parent_id, **summarize_for_prompt(result))
            keep_incumbent(review()[1])

        def hypotheses(source, key, parent_key):
            changed = {}
            for component, function, updated, previous in zip(COMPONENTS, ast.parse(source).body, key, parent_key):
                if updated != previous:
                    changed[component] = " ".join((ast.get_docstring(function) or "").split())[:200]
            return changed

        for row in state["history"]:
            record(bridge_read(context.wait_evaluation, row["request_id"]))
        keep_incumbent(review()[1])

        issue = None
        initial_experience_pending = bool(marker and not history)
        consecutive_api_failures = 0
        last_recovery_reply = None
        repeated_recovery_replies = 0
        while True:
            state = bridge_read(context.status)
            if not state["evaluation_dispatch_allowed"]:
                write_event(folder, "budget_stop", evaluations_remaining=state["evaluations_remaining"],
                            model_stop_reasons=state["model_stop_reasons"])
                break

            archive, trials = review()
            by_request = {result["request_id"]: result for result in history}
            inert = inert_edits(trials, sources, initial, context.baseline_candidate)
            step = next_step(trials, start_score, start_cells, opening, state["evaluations_remaining"],
                             submitted | blocked,
                             [{"key": entry["key"], "changed": sorted(entry["changed"])} for entry in shelf])

            # Actions that need no model call: retry an infrastructure failure, or combine verified gains.
            if step["action"] == "resubmit":
                target = step["target"]
                source, previous = sources[target], attempts.get(target, {})
                key = function_keys(source)
                retry_id = evaluation_id(key, target)
                evaluate(source, key, previous.get("parent_id", context.baseline_candidate), "retry",
                         step["decision"], previous.get("changed_components", {}), request_id=retry_id,
                         retry_of=target)
                continue
            if step["action"] == "shelved":
                spare = shelf.pop(step["index"])
                try:
                    check_proposal(spare["source"], initial, initial, {"intent": spare["intent"]}, True,
                                   submitted, inert, recorded_ids=recorded_ids)
                except (ValueError, SyntaxError) as error:
                    write_event(folder, "shelved_proposal_dropped", reason=str(error)[:400])
                    continue
                evaluate(spare["source"], spare["key"], spare["parent_id"], spare["intent"], step["decision"],
                         spare["changed"], from_shelf=True)
                continue
            if step["action"] == "recombine":
                try:
                    source = validate_program(splice(sources[step["loss"]], sources[step["decoder"]]), DECODER_TASK)
                    key = function_keys(source)
                    if key != step["key"] or key in submitted or evaluation_id(key) in recorded_ids:
                        raise ValueError("The splice does not reproduce both evaluated functions")
                except (ValueError, SyntaxError, KeyError) as error:
                    blocked.add(step["key"])
                    write_event(folder, "recombination_rejected", reason=str(error)[:400], **step["decision"])
                    continue
                parent = by_request[step["parent"]]
                evaluate(source, key, parent["candidate_id"], "recombine", step["decision"],
                         hypotheses(source, key, function_keys(sources[step["parent"]])),
                         recombined_from=[by_request[step["loss"]]["candidate_id"],
                                          by_request[step["decoder"]]["candidate_id"]])
                continue

            parent = by_request.get(step["parent"])
            if parent is not None and parent["request_id"] not in sources:
                parent = None  # the bridge returned no source for it: build on the official start instead
            intent, decision = step["intent"], step["decision"]
            parent_id = parent["candidate_id"] if parent else context.baseline_candidate
            parent_source = sources[parent["request_id"]] if parent else initial

            recovery = rejections >= 4 * RELAX_AFTER
            if not recovery:
                last_recovery_reply = None
                repeated_recovery_replies = 0
            if not state["model_dispatch_allowed"] or consecutive_api_failures >= MODEL_ERROR_LIMIT or recovery:
                spare = next((entry for entry in shelf if entry["key"] not in submitted
                              and evaluation_id(entry["key"]) not in recorded_ids), None)
                if spare is not None:
                    shelf.remove(spare)
                    evaluate(spare["source"], spare["key"], spare["parent_id"], spare["intent"],
                             {"reason": "shelved_proposal_without_model"}, spare["changed"], from_shelf=True)
                    continue
                if (not state["model_dispatch_allowed"] or consecutive_api_failures >= MODEL_ERROR_LIMIT
                        or repeated_recovery_replies >= REPEATED_RECOVERY_LIMIT):
                    reason = ("model_budget" if not state["model_dispatch_allowed"] else "model_unavailable"
                              if consecutive_api_failures >= MODEL_ERROR_LIMIT else "same_unusable_recovery_reply")
                    write_event(folder, "budget_stop", evaluations_remaining=state["evaluations_remaining"],
                                reason=reason, model_stop_reasons=state["model_stop_reasons"])
                    break
                intent = "diversify"
                decision = {"reason": "recover_after_rejected_proposals",
                            "instruction": "Propose an untested valid change. You may change either function or both."}
                step = {**step, "intent": intent, "decision": decision, "required_component": None, "avoid": []}

            relaxed = rejections >= RELAX_AFTER
            if relaxed:
                # Steer away from what keeps being rejected: ask for the other function, then for anything valid.
                stuck = next(iter(rejected_components)) if len(rejected_components) == 1 else "loss"
                step = {**step, "avoid": [], "required_component":
                        OTHER_COMPONENT[stuck] if rejections < 2 * RELAX_AFTER and step["intent"] in (
                            "refine", "diversify") else None}
            summaries = {row["request_id"]: row for row in archive}
            parent_feedback = summaries[parent["request_id"]] if parent else None
            chosen, _ = final_choice(trials, start_score)
            incumbent_feedback = (summaries[chosen["request_id"]]
                                  if chosen and (parent is None or chosen["request_id"] != parent["request_id"])
                                  else None)
            displayed = {row["request_id"] for row in (parent_feedback, incumbent_feedback) if row is not None}
            previous_results = [row for row in archive if row["request_id"] not in displayed][-2:]
            displayed.update(row["request_id"] for row in previous_results)
            focus = parent_feedback or incumbent_feedback
            if archive and any("failure" in cell for cell in archive[-1]["cells"]):
                focus = archive[-1]
            request = {
                "search_intent": intent,
                "decision": decision,
                "required_component": step["required_component"],
                "avoid_mechanism_families": step["avoid"],
                "mechanism_ledger": ledger(trials),
                "parent_id": parent_id,
                "parent_program": parent_source,
                "training_cells": context.evaluation_cells,
                "evaluations_remaining": state["evaluations_remaining"],
                "published_start_score": start_score,
                "parent_feedback": parent_feedback,
                "incumbent_feedback": incumbent_feedback,
                "recent_feedback": previous_results,
                "related_experiment": related_experiment(focus, archive, displayed),
                "previous_generation_issue": issue,
            }
            if parent and intent in ("repair", "refine", "diversify"):
                parent_trial = next(t for t in trials if t["request_id"] == parent["request_id"])
                alternative = alternative_on_weakest_case(
                    parent_trial, trials, sources, step.get("component") or step.get("required_component"))
                if alternative:
                    request["measured_alternative"] = alternative
            system_prompt = instructions.rstrip() + "\n\n" + interface
            if initial_experience_pending:
                initial_experience_pending = False
                if claim_once(context, "loss-decoder-initial-prompt-sent"):
                    system_prompt += "\n\n" + marker + experience
            try:
                response = ask_model(context, system_prompt, request)
            except AgentAPIError as error:
                issue = str(error)
                write_event(folder, "model_error", parent_id=parent_id, reason=issue)
                consecutive_api_failures += 1
                time.sleep(min(60, 2 ** consecutive_api_failures))
                continue
            consecutive_api_failures = 0

            source = None
            try:
                source = generated_program(response)
                structures = {function_keys(text, ignore_numbers=True) for text in sources.values()}
                key, parent_key = check_proposal(source, parent_source, initial, step, relaxed, submitted,
                                                 [] if recovery else inert,
                                                 structures, recorded_ids)
            except (ValueError, SyntaxError) as error:
                issue = str(error)[:400]
                rejections += 1
                if recovery:
                    if source is not None:
                        reply_key = function_keys(source)
                    else:
                        try:
                            reply_key = model_text(response)
                        except ValueError:
                            reply_key = issue
                    repeated_recovery_replies = repeated_recovery_replies + 1 if reply_key == last_recovery_reply else 1
                    last_recovery_reply = reply_key
                if source is not None:
                    edited = [name for name, new, old in zip(COMPONENTS, function_keys(source),
                                                              function_keys(parent_source)) if new != old]
                    rejected_components.update(edited)
                write_event(folder, "invalid_proposal", parent_id=parent_id, reason=issue)
                continue

            if (decision["reason"] == "opening_control" and bridge_read(context.status)["model_dispatch_allowed"]
                    and claim_once(context, f"loss-decoder-opening-review-{intent}")):
                review_inputs = {"search_intent": intent, "parent_program": parent_source,
                                 "official_interface": interface}
                choices = {"A": source}
                keys = {"A": key}
                try:
                    alternative = generated_program(ask_model(context, OPENING_ALTERNATIVE, {
                        **review_inputs, "proposal_a": source,
                    }))
                    keys["B"], _ = check_proposal(alternative, parent_source, initial, step, True,
                                                  submitted | {key}, inert, recorded_ids=recorded_ids)
                    if function_keys(alternative, ignore_numbers=True) == function_keys(source, ignore_numbers=True):
                        raise ValueError("Opening alternative only changes numeric literals")
                    choices["B"] = alternative
                except (AgentAPIError, ValueError, SyntaxError) as error:
                    write_event(folder, "opening_alternative_rejected", parent_id=parent_id, reason=str(error)[:400])
                verdict, reason = "A", ""
                if bridge_read(context.status)["model_dispatch_allowed"]:
                    try:
                        from opening_gradients import opening_gradients
                        gradients = opening_gradients(intent, context, parent_source, choices)
                        if gradients is not None:
                            review_inputs["initial_batch_gradients"] = gradients
                        verdict, reason = review_opening(context, choices, review_inputs)
                    except (AgentAPIError, ValueError) as error:
                        reason = str(error)[:400]
                write_event(folder, "opening_review", parent_id=parent_id, search_intent=intent,
                            proposals=list(choices), verdict=verdict, reason=reason)
                if verdict == "revise":
                    issue = f"Opening review found this concrete issue: {reason} Propose a different mechanism."
                    continue
                # A proposal the review sets aside was never measured: keep it available, do not ban it.
                for name, program in choices.items():
                    if name != verdict:
                        entry = {"source": program, "key": keys[name], "parent_id": parent_id, "intent": intent,
                                 "changed": hypotheses(program, keys[name], parent_key)}
                        shelf.append(entry)
                        write_event(folder, "shelved", source=program, parent_id=parent_id,
                                    search_intent=intent, changed_components=entry["changed"])
                source, key = choices[verdict], keys[verdict]

            evaluate(source, key, parent_id, intent, decision, hypotheses(source, key, parent_key))
            issue = None

        archive, trials = review()
        scored = {summary["request_id"] for summary in archive if officially_scored(summary, schedule)}
        chosen, selection = final_choice(trials, start_score, scored_requests=scored)
        if chosen is None:
            if not RETURN_START_WHEN_UNBEATEN:
                raise RuntimeError("No candidate completed an official training evaluation"
                                   + (f": {issue}" if issue else ""))
            write_event(folder, "selected_published_start", selection=selection)
            return context.baseline_candidate
        if RETURN_START_WHEN_UNBEATEN and not selection["beats_published_start"]:
            write_event(folder, "selected_published_start", selection=selection)
            return context.baseline_candidate
        best = next(result for result in history if result["request_id"] == chosen["request_id"])
        if chosen["request_id"] != saved[0] and chosen["request_id"] in sources:
            save_best(folder, {"source": sources[chosen["request_id"]]})
        write_event(folder, "selected", selection=selection, **summarize(best))
        return best["candidate_id"]
