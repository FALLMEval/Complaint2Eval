import asyncio
import json
import os
import re
from datetime import datetime
from itertools import islice
from typing import Any, Dict, List, Tuple

from .model_interface import ModelInterface
from .prepare_json_string import prepare_json_string


def prompt_deduplicate(smallchunkQ: str) -> str:
    """Generate deduplication prompt"""
    return f"""
You are an evaluation auditor tasked with deduplicating financial evaluation questions.

## Instructions
- Input: A numbered list of questions.
- Task: Remove duplicates.
  - If multiple questions ask about essentially the same evaluation criterion, keep only one generalized version.
  - Keep the representative version as the `question`.
  - Attach all original numbering identifiers in an array called `source_ids`.
- Output: **valid JSON only**.

## JSON Schema
{{"deduplicated_questions": [
  {{"question": "string", "source_ids": ["1","2", "..."]}},
  ...
}}


## Requirements
- Each duplicate group becomes one entry in `deduplicated_questions`.
- Do not drop any question completely; each original item must be represented in some `source_ids`.
- Representative question should be phrased clearly and generally (avoid redundant wording).

## Input
{smallchunkQ}

"""


def chunk_by(seq, size):
    it = iter(seq)
    while True:
        block = list(islice(it, size))
        if not block:
            break
        yield block


def build_leaf_items(results_full: List[Dict]) -> List[Dict]:
    """Flatten questions under each Complaint into leaf items; hide Complaint identifiers from the LLM (only reflected in the prompt)."""
    items: List[Dict] = []
    for comp in results_full:
        cid = str(comp["Complaint_Index"])
        for i, q in enumerate(comp["evaluation_criteria"], start=1):
            items.append(
                {
                    "unit_id": f"C{cid}",  # unit = Complaint
                    "current_id": f"C{cid}.{i}",  # current ID (first round = original)
                    "original_ids": [
                        f"{cid}.{i}"
                    ],  # leaf ancestry (original source IDs)
                    "question": q["question"],
                    "iter": 0,
                }
            )
    return items


def prompt_deduplicate_from_items(items: List[Dict]) -> Tuple[str, Dict[str, str]]:
    """Build the prompt for a chunk and the index mapping num->current_id (the LLM does not see any Cxxx identifiers)."""
    # items_sorted = sorted(items, key=lambda x: x["current_id"])
    items_sorted = sorted(items, key=lambda x: x["unit_id"], reverse=True)
    id_map = {str(i + 1): it["current_id"] for i, it in enumerate(items_sorted)}
    numbered = [f"{i + 1}. {it['question']}" for i, it in enumerate(items_sorted)]
    smallchunkQ = "\n".join(numbered)
    prompt = prompt_deduplicate(smallchunkQ)
    return prompt, id_map


def normalize_source_ids(raw_ids, id_map: Dict[str, str]) -> List[str]:
    """Support both numbers/strings; drop IDs not in the mapping, and (optionally) log missing ones."""
    mapped = []
    missing = 0
    for x in raw_ids:
        k = str(x).strip()
        if k in id_map:
            mapped.append(id_map[k])
        else:
            missing += 1
    return mapped


def group_keys_of_iteration(items: List[Dict], i: int) -> List[str]:
    """Extract group unit_ids produced in iteration i: G<i>-k (or G1-k for the first iteration)."""
    return sorted({it["unit_id"] for it in items if it["iter"] == i})


async def run_pipeline_deduplicate(
    results_full: List[Dict],
    dedup_model: str,
    max_iterations: int = 5,
    max_concurrent: int = 10,
):
    """
    Dedup pipeline with deferred fallback:
    - Iter 1: batch every 5 Complaints (unit_id=Cxxx); produce groups G1-1, G1-2, ...
    - Iter 2+: batch every 5 groups from the previous iteration (unit_id=G<i-1>-k); produce G<i>-1, G<i>-2, ...
    - Any items missed by the model in iteration i are NOT immediately re-run; instead they are queued
      as separate groups (e.g., G<i>-k-FB) and injected into iteration (i+1) grouping.
    - If an iteration would end with only one group left but there are queued fallback groups,
      run another iteration including those fallback groups.
    """

    interface = ModelInterface()

    # Iter 0: expand to leaf items (questions)
    items = build_leaf_items(
        results_full
    )  # each: {unit_id, current_id, original_ids, question, iter=0}

    iteration = 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Queue of fallback groups to inject into *next* iteration.
    # Structure: { fb_tag (str): List[Dict] of it_items from previous iteration }
    queued_fb: Dict[str, List[Dict]] = {}

    while True:
        # ------------------------------------------------------------
        # (A) Inject fallback groups from the previous iteration
        # ------------------------------------------------------------
        if iteration > 1 and queued_fb:
            # We need to materialize each queued FB group as items that belong to prev_iter,
            # so they are picked up when batching prev_iter groups for this iteration.
            prev_iter = iteration - 1
            for fb_tag, fb_block in queued_fb.items():
                # Synthesize per-group numbering for current_id within this FB group.
                for idx, it_item in enumerate(fb_block, start=1):
                    # We carry the text & original_ids forward, but assign unit_id = fb_tag
                    # and set iter = prev_iter so they are grouped this iteration.
                    items.append(
                        {
                            "unit_id": fb_tag,
                            "current_id": f"{fb_tag}.{idx}.fallback",  # traceable id
                            "original_ids": it_item["original_ids"],
                            "question": it_item["question"],
                            "iter": prev_iter,
                        }
                    )
            # Clear the queue after injection
            queued_fb = {}

        # ------------------------------------------------------------
        # (B) Build chunks for the current iteration
        # ------------------------------------------------------------
        if iteration == 1:
            unit_order = sorted({it["unit_id"] for it in items if it["iter"] == 0})
            unit_batches = list(chunk_by(unit_order, 5))
            chunks: List[Tuple[str, List[Dict]]] = []
            for b_idx, units in enumerate(unit_batches, start=1):
                block = [
                    it for it in items if (it["unit_id"] in units and it["iter"] == 0)
                ]
                if block:
                    chunks.append((f"G1-{b_idx}", block))
        else:
            prev_iter = iteration - 1
            prev_group_keys = group_keys_of_iteration(
                items, prev_iter
            )  # includes injected FB groups
            group_batches = list(chunk_by(prev_group_keys, 5))
            chunks = []
            for b_idx, units in enumerate(group_batches, start=1):
                block = [
                    it
                    for it in items
                    if (it["unit_id"] in units and it["iter"] == prev_iter)
                ]
                if block:
                    chunks.append((f"G{iteration}-{b_idx}", block))

        if not chunks:
            # Nothing left to process
            break

        # ------------------------------------------------------------
        # (C) First and only pass this iteration (no immediate second pass):
        #     Prepare prompts and call the model
        # ------------------------------------------------------------
        prompts, id_maps = [], []
        for tag, block in chunks:
            p, m = prompt_deduplicate_from_items(block)
            prompts.append(p)
            id_maps.append(m)

        batch_results = await interface.batch_call(
            prompts=prompts,
            models=[dedup_model],
            temperature=0,
            max_concurrent=max_concurrent,
        )

        filename = f"batch_results_logs/batch_results_stage2_{tag}_{iteration}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)

        # ------------------------------------------------------------
        # (D) Parse outputs, build next-round items, and queue fallbacks
        # ------------------------------------------------------------
        new_items: List[Dict] = []
        per_tag_counter: Dict[str, int] = {}
        total_before = sum(len(block) for _, block in chunks)

        # Temporary store of FB for *this* iteration; to be injected next iteration
        next_queued_fb: Dict[str, List[Dict]] = {}

        for (tag, block), id_map, res in zip(chunks, id_maps, batch_results):
            if tag not in per_tag_counter:
                per_tag_counter[tag] = 0

            # Robustly extract the model's textual payload
            model_text = res.get("response", res)

            # If parsing fails, queue the entire block for fallback (to run next iteration)
            try:
                parsed = prepare_json_string(model_text)
            except Exception:
                fb_tag = f"{tag}-FB"
                next_queued_fb.setdefault(fb_tag, [])
                # Keep the exact items (they currently belong to prev_iter)
                next_queued_fb[fb_tag].extend(block)
                continue

            lookup = {it_item["current_id"]: it_item for it_item in block}
            seen_current_ids = set()
            deduped_list = parsed.get("deduplicated_questions", []) or []

            # Add deduplicated entries to new_items (belong to this iteration under main tag)
            for entry in deduped_list:
                raw_ids = entry.get("source_ids", [])
                mapped_current_ids = normalize_source_ids(raw_ids, id_map)
                seen_current_ids.update(mapped_current_ids)

                original_ids_acc, questions_acc = [], []
                for cid in mapped_current_ids:
                    it_item = lookup.get(cid)
                    if it_item:
                        original_ids_acc.extend(it_item["original_ids"])
                        questions_acc.append(it_item["question"])

                if not original_ids_acc:
                    continue

                per_tag_counter[tag] += 1
                new_items.append(
                    {
                        "unit_id": tag,
                        "current_id": f"{tag}.{per_tag_counter[tag]}",
                        "original_ids": sorted(set(original_ids_acc)),
                        "question": entry.get("question")
                        or (questions_acc[0] if questions_acc else ""),
                        "iter": iteration,
                    }
                )

            # Queue missed items into a fallback group (to be injected next iteration)
            missing_ids = set(lookup.keys()) - seen_current_ids
            if missing_ids:
                fb_tag = f"{tag}-FB"
                next_queued_fb.setdefault(fb_tag, [])
                for cid in sorted(missing_ids):
                    next_queued_fb[fb_tag].append(lookup[cid])

        # Append the current iteration's outputs
        items.extend(new_items)

        # ------------------------------------------------------------
        # (E) Decide whether to terminate or continue
        # ------------------------------------------------------------
        # Count how many *groups* this iteration produced (for the next iteration).
        next_keys = group_keys_of_iteration(items, iteration)

        total_after = len(new_items)
        prev_keys = group_keys_of_iteration(
            items, iteration - 1 if iteration > 1 else 0
        )
        print(
            f"[Iter {iteration}] items: {total_before} -> {total_after} | groups: {len(prev_keys)} -> {len(next_keys)} | queued_fallback: {len(next_queued_fb)}"
        )

        # Core rule:
        # - If only one group remains *and* there are NO queued fallbacks, we can stop.
        # - If only one group remains but we DO have queued fallbacks, we must run one more iteration,
        #   because next iteration will include that single group + all FB groups.
        should_stop = (len(next_keys) <= 1) and (not next_queued_fb)
        if should_stop:
            # No pending FB groups and only one group remains => done
            break

        if iteration >= max_iterations:
            print(f"Reached max_iterations = {max_iterations}, stopping.")
            break

        # Carry queued fallbacks to the next iteration
        queued_fb = next_queued_fb
        iteration += 1

    # ------------------------------------------------------------
    # (F) Final aggregation
    # ------------------------------------------------------------
    last_iter = max((it["iter"] for it in items), default=0)
    final_items = [it for it in items if it["iter"] == last_iter]

    final_comparison = [
        {
            "merged_question": it["question"],
            "current_id": it["current_id"],
            "original_ids": it["original_ids"],
            "source_count": len(it["original_ids"]),
        }
        for it in final_items
    ]
    final_question_list = [it["question"] for it in final_items]

    return items, final_comparison, final_question_list


# -------------------export results -------------------
def get_original_question_text(original_id: str, results_full: List[Dict]) -> str:
    m = re.match(r"^[Cc]?(\d+)\.(\d+)$", original_id.strip())
    if not m:
        return f"[Unrecognized original_id format: {original_id}]"

    complaint_id = int(m.group(1))
    question_idx = int(m.group(2)) - 1

    for item in results_full:
        if int(item.get("Complaint_Index", -1)) == complaint_id:
            ecs = item.get("evaluation_criteria", [])
            if 0 <= question_idx < len(ecs):
                q = ecs[question_idx]
                return (
                    q.get("question")
                    or q.get("text")
                    or f"[Empty question at {original_id}]"
                )
            return f"[Question index out of range: {original_id}]"
    return f"[Complaint not found: {original_id}]"


def export_results(items, final_comparison, dedup_model, results_full):
    """Export results to different file formats in organized folder structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = dedup_model.replace("/", "_")

    # Create folder structure
    main_folder = "stage2_deduplication"
    model_folder = os.path.join("pipeline_output", main_folder, model_name)

    # Create directories if they don't exist
    os.makedirs(model_folder, exist_ok=True)

    # 1. Export complete data (items + final_comparison) to JSON format
    complete_data_filename = os.path.join(
        model_folder, f"Stage2_complete_results_{timestamp}.json"
    )
    with open(complete_data_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": dedup_model,
                "items": items,
                "final_comparison": final_comparison,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 2. Export final_comparison to MD format
    final_md_filename = os.path.join(model_folder, f"final_comparison_{timestamp}.md")
    with open(final_md_filename, "w", encoding="utf-8") as f:
        f.write("# Final Results\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {dedup_model}\n")
        f.write(f"**Final Question Count**: {len(final_comparison)}\n\n")
        f.write("---\n\n")

        for i, comp in enumerate(final_comparison, 1):
            f.write(f"## {i}. Question `{comp['current_id']}`\n\n")
            f.write(f"**Final Question**: {comp['merged_question']}\n\n")
            f.write(f"**Source Count**: {comp['source_count']} original questions\n\n")
            f.write(f"*Original Questions*:\n")
            for orig_id in comp["original_ids"]:
                original_text = get_original_question_text(orig_id, results_full)
                f.write(f"- `{orig_id}`: {original_text}\n")
            f.write("---\n\n")

    # 3. Export last round questions to MD format
    if items:
        last_iter = max(item["iter"] for item in items)
        last_round_items = [item for item in items if item["iter"] == last_iter]

        last_round_filename = os.path.join(
            model_folder, f"last_round_questions_{timestamp}.md"
        )
        with open(last_round_filename, "w", encoding="utf-8") as f:
            f.write(f"# Round {last_iter} Deduplication Results\n\n")
            f.write(f"**Model**: {dedup_model}\n")
            f.write(f"**Question Count**: {len(last_round_items)}\n\n")
            f.write("---\n\n")

            for i, item in enumerate(last_round_items, 1):
                f.write(f"## {i}. `{item['current_id']}`\n\n")
                f.write(f"**Question Content**: {item['question']}\n\n")
                f.write("---\n\n")

    print(f"\n✅ Export completed:")
    print(f"📁 Main folder: {main_folder}")
    print(f"📁 Model folder: {model_folder}")
    print(f"💾 Complete data (JSON): {complete_data_filename}")
    print(f"📋 Final comparison (MD): {final_md_filename}")
    print(f"🎯 Last round questions (MD): {last_round_filename}")
