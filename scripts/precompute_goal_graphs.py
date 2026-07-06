"""Precompute GoalGraphs for R2R / GOAT / SOON (strict GoalNode schema).

Usage:
    # GOAT val_seen: LLM parse + CLIP text/image embeddings
    python scripts/precompute_goal_graphs.py --task goat --split val_seen \\
        --llm-url http://localhost:8000/v1 --embed-clip --no-train

    # Category-only (no vLLM)
    python scripts/precompute_goal_graphs.py --task goat --split val_seen --no-llm --embed-clip

    # Re-embed existing JSON
    python scripts/precompute_goal_graphs.py --embed-only --no-train

    # One-shot GOAT+SOON val (conda env: goat)
    bash scripts/run_goal_graphs_goat_soon.sh
"""

from __future__ import annotations

import argparse
import gzip
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
sys.path.insert(0, str(ROOT))

from conftopo.core.goal_graph_preprocess import (  # noqa: E402
    DESCRIPTION_PARSER_SYSTEM_PROMPT,
    INSTRUCTION_PARSER_PROMPT,
    build_goal_node_from_goat_task,
    build_goal_node_from_soon_layers,
    embed_goal_graphs,
    extract_json_from_llm_text,
    validate_goal_node,
)
from conftopo.core.instruction_graph import GoalNode, InstructionGraph, SubGoal  # noqa: E402


class GoalGraphLLM:
    """HTTP client for offline GoalGraph LLM parsing (vLLM OpenAI API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=2,
        )
        self.model = model
        self._instruction_cache: Dict[str, dict] = {}
        self._description_cache: Dict[str, dict] = {}

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def parse_instruction(self, instruction: str, use_cache: bool = True) -> dict:
        if use_cache and instruction in self._instruction_cache:
            return self._instruction_cache[instruction]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INSTRUCTION_PARSER_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        except Exception:
            result = {"sub_goals": []}
        if use_cache:
            self._instruction_cache[instruction] = result
        return result

    def parse_description(self, description: str, use_cache: bool = True) -> Optional[dict]:
        if use_cache and description in self._description_cache:
            return self._description_cache[description]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DESCRIPTION_PARSER_SYSTEM_PROMPT},
                    {"role": "user", "content": description},
                ],
                temperature=0.0,
                max_tokens=384,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            result = extract_json_from_llm_text(text)
        except Exception:
            result = None
        if use_cache and result is not None:
            self._description_cache[description] = result
        return result


def _llm_description_fn(llm_client: Optional[GoalGraphLLM]):
    if llm_client is None:
        return None
    return llm_client.parse_description


def _warn_goal_node(context: str, goal: GoalNode) -> None:
    for msg in validate_goal_node(goal):
        print(f"  [WARN] {context}: {msg}", flush=True)


# ==================== R2R ====================

def process_r2r(data_dir: str, output_dir: str, splits: List[str], llm_client: Optional[GoalGraphLLM] = None):
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        input_path = os.path.join(data_dir, split, f"{split}.json.gz")
        if not os.path.exists(input_path):
            print(f"  [SKIP] {input_path}")
            continue

        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data.get("episodes", data if isinstance(data, list) else [])
        results = {}

        for i, ep in enumerate(episodes):
            ep_id = ep.get("episode_id", i)
            instr = ep.get("instruction", {})
            text = instr.get("instruction_text", "") if isinstance(instr, dict) else str(instr)

            ig = InstructionGraph(goal_type="route")
            sg_list = None
            if llm_client and text:
                try:
                    parsed = llm_client.parse_instruction(text)
                    sg_list = parsed if isinstance(parsed, list) else parsed.get("sub_goals", [parsed])
                except Exception:
                    sg_list = None

            if not sg_list:
                sg_list = [{
                    "action": "go_forward",
                    "landmark": text[:80] if text else None,
                    "spatial_relation": "towards",
                    "implied_room": None,
                    "termination_condition": "end",
                }]

            for j, sg_data in enumerate(sg_list):
                if isinstance(sg_data, str):
                    continue
                ig.sub_goals.append(SubGoal(
                    id=j,
                    action=sg_data.get("action", "go_forward"),
                    landmark=sg_data.get("landmark"),
                    spatial_relation=sg_data.get("spatial_relation", "at"),
                    implied_room=sg_data.get("implied_room"),
                    termination_condition=sg_data.get("termination_condition", ""),
                ))

            results[str(ep_id)] = ig.to_dict()
            if (i + 1) % 500 == 0:
                print(f"  [{split}] {i+1}/{len(episodes)}")

        out_path = os.path.join(output_dir, f"{split}_goal_graphs.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"  [{split}] {len(results)} episodes -> {out_path}")


# ==================== GOAT ====================

def process_goat(goat_root: str, output_dir: str, splits: List[str], llm_client: Optional[GoalGraphLLM] = None):
    os.makedirs(output_dir, exist_ok=True)
    parse_desc = _llm_description_fn(llm_client)

    split_map = {
        "train": "train",
        "val": "val_seen",
        "val_seen": "val_seen",
        "val_unseen": "val_unseen",
        "val_seen_synonyms": "val_seen_synonyms",
    }

    for split in splits:
        folder = split_map.get(split, split)
        content_dir = os.path.join(goat_root, folder, "content")
        if not os.path.isdir(content_dir):
            print(f"  [SKIP] {content_dir}")
            continue

        results = {}
        scene_files = sorted(glob.glob(os.path.join(content_dir, "*.json.gz")))
        ep_count = 0
        warn_count = 0

        for si, sf in enumerate(scene_files):
            scene_name = os.path.basename(sf).replace(".json.gz", "")
            with gzip.open(sf, "rt", encoding="utf-8") as f:
                data = json.load(f)

            goals = data.get("goals", {})
            for ep in data.get("episodes", []):
                ep_id = ep.get("episode_id", 0)
                key = f"{scene_name}_{ep_id}"
                ig = InstructionGraph(goal_type="object_goal")

                for task in ep.get("tasks", []):
                    if len(task) < 2:
                        continue
                    category, modality = task[0], task[1]
                    ref = task[2] if len(task) > 2 else None
                    goal_image_id = int(task[3]) if len(task) > 3 and task[1] == "image" else None
                    try:
                        gn = build_goal_node_from_goat_task(
                            category, modality, ref, goals, parse_desc, goal_image_id,
                        )
                        warns = validate_goal_node(gn)
                        if warns:
                            warn_count += 1
                            if warn_count <= 20:
                                _warn_goal_node(key, gn)
                        ig.goal_nodes.append(gn)
                    except Exception as exc:
                        print(f"  [WARN] GOAT parse failed {key}: {exc}", flush=True)
                        continue

                if ig.goal_nodes:
                    results[key] = ig.to_dict()
                    ep_count += 1

            if (si + 1) % 5 == 0 or si == len(scene_files) - 1:
                print(f"  [{folder}] {si+1}/{len(scene_files)} scenes, {ep_count} eps", flush=True)

        out_path = os.path.join(output_dir, f"{folder}_goal_graphs.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"  [{folder}] DONE {ep_count} episodes from {len(scene_files)} scenes -> {out_path}", flush=True)
        if warn_count:
            print(f"  [{folder}] validation warnings on {warn_count} goal nodes", flush=True)


# ==================== SOON ====================

SOON_FILES = {
    "train": "train.json",
    "val_unseen_instrs": "val_unseen_instrs.json",
    "val_unseen_house": "val_unseen_house.json",
    "test_release": "test_release.json",
}


def _iter_soon_instruction_groups(item: dict) -> List[Tuple[List[str], int]]:
    instructions = item.get("instructions") or []
    if not instructions:
        return []
    if isinstance(instructions[0], str):
        return [(instructions, 0)]
    groups: List[Tuple[List[str], int]] = []
    for gi, group in enumerate(instructions):
        if isinstance(group, list) and group:
            groups.append((group, gi))
    return groups


def _soon_episode_key(item: dict, group: List[str], group_idx: int, item_idx: int) -> str:
    scan = item.get("scan")
    if not scan and item.get("bboxes"):
        scan = item["bboxes"][0].get("scan", "unknown")
    if not scan:
        scan = "unknown"
    instr_id = item.get("instr_id", item_idx)
    uid = str(group[5]) if len(group) > 5 else str(group_idx)
    return f"{scan}_{instr_id}_{uid}"


def process_soon(
    soon_dir: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
    llm_client: Optional[GoalGraphLLM] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    parse_desc = _llm_description_fn(llm_client)

    if splits:
        selected = {}
        for name in splits:
            if name in SOON_FILES:
                selected[name] = SOON_FILES[name]
            elif name.endswith(".json"):
                selected[name.replace(".json", "")] = name
            else:
                selected[name] = SOON_FILES.get(name, f"{name}.json")
    else:
        selected = dict(SOON_FILES)

    for split_name, json_name in selected.items():
        input_path = os.path.join(soon_dir, json_name)
        if not os.path.isfile(input_path):
            print(f"  [SKIP] {input_path}")
            continue

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        results: Dict[str, dict] = {}
        for item_idx, item in enumerate(data):
            if (item_idx + 1) % 50 == 0:
                print(f"  [{split_name}] {item_idx+1}/{len(data)}")

            obj_name = None
            bboxes = item.get("bboxes") or []
            if bboxes:
                obj_name = bboxes[0].get("obj_name")

            for group, gi in _iter_soon_instruction_groups(item):
                key = _soon_episode_key(item, group, gi, item_idx)
                try:
                    gn = build_goal_node_from_soon_layers(group, obj_name, parse_desc)
                    _warn_goal_node(key, gn)
                    ig = InstructionGraph(goal_type="object_goal")
                    ig.goal_nodes.append(gn)
                    results[key] = ig.to_dict()
                except Exception as exc:
                    print(f"  [WARN] SOON parse failed key={key}: {type(exc).__name__}: {exc}")

        out_path = os.path.join(output_dir, f"{split_name}_goal_graphs.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"  [{split_name}] {len(results)} goals from {len(data)} items -> {out_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Precompute GoalGraph JSON for R2R/GOAT/SOON")
    parser.add_argument("--task", choices=["r2r", "goat", "soon", "all"], default="all")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["r2r", "goat", "soon"],
        help="Run only these tasks (overrides --task)",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--r2r-dir", default=str(DATA_ROOT / "datasets/r2r"))
    parser.add_argument("--goat-dir", default=str(DATA_ROOT / "datasets/goat/hm3d/v1"))
    parser.add_argument("--soon-dir", default=str(DATA_ROOT / "datasets/soon"))
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "goal_graphs"))
    parser.add_argument("--scene-root", default=str(DATA_ROOT / "scene_datasets/hm3d"))
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--embed-clip", action="store_true")
    parser.add_argument("--embed-clip-train", action="store_true")
    parser.add_argument("--no-embed-images", action="store_true", help="Skip habitat image rendering for image goals")
    parser.add_argument(
        "--force-reembed-images",
        action="store_true",
        help="Re-render all image goals even if target_embedding already exists",
    )
    parser.add_argument(
        "--clip-cache",
        default=str(DATA_ROOT.parent / "clip"),
        help="Directory containing CLIP weights. Defaults to <workspace>/clip/ where ViT-B-32.pt lives.",
    )
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--goat-image-cache-dir",
        default=None,
        help="Directory with official GOAT image cache pickles (e.g. data/goat-assets/goal_cache/iin)",
    )
    parser.add_argument(
        "--goat-image-encoder",
        default="CLIP",
        help="Encoder name in cache filename, e.g. CLIP → {scene}_CLIP_goat_embedding.pkl",
    )
    args = parser.parse_args()

    r2r_default = ["val_seen", "val_unseen"] if args.no_train else ["train", "val_seen", "val_unseen"]
    goat_default = (
        ["val_seen", "val_unseen", "val_seen_synonyms"]
        if args.no_train
        else ["train", "val_seen", "val_unseen", "val_seen_synonyms"]
    )
    soon_default = ["val_unseen_instrs", "val_unseen_house"] if args.no_train else None

    if args.embed_only:
        print("\n=== CLIP embeddings (embed-only) ===")
        embed_goal_graphs(
            args.output_dir,
            clip_cache=args.clip_cache,
            include_train=args.embed_clip_train,
            goat_root=args.goat_dir,
            scene_root=args.scene_root,
            embed_images=not args.no_embed_images,
            skip_existing_image_embed=not args.force_reembed_images,
            goat_image_cache_dir=args.goat_image_cache_dir,
            goat_image_encoder=args.goat_image_encoder,
        )
        print("\nDone!")
        return

    llm_client: Optional[GoalGraphLLM] = None
    if not args.no_llm:
        try:
            llm_client = GoalGraphLLM(base_url=args.llm_url, model=args.llm_model)
            if llm_client.is_available():
                print(f"LLM API OK at {args.llm_url} (model={args.llm_model})")
            else:
                print(f"LLM API not reachable at {args.llm_url} -> fallback mode")
                print("  Start: bash ETPNav/scripts/start_vllm.sh 8000  # uses conda env vllm")
                llm_client = None
        except ImportError as exc:
            print(f"openai not installed ({exc}) -> fallback mode")

    active_tasks = args.tasks if args.tasks else (
        ["r2r", "goat", "soon"] if args.task == "all" else [args.task]
    )

    if "r2r" in active_tasks:
        print("\n=== R2R ===")
        splits = [args.split] if args.split else r2r_default
        process_r2r(args.r2r_dir, os.path.join(args.output_dir, "r2r"), splits, llm_client)

    if "goat" in active_tasks:
        print("\n=== GOAT ===")
        splits = [args.split] if args.split else goat_default
        process_goat(args.goat_dir, os.path.join(args.output_dir, "goat"), splits, llm_client)

    if "soon" in active_tasks:
        if os.path.isdir(args.soon_dir):
            print("\n=== SOON ===")
            soon_splits = [args.split] if args.split else soon_default
            process_soon(args.soon_dir, os.path.join(args.output_dir, "soon"), splits=soon_splits, llm_client=llm_client)
        else:
            print(f"\n=== SOON [SKIP] no data at {args.soon_dir} ===")

    if args.embed_clip:
        print("\n=== CLIP embeddings ===")
        embed_goal_graphs(
            args.output_dir,
            clip_cache=args.clip_cache,
            include_train=args.embed_clip_train,
            goat_root=args.goat_dir,
            scene_root=args.scene_root,
            embed_images=not args.no_embed_images,
            skip_existing_image_embed=not args.force_reembed_images,
            goat_image_cache_dir=args.goat_image_cache_dir,
            goat_image_encoder=args.goat_image_encoder,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
