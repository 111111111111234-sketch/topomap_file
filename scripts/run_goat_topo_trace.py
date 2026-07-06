from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from run_goat_minimal import (
    ROOT,
    PretrainedPointNavController,
    controller_step,
    find_scene_file,
    load_goal_graph,
    normalize_quat,
    pick_episode,
    quat_to_heading,
    rgb_to_embedding,
    make_sim,
)
from conftopo.agents import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.perception import GoatModalityClipEncoder, encode_agent_rgb_embed
from conftopo.navigation import CollisionLikeTracker, PathfinderExecutor


DEFAULT_ENV_LANDMARK_LABELS = [
    "door",
    "window",
    "stairs",
    "wall",
    "corner",
    "doorway",
    "corridor",
    "hallway",
    "entrance",
    "opening",
    "kitchen",
    "bedroom",
    "bathroom",
    "toilet",
    "living room",
    "dining room",
    "table",
    "sofa",
    "bed",
    "counter",
    "cabinet",
    "shelf",
    "bookshelf",
    "desk",
    "picture",
    "painting",
    "mirror",
    "TV",
    "fireplace",
    "plant",
    "lamp",
    "sink",
    "refrigerator",
    "oven",
    "bathtub",
    "washer",
]


def _tolist(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_tolist(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tolist(v) for v in value)
    if isinstance(value, dict):
        return {k: _tolist(v) for k, v in value.items() if k != "per_view_goal_sims"}
    return value


def top_scores(scores: list, k: int = 5) -> list[dict[str, Any]]:
    return [{"label": label, "score": float(score)} for label, score in scores[:k]]


def _agent_perception_dict(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    """Support legacy ``_cur_perception``, refactored ``_last_clip_hint``, and ``_last_packet.report``."""
    if hasattr(agent, "_cur_perception"):
        return dict(agent._cur_perception or {})
    hint = getattr(agent, "_last_clip_hint", None)
    if hint is not None:
        raw = getattr(hint, "raw", {}) or {}
        return {
            "room_label": str(getattr(hint, "room_label", "unknown")),
            "room_confidence": float(getattr(hint, "room_confidence", 0.0)),
            "best_goal_sim": float(getattr(hint, "best_goal_sim", 0.0)),
            "best_landmark_sim": float(getattr(hint, "best_landmark_sim", 0.0)),
            "room_scores": list(raw.get("room_scores", [])),
            "goal_scores": list(getattr(hint, "goal_scores", [])),
            "landmark_scores": list(getattr(hint, "landmark_scores", [])),
            "best_view_idx": 0,
        }
    packet = getattr(agent, "_last_packet", None)
    if packet is not None and getattr(packet, "report", None) is not None:
        report = packet.report
        return {
            "room_label": report.room_label,
            "room_confidence": float(report.room_confidence),
            "best_goal_sim": float(report.best_goal_sim),
            "best_landmark_sim": float(report.best_landmark_sim),
            "room_scores": list(report.room_scores),
            "goal_scores": list(report.goal_scores),
            "landmark_scores": list(report.landmark_scores),
            "best_view_idx": int(getattr(report, "best_view_idx", 0)),
            "fresh_vlm": bool(getattr(packet, "fresh_vlm", False)),
            "trigger_reason": str(getattr(packet, "trigger_reason", "")),
            "object_count": len(report.objects),
        }
    return {}


def snapshot_perception(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    p = _agent_perception_dict(agent)
    return {
        "room_label": p.get("room_label", "unknown"),
        "room_confidence": float(p.get("room_confidence", 0.0)),
        "best_goal_sim": float(p.get("best_goal_sim", 0.0)),
        "best_landmark_sim": float(p.get("best_landmark_sim", 0.0)),
        "room_scores": top_scores(p.get("room_scores", [])),
        "goal_scores": top_scores(p.get("goal_scores", [])),
        "landmark_scores": top_scores(p.get("landmark_scores", [])),
        "best_view_idx": int(p.get("best_view_idx", 0)),
    }


def snapshot_topo(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    """Capture topological map with dual-layer structure: semantic + navigation layers."""
    from conftopo.core.dynamic_topo_map import NodeType

    def _display_label(node: Any) -> str:
        label = str(getattr(node, "label", "") or "").strip()
        attrs = getattr(node, "attributes", {}) or {}
        node_type = getattr(node, "node_type", None)
        if node_type == NodeType.ROOM:
            room_probs = attrs.get("room_probs") or {}
            if isinstance(room_probs, dict) and room_probs:
                best_label = max(
                    room_probs.items(),
                    key=lambda item: float(item[1]) if item[1] is not None else 0.0,
                )[0]
                best_text = str(best_label).strip()
                if best_text:
                    return best_text
        return label
    
    topo = agent.topo_map
    
    # 结构层节点（语义知识：房间、物体、地标、传送点）
    structure_nodes_list = []
    try:
        for node in topo.structure_layer.nodes():
            structure_nodes_list.append({
                "id": node.node_id,
                "type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
                "position": np.asarray(node.position).round(4).tolist(),
                "confidence": float(node.confidence),
                "label": _display_label(node),
                "raw_label": node.label,
                "step_id": int(node.step_id),
                "visit_count": int(node.visit_count),
                "attributes": dict(node.attributes) if node.attributes else {},
            })
    except Exception as e:
        # Fallback if structure_layer not fully initialized
        structure_nodes_list = []
    
    # 导航层节点（可执行路径：waypoint、frontier、candidate、approach）
    navigation_nodes_list = []
    try:
        for node in topo.navigation_layer.nodes():
            navigation_nodes_list.append({
                "id": node.node_id,
                "type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
                "position": np.asarray(node.position).round(4).tolist(),
                "confidence": float(node.confidence),
                "step_id": int(node.step_id),
                "visit_count": int(node.visit_count),
                "attributes": dict(node.attributes) if node.attributes else {},
            })
    except Exception as e:
        # Fallback if navigation_layer not fully initialized
        navigation_nodes_list = []
    
    # 跨层链接（结构意图 → 导航目标的映射）
    cross_layer_edges = []
    try:
        for u, v, data in topo.cross_layer.edges():
            edge_type = data.get("edge_type", "")
            if hasattr(edge_type, "value"):
                edge_type = edge_type.value
            cross_layer_edges.append({
                "source": u,
                "target": v,
                "type": str(edge_type),
                "relation": data.get("relation", ""),
            })
    except Exception as e:
        cross_layer_edges = []
    
    # 完整图数据（保持向后兼容）
    all_nodes = []
    for node in topo._nodes.values():
        all_nodes.append({
            "id": node.node_id,
            "type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            "position": np.asarray(node.position).round(4).tolist(),
            "confidence": float(node.confidence),
            "label": _display_label(node),
            "raw_label": node.label,
            "step_id": int(node.step_id),
            "visit_count": int(node.visit_count),
            "attributes": dict(node.attributes) if node.attributes else {},
        })
    
    all_edges = []
    for u, v, data in topo.graph.edges(data=True):
        edge_type = data.get("edge_type", "")
        if hasattr(edge_type, "value"):
            edge_type = edge_type.value
        edge_entry = {"source": u, "target": v, "type": edge_type, "weight": float(data.get("weight", 1.0))}
        for extra_key in ["distance_m", "traversability", "visited_count", "evidence", "directions", "description_type", "connected_rooms", "via_landmark", "passage_type", "confidence", "transitions"]:
            if extra_key in data:
                edge_entry[extra_key] = data[extra_key]
        all_edges.append(edge_entry)
    
    # 层级统计
    layers_summary = {
        "structure_layer_nodes": len(structure_nodes_list),
        "navigation_layer_nodes": len(navigation_nodes_list),
        "cross_layer_links": len(cross_layer_edges),
        "total_graph_nodes": len(all_nodes),
        "total_graph_edges": len(all_edges),
    }
    
    return {
        "nodes": all_nodes,  # 向后兼容
        "edges": all_edges,  # 向后兼容
        "structure_layer": {
            "nodes": structure_nodes_list,
            "node_count": len(structure_nodes_list),
        },
        "navigation_layer": {
            "nodes": navigation_nodes_list,
            "node_count": len(navigation_nodes_list),
        },
        "cross_layer": {
            "edges": cross_layer_edges,
            "edge_count": len(cross_layer_edges),
        },
        "layers_summary": layers_summary,
    }


def landmark_names(raw) -> list[str]:
    names = []
    if isinstance(raw, dict):
        raw = list(raw.keys())
    for item in raw or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("object") or item.get("label")
            if name:
                names.append(str(name))
        else:
            names.append(str(item))
    return names


def parse_env_landmarks(value: str) -> list[str]:
    value = (value or "default").strip()
    if value.lower() in {"", "none", "off", "false", "0"}:
        return []
    if value.lower() == "default":
        return list(DEFAULT_ENV_LANDMARK_LABELS)
    labels = [item.strip() for item in value.split(",")]
    return [label for label in labels if label]


def dedupe_labels(labels: list[str]) -> list[str]:
    out = []
    seen = set()
    for label in labels:
        key = label.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(label.strip())
    return out


def configure_landmark_perception(
    agent: ConfTopoGOATAgent,
    first_goal: GoalNode | None,
    encoder: GoatModalityClipEncoder | None,
    env_landmark_labels: list[str],
) -> list[str]:
    goal_lm_names = landmark_names(first_goal.landmarks) if isinstance(first_goal, GoalNode) else []
    all_labels = dedupe_labels(goal_lm_names + env_landmark_labels)
    agent.set_environment_landmark_labels(env_landmark_labels)
    if not all_labels or encoder is None:
        return all_labels

    embeds_by_label: dict[str, np.ndarray] = {}
    if isinstance(first_goal, GoalNode) and goal_lm_names:
        goal_embeds = first_goal.landmark_embeddings
        if goal_embeds is None:
            goal_embeds = encoder.encode_text(goal_lm_names)
        for label, embed in zip(goal_lm_names, goal_embeds):
            embeds_by_label[label.strip().lower()] = np.asarray(embed, dtype=np.float32)

    missing = [label for label in all_labels if label.strip().lower() not in embeds_by_label]
    if missing:
        missing_embeds = encoder.encode_text(missing)
        for label, embed in zip(missing, missing_embeds):
            embeds_by_label[label.strip().lower()] = np.asarray(embed, dtype=np.float32)

    all_embeds = np.asarray([embeds_by_label[label.strip().lower()] for label in all_labels], dtype=np.float32)
    agent.perceiver.set_landmark_labels(all_labels, all_embeds)
    return all_labels


def current_goal_summary(ig) -> dict[str, Any]:
    goal = ig.get_current_goal()
    if isinstance(goal, GoalNode):
        return {
            "target_object": goal.target_object,
            "goal_type": goal.goal_type,
            "attributes": goal.attributes,
            "room_prior": goal.room_prior,
            "landmarks": landmark_names(goal.landmarks),
            "has_target_embedding": goal.target_embedding is not None,
            "target_embedding_dim": None if goal.target_embedding is None else int(goal.target_embedding.shape[-1]),
        }
    return {"goal": str(goal)}


GOAL_MODALITY_ALIASES = {
    "object": {"object", "category"},
    "instruction": {"instruction", "description"},
    "description": {"instruction", "description"},
    "image": {"image"},
}


def goal_node_modality(goal: GoalNode) -> str:
    goal_type = (goal.goal_type or "").lower()
    if goal_type == "category":
        return "object"
    if goal_type == "description":
        return "instruction"
    return goal_type or "unknown"


def available_goal_modalities(ig: InstructionGraph) -> list[str]:
    seen = []
    for goal in ig.goal_nodes:
        modality = goal_node_modality(goal)
        if modality not in seen:
            seen.append(modality)
    return seen


def select_goal_modality(ig: InstructionGraph, requested: str) -> tuple[InstructionGraph, dict[str, Any]]:
    requested = (requested or "auto").lower()
    if requested == "auto":
        current = ig.get_current_goal()
        return ig, {
            "requested_goal_modality": "auto",
            "selected_goal_modality": goal_node_modality(current) if isinstance(current, GoalNode) else "unknown",
            "selected_goal_index": ig.current_idx,
            "available_goal_modalities": available_goal_modalities(ig),
        }

    aliases = GOAL_MODALITY_ALIASES[requested]
    for idx, goal in enumerate(ig.goal_nodes):
        if (goal.goal_type or "").lower() in aliases:
            selected = InstructionGraph(goal_type=ig.goal_type, goal_nodes=[goal])
            selected.set_current_goal_by_index(0)
            return selected, {
                "requested_goal_modality": requested,
                "selected_goal_modality": goal_node_modality(goal),
                "selected_goal_index": idx,
                "available_goal_modalities": available_goal_modalities(ig),
            }

    available = ", ".join(available_goal_modalities(ig)) or "none"
    raise ValueError(f"Goal modality '{requested}' not found for this episode. Available: {available}")


def planar_distance(a, b) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((aa - bb)[[0, 2]]))


def semantic_decisions(perception: dict, thresholds: dict) -> dict[str, list[dict[str, Any]]]:
    mapping = {
        'objects': ('goal_scores', 'object'),
        'rooms': ('room_scores', 'room'),
        'landmarks': ('landmark_scores', 'landmark'),
    }
    out = {}
    for name, (score_key, threshold_key) in mapping.items():
        threshold = float(thresholds.get(threshold_key, 0.0))
        rows = []
        for item in perception.get(score_key, [])[:5]:
            score = float(item.get('score', 0.0))
            rows.append({
                'label': item.get('label', ''),
                'score': score,
                'threshold': threshold,
                'decision': 'accepted' if score >= threshold else 'below_threshold',
            })
        out[name] = rows
    return out


def build_selection_debug(target_node_id, candidate_scores, perception, thresholds, sticky_debug):
    ranked = sorted(candidate_scores, key=lambda x: float(x.get('score', 0.0)), reverse=True)
    selected_rank = None
    for idx, item in enumerate(ranked, start=1):
        if item.get('node_id') == target_node_id:
            selected_rank = idx
            break
    return {
        'selected_target': target_node_id,
        'selected_rank': selected_rank,
        'top_candidate_scores': ranked[:5],
        'all_candidate_scores': ranked,
        'semantic_decisions': semantic_decisions(perception, thresholds),
        'sticky_debug': sticky_debug or {},
        'skipped_candidates': (sticky_debug or {}).get('skipped_candidates', []),
        'blocked_targets': (sticky_debug or {}).get('blocked_targets', {}),
        'navigation_event': (sticky_debug or {}).get('navigation_event', {}),
        'navigation_debug': (sticky_debug or {}).get('navigation_debug', {}),
        'reachable_candidates': (sticky_debug or {}).get('reachable_candidates', []),
        'unreachable_candidates': (sticky_debug or {}).get('unreachable_candidates', []),
        'thresholds': thresholds,
    }


def max_score(trace: dict, key: str) -> float:
    vals = []
    for st in trace.get("steps", []):
        scores = st.get("perception", {}).get(key, [])
        if scores:
            vals.append(float(scores[0]["score"]))
    return max(vals) if vals else 0.0



def count_target_switches(steps: list[dict[str, Any]]) -> int:
    prev = None
    switches = 0
    for st in steps:
        cur = st.get("target_node_id")
        if prev is not None and cur != prev:
            switches += 1
        prev = cur
    return switches


def path_length(steps: list[dict[str, Any]]) -> float:
    total = 0.0
    prev = None
    for st in steps:
        pos = st.get("position")
        if pos is not None and prev is not None:
            total += float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - np.asarray(prev, dtype=np.float32)))
        prev = pos
    return total


def memory_reuse_count(steps: list[dict[str, Any]]) -> int:
    count = 0
    for st in steps:
        target = st.get("target_node_id") or ""
        if target.startswith(("obj_", "roo_", "lan_")):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the agent stops")
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--goal-modality", choices=["auto", "object", "instruction", "description", "image"], default="auto")
    parser.add_argument("--output", default="data/logs/goat_topo/topo_trace_semantic.json")
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-image-model", default="RN50", help="CLIP model for image-goal rgb_embed (GOAT official: RN50)")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--object-threshold", type=float, default=None)
    parser.add_argument("--heavy-enabled", action="store_true")
    parser.add_argument("--heavy-interval", type=int, default=None)
    parser.add_argument("--object-detection-threshold", type=float, default=None)
    parser.add_argument("--groundingdino-config", default=None)
    parser.add_argument("--groundingdino-checkpoint", default=None)
    parser.add_argument("--groundingdino-device", default=None)
    parser.add_argument("--room-threshold", type=float, default=None)
    parser.add_argument("--landmark-threshold", type=float, default=None)
    parser.add_argument(
        "--env-landmarks",
        default="default",
        help="Comma-separated scene landmark labels, 'default' for built-ins, or 'none' to disable.",
    )
    parser.add_argument("--use-placeholder-embed", action="store_true")
    parser.add_argument(
        "--controller-mode",
        choices=["pathfinder", "pretrained_pointnav"],
        default="pathfinder",
        help="Low-level controller backend.",
    )
    parser.add_argument(
        "--pointnav-model-path",
        default="ETPNav/data/ddppo-models/gibson-2plus-se-resneXt50-rgb.pth",
        help="Checkpoint path for pretrained PointNav policy.",
    )
    parser.add_argument("--pointnav-input-type", choices=["rgb", "depth", "rgbd"], default="rgb")
    parser.add_argument("--pointnav-resolution", type=int, default=256)
    parser.add_argument("--pointnav-stop-radius", type=float, default=0.35)
    parser.add_argument("--pointnav-gpu-id", type=int, default=0)
    args = parser.parse_args()
    env_landmark_labels = parse_env_landmarks(args.env_landmarks)

    dataset_dir = ROOT / args.dataset_dir
    scene_path, episode = pick_episode(dataset_dir, args.split, args.scene, args.episode_index)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])
    ig, goal_modality_debug = select_goal_modality(ig, args.goal_modality)

    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_image_model = args.clip_image_model
    config.perception.clip_device = args.clip_device
    if args.object_threshold is not None:
        config.perception.object_threshold = args.object_threshold
    config.perception.heavy_enabled = bool(args.heavy_enabled)
    if args.heavy_interval is not None:
        config.perception.heavy_interval = args.heavy_interval
    if args.object_detection_threshold is not None:
        config.perception.object_detection_threshold = args.object_detection_threshold
    if args.groundingdino_config is not None:
        config.perception.groundingdino_config = args.groundingdino_config
    if args.groundingdino_checkpoint is not None:
        config.perception.groundingdino_checkpoint = args.groundingdino_checkpoint
    if args.groundingdino_device is not None:
        config.perception.groundingdino_device = args.groundingdino_device
    if args.room_threshold is not None:
        config.perception.room_threshold = args.room_threshold
    if args.landmark_threshold is not None:
        config.perception.landmark_threshold = args.landmark_threshold

    agent = ConfTopoGOATAgent(config)
    agent.set_goal(ig)
    first_goal = ig.get_current_goal()
    if isinstance(first_goal, GoalNode):
        agent.set_new_goal(first_goal)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = GoatModalityClipEncoder(
            config.perception.clip_model,
            config.perception.clip_image_model,
            config.perception.clip_device,
        )
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)
        active_landmark_labels = configure_landmark_perception(agent, first_goal, encoder, env_landmark_labels)
    else:
        active_landmark_labels = []

    frame_dir = ROOT / args.frame_dir if args.frame_dir else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)

    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)

    import habitat_sim
    state = habitat_sim.AgentState()
    state.position = np.array(episode["start_position"], dtype=np.float32)
    q = normalize_quat(episode["start_rotation"])
    if abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6 and (abs(q[1]) > 1e-6 or abs(q[3]) > 1e-6):
        state.rotation = np.quaternion(q[3], q[0], q[1], q[2])
    else:
        state.rotation = np.quaternion(q[0], q[1], q[2], q[3])
    sim_agent.set_state(state)
    origin = np.array(state.position, dtype=np.float32)
    nav_executor = PathfinderExecutor()
    collision_tracker = CollisionLikeTracker()
    previous_low_action = None
    pointnav_controller = None
    if args.controller_mode == "pretrained_pointnav":
        if not args.pointnav_model_path:
            raise ValueError("--pointnav-model-path is required when --controller-mode=pretrained_pointnav")
        model_path = Path(args.pointnav_model_path)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        pointnav_controller = PretrainedPointNavController(
            model_path=str(model_path),
            input_type=args.pointnav_input_type,
            resolution=args.pointnav_resolution,
            stop_radius=args.pointnav_stop_radius,
            pth_gpu_id=args.pointnav_gpu_id,
        )

    trace: dict[str, Any] = {
        "split": args.split,
        "scene": args.scene,
        "episode_index": args.episode_index,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "origin_world": origin.round(4).tolist(),
        "start_rotation": episode["start_rotation"],
        "num_tasks": len(episode.get("tasks", [])),
        "tasks": episode.get("tasks", []),
        "goal_type": ig.goal_type,
        **goal_modality_debug,
        "current_goal": current_goal_summary(ig),
        "environment_landmarks": env_landmark_labels,
        "active_landmark_labels": active_landmark_labels,
        "embedding_source": "placeholder" if args.use_placeholder_embed else f"clip:{args.clip_model}",
        "thresholds": {
            "object": config.perception.object_threshold,
            "room": config.perception.room_threshold,
            "landmark": config.perception.landmark_threshold,
        },
        "coordinate_frame": "episode_start_relative",
        "pose_sources": {
            "position": "episode_start_relative_from_gt_world",
            "heading": "habitat_gt_heading",
            "world_position": "habitat_gt_world_position",
            "world_heading": "habitat_gt_heading",
        },
        "controller": {
            "mode": args.controller_mode,
            "pointnav_model_path": args.pointnav_model_path,
            "pointnav_input_type": args.pointnav_input_type,
            "pointnav_resolution": args.pointnav_resolution,
            "pointnav_stop_radius": args.pointnav_stop_radius,
            "pointnav_gpu_id": args.pointnav_gpu_id,
        },
        "steps": [],
    }

    step = 0
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            obs = sim.get_sensor_observations()
            state = sim_agent.get_state()
            rgb = obs.get("color_sensor")
            if rgb is not None and rgb.shape[-1] == 4:
                rgb_save = rgb[..., :3]
            else:
                rgb_save = rgb
            frame_rel = None
            if frame_dir is not None and rgb_save is not None:
                frame_path = frame_dir / f"rgb_{step:04d}.png"
                imageio.imwrite(frame_path, rgb_save)
                frame_rel = str(frame_path.relative_to(ROOT))
            rgb_embed = encode_agent_rgb_embed(
                encoder, rgb, agent,
                use_placeholder=args.use_placeholder_embed,
                placeholder_fn=rgb_to_embedding,
            )
            conf_obs = {
                "rgb": rgb,
                "rgb_embed": rgb_embed,
                "position": np.array(state.position, dtype=np.float32),
                "heading": quat_to_heading([state.rotation.real, *list(state.rotation.imag)]),
            }
            out = agent.step(conf_obs)
            world_position = np.asarray(state.position, dtype=np.float32)
            world_heading = float(conf_obs["heading"])
            rel_position = world_position - origin
            collision_debug = collision_tracker.update(previous_low_action, rel_position)
            navigation_event = {}
            navigation_debug = {}

            target = out.get("target_position")
            target_node_id = out.get("target_node_id")
            candidate_ids = out.get("candidate_ids", []) or []
            scores = np.asarray(out.get("scores", []), dtype=np.float32).tolist()
            perception = snapshot_perception(agent)

            reachable_debug = {"selected": None, "reachable_candidates": [], "unreachable_candidates": []}
            if candidate_ids:
                reachable_debug = nav_executor.select_reachable_candidate(sim, origin, agent.topo_map, candidate_ids, scores, top_k=5)
                selected = reachable_debug.get("selected")
                if selected is not None and selected.get("node_id") != target_node_id:
                    node = agent.topo_map.get_node(selected["node_id"])
                    if node is not None:
                        target_node_id = node.node_id
                        target = node.position.copy()

            if agent.should_stop():
                low_action = "stop"
                navigation_debug = {"reachable": True, "reason": "goal_similarity_stop", "best_goal_sim": float(perception.get("best_goal_sim", 0.0))}
            elif collision_debug.get("collision_like") and target_node_id is not None:
                navigation_event = agent.on_navigation_event(target_node_id, "collision_like")
                low_action = "target_reached"
                navigation_debug = {"reachable": False, "reason": "collision_like", "release_reason": "collision_like"}
            elif target is None:
                low_action = "stop"
                navigation_debug = {"reachable": False, "reason": "no_target"}
            else:
                if args.controller_mode == "pretrained_pointnav":
                    low_action, navigation_debug = pointnav_controller.step(
                        rgb=rgb,
                        target_rel=np.asarray(target, dtype=np.float32),
                        agent_rel=rel_position,
                        agent_heading=world_heading,
                    )
                else:
                    low_action, navigation_debug = nav_executor.step(sim, np.asarray(target), origin, target_node_id=target_node_id)
                    if low_action == "unreachable":
                        navigation_event = agent.on_navigation_event(target_node_id, navigation_debug.get("reason", "unreachable"))
                        fallback = reachable_debug.get("selected")
                        if fallback is not None and fallback.get("node_id") != target_node_id:
                            node = agent.topo_map.get_node(fallback["node_id"])
                            if node is not None:
                                target_node_id = node.node_id
                                target = node.position.copy()
                                low_action, navigation_debug = nav_executor.step(sim, np.asarray(target), origin, target_node_id=target_node_id)
                        if low_action == "unreachable":
                            low_action = "turn_left"
                if low_action == "target_reached":
                    target_dist = planar_distance(target, rel_position)
                    navigation_debug["final_target_distance"] = target_dist
                    if navigation_debug.get("reason") == "target_reached" or target_dist <= nav_executor.reach_radius:
                        navigation_event = agent.on_navigation_event(target_node_id, "target_reached")
                    else:
                        low_action = "move_forward"
                        navigation_debug["low_action"] = low_action
                        navigation_debug["reason"] = "intermediate_waypoint_reached"

            navigation_debug["collision_like"] = bool(collision_debug.get("collision_like"))
            navigation_debug["stuck_steps"] = int(collision_debug.get("stuck_steps", 0))
            navigation_debug["movement"] = collision_debug.get("movement")
            navigation_debug["reachable_candidates"] = reachable_debug.get("reachable_candidates", [])
            navigation_debug["unreachable_candidates"] = reachable_debug.get("unreachable_candidates", [])

            candidate_scores = [{"node_id": node_id, "score": float(score)} for node_id, score in zip(candidate_ids, scores)]
            sticky_debug = dict(out.get("sticky_debug", {}) or {})
            sticky_debug["navigation_debug"] = navigation_debug
            sticky_debug["reachable_candidates"] = reachable_debug.get("reachable_candidates", [])
            sticky_debug["unreachable_candidates"] = reachable_debug.get("unreachable_candidates", [])
            if navigation_event:
                sticky_debug["navigation_event"] = navigation_event
            if reachable_debug.get("unreachable_candidates"):
                existing_skipped = list(sticky_debug.get("skipped_candidates", []))
                for row in reachable_debug.get("unreachable_candidates", []):
                    existing_skipped.append({"node_id": row.get("node_id"), "type": row.get("type", ""), "reason": row.get("reason", "unreachable"), "score": row.get("score")})
                sticky_debug["skipped_candidates"] = existing_skipped
            thresholds = {
                "object": config.perception.object_threshold,
                "room": config.perception.room_threshold,
                "landmark": config.perception.landmark_threshold,
            }
            selection_debug = build_selection_debug(target_node_id, candidate_scores, perception, thresholds, sticky_debug)
            trace["steps"].append({
                "step": step,
                "rgb_frame": frame_rel,
                "agent_action": out.get("action"),
                "low_action": low_action,
                "target_node_id": target_node_id,
                "target_position": None if target is None else np.asarray(target).round(4).tolist(),
                "position": rel_position.round(4).tolist(),
                "heading": world_heading,
                "world_position": world_position.round(4).tolist(),
                "world_heading": world_heading,
                "rgb_embedding": np.asarray(rgb_embed, dtype=np.float32).round(6).tolist(),
                "candidate_ids": candidate_ids,
                "candidate_scores": candidate_scores,
                "sticky_debug": sticky_debug,
                "navigation_event": navigation_event,
                "navigation_debug": navigation_debug,
                "collision_like": bool(collision_debug.get("collision_like")),
                "stuck_steps": int(collision_debug.get("stuck_steps", 0)),
                "selection_debug": selection_debug,
                "perception": perception,
                "memory": agent.memory_stats,
                "topo": snapshot_topo(agent),
            })
            if low_action == "stop":
                break
            if low_action not in ("target_reached", "unreachable"):
                sim.step(low_action)
            previous_low_action = low_action
            step += 1
    finally:
        sim.close()

    trace["final_memory"] = agent.memory_stats
    trace["final_summary"] = {
        "object_nodes": agent.memory_stats["objects"],
        "room_nodes": agent.memory_stats["rooms"],
        "landmark_nodes": agent.memory_stats["landmarks"],
        "max_object_score": max_score(trace, "goal_scores"),
        "max_room_score": max_score(trace, "room_scores"),
        "max_landmark_score": max_score(trace, "landmark_scores"),
        "thresholds": trace["thresholds"],
        "episode_length": len(trace["steps"]),
        "collision_like_count": sum(1 for st in trace["steps"] if st.get("collision_like")),
        "frontier_visited_count": agent.memory_stats.get("consumed_frontiers", 0),
        "semantic_node_count": agent.memory_stats.get("objects", 0) + agent.memory_stats.get("rooms", 0) + agent.memory_stats.get("landmarks", 0),
        "heavy_perception_calls": agent.memory_stats.get("heavy_perception_calls", 0),
        "object_merge_count": agent.memory_stats.get("object_merge_count", 0),
        "mean_object_confidence": agent.memory_stats.get("mean_object_confidence", 0.0),
        "memory_reuse_count": memory_reuse_count(trace["steps"]),
        "target_switch_count": count_target_switches(trace["steps"]),
        "path_length_relative": path_length(trace["steps"]),
        "sr_proxy": bool(agent.memory_stats["objects"] > 0 or agent.memory_stats["landmarks"] > 0),
        "spl_proxy": 0.0,
        "failure_reason": "" if (agent.memory_stats["objects"] > 0 and (not trace["current_goal"].get("landmarks") or agent.memory_stats["landmarks"] > 0)) else "semantic_node_missing_check_top_scores_and_thresholds",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_tolist(trace), indent=2))
    print(json.dumps({
        "ok": True,
        "output": str(output),
        "steps": len(trace["steps"]),
        "requested_goal_modality": trace["requested_goal_modality"],
        "selected_goal_modality": trace["selected_goal_modality"],
        "current_goal": trace["current_goal"],
        "final_memory": trace["final_memory"],
        "final_summary": trace["final_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
