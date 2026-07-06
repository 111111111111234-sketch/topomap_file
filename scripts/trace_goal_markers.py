"""Resolve and draw GOAT goal GT markers for trace visualizations."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GOAL_MARKER_COLORS = ("#e11d48", "#db2777", "#c026d3", "#7c3aed", "#ea580c", "#0891b2")


def resolve_repo_path(path_like: str | Path | None, root: Path = ROOT) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (root / path).exists():
        return root / path
    parts = path.parts
    if "data" in parts:
        candidate = root.joinpath(*parts[parts.index("data") :])
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _scene_basename(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id", ""))
    return scene_id.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")


def _scene_goal_key_prefixes(episode: dict[str, Any]) -> list[str]:
    scene_file = str(episode.get("scene_id", "")).split("/")[-1]
    candidates = [scene_file]
    scene_base = _scene_basename(episode)
    if scene_base and scene_base not in candidates:
        candidates.append(scene_base)
    return candidates


def _goal_entries_for_key(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    key_suffix: str | None,
) -> list[dict[str, Any]]:
    if not key_suffix:
        return []
    for prefix in _scene_goal_key_prefixes(episode):
        entries = dataset_goals.get(f"{prefix}_{key_suffix}")
        if isinstance(entries, list):
            return [g for g in entries if isinstance(g, dict)]
    return []


def _resolve_goal_entries(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    task_index: int,
) -> list[dict[str, Any]]:
    tasks = episode.get("tasks") or []
    if task_index >= len(tasks) or len(tasks[task_index]) < 2:
        return []
    goal_category, goal_type, *rest = tasks[task_index]
    goal_inst_id = rest[0] if rest else None

    goal_entries = _goal_entries_for_key(dataset_goals, episode, str(goal_category))
    if not goal_entries:
        return []

    for child_cat in goal_entries[0].get("children_object_categories") or []:
        goal_entries.extend(_goal_entries_for_key(dataset_goals, episode, str(child_cat)))

    if goal_type == "object":
        return [g for g in goal_entries if isinstance(g, dict)]
    if goal_inst_id is None:
        return []

    instance_key_entries = _goal_entries_for_key(dataset_goals, episode, str(goal_inst_id))
    if instance_key_entries:
        return instance_key_entries
    return [
        g
        for g in goal_entries
        if isinstance(g, dict) and str(g.get("object_id")) == str(goal_inst_id)
    ]


def _dedupe_positions(positions: list[list[float]]) -> list[list[float]]:
    seen: set[tuple] = set()
    out: list[list[float]] = []
    for pos in positions:
        key = tuple(np.round(np.asarray(pos, dtype=np.float32), 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append([float(x) for x in pos])
    return out


def world_to_relative(world_pos: Any, origin_world: Any) -> np.ndarray:
    rel = np.asarray(world_pos, dtype=np.float32) - np.asarray(origin_world, dtype=np.float32)
    rel[1] = 0.0
    return rel


def relative_to_world(relative_pos: Any, origin_world: Any) -> np.ndarray:
    return np.asarray(relative_pos, dtype=np.float32) + np.asarray(origin_world, dtype=np.float32)


def _positions_from_entries(
    goal_entries: list[dict[str, Any]],
    origin_world: list[float] | np.ndarray | None,
) -> tuple[list[list[float]], list[list[float]]]:
    world_positions: list[list[float]] = []
    for entry in goal_entries:
        pos_raw = entry.get("position")
        if pos_raw is None:
            continue
        world_positions.append(np.asarray(pos_raw, dtype=np.float32).round(4).tolist())
    world_positions = _dedupe_positions(world_positions)
    if origin_world is None:
        return world_positions, []
    relative_positions = [
        world_to_relative(pos, origin_world).round(4).tolist() for pos in world_positions
    ]
    return world_positions, relative_positions


def _viewpoints_from_entries(
    goal_entries: list[dict[str, Any]],
    origin_world: list[float] | np.ndarray | None,
) -> tuple[list[list[float]], list[list[float]]]:
    world_positions: list[list[float]] = []
    for entry in goal_entries:
        for view_point in entry.get("view_points") or []:
            agent_state = view_point.get("agent_state") if isinstance(view_point, dict) else None
            pos_raw = None
            if isinstance(agent_state, dict):
                pos_raw = agent_state.get("position")
            if pos_raw is None and isinstance(view_point, dict):
                pos_raw = view_point.get("position")
            if pos_raw is None:
                continue
            world_positions.append(np.asarray(pos_raw, dtype=np.float32).round(4).tolist())
    world_positions = _dedupe_positions(world_positions)
    if origin_world is None:
        return world_positions, []
    relative_positions = [
        world_to_relative(pos, origin_world).round(4).tolist() for pos in world_positions
    ]
    return world_positions, relative_positions


def _instances_from_task_summaries(trace: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for summary in trace.get("task_summaries") or []:
        world = summary.get("goal_instance_positions_world")
        relative = summary.get("goal_instance_positions")
        if not world and not relative:
            continue
        instances.append(
            {
                "goal_index": int(summary.get("goal_index", len(instances))),
                "target_object": str(
                    summary.get("target_object")
                    or summary.get("target")
                    or summary.get("goal")
                    or ""
                ),
                "positions_world": world or [],
                "positions_world_same_floor": summary.get(
                    "goal_instance_positions_world_same_floor",
                    [],
                ),
                "positions_world_off_floor": summary.get(
                    "goal_instance_positions_world_off_floor",
                    [],
                ),
                "positions_relative": relative or [],
                "view_points_world": summary.get("view_points_world", []),
                "view_points": summary.get("view_points", []),
            }
        )
    return instances


def _instances_from_episode(trace: dict[str, Any]) -> list[dict[str, Any]]:
    episode_file = resolve_repo_path(trace.get("episode_file"))
    if episode_file is None:
        return []

    try:
        with gzip.open(episode_file, "rt") as handle:
            dataset = json.load(handle)
    except Exception:
        return []

    episode_id = trace.get("episode_id")
    episode = None
    for item in dataset.get("episodes", []):
        if str(item.get("episode_id")) == str(episode_id):
            episode = item
            break
    if episode is None:
        return []

    origin_world = trace.get("origin_world")
    goals = trace.get("goals") or []
    instances: list[dict[str, Any]] = []
    for goal_index, target_object in enumerate(goals):
        entries = _resolve_goal_entries(dataset.get("goals", {}), episode, goal_index)
        world_positions, relative_positions = _positions_from_entries(entries, origin_world)
        view_points_world, view_points = _viewpoints_from_entries(entries, origin_world)
        if not world_positions:
            continue
        instances.append(
            {
                "goal_index": goal_index,
                "target_object": str(target_object),
                "positions_world": world_positions,
                "positions_relative": relative_positions,
                "view_points_world": view_points_world,
                "view_points": view_points,
            }
        )
    return instances


def _merge_missing_viewpoints(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not primary or not fallback:
        return primary

    merged: list[dict[str, Any]] = []
    fallback_by_goal = {
        (item.get("goal_index"), str(item.get("target_object", ""))): item for item in fallback
    }
    for item in primary:
        key = (item.get("goal_index"), str(item.get("target_object", "")))
        fallback_item = fallback_by_goal.get(key)
        if fallback_item is None:
            merged.append(item)
            continue
        if item.get("view_points_world") or item.get("view_points"):
            merged.append(item)
            continue
        merged.append(
            {
                **fallback_item,
                **item,
                "view_points_world": fallback_item.get("view_points_world", []),
                "view_points": fallback_item.get("view_points", []),
            }
        )
    return merged


def load_goal_instances(trace: dict[str, Any]) -> list[dict[str, Any]]:
    cached = trace.get("_goal_instances_cache")
    if isinstance(cached, list):
        return cached

    instances = trace.get("goal_instances")
    if isinstance(instances, list) and instances:
        if not any(item.get("view_points_world") or item.get("view_points") for item in instances):
            instances = _merge_missing_viewpoints(instances, _instances_from_episode(trace))
        trace["_goal_instances_cache"] = instances
        return instances

    instances = _instances_from_task_summaries(trace)
    if instances:
        if not any(item.get("view_points_world") or item.get("view_points") for item in instances):
            instances = _merge_missing_viewpoints(instances, _instances_from_episode(trace))
        trace["_goal_instances_cache"] = instances
        return instances

    instances = _instances_from_episode(trace)
    trace["_goal_instances_cache"] = instances
    return instances


def _same_floor_world_positions(
    trace: dict[str, Any] | None,
    instance: dict[str, Any],
) -> list[np.ndarray]:
    explicit = instance.get("positions_world_same_floor")
    if explicit:
        return [np.asarray(pos, dtype=np.float32) for pos in explicit]

    world = [
        np.asarray(pos, dtype=np.float32)
        for pos in instance.get("positions_world", []) or []
    ]
    if not world or trace is None or trace.get("origin_world") is None:
        return world

    floor_y = float(np.asarray(trace["origin_world"], dtype=np.float32)[1])
    tolerance = float(trace.get("same_floor_y_tolerance", 2.0) or 2.0)
    return [pos for pos in world if abs(float(pos[1]) - floor_y) <= tolerance]


def goal_color_map(trace: dict[str, Any]) -> dict[str, str]:
    goals = trace.get("goals") or []
    if not goals:
        seen: list[str] = []
        for step in trace.get("steps", []):
            name = step.get("goal")
            if name and name not in seen:
                seen.append(str(name))
        goals = seen
    return {str(goal): GOAL_MARKER_COLORS[i % len(GOAL_MARKER_COLORS)] for i, goal in enumerate(goals)}


def _goal_viewpoint_positions(
    instance: dict[str, Any],
    *,
    use_world: bool,
    trace: dict[str, Any] | None = None,
    origin_world: Any = None,
    pos_fn: Callable[[Any], np.ndarray] | None = None,
    same_floor_only: bool = True,
) -> list[np.ndarray]:
    world_positions = [
        np.asarray(pos, dtype=np.float32)
        for pos in instance.get("view_points_world", []) or []
    ]
    if use_world:
        if world_positions:
            out = world_positions
            if same_floor_only and trace is not None and trace.get("origin_world") is not None:
                floor_y = float(np.asarray(trace["origin_world"], dtype=np.float32)[1])
                tolerance = float(trace.get("same_floor_y_tolerance", 2.0) or 2.0)
                out = [p for p in out if abs(float(p[1]) - floor_y) <= tolerance]
            return out
        relative = instance.get("view_points") or []
        if pos_fn is not None:
            return [pos_fn(p) for p in relative]
        if origin_world is not None:
            return [relative_to_world(p, origin_world) for p in relative]
        return [np.asarray(p, dtype=np.float32) for p in relative]

    if world_positions and origin_world is not None:
        out = world_positions
        if same_floor_only and trace is not None and trace.get("origin_world") is not None:
            floor_y = float(np.asarray(trace["origin_world"], dtype=np.float32)[1])
            tolerance = float(trace.get("same_floor_y_tolerance", 2.0) or 2.0)
            out = [p for p in out if abs(float(p[1]) - floor_y) <= tolerance]
        return [world_to_relative(p, origin_world) for p in out]
    relative = instance.get("view_points") or []
    return [np.asarray(p, dtype=np.float32) for p in relative]


def plot_positions(
    instance: dict[str, Any],
    *,
    use_world: bool,
    trace: dict[str, Any] | None = None,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
    pos_fn: Callable[[Any], np.ndarray] | None = None,
) -> list[np.ndarray]:
    world_positions = _same_floor_world_positions(trace, instance)
    if use_world:
        if world_positions:
            return world_positions
        relative = instance.get("positions_relative") or []
        if pos_fn is not None:
            return [pos_fn(p) for p in relative]
        if position_frame == "episode_start_relative" and origin_world is not None:
            return [relative_to_world(p, origin_world) for p in relative]
        return [np.asarray(p, dtype=np.float32) for p in relative]

    if world_positions and origin_world is not None:
        return [world_to_relative(p, origin_world) for p in world_positions]
    relative = instance.get("positions_relative") or []
    if relative:
        return [np.asarray(p, dtype=np.float32) for p in relative]
    if origin_world is None:
        return []
    return [world_to_relative(p, origin_world) for p in world_positions]


def extend_xz_limits(
    xs: list[float],
    zs: list[float],
    trace: dict[str, Any],
    *,
    use_world: bool,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
) -> None:
    for instance in load_goal_instances(trace):
        for pos in plot_positions(
            instance,
            use_world=use_world,
            trace=trace,
            origin_world=origin_world,
            position_frame=position_frame,
        ):
            xs.append(float(pos[0]))
            zs.append(float(pos[2]))
        for pos in _goal_viewpoint_positions(
            instance,
            use_world=use_world,
            trace=trace,
            origin_world=origin_world,
        ):
            xs.append(float(pos[0]))
            zs.append(float(pos[2]))


def draw_goal_position_markers(
    ax,
    trace: dict[str, Any],
    step: dict[str, Any],
    *,
    use_world: bool = False,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
    pos_fn: Callable[[Any], np.ndarray] | None = None,
    highlight_current: bool = True,
    show_labels: bool = True,
    show_instances: bool = True,
    show_viewpoints: bool = True,
) -> None:
    instances = load_goal_instances(trace)
    if not instances:
        return

    colors = goal_color_map(trace)
    current_index = step.get("goal_index")
    current_name = str(step.get("goal") or "")

    def is_current(instance: dict[str, Any]) -> bool:
        return (
            current_index is not None and instance.get("goal_index") == current_index
        ) or (bool(current_name) and instance.get("target_object") == current_name)

    labeled_current_instance = False
    labeled_other_instance = False
    labeled_current_viewpoint = False
    labeled_other_viewpoint = False

    for instance in instances:
        current = is_current(instance)
        if highlight_current and not current:
            continue

        color = colors.get(str(instance.get("target_object", "")), GOAL_MARKER_COLORS[0])
        alpha = 1.0 if current or not highlight_current else 0.45
        if show_instances:
            positions = plot_positions(
                instance,
                use_world=use_world,
                trace=trace,
                origin_world=origin_world,
                position_frame=position_frame,
                pos_fn=pos_fn,
            )
            if positions:
                size = 190 if current else 110
                label = None
                if current and not labeled_current_instance:
                    label = "goal instance GT ({})".format(instance.get("target_object", "goal"))
                    labeled_current_instance = True
                elif highlight_current and not current and not labeled_other_instance:
                    label = "other goal instances"
                    labeled_other_instance = True

                xs = [float(p[0]) for p in positions]
                zs = [float(p[2]) for p in positions]
                ax.scatter(
                    xs,
                    zs,
                    marker="H",
                    s=size,
                    color=color,
                    edgecolors="white",
                    linewidths=1.0,
                    alpha=alpha,
                    label=label,
                    zorder=6,
                )
                if show_labels and current:
                    for pos in positions:
                        ax.text(
                            float(pos[0]),
                            float(pos[2]) + 0.14,
                            str(instance.get("target_object", "goal")),
                            fontsize=7,
                            ha="center",
                            color=color,
                            zorder=7,
                        )

        if show_viewpoints:
            view_positions = _goal_viewpoint_positions(
                instance,
                use_world=use_world,
                trace=trace,
                origin_world=origin_world,
                pos_fn=pos_fn,
            )
            if view_positions:
                label = None
                if current and not labeled_current_viewpoint:
                    label = "official goal viewpoints"
                    labeled_current_viewpoint = True
                elif highlight_current and not current and not labeled_other_viewpoint:
                    label = "other goal viewpoints"
                    labeled_other_viewpoint = True

                ax.scatter(
                    [float(p[0]) for p in view_positions],
                    [float(p[2]) for p in view_positions],
                    marker="o",
                    s=42 if current else 28,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1 if current else 0.8,
                    alpha=0.95 if current else 0.45,
                    label=label,
                    zorder=4,
                )

    if highlight_current:
        for instance in instances:
            if is_current(instance):
                continue
            color = colors.get(str(instance.get("target_object", "")), GOAL_MARKER_COLORS[0])
            if show_instances:
                positions = plot_positions(
                    instance,
                    use_world=use_world,
                    trace=trace,
                    origin_world=origin_world,
                    position_frame=position_frame,
                    pos_fn=pos_fn,
                )
                if positions:
                    ax.scatter(
                        [float(p[0]) for p in positions],
                        [float(p[2]) for p in positions],
                        marker="H",
                        s=110,
                        color=color,
                        edgecolors="white",
                        linewidths=0.8,
                        alpha=0.45,
                        label=None if labeled_other_instance else "other goal instances",
                        zorder=5,
                    )
                    labeled_other_instance = True
            if show_viewpoints:
                view_positions = _goal_viewpoint_positions(
                    instance,
                    use_world=use_world,
                    trace=trace,
                    origin_world=origin_world,
                    pos_fn=pos_fn,
                )
                if view_positions:
                    ax.scatter(
                        [float(p[0]) for p in view_positions],
                        [float(p[2]) for p in view_positions],
                        marker="o",
                        s=28,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.8,
                        alpha=0.45,
                        label=None if labeled_other_viewpoint else "other goal viewpoints",
                        zorder=4,
                    )
                    labeled_other_viewpoint = True
