from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from trace_goal_markers import draw_goal_position_markers, extend_xz_limits, load_goal_instances

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"waypoint_visited": "#2563eb", "waypoint_frontier": "#f97316", "waypoint_candidate": "#06b6d4", "object": "#16a34a", "room": "#dc2626", "landmark": "#9333ea"}
MARKERS = {"waypoint_visited": "o", "waypoint_frontier": "^", "waypoint_candidate": "v", "object": "D", "room": "s", "landmark": "P"}
LABELS = {"waypoint_visited": "visited waypoint", "waypoint_frontier": "frontier", "waypoint_candidate": "candidate/ghost", "object": "object", "room": "room", "landmark": "landmark"}


def semantic_node_text(node):
    """Render semantic labels for structure nodes only."""
    node_type = str(node.get("type", ""))
    if node_type not in {"object", "room", "landmark"}:
        return None
    label = str(node.get("label", "") or "").strip()
    return label if label else str(node.get("id", "")).strip() or None


def perception_objects(step):
    perception = step.get("perception") or {}
    objects = perception.get("vlm_objects") or perception.get("objects") or []
    return objects if isinstance(objects, list) else []


def _bbox_pixels(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))))
    y1, y2 = sorted((max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return x1, y1, x2, y2


def _semantic_bbox_pixels(obj, width, height):
    bearing = str(obj.get("bearing", "") or "").lower()
    range_bin = str(obj.get("range_bin", obj.get("range", "")) or "").lower()
    if bearing in {"left", "left_front"}:
        cx = 0.28
    elif bearing in {"right", "right_front"}:
        cx = 0.72
    elif bearing in {"center", "front"}:
        cx = 0.50
    else:
        return None
    if range_bin in {"very_near", "close"}:
        bw, bh, cy = 0.36, 0.46, 0.58
    elif range_bin == "near":
        bw, bh, cy = 0.28, 0.34, 0.56
    elif range_bin == "mid":
        bw, bh, cy = 0.22, 0.26, 0.52
    else:
        bw, bh, cy = 0.18, 0.20, 0.50
    return _bbox_pixels(
        [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
        width,
        height,
    )


def draw_rgb_bboxes(ax, rgb, step, goal_label=None, max_boxes=8):
    objects = []
    for obj in perception_objects(step):
        attrs = obj.get("attributes") or {}
        bbox = _bbox_pixels(
            obj.get("bbox") or obj.get("display_bbox") or attrs.get("display_bbox"),
            rgb.shape[1],
            rgb.shape[0],
        )
        estimated = False
        if obj.get("bbox") is None:
            estimated = bbox is not None or bool(attrs.get("estimated_bbox"))
        if bbox is None:
            bbox = _semantic_bbox_pixels(obj, rgb.shape[1], rgb.shape[0])
            estimated = bbox is not None
        if bbox is None:
            continue
        try:
            conf = float(obj.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        objects.append((conf, bbox, obj, estimated))
    if not objects:
        return
    goal_key = str(goal_label or step.get("goal") or "").lower().strip()
    objects.sort(key=lambda item: item[0], reverse=True)
    for conf, (x1, y1, x2, y2), obj, estimated in objects[:max_boxes]:
        label = str(obj.get("label", "object") or "object").strip()
        is_goal = bool(goal_key and goal_key in label.lower())
        color = "#22c55e" if is_goal else ("#06b6d4" if conf >= 0.75 else "#94a3b8")
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.0 if is_goal else 1.5,
                linestyle="--" if estimated else "-",
                alpha=0.95,
                zorder=5,
            )
        )
        caption = f"{label} {conf:.2f}" + (" ~" if estimated else "")
        text_y = max(0.0, y1 - 3.0)
        va = "bottom" if text_y > 1.0 else "top"
        ax.text(
            x1,
            text_y,
            caption,
            color="white",
            fontsize=7.5,
            va=va,
            ha="left",
            bbox={"facecolor": color, "edgecolor": color, "boxstyle": "round,pad=0.15", "alpha": 0.82},
            zorder=6,
        )


def resolve_repo_path(path_like):
    """Resolve trace paths that may come from another workspace mount."""
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (ROOT / path).exists():
        return ROOT / path
    parts = path.parts
    if "data" in parts:
        candidate = ROOT.joinpath(*parts[parts.index("data"):])
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def load_origin_world(trace):
    if trace.get("origin_world") is not None:
        return np.asarray(trace["origin_world"], dtype=np.float32), "trace.origin_world"
    episode_file = resolve_repo_path(trace.get("episode_file"))
    episode_id = trace.get("episode_id")
    if episode_file is None or episode_id is None:
        return None, "missing origin_world and episode_file"
    try:
        with gzip.open(episode_file, "rt") as f:
            data = json.load(f)
        for episode in data.get("episodes", []):
            if str(episode.get("episode_id")) == str(episode_id):
                return np.asarray(episode["start_position"], dtype=np.float32), f"{episode_file}:episode_{episode_id}"
    except Exception as exc:
        return None, f"failed to read origin from episode_file: {exc}"
    return None, f"episode_id={episode_id} not found in {episode_file}"


def infer_position_frame(trace, origin_world=None):
    frame = trace.get("coordinate_frame")
    if frame:
        return frame, "trace.coordinate_frame"
    steps = trace.get("steps") or []
    if not steps:
        return "episode_start_relative", "default: no steps"
    first = np.asarray(steps[0].get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    if origin_world is not None:
        origin = np.asarray(origin_world, dtype=np.float32)
        if np.linalg.norm(first[[0, 2]] - origin[[0, 2]]) < 0.5:
            return "world", "inferred: first step near origin_world"
    if np.linalg.norm(first[[0, 2]]) < 0.5:
        return "episode_start_relative", "inferred: first step near [0, 0]"
    return "world", "inferred: legacy trace without coordinate_frame"


def infer_topo_position_frame(trace, origin_world=None, position_frame="episode_start_relative"):
    frame = trace.get("topo_coordinate_frame")
    if frame:
        return frame, "trace.topo_coordinate_frame"
    steps = trace.get("steps") or []
    if not steps:
        return position_frame, "default: no steps"
    first_step = steps[0]
    first_agent = np.asarray(first_step.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    first_node = None
    for node in (first_step.get("topo") or first_step.get("topo_snapshot") or {}).get("nodes", []):
        if node.get("type") == "waypoint_visited":
            first_node = np.asarray(node.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            break
    if first_node is None:
        return position_frame, "default: no visited waypoint"
    if np.linalg.norm(first_node[[0, 2]] - first_agent[[0, 2]]) < 0.5:
        return position_frame, "inferred: topo near agent position"
    if origin_world is not None:
        origin = np.asarray(origin_world, dtype=np.float32)
        if np.linalg.norm(first_node[[0, 2]] - origin[[0, 2]]) < 0.5:
            return "world", "inferred: topo near origin_world"
    return position_frame, "inferred: fallback to agent frame"


def infer_navmesh_path(trace, override=None):
    if override:
        path = resolve_repo_path(override)
        return path, "override" if path else f"override not found: {override}"
    scene_file = resolve_repo_path(trace.get("scene_file"))
    if scene_file is not None:
        candidates = []
        name = scene_file.name
        if name.endswith(".basis.glb"):
            candidates.append(scene_file.with_name(name.replace(".basis.glb", ".basis.navmesh")))
        if scene_file.suffix == ".glb":
            candidates.append(scene_file.with_suffix(".navmesh"))
        candidates.append(scene_file.with_name(scene_file.stem + ".navmesh"))
        for candidate in candidates:
            if candidate.exists():
                return candidate, "scene_file"
    scene = trace.get("scene") or (Path(str(trace.get("scene_file", ""))).name.split(".")[0] if trace.get("scene_file") else "")
    scene = scene.replace(".basis", "")
    if scene:
        matches = sorted((ROOT / "data/scene_datasets").glob(f"**/{scene}*.navmesh"))
        if matches:
            return matches[0], "scene search"
    return None, "navmesh not found"


def infer_slice_height(trace, origin_world, bounds_min, pathfinder=None):
    steps = trace.get("steps") or []
    ys = []
    for step in steps:
        world = step.get("world_position")
        if world is None:
            continue
        world_arr = np.asarray(world, dtype=np.float32)
        if pathfinder is not None:
            world_arr = np.asarray(pathfinder.snap_point(world_arr), dtype=np.float32)
        ys.append(float(world_arr[1]))
    if ys:
        return float(np.median(ys)) + 0.02
    if origin_world is not None:
        return float(np.asarray(origin_world, dtype=np.float32)[1]) + 0.02
    return float(bounds_min[1] + 0.1)


def agent_world_position(step, origin_world=None, position_frame="episode_start_relative"):
    if step.get("world_position") is not None:
        return np.asarray(step["world_position"], dtype=np.float32)
    return position_to_world(step.get("position", [0.0, 0.0, 0.0]), origin_world, position_frame)


def world_xz_limits(trace, idx, origin_world, position_frame, topo_position_frame=None, pad=2.5):
    topo_position_frame = topo_position_frame or position_frame
    xs, zs = [], []
    for step in trace["steps"][: idx + 1]:
        pos = agent_world_position(step, origin_world, position_frame)
        xs.append(float(pos[0]))
        zs.append(float(pos[2]))
    for step in trace["steps"][: idx + 1]:
        for node in step.get("topo", {}).get("nodes", []):
            pos = position_to_world(node["position"], origin_world, topo_position_frame)
            xs.append(float(pos[0]))
            zs.append(float(pos[2]))
    extend_xz_limits(
        xs,
        zs,
        trace,
        use_world=True,
        origin_world=origin_world,
        position_frame=position_frame,
    )
    return (min(xs) - pad, max(xs) + pad), (min(zs) - pad, max(zs) + pad)


def get_cached_topdown(pathfinder, resolution, slice_height, cache):
    key = round(float(slice_height), 3)
    if key not in cache:
        cache[key] = np.asarray(pathfinder.get_topdown_view(resolution, float(slice_height)), dtype=bool)
    return cache[key]


def load_gt_topdown(trace, resolution, navmesh_override=None, disabled=False):
    if disabled:
        return {"available": False, "reason": "disabled by --no-gt-topdown"}
    navmesh_path, navmesh_source = infer_navmesh_path(trace, navmesh_override)
    origin_world, origin_source = load_origin_world(trace)
    position_frame, position_frame_source = infer_position_frame(trace, origin_world)
    topo_position_frame, topo_position_frame_source = infer_topo_position_frame(trace, origin_world, position_frame)
    if navmesh_path is None:
        return {"available": False, "reason": navmesh_source, "origin_world": origin_world, "origin_source": origin_source, "position_frame": position_frame, "position_frame_source": position_frame_source, "topo_position_frame": topo_position_frame, "topo_position_frame_source": topo_position_frame_source}
    if origin_world is None:
        return {"available": False, "reason": origin_source, "navmesh_path": str(navmesh_path), "position_frame": position_frame, "position_frame_source": position_frame_source, "topo_position_frame": topo_position_frame, "topo_position_frame_source": topo_position_frame_source}
    try:
        import habitat_sim
        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh_path)):
            return {"available": False, "reason": f"failed to load navmesh: {navmesh_path}", "origin_world": origin_world}
        bounds_min, bounds_max = pathfinder.get_bounds()
        slice_height = infer_slice_height(trace, origin_world, bounds_min, pathfinder=pathfinder)
        topdown = np.asarray(pathfinder.get_topdown_view(resolution, slice_height), dtype=bool)
        return {
            "available": True,
            "topdown": topdown,
            "bounds_min": np.asarray(bounds_min, dtype=np.float32),
            "bounds_max": np.asarray(bounds_max, dtype=np.float32),
            "slice_height": slice_height,
            "pathfinder": pathfinder,
            "topdown_cache": {round(float(slice_height), 3): topdown},
            "origin_world": origin_world,
            "origin_source": origin_source,
            "position_frame": position_frame,
            "position_frame_source": position_frame_source,
            "topo_position_frame": topo_position_frame,
            "topo_position_frame_source": topo_position_frame_source,
            "navmesh_path": str(navmesh_path),
            "navmesh_source": navmesh_source,
            "resolution": float(resolution),
        }
    except Exception as exc:
        return {"available": False, "reason": f"topdown unavailable: {exc}", "origin_world": origin_world, "navmesh_path": str(navmesh_path)}


def relative_to_world(position, origin_world):
    return np.asarray(position, dtype=np.float32) + np.asarray(origin_world, dtype=np.float32)


def position_to_world(position, origin_world, position_frame):
    arr = np.asarray(position, dtype=np.float32)
    if position_frame == "episode_start_relative":
        return relative_to_world(arr, origin_world)
    return arr


def position_between_frames(position, origin_world, from_frame, to_frame):
    arr = np.asarray(position, dtype=np.float32)
    if from_frame == to_frame or origin_world is None:
        return arr
    world = position_to_world(arr, origin_world, from_frame)
    if to_frame == "episode_start_relative":
        return world - np.asarray(origin_world, dtype=np.float32)
    return world


def collect_limits(steps, trace=None):
    xs, zs = [], []
    origin_world = None
    position_frame = "episode_start_relative"
    topo_position_frame = "episode_start_relative"
    if trace is not None:
        origin_world, _ = load_origin_world(trace)
        position_frame, _ = infer_position_frame(trace, origin_world)
        topo_position_frame, _ = infer_topo_position_frame(trace, origin_world, position_frame)
    for st in steps:
        p = st["position"]
        xs.append(p[0]); zs.append(p[2])
        if st.get("target_position"):
            t = st["target_position"]
            xs.append(t[0]); zs.append(t[2])
        for n in st["topo"]["nodes"]:
            pos = position_between_frames(
                n["position"],
                origin_world,
                topo_position_frame,
                position_frame,
            )
            xs.append(float(pos[0])); zs.append(float(pos[2]))
    if trace is not None:
        for p in trace.get("trajectory", {}).get("path_points_relative", []):
            xs.append(p[0]); zs.append(p[2])
        extend_xz_limits(xs, zs, trace, use_world=False)
    return (min(xs)-1.0, max(xs)+1.0), (min(zs)-1.0, max(zs)+1.0)


def fmt_scores(title, scores):
    if not scores:
        return [title + ": n/a"]
    lines = [title]
    for item in scores[:3]:
        if isinstance(item, dict):
            label = item.get("label", item.get("name", "n/a"))
            score = item.get("score", item.get("confidence", 0.0))
        else:
            try:
                label, score = item
            except (TypeError, ValueError):
                label, score = str(item), 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        lines.append("  {}: {:.3f}".format(label, score))
    return lines


def fmt_candidates(candidates):
    if not candidates:
        return ["Top candidates: n/a"]
    lines = ["Top candidates"]
    for item in candidates[:3]:
        lines.append("  {}: {:.3f}".format(item.get("node_id", "n/a"), float(item.get("score", 0.0))))
    return lines


def fmt_semantic_decisions(decisions):
    if not decisions:
        return ["Semantic decisions: n/a"]
    lines = ["Semantic decisions"]
    for key, short in [("objects", "obj"), ("rooms", "room"), ("landmarks", "lm")]:
        rows = decisions.get(key, [])
        if not rows:
            lines.append("  {}: n/a".format(short))
            continue
        item = rows[0]
        lines.append(
            "  {} {} {:.3f}/{:.3f} {}".format(
                short,
                item.get("label", "n/a"),
                float(item.get("score", 0.0)),
                float(item.get("threshold", 0.0)),
                item.get("decision", ""),
            )
        )
    return lines


def fig_to_rgb(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()


def resize_to(img, shape):
    from PIL import Image
    h, w = shape[:2]
    return np.asarray(Image.fromarray(img).resize((w, h)))


def topo_nodes_by_id(step):
    return {n["id"]: n for n in step["topo"]["nodes"]}


def draw_trajectory_route_axis(ax, trace, idx=None, use_world=False, origin_world=None, position_frame="episode_start_relative"):
    trajectory = trace.get("trajectory") or {}
    route = trajectory.get("path_points_relative") or []
    if len(route) < 2:
        return

    def pos(value):
        arr = np.asarray(value, dtype=np.float32)
        return position_to_world(arr, origin_world, position_frame) if use_world and origin_world is not None else arr

    points = np.asarray([pos(p) for p in route], dtype=np.float32)
    ax.plot(points[:, 0], points[:, 2], color="#7c3aed", linewidth=1.6, alpha=0.78, label="planned loop route", zorder=2)
    ax.scatter([points[0, 0]], [points[0, 2]], marker="h", s=110, color="#22c55e", edgecolors="black", linewidths=0.7, label="loop start/end", zorder=4)
    anchors = trajectory.get("waypoints_relative") or []
    if anchors:
        anchor_points = np.asarray([pos(p) for p in anchors], dtype=np.float32)
        ax.scatter(anchor_points[:, 0], anchor_points[:, 2], marker="D", s=55, color="#a855f7", edgecolors="black", linewidths=0.55, label="loop anchors", zorder=4)
    if idx is not None and 0 <= idx < len(trace.get("steps", [])):
        progress = trace["steps"][idx].get("trajectory_progress") or {}
        route_index = int(progress.get("route_index", -1))
        if 0 <= route_index < len(route):
            target = pos(route[route_index])
            ax.scatter([target[0]], [target[2]], marker="X", s=95, color="#c084fc", edgecolors="black", linewidths=0.7, label="loop target", zorder=5)


_ACTIVE_STATE_STYLE = {
    "folded": ("#9ca3af", "ActiveTopo: folded"),
    "suppressed": ("#4b5563", "ActiveTopo: suppressed"),
    "verify": ("#f59e0b", "ActiveTopo: verify"),
}


def overlay_active_topo(ax, st, has_node, pos_of_id):
    """Ring-highlight nodes by their ActiveTopo state (folded/suppressed/verify).

    Reads the compact per-node buckets emitted in ``step.debug.active_topo_nodes``.
    Behaves as a no-op for older traces that lack the field."""
    buckets = (st.get("debug") or {}).get("active_topo_nodes") or {}
    for state, (color, label) in _ACTIVE_STATE_STYLE.items():
        pts = [pos_of_id(i) for i in (buckets.get(state) or []) if has_node(i)]
        if not pts:
            continue
        ax.scatter(
            [p[0] for p in pts],
            [p[2] for p in pts],
            s=230,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=1.7,
            alpha=0.95,
            label=label,
            zorder=6,
        )


def fmt_active_topo(debug):
    av = (debug or {}).get("active_topo")
    if not av:
        return []
    dist = av.get("activity_distribution", {}) or {}
    ve = (debug or {}).get("verify_events", {}) or {}
    rec = (debug or {}).get("recovery", {}) or {}
    return [
        "ActiveTopo",
        "active/persist: {}/{}".format(av.get("active_node_count", "?"), av.get("persistent_node_count", "?")),
        "folded: {}  suppressed: {}".format(dist.get("folded", 0), dist.get("suppressed", 0)),
        "verify nodes: {}".format(dist.get("verify", 0)),
        "verify ok/fail: {}/{}".format(ve.get("success_count", 0), ve.get("fail_count", 0)),
        "needs_verify: {}".format(ve.get("needs_verify_now", False)),
        "recovery lvl: {}  exhausted: {}".format(rec.get("level", 0), rec.get("exhausted_regions", 0)),
        "",
    ]


def draw_topo_on_axis(ax, trace, idx, xlim, zlim, title, use_world=False, origin_world=None, position_frame="episode_start_relative", topo_position_frame=None, show_info_labels=True):
    st = trace["steps"][idx]
    nodes = topo_nodes_by_id(st)
    topo_position_frame = topo_position_frame or position_frame

    def pos(value):
        arr = np.asarray(value, dtype=np.float32)
        if use_world and origin_world is not None:
            return position_to_world(arr, origin_world, topo_position_frame)
        return position_between_frames(arr, origin_world, topo_position_frame, position_frame)

    def agent_pos(step):
        if use_world and origin_world is not None:
            return agent_world_position(step, origin_world, position_frame)
        return np.asarray(step["position"], dtype=np.float32)

    for e in st["topo"]["edges"]:
        a, b = nodes.get(e["source"]), nodes.get(e["target"])
        if not a or not b:
            continue
        pa, pb = pos(a["position"]), pos(b["position"])
        nav = e.get("type") == "navigable"
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color="#6b7280", linewidth=1.1 if nav else 0.7, linestyle="-" if nav else "--", alpha=0.65, zorder=1)

    selection = st.get("selection_debug", {})
    rank_labels = {}
    for rank, item in enumerate(selection.get("top_candidate_scores", [])[:5], start=1):
        node_id = item.get("node_id")
        if node_id:
            rank_labels[node_id] = "#{} {:.3f}".format(rank, float(item.get("score", 0.0)))

    def is_anchored_object(node):
        if node.get("type") != "object":
            return False
        attrs = node.get("attributes", {}) or {}
        role = str(attrs.get("semantic_role", "")).strip()
        if role == "object_anchor":
            return True
        if role not in {"object", "context_object"}:
            return False
        return bool(
            attrs.get("anchor_waypoint_id")
            or attrs.get("best_anchor_waypoint_id")
            or attrs.get("approach_point_id")
        )

    object_anchors = [n for n in nodes.values() if is_anchored_object(n)]
    other_objects = [n for n in nodes.values() if n["type"] == "object" and not is_anchored_object(n)]
    draw_groups = [
        ("waypoint_visited", [n for n in nodes.values() if n["type"] == "waypoint_visited"]),
        ("waypoint_frontier", [n for n in nodes.values() if n["type"] == "waypoint_frontier"]),
        ("waypoint_candidate", [n for n in nodes.values() if n["type"] == "waypoint_candidate"]),
        ("object", other_objects),
        ("room", [n for n in nodes.values() if n["type"] == "room"]),
        ("landmark", [n for n in nodes.values() if n["type"] == "landmark"]),
    ]
    for typ, group in draw_groups:
        if not group:
            continue
        positions = [pos(n["position"]) for n in group]
        ax.scatter(
            [p[0] for p in positions],
            [p[2] for p in positions],
            s=[55 + 140 * float(n.get("confidence", 0.5)) for n in group],
            marker=MARKERS[typ],
            color=COLORS[typ],
            edgecolors="black",
            linewidths=0.55,
            alpha=0.9,
            label=LABELS[typ],
            zorder=3,
        )
        if show_info_labels:
            for n, p in zip(group, positions):
                label = semantic_node_text(n)
                if label:
                    ax.text(p[0], p[2] + 0.08, label, fontsize=6.5, ha="center", zorder=5)
                if label and n["id"] in rank_labels:
                    ax.text(
                        p[0],
                        p[2] - 0.16,
                        rank_labels[n["id"]],
                        fontsize=6.5,
                        ha="center",
                        color="#111827",
                        bbox={"facecolor": "#fef3c7", "edgecolor": "#f59e0b", "boxstyle": "round,pad=0.16", "alpha": 0.78},
                        zorder=6,
                    )
    if object_anchors:
        anchor_positions = [pos(n["position"]) for n in object_anchors]
        ax.scatter(
            [p[0] for p in anchor_positions],
            [p[2] for p in anchor_positions],
            s=[120 + 180 * float(n.get("confidence", 0.5)) for n in object_anchors],
            marker="*",
            color="#16a34a",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.95,
            label="anchored object (VLM)",
            zorder=4,
        )
        if show_info_labels:
            for n, p in zip(object_anchors, anchor_positions):
                label = str(n.get("label", "") or "").strip() or "object"
                ax.text(p[0], p[2] + 0.12, label, fontsize=7, ha="center", color="#14532d", zorder=6)

    path = np.array([agent_pos(s) for s in trace["steps"][:idx+1]])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.4, alpha=0.85, label="agent path", zorder=2)
    draw_goal_position_markers(
        ax,
        trace,
        st,
        use_world=use_world,
        origin_world=origin_world,
        position_frame=position_frame,
        pos_fn=pos,
        show_labels=show_info_labels,
    )
    cur = agent_pos(st)
    ax.scatter([cur[0]], [cur[2]], marker="*", s=220, color="#facc15", edgecolors="black", linewidths=1.0, label="agent", zorder=5)
    if st.get("target_position"):
        tgt = pos(st["target_position"])
        ax.scatter([tgt[0]], [tgt[2]], marker="X", s=130, color="#ef4444", edgecolors="black", linewidths=0.8, label="planned target", zorder=4)
        ax.plot([cur[0], tgt[0]], [cur[2], tgt[2]], color="#ef4444", linewidth=0.95, linestyle=":", alpha=0.7, zorder=1)

    overlay_active_topo(ax, st, lambda i: i in nodes, lambda i: pos(nodes[i]["position"]))

    ax.set_title(title)
    ax.set_xlabel("world x" if use_world else "relative x")
    ax.set_ylabel("world z" if use_world else "relative z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=6.5)


def draw_topo_frame(trace, idx, xlim, zlim, out_path=None):
    st = trace["steps"][idx]
    nodes = {n["id"]: n for n in st["topo"]["nodes"]}
    origin_world, _ = load_origin_world(trace)
    position_frame, _ = infer_position_frame(trace, origin_world)
    topo_position_frame, _ = infer_topo_position_frame(trace, origin_world, position_frame)

    def node_pos(node):
        return position_between_frames(
            node["position"],
            origin_world,
            topo_position_frame,
            position_frame,
        )

    fig, (ax, info) = plt.subplots(1, 2, figsize=(13, 6), dpi=130, gridspec_kw={"width_ratios": [2.1, 1]})
    for e in st["topo"]["edges"]:
        a, b = nodes.get(e["source"]), nodes.get(e["target"])
        if not a or not b:
            continue
        nav = e.get("type") == "navigable"
        pa, pb = node_pos(a), node_pos(b)
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color="#6b7280", linewidth=1.2 if nav else 0.8, linestyle="-" if nav else "--", alpha=0.65, zorder=1)
    selection = st.get("selection_debug", {})
    rank_labels = {}
    for rank, item in enumerate(selection.get("top_candidate_scores", [])[:5], start=1):
        node_id = item.get("node_id")
        if node_id:
            rank_labels[node_id] = "#{} {:.3f}".format(rank, float(item.get("score", 0.0)))
    for typ in ["waypoint_visited", "waypoint_frontier", "object", "room", "landmark"]:
        group = [n for n in nodes.values() if n["type"] == typ]
        if not group:
            continue
        group_pos = [node_pos(n) for n in group]
        ax.scatter([p[0] for p in group_pos], [p[2] for p in group_pos], s=[70 + 180 * float(n.get("confidence", 0.5)) for n in group], marker=MARKERS[typ], color=COLORS[typ], edgecolors="black", linewidths=0.6, alpha=0.9, label=LABELS[typ], zorder=3)
        for n in group:
            p = node_pos(n)
            label = semantic_node_text(n)
            if label:
                ax.text(p[0], p[2] + 0.08, label, fontsize=7, ha="center")
            if label and n["id"] in rank_labels:
                ax.text(
                    p[0],
                    p[2] - 0.16,
                    rank_labels[n["id"]],
                    fontsize=7,
                    ha="center",
                    color="#111827",
                    bbox={"facecolor": "#fef3c7", "edgecolor": "#f59e0b", "boxstyle": "round,pad=0.18", "alpha": 0.78},
                    zorder=6,
                )
    path = np.array([s["position"] for s in trace["steps"][:idx+1]])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.6, alpha=0.8, label="agent path", zorder=2)
    draw_goal_position_markers(ax, trace, st, use_world=False, show_labels=True)
    cur = st["position"]
    ax.scatter([cur[0]], [cur[2]], marker="*", s=260, color="#facc15", edgecolors="black", linewidths=1.0, label="agent", zorder=5)
    if st.get("target_position"):
        tgt = st["target_position"]
        ax.scatter([tgt[0]], [tgt[2]], marker="X", s=150, color="#ef4444", edgecolors="black", linewidths=0.8, label="planned target", zorder=4)
        ax.plot([cur[0], tgt[0]], [cur[2], tgt[2]], color="#ef4444", linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    overlay_active_topo(ax, st, lambda i: i in nodes, lambda i: node_pos(nodes[i]))
    ax.set_title("DynamicTopoMap | step {}".format(st.get("step")))
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_xlim(*xlim); ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.22); ax.legend(loc="upper right", fontsize=7)
    mem, goal, perception = st["memory"], trace.get("current_goal", {}), st.get("perception", {})
    sticky = st.get("sticky_debug", {})
    selection = st.get("selection_debug", {})
    event = selection.get("navigation_event") or st.get("navigation_event") or {}
    blocked = selection.get("blocked_targets") or sticky.get("blocked_targets") or {}
    skipped = selection.get("skipped_candidates") or sticky.get("skipped_candidates") or []
    lines = ["Topo Memory", "total nodes: {}".format(mem["total_nodes"]), "visited: {}".format(mem["visited_waypoints"]), "frontiers: {}".format(mem["frontiers"]), "objects: {}".format(mem["objects"]), "rooms: {}".format(mem["rooms"]), "landmarks: {}".format(mem.get("landmarks", 0)), "", "Goal/Target", "goal: {}/object".format(goal.get("target_object", "n/a")), "target: {}".format(st.get("target_node_id") or "n/a"), "selected rank: {}".format(selection.get("selected_rank") or "n/a"), "sticky: {}".format(sticky.get("sticky_target_id") or "n/a"), "release: {}".format(sticky.get("sticky_release_reason") or ""), "event: {}".format(event.get("action") or event.get("reason") or ""), "blocked: {}".format(len(blocked)), "skipped: {}".format(len(skipped)), "", "Action", "low: {}".format(st.get("low_action")), "agent: {}".format(st.get("agent_action")), ""]
    lines.extend(fmt_active_topo(st.get("debug", {})))
    progress = st.get("trajectory_progress")
    if progress:
        lines.extend([
            "Trajectory",
            "route: {}/{}".format(progress.get("route_index", "n/a"), progress.get("route_points", "n/a")),
            "target dist: {:.2f}".format(float(progress.get("target_distance", 0.0))),
            "to start: {:.2f}".format(float(progress.get("distance_to_start", 0.0))),
            "completed: {}".format(progress.get("completed", False)),
            "return ok: {}".format(trace.get("final_summary", {}).get("trajectory_returned_to_start", "n/a")),
            "",
        ])
    lines.extend(fmt_candidates(selection.get("top_candidate_scores", [])))
    lines.append("")
    lines.extend(fmt_semantic_decisions(selection.get("semantic_decisions", {})))
    lines.append("")
    lines.extend(fmt_scores("Top object CLIP", perception.get("goal_scores", [])))
    lines.extend(fmt_scores("Top room CLIP", perception.get("room_scores", [])))
    lines.extend(fmt_scores("Top landmark CLIP", perception.get("landmark_scores", [])))
    info.axis("off"); info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig); return None
    img = fig_to_rgb(fig); plt.close(fig); return img


def draw_first_person_frame(trace, idx):
    st = trace["steps"][idx]
    frame = st.get("rgb_frame")
    if frame:
        frame_path = resolve_repo_path(frame)
        rgb = imageio.imread(frame_path) if frame_path is not None else np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    goal = trace.get("current_goal", {})
    goal_label = st.get("goal") or goal.get("target_object")
    fig, (ax, info) = plt.subplots(1, 2, figsize=(8, 4.5), dpi=130, gridspec_kw={"width_ratios": [1.25, 1]})
    ax.imshow(rgb)
    draw_rgb_bboxes(ax, rgb, st, goal_label=goal_label)
    ax.axis("off"); ax.set_title("First person RGB + VLM boxes | step {}".format(st.get("step")))
    p = st.get("perception", {})
    selection = st.get("selection_debug", {})
    lines = ["GOAT Input", "{}/object".format(goal_label or "n/a"), "", "VLM", "boxes: {}".format(len(perception_objects(st))), "fresh: {}".format(p.get("fresh_vlm", False)), "", "Action", "low: {}".format(st.get("low_action")), "target: {}".format(st.get("target_node_id") or "n/a"), "selected rank: {}".format(selection.get("selected_rank") or "n/a"), ""]
    progress = st.get("trajectory_progress")
    if progress:
        lines.extend([
            "Trajectory",
            "route: {}/{}".format(progress.get("route_index", "n/a"), progress.get("route_points", "n/a")),
            "target dist: {:.2f}".format(float(progress.get("target_distance", 0.0))),
            "to start: {:.2f}".format(float(progress.get("distance_to_start", 0.0))),
            "return ok: {}".format(trace.get("final_summary", {}).get("trajectory_returned_to_start", "n/a")),
            "",
        ])
    lines.extend(fmt_candidates(selection.get("top_candidate_scores", [])))
    lines.append("")
    lines.extend(fmt_scores("Top object CLIP", p.get("goal_scores", [])))
    lines.extend(fmt_scores("Top room CLIP", p.get("room_scores", [])))
    lines.extend(fmt_scores("Top landmark CLIP", p.get("landmark_scores", [])))
    info.axis("off"); info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout(); img = fig_to_rgb(fig); plt.close(fig); return img


def draw_gt_topdown_axis(ax, trace, idx, gt_ctx, show_planned_route=False):
    if not gt_ctx.get("available"):
        ax.axis("off")
        reason = gt_ctx.get("reason", "GT topdown unavailable")
        ax.text(0.5, 0.5, "GT topdown unavailable\n{}".format(reason), ha="center", va="center", fontsize=10)
        ax.set_title("GT topdown + topo overlay")
        return

    bounds_min = gt_ctx["bounds_min"]
    bounds_max = gt_ctx["bounds_max"]
    origin_world = gt_ctx["origin_world"]
    position_frame = gt_ctx.get("position_frame", "episode_start_relative")
    topo_position_frame = gt_ctx.get("topo_position_frame", position_frame)
    resolution = float(gt_ctx.get("resolution", 0.05))
    pathfinder = gt_ctx.get("pathfinder")
    cache = gt_ctx.setdefault("topdown_cache", {})
    st = trace["steps"][idx]
    world_pos = agent_world_position(st, origin_world, position_frame)
    if pathfinder is not None:
        slice_height = float(np.asarray(pathfinder.snap_point(world_pos), dtype=np.float32)[1]) + 0.02
        topdown = get_cached_topdown(pathfinder, resolution, slice_height, cache)
        title = "GT topdown + topo overlay | slice_y={:.2f}".format(slice_height)
    else:
        topdown = gt_ctx["topdown"]
        slice_height = float(gt_ctx.get("slice_height", bounds_min[1] + 0.1))
        title = "GT topdown + topo overlay | slice_y={:.2f}".format(slice_height)
    extent = [bounds_min[0], bounds_max[0], bounds_max[2], bounds_min[2]]
    ax.imshow(topdown, cmap="gray", origin="upper", extent=extent, alpha=0.58)
    xlim, zlim = world_xz_limits(trace, idx, origin_world, position_frame, topo_position_frame, pad=2.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    if show_planned_route:
        draw_trajectory_route_axis(
            ax,
            trace,
            idx,
            use_world=True,
            origin_world=origin_world,
            position_frame=position_frame,
        )
    draw_topo_on_axis(
        ax,
        trace,
        idx,
        xlim,
        zlim,
        title,
        use_world=True,
        origin_world=origin_world,
        position_frame=position_frame,
        topo_position_frame=topo_position_frame,
        show_info_labels=False,
    )


def goal_header(trace, idx, gt_ctx):
    st = trace["steps"][idx]
    goal = dict(trace.get("current_goal", {}) or {})
    if st.get("goal"):
        goal["target_object"] = st.get("goal")
    instruction = trace.get("instruction") or goal.get("instruction") or goal.get("description") or ""
    target = goal.get("target_object") or goal.get("goal") or "n/a"
    goal_type = goal.get("goal_type", trace.get("goal_type", "n/a"))
    goal_modality = trace.get("selected_goal_modality") or goal_type
    requested_modality = trace.get("requested_goal_modality", "auto")
    attrs = ", ".join(goal.get("attributes", [])[:4]) if isinstance(goal.get("attributes"), list) else ""
    rooms = ", ".join(goal.get("room_prior", [])[:4]) if isinstance(goal.get("room_prior"), list) else ""
    landmarks = ", ".join(goal.get("landmarks", [])[:4]) if isinstance(goal.get("landmarks"), list) else ""
    origin = gt_ctx.get("origin_world")
    origin_text = "n/a" if origin is None else np.asarray(origin).round(4).tolist()
    graph_state = "GoalGraph: current_goal present" if trace.get("current_goal") else "GoalGraph: n/a"
    trajectory = trace.get("trajectory") or {}
    trajectory_text = "Trajectory: {}".format(trajectory.get("kind", "agent_policy"))
    return (
        "Scene: {scene} | Episode: {episode} | Step: {step} | "
        "Input: {modality} (requested: {requested_modality}) | Instruction: {instruction} | Goal: {target} ({goal_type}) | attrs: {attrs} | rooms: {rooms} | landmarks: {landmarks}\n"
        "{graph_state} | {trajectory} | embedding: {embedding} | Trace frame: {frame} | Overlay frame: world via origin_world | origin_world: {origin}"
    ).format(
        scene=trace.get("scene") or Path(str(trace.get("scene_file", "n/a"))).stem,
        episode=trace.get("episode_id", "n/a"),
        step=st.get("step", idx),
        modality=goal_modality,
        requested_modality=requested_modality,
        instruction=instruction or "n/a",
        target=target,
        goal_type=goal_type,
        attrs=attrs or "n/a",
        rooms=rooms or "n/a",
        landmarks=landmarks or "n/a",
        graph_state=graph_state,
        trajectory=trajectory_text,
        embedding=trace.get("embedding_source", "n/a"),
        frame=gt_ctx.get("position_frame", trace.get("coordinate_frame", "n/a")),
        origin=origin_text,
    )


def draw_audit_frame(trace, idx, xlim, zlim, gt_ctx, show_planned_route=False):
    st = trace["steps"][idx]
    frame = st.get("rgb_frame")
    if frame:
        frame_path = resolve_repo_path(frame)
        rgb = imageio.imread(frame_path) if frame_path is not None else np.zeros((360, 480, 3), dtype=np.uint8)
    else:
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)

    fig = plt.figure(figsize=(19.0, 10.5), dpi=125)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.14, 1.0, 1.35], width_ratios=[0.9, 1.35])
    header_ax = fig.add_subplot(gs[0, :])
    fp_ax = fig.add_subplot(gs[1:, 0])
    gt_ax = fig.add_subplot(gs[1, 1])
    topo_ax = fig.add_subplot(gs[2, 1])

    header_ax.axis("off")
    header_ax.text(0.01, 0.5, goal_header(trace, idx, gt_ctx), va="center", ha="left", fontsize=10, family="monospace")

    fp_ax.imshow(rgb)
    draw_rgb_bboxes(fp_ax, rgb, st, goal_label=st.get("goal") or (trace.get("current_goal") or {}).get("target_object"), max_boxes=10)
    fp_ax.axis("off")
    fp_ax.set_title("First person RGB + VLM boxes | step {}".format(st.get("step")))

    draw_gt_topdown_axis(gt_ax, trace, idx, gt_ctx, show_planned_route=show_planned_route)
    origin_world, _ = load_origin_world(trace)
    position_frame, _ = infer_position_frame(trace, origin_world)
    topo_position_frame, _ = infer_topo_position_frame(trace, origin_world, position_frame)
    draw_topo_on_axis(
        topo_ax,
        trace,
        idx,
        xlim,
        zlim,
        "DynamicTopoMap realtime update",
        origin_world=origin_world,
        position_frame=position_frame,
        topo_position_frame=topo_position_frame,
        show_info_labels=True,
    )

    fig.tight_layout()
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/logs/goat_topo/topo_trace_semantic.json")
    ap.add_argument("--out-dir", default="data/logs/goat_topo/viz_semantic")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--topdown-resolution", type=float, default=0.05)
    ap.add_argument("--navmesh-path", default=None)
    ap.add_argument("--no-gt-topdown", action="store_true")
    ap.add_argument("--show-planned-route", action="store_true")
    args = ap.parse_args()
    trace = json.load(open(ROOT / args.trace if not Path(args.trace).is_absolute() else args.trace))
    load_goal_instances(trace)
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlim, zlim = collect_limits(trace["steps"], trace)
    gt_ctx = load_gt_topdown(trace, args.topdown_resolution, args.navmesh_path, args.no_gt_topdown)
    final_png = out_dir / "topo_map_final.png"
    draw_topo_frame(trace, len(trace["steps"])-1, xlim, zlim, final_png)
    key_frames = []
    for i in sorted(set([0, min(10, len(trace["steps"])-1), min(25, len(trace["steps"])-1), min(50, len(trace["steps"])-1), len(trace["steps"])-1])):
        p = out_dir / "topo_map_step_{:03d}.png".format(i)
        draw_topo_frame(trace, i, xlim, zlim, p); key_frames.append(str(p))
    stride = max(1, args.stride)
    topo_frames, fp_frames, dual_frames, audit_frames = [], [], [], []
    for i in range(0, len(trace["steps"]), stride):
        topo = draw_topo_frame(trace, i, xlim, zlim)
        fp = draw_first_person_frame(trace, i)
        fp = resize_to(fp, topo.shape)
        dual = np.concatenate([fp, topo], axis=1)
        audit = draw_audit_frame(trace, i, xlim, zlim, gt_ctx, show_planned_route=args.show_planned_route)
        topo_frames.append(topo); fp_frames.append(fp); dual_frames.append(dual)
        audit_frames.append(audit)
    paths = {
        "first_person_video": out_dir / "first_person_semantic.mp4",
        "topo_video": out_dir / "topo_map_semantic_growth.mp4",
        "dual_video": out_dir / "goat_semantic_dual_view.mp4",
        "audit_video": out_dir / "goat_semantic_audit_view.mp4",
        "gif": out_dir / "topo_map_semantic_growth.gif",
    }
    imageio.mimsave(paths["first_person_video"], fp_frames, fps=args.fps)
    imageio.mimsave(paths["topo_video"], topo_frames, fps=args.fps)
    imageio.mimsave(paths["dual_video"], dual_frames, fps=args.fps)
    imageio.mimsave(paths["audit_video"], audit_frames, fps=args.fps)
    imageio.mimsave(paths["gif"], topo_frames, duration=1.0/args.fps)
    summary = {"trace": args.trace, "steps": len(trace["steps"]), "final_png": str(final_png), "key_frames": key_frames, "final_memory": trace.get("final_memory", {}), "final_summary": trace.get("final_summary", {})}
    summary["gt_topdown"] = {
        "available": bool(gt_ctx.get("available")),
        "reason": gt_ctx.get("reason", ""),
        "navmesh_path": gt_ctx.get("navmesh_path", ""),
        "navmesh_source": gt_ctx.get("navmesh_source", ""),
        "origin_source": gt_ctx.get("origin_source", ""),
        "origin_world": None if gt_ctx.get("origin_world") is None else np.asarray(gt_ctx.get("origin_world")).round(4).tolist(),
        "resolution": gt_ctx.get("resolution", args.topdown_resolution),
        "position_frame": gt_ctx.get("position_frame", trace.get("coordinate_frame", "")),
        "position_frame_source": gt_ctx.get("position_frame_source", ""),
        "topo_position_frame": gt_ctx.get("topo_position_frame", trace.get("topo_coordinate_frame", "")),
        "topo_position_frame_source": gt_ctx.get("topo_position_frame_source", ""),
        "show_planned_route": bool(args.show_planned_route),
    }
    summary.update({k: str(v) for k, v in paths.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
