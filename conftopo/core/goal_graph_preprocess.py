"""GoalGraph precompute helpers aligned with InstructionGraph / GoalNode schema.

Design contract (see conftopo/docs/CONFTOPO_DESIGN_SUMMARY.md):
- GoalGraph = LLM-parsed task prior; no physical instances.
- target_object = clean category noun only; attributes live in attributes[].
- description field stores raw language text for description modality only.
- goal_type in {category, description, image}.
- normalize_goal_node() only cleans schema; semantic parsing is LLM/offline.
"""

from __future__ import annotations

import gzip
import glob
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from conftopo.core.instruction_graph import (
    GoalNode,
    InstructionGraph,
    Relation,
    SubGoal,
    normalize_goal_node,
)

GOAT_MODALITY_TO_GOAL_TYPE = {
    "object": "category",
    "category": "category",
    "description": "description",
    "instruction": "description",
    "image": "image",
}

VALID_GOAL_TYPES = frozenset({"category", "description", "image"})

DESCRIPTION_PARSER_SYSTEM_PROMPT = """You parse object-goal navigation instructions into structured task priors.

Output valid JSON with exactly these keys:
- target_object: a single clean English noun/category ONLY (e.g. "chair", "sink", "picture").
  Do NOT include adjectives, colors, materials, sizes, or spatial phrases.
- attributes: list of visual descriptors (color, material, size, shape, etc.)
- room_prior: list of room types (e.g. "kitchen", "bedroom")
- landmarks: list of nearby reference objects (e.g. "dining table", "window")
- relations: list of {"relation_type": "...", "reference": "..."}
  relation_type one of: near, left_of, right_of, in_front_of, behind, on, under

Example input: "find the red wooden chair near the dining table in the living room"
Example output:
{
  "target_object": "chair",
  "attributes": ["red", "wooden"],
  "room_prior": ["living room"],
  "landmarks": ["dining table"],
  "relations": [{"relation_type": "near", "reference": "dining table"}]
}

Output JSON only. No markdown. No explanation."""

INSTRUCTION_PARSER_PROMPT = """You are a navigation instruction parser. Given a natural language navigation instruction, decompose it into a structured sequence of sub-goals.

Output a JSON object with:
- "sub_goals": a list of objects, each containing:
  - "id": integer index starting from 0
  - "action": the movement action (e.g., "turn_left", "turn_right", "go_forward", "go_up", "go_down", "stop")
  - "landmark": the key landmark or reference object mentioned (null if none)
  - "spatial_relation": spatial relationship to the landmark (e.g., "past", "towards", "left_of", "right_of", "at", null)
  - "implied_room": room type if mentioned (e.g., "kitchen", null)
  - "termination_condition": what signals this sub-goal is complete

Output valid JSON only."""


def extract_json_from_llm_text(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    for tag in ("think", "redacted_thinking"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            chunk = part.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                text = chunk
                break
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def normalize_attributes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for key, val in value.items():
            if isinstance(val, bool) and val:
                out.append(str(key))
            elif isinstance(val, (list, tuple)):
                out.extend(str(x).strip() for x in val if str(x).strip())
            elif val is not None and not isinstance(val, dict):
                out.append(str(val).strip())
        return [x for x in out if x]
    return normalize_str_list(value)


def relations_from_llm(parsed_relations: Any) -> List[Relation]:
    out: List[Relation] = []
    if not parsed_relations:
        return out
    if isinstance(parsed_relations, dict):
        for rel_type, refs in parsed_relations.items():
            for ref in normalize_str_list(refs):
                out.append(Relation(str(rel_type), ref[:120]))
        return out
    for item in parsed_relations:
        if isinstance(item, str) and item.strip():
            out.append(Relation("near", item.strip()[:120]))
            continue
        if not isinstance(item, dict):
            continue
        ref = (
            item.get("reference")
            or item.get("object")
            or item.get("landmark")
            or item.get("target")
            or item.get("name")
        )
        if not ref:
            continue
        rel_type = item.get("relation_type") or item.get("type") or "near"
        out.append(Relation(str(rel_type), str(ref)[:120]))
    return out


def apply_parsed_description(parsed: dict) -> Tuple[List[str], List[str], List[str], List[Relation], Optional[str]]:
    attributes = normalize_attributes(parsed.get("attributes"))
    room_prior = normalize_str_list(parsed.get("room_prior"))
    landmarks = normalize_str_list(parsed.get("landmarks"))
    relations = relations_from_llm(parsed.get("relations"))
    target_object = parsed.get("target_object")
    if target_object is not None:
        target_object = str(target_object).strip()[:80] or None
    return attributes, room_prior, landmarks, relations, target_object


def goat_goal_type(modality: str) -> str:
    return GOAT_MODALITY_TO_GOAL_TYPE.get((modality or "").lower(), "category")


def validate_goal_node(goal: GoalNode) -> List[str]:
    """Return non-fatal validation warnings for a GoalNode."""
    warnings: List[str] = []
    if not (goal.target_object or "").strip():
        warnings.append("empty target_object")
    if goal.goal_type not in VALID_GOAL_TYPES:
        warnings.append(f"unexpected goal_type={goal.goal_type!r}")
    if goal.goal_type == "description" and not (goal.description or "").strip():
        warnings.append("description modality missing description text")
    if goal.goal_type == "image" and goal.description:
        warnings.append("image modality should not set description")
    if goal.goal_type == "category" and goal.description:
        warnings.append("category modality should not set description")
    if goal.attributes and goal.goal_type == "category":
        warnings.append("category modality should not set attributes (use description)")
    words = (goal.target_object or "").split()
    if len(words) > 2:
        warnings.append(f"target_object looks like phrase not noun: {goal.target_object!r}")
    return warnings


def find_goat_goal_entry(goals: dict, category: str, ref: Any) -> Optional[dict]:
    """Resolve GOAT goals dict entry from category and optional object_id ref."""
    cat = (category or "").strip()

    if ref is not None:
        ref_s = str(ref).strip()
        # 1) Exact object_id match (GOAT-Bench canonical)
        for entries in goals.values():
            if entries and entries[0].get("object_id") == ref_s:
                return entries[0]
        # 2) Goal key suffix match, e.g. ...basis.glb_heater with object_id heater_128
        for key, entries in goals.items():
            if not entries:
                continue
            key_us = key.replace(" ", "_")
            if key.endswith(f"_{ref_s}") or key_us.endswith(f"_{ref_s}"):
                return entries[0]
        return None

    cat_us = cat.replace(" ", "_")
    for key, entries in goals.items():
        if not entries:
            continue
        key_us = key.replace(" ", "_")
        if key_us.endswith(f"_{cat_us}"):
            return entries[0]
    for key, entries in goals.items():
        if cat in key and entries:
            return entries[0]
    return None


def find_goat_goal_entry_for_image(
    goals: dict,
    category: str,
    ref: Any,
    goal_image_id: int,
) -> Optional[dict]:
    """Pick a goal entry whose image_goals[goal_image_id] exists."""
    entry = find_goat_goal_entry(goals, category, ref)
    if entry is not None:
        imgs = entry.get("image_goals") or []
        if 0 <= goal_image_id < len(imgs):
            return entry

    cat = (category or "").strip().lower()
    cat_us = cat.replace(" ", "_")
    best: Optional[dict] = None
    best_n = -1
    for key, entries in goals.items():
        if not entries:
            continue
        candidate = entries[0]
        imgs = candidate.get("image_goals") or []
        if goal_image_id >= len(imgs):
            continue
        oc = str(candidate.get("object_category", "")).lower()
        key_us = key.replace(" ", "_")
        if oc == cat or key_us.endswith(f"_{cat_us}"):
            if len(imgs) > best_n:
                best = candidate
                best_n = len(imgs)
    return best


def build_goal_node_from_goat_task(
    category: str,
    modality: str,
    ref: Any,
    goals: dict,
    llm_parse_description: Optional[Callable[[str], Optional[dict]]] = None,
    goal_image_id: Optional[int] = None,
) -> GoalNode:
    """Build a schema-normalized GoalNode from one GOAT sub-task."""
    del goal_image_id  # consumed during image-embedding pass, not stored on GoalNode
    entry = find_goat_goal_entry(goals, category, ref)
    goal_type = goat_goal_type(modality)
    target_object = (category or "object").strip()
    raw_description: Optional[str] = None
    attributes: List[str] = []
    room_prior: List[str] = []
    landmarks: List[str] = []
    relations: List[Relation] = []

    if entry:
        target_object = str(entry.get("object_category") or target_object).strip()
        lang = (entry.get("lang_desc") or entry.get("description") or "").strip()

        if goal_type == "description":
            if lang:
                raw_description = lang[:500]
            elif ref is not None:
                raw_description = str(ref).strip()[:500]
            if raw_description and llm_parse_description:
                parsed = llm_parse_description(raw_description)
                if parsed:
                    attributes, room_prior, landmarks, relations, parsed_target = apply_parsed_description(parsed)
                    if parsed_target:
                        target_object = parsed_target
        elif goal_type == "category":
            pass
        elif goal_type == "image":
            pass
    elif goal_type == "description" and ref is not None:
        raw_description = str(ref).strip()[:500]

    node = normalize_goal_node(
        GoalNode(
            target_object=target_object,
            description=raw_description,
            attributes=attributes,
            room_prior=room_prior,
            landmarks=landmarks,
            relations=relations,
            goal_type=goal_type,
        )
    )
    return node


def relations_from_spatial_text(text: str) -> List[Relation]:
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
    lowered = raw.lower()
    rules = [
        ("left_of", ["left of", "left side"]),
        ("right_of", ["right of", "right side"]),
        ("in_front_of", ["in front", "opposite to", "opposite"]),
        ("behind", ["behind"]),
        ("on", [" on ", "above"]),
        ("under", ["under", "below"]),
        ("near", ["near", "next to", "beside"]),
    ]
    for rel_type, keywords in rules:
        if any(k in lowered for k in keywords):
            return [Relation(rel_type, raw[:120])]
    return [Relation("near", raw[:120])]


def split_csv_phrases(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).replace("\n", " ").split(",") if p.strip()]


def build_goal_node_from_soon_layers(
    layers: List[str],
    obj_name: Optional[str] = None,
    llm_parse_description: Optional[Callable[[str], Optional[dict]]] = None,
) -> GoalNode:
    """Map FAO/SOON 6-layer instruction group to GoalNode."""
    layers = list(layers)
    attr_text = layers[0] if len(layers) > 0 else ""
    rel_text = layers[1] if len(layers) > 1 else ""
    room_text = layers[2] if len(layers) > 2 else ""
    landmark_text = layers[3] if len(layers) > 3 else ""
    full_text = layers[4] if len(layers) > 4 else ""

    attributes = split_csv_phrases(attr_text)
    relations = relations_from_spatial_text(rel_text)
    room_prior = [str(room_text).strip()] if str(room_text).strip() else []
    landmarks = [str(landmark_text).strip()] if str(landmark_text).strip() else []
    raw_description = str(full_text).strip()[:500] if str(full_text).strip() else None
    target_object = (obj_name or "object").strip()

    if llm_parse_description and raw_description:
        parsed = llm_parse_description(raw_description)
        if parsed:
            attrs, rp, lms, rels, parsed_target = apply_parsed_description(parsed)
            if parsed_target:
                target_object = parsed_target
            if attrs:
                attributes = attrs
            if rp:
                room_prior = rp
            if lms:
                landmarks = lms
            if rels:
                relations = rels

    return normalize_goal_node(
        GoalNode(
            target_object=target_object,
            description=raw_description,
            attributes=attributes,
            room_prior=room_prior,
            landmarks=landmarks,
            relations=relations,
            goal_type="description",
        )
    )


def clip_target_text(goal: GoalNode) -> str:
    """Text used for CLIP target_embedding on category/description goals."""
    target = (goal.target_object or "").strip()
    if goal.goal_type == "image":
        return target
    if goal.attributes:
        return " ".join(str(a) for a in goal.attributes[:4]) + " " + target
    return target


@dataclass
class GoatImageGoalRef:
    scene_name: str
    episode_id: Any
    goal_index: int
    category: str
    object_id: Optional[str]
    goal_image_id: int


def iter_goat_image_goal_refs(goat_root: Path, split: str) -> Iterable[GoatImageGoalRef]:
    split_map = {
        "train": "train",
        "val": "val_seen",
        "val_seen": "val_seen",
        "val_unseen": "val_unseen",
        "val_seen_synonyms": "val_seen_synonyms",
    }
    folder = split_map.get(split, split)
    content_dir = goat_root / folder / "content"
    if not content_dir.is_dir():
        return
    for sf in sorted(content_dir.glob("*.json.gz")):
        scene_name = sf.name.replace(".json.gz", "")
        with gzip.open(sf, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for ep in data.get("episodes", []):
            ep_id = ep.get("episode_id", 0)
            for gi, task in enumerate(ep.get("tasks", [])):
                if len(task) < 4 or task[1] != "image":
                    continue
                yield GoatImageGoalRef(
                    scene_name=scene_name,
                    episode_id=ep_id,
                    goal_index=gi,
                    category=str(task[0]),
                    object_id=str(task[2]) if task[2] is not None else None,
                    goal_image_id=int(task[3]),
                )


def resolve_goat_image_params(goals: dict, ref: GoatImageGoalRef) -> Optional[dict]:
    entry = find_goat_goal_entry_for_image(
        goals, ref.category, ref.object_id, ref.goal_image_id,
    )
    if not entry:
        return None
    image_goals = entry.get("image_goals") or []
    if ref.goal_image_id < 0 or ref.goal_image_id >= len(image_goals):
        return None
    return image_goals[ref.goal_image_id]


def find_scene_file(scene_id: str, scene_root: Path) -> Path:
    scene_name = Path(scene_id).name.replace(".basis.glb", "")
    matches = sorted(scene_root.glob(f"**/{scene_name}.basis.glb"))
    if not matches:
        raise FileNotFoundError(f"Scene {scene_name} not found under {scene_root}")
    return matches[0]


def _quaternion_from_coeff(coeffs: List[float]):
    import quaternion as qt

    quat = qt.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def render_goat_image_goal(sim: Any, img_params: dict) -> np.ndarray:
    """Render GOAT instance image goal using habitat-sim (GOAT-Bench protocol)."""
    import habitat_sim
    from habitat_sim.agent import AgentState, SixDOFPose

    sensor_uuid = "goal_graph_image_sensor"
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = sensor_uuid
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = img_params["image_dimensions"]
    spec.hfov = float(img_params["hfov"])
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sim.add_sensor(spec)

    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    agent.set_state(
        AgentState(
            position=agent_state.position,
            rotation=agent_state.rotation,
            sensor_states={
                **agent_state.sensor_states,
                sensor_uuid: SixDOFPose(
                    position=np.array(img_params["position"], dtype=np.float32),
                    rotation=_quaternion_from_coeff(img_params["rotation"]),
                ),
            },
        ),
        infer_sensor_states=False,
    )

    sim._sensors[sensor_uuid].draw_observation()
    rgb = sim._sensors[sensor_uuid].get_observation()
    rgb = np.asarray(rgb)[..., :3]

    del sim._sensors[sensor_uuid]
    import habitat_sim as hsim

    hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
    del agent._sensors[sensor_uuid]
    agent.agent_config.sensor_specifications = [
        s for s in agent.agent_config.sensor_specifications if s.uuid != sensor_uuid
    ]
    return rgb.astype(np.uint8)


def _quiet_habitat_logs() -> None:
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


class _SceneSimSession:
    """Hold at most one habitat-sim instance to avoid GL context teardown crashes."""

    def __init__(self) -> None:
        self._scene_key: Optional[str] = None
        self._sim: Any = None

    def get(self, scene_file: Path) -> Any:
        key = str(scene_file)
        if self._scene_key == key and self._sim is not None:
            return self._sim
        self.close()
        _quiet_habitat_logs()
        self._sim = make_minimal_habitat_sim(scene_file)
        self._scene_key = key
        return self._sim

    def close(self) -> None:
        if self._sim is None:
            return
        try:
            self._sim.close()
        except Exception:
            pass
        self._sim = None
        self._scene_key = None


def make_minimal_habitat_sim(scene_file: Path) -> Any:
    """Create habitat-sim with a placeholder COLOR sensor.

    Newer habitat-sim requires COLOR sensor data at init before dynamic sensors
    can be added (GOAT instance-image-goal rendering protocol).
    """
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_file)
    sim_cfg.enable_physics = False

    placeholder = habitat_sim.CameraSensorSpec()
    placeholder.uuid = "color_sensor"
    placeholder.sensor_type = habitat_sim.SensorType.COLOR
    placeholder.resolution = [256, 256]
    placeholder.position = [0.0, 1.25, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [placeholder]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def resolve_clip_pretrained(clip_cache: Path) -> str:
    candidates = [
        clip_cache / "ViT-B-32-laion2b_s34b_b79k.bin",
        clip_cache / "open_clip_pytorch_model.bin",
        clip_cache.parent / "clip" / "ViT-B-32-laion2b_s34b_b79k.bin",
        # OpenAI .pt format — open_clip accepts a file path as pretrained
        clip_cache / "ViT-B-32.pt",
        clip_cache.parent / "clip" / "ViT-B-32.pt",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return "laion2b_s34b_b79k"


def _goat_image_cache_pkl_name(scene: str, encoder: str) -> str:
    """Official GOAT image cache filename: ``{scene}_{encoder}_goat_embedding.pkl``."""
    return f"{scene}_{encoder}_goat_embedding.pkl"


def load_goat_scene_image_cache(
    cache_dir: Path,
    split: str,
    scene: str,
    encoder: str = "CLIP",
) -> Optional[dict]:
    """Load one scene's official GOAT image cache pickle.

    Tries ``{cache_dir}/{split}_embeddings/{scene}_{encoder}_goat_embedding.pkl``
    then falls back to ``{cache_dir}/{scene}_{encoder}_goat_embedding.pkl``.
    Returns ``None`` if not found.
    """
    candidates = [
        cache_dir / f"{split}_embeddings" / _goat_image_cache_pkl_name(scene, encoder),
        cache_dir / _goat_image_cache_pkl_name(scene, encoder),
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as exc:
                print(f"    [WARN] cannot load cache {p}: {exc}")
                return None
    return None


def lookup_official_image_embedding(
    scene_cache: dict,
    scene: str,
    object_id: str,
    goal_image_id: int,
) -> Optional[np.ndarray]:
    """Lookup ``cache["{scene}_{object_id}"][goal_image_id]["embedding"]``.

    Mirrors ``CacheInstanceImageGoalSensor.get_observation`` from goat-bench.
    Returns None on miss.
    """
    key = f"{scene}_{object_id}"
    entries = scene_cache.get(key)
    if not entries or not isinstance(entries, list):
        return None
    if goal_image_id < 0 or goal_image_id >= len(entries):
        return None
    entry = entries[goal_image_id]
    emb = entry.get("embedding") if isinstance(entry, dict) else None
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def _ensure_goat_scene_data(
    cache: Dict[str, dict], content_dir: Path, scene_name: str,
) -> dict:
    """Load and cache a scene's GOAT content JSON (lazy)."""
    if scene_name not in cache:
        sf = content_dir / f"{scene_name}.json.gz"
        if sf.is_file():
            with gzip.open(sf, "rt", encoding="utf-8") as gf:
                cache[scene_name] = json.load(gf)
        else:
            cache[scene_name] = {}
    return cache[scene_name]


def _find_episode_scene_id(scene_data: dict, ep_id: str) -> Optional[str]:
    for ep in scene_data.get("episodes", []):
        if str(ep.get("episode_id")) == str(ep_id):
            return ep.get("scene_id")
    return None


def _resolve_image_task_ids(
    goat_scene_cache: Dict[str, dict],
    goat_root_path: Optional[Path],
    content_dir: Optional[Path],
    scene_name: str,
    ep_id: str,
    goal_index: int,
    gn: dict,
) -> Tuple[Optional[str], int]:
    """Extract ``(object_id, goal_image_id)`` from GOAT content for one image task.

    Returns ``(None, 0)`` if the task cannot be resolved.
    """
    if content_dir is None:
        if goat_root_path is not None:
            for folder in (scene_name, "val_seen", "val_unseen", "val_seen_synonyms", "train"):
                candidate = goat_root_path / folder / "content"
                if candidate.is_dir():
                    content_dir = candidate
                    break
    if content_dir is None:
        return None, 0
    scene_data = _ensure_goat_scene_data(goat_scene_cache, content_dir, scene_name)
    if not scene_data:
        return None, 0
    for ep in scene_data.get("episodes", []):
        if str(ep.get("episode_id")) != str(ep_id):
            continue
        tasks = ep.get("tasks", [])
        if goal_index >= len(tasks):
            return None, 0
        task = tasks[goal_index]
        if len(task) < 4 or task[1] != "image":
            return None, 0
        obj_id = str(task[2]) if task[2] is not None else None
        return obj_id, int(task[3])
    return None, 0


def embed_goal_graphs(
    goal_graph_dir: str | Path,
    clip_cache: Optional[str | Path] = None,
    include_train: bool = False,
    goat_root: Optional[str | Path] = None,
    scene_root: Optional[str | Path] = None,
    embed_images: bool = True,
    skip_existing_image_embed: bool = True,
    goat_image_cache_dir: Optional[str | Path] = None,
    goat_image_encoder: str = "CLIP",
) -> None:
    """Add CLIP embeddings to serialized GoalGraph JSON files."""
    try:
        import open_clip
        import torch
    except ImportError:
        print("open_clip not installed, skip embedding")
        return

    goal_graph_dir = Path(goal_graph_dir)
    if clip_cache is None:
        clip_cache = goal_graph_dir.parent.parent / "models" / "clip"
    clip_cache = Path(clip_cache)
    clip_cache.mkdir(parents=True, exist_ok=True)
    pretrained = resolve_clip_pretrained(clip_cache)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # open_clip loading OpenAI .pt (TorchScript archive) requires weights_only=False.
    # Patch torch.load temporarily so open_clip can load the local file without error.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **dict(kw, weights_only=False))
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained, cache_dir=str(clip_cache),
        )
    finally:
        torch.load = _orig_torch_load
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text_cache: Dict[str, List[float]] = {}
    image_cache: Dict[str, List[float]] = {}

    def encode_texts(texts: List[str]) -> None:
        unique = list({t for t in texts if t and t not in text_cache})
        if not unique:
            return
        batch_size = 64
        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for text, feat in zip(batch, feats.cpu().numpy()):
                text_cache[text] = feat.astype(np.float32).tolist()

    def encode_image(rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    json_files = sorted(goal_graph_dir.glob("**/*_goal_graphs.json"))
    for jf in json_files:
        basename = jf.name
        if not include_train and (
            "train_goal_graphs" in basename or "test_release_goal_graphs" in basename
        ):
            print(f"  [SKIP embed] {jf} (use --embed-clip-train to include)")
            continue
        print(f"  Embedding {jf}...")
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        texts_to_encode: List[str] = []
        for ep_data in data.values():
            if ep_data.get("goal_type") == "route":
                for sg in ep_data.get("sub_goals", []):
                    if isinstance(sg.get("landmark"), str) and sg["landmark"]:
                        texts_to_encode.append(sg["landmark"])
                    if isinstance(sg.get("implied_room"), str) and sg["implied_room"]:
                        texts_to_encode.append(sg["implied_room"])
                continue
            for gn in ep_data.get("goal_nodes", []):
                goal_type = gn.get("goal_type", "category")
                if goal_type != "image":
                    texts_to_encode.append(clip_target_text(GoalNode(
                        target_object=gn.get("target_object", ""),
                        attributes=gn.get("attributes", []),
                        goal_type=goal_type,
                    )))
                else:
                    t = (gn.get("target_object") or "").strip()
                    if t:
                        texts_to_encode.append(t)
                for rp in gn.get("room_prior", []):
                    if isinstance(rp, str) and rp:
                        texts_to_encode.append(rp)
                for lm in gn.get("landmarks", []):
                    if isinstance(lm, str) and lm:
                        texts_to_encode.append(lm)

        encode_texts(texts_to_encode)

        split_name = basename.replace("_goal_graphs.json", "")
        goat_scene_cache: Dict[str, dict] = {}
        official_image_caches: Dict[str, Optional[dict]] = {}
        sim_session = _SceneSimSession()
        scene_root_path = Path(scene_root) if scene_root else None
        goat_root_path = Path(goat_root) if goat_root else None
        image_cache_path = Path(goat_image_cache_dir) if goat_image_cache_dir else None
        has_official_cache = image_cache_path is not None and image_cache_path.is_dir()
        can_render_images = (
            embed_images
            and goat_root_path is not None
            and scene_root_path is not None
            and scene_root_path.is_dir()
        )

        content_dir = None
        if can_render_images:
            for folder in (split_name, "val_seen", "val_unseen", "train"):
                candidate = goat_root_path / folder / "content"
                if candidate.is_dir():
                    content_dir = candidate
                    break

        img_official_hit = img_render = img_text_fallback = img_skipped = img_failed = 0

        for ep_key, ep_data in data.items():
            if ep_data.get("goal_type") == "route":
                for sg in ep_data.get("sub_goals", []):
                    lm = sg.get("landmark")
                    if isinstance(lm, str) and lm in text_cache:
                        sg["landmark_embedding"] = text_cache[lm]
                continue

            for gi, gn in enumerate(ep_data.get("goal_nodes", [])):
                goal_type = gn.get("goal_type", "category")
                if goal_type != "image":
                    text = clip_target_text(GoalNode(
                        target_object=gn.get("target_object", ""),
                        attributes=gn.get("attributes", []),
                        goal_type=goal_type,
                    ))
                    if text in text_cache:
                        gn["target_embedding"] = text_cache[text]
                        gn["embedding_source"] = "text_clip"
                else:
                    if skip_existing_image_embed and gn.get("target_embedding"):
                        img_skipped += 1
                    else:
                        scene_name, _, ep_id_s = ep_key.rpartition("_")
                        if not scene_name:
                            img_failed += 1
                        else:
                            object_id, goal_image_id = _resolve_image_task_ids(
                                goat_scene_cache, goat_root_path, content_dir,
                                scene_name, ep_id_s, gi, gn,
                            )
                            embedded = False

                            # --- Priority 1: Official GOAT image cache ---
                            if has_official_cache and object_id is not None:
                                if scene_name not in official_image_caches:
                                    official_image_caches[scene_name] = load_goat_scene_image_cache(
                                        image_cache_path, split_name, scene_name, goat_image_encoder,
                                    )
                                sc = official_image_caches[scene_name]
                                if sc is not None:
                                    emb = lookup_official_image_embedding(
                                        sc, scene_name, object_id, goal_image_id,
                                    )
                                    if emb is not None:
                                        gn["target_embedding"] = emb.tolist()
                                        gn["embedding_source"] = "goat_official_cache"
                                        gn["match_status"] = "official_cache_hit"
                                        gn["image_cache_key"] = f"{scene_name}_{object_id}"
                                        gn["goal_image_id"] = goal_image_id
                                        img_official_hit += 1
                                        embedded = True

                            # --- Priority 2: Habitat render + CLIP (old path) ---
                            if not embedded and can_render_images and content_dir is not None:
                                scene_data = _ensure_goat_scene_data(
                                    goat_scene_cache, content_dir, scene_name,
                                )
                                if scene_data and object_id is not None:
                                    ref = GoatImageGoalRef(
                                        scene_name=scene_name,
                                        episode_id=ep_id_s,
                                        goal_index=gi,
                                        category=gn.get("target_object", ""),
                                        object_id=object_id,
                                        goal_image_id=goal_image_id,
                                    )
                                    img_params = resolve_goat_image_params(
                                        scene_data.get("goals", {}), ref,
                                    )
                                    if img_params is not None:
                                        render_key = f"{scene_name}:{object_id}:{goal_image_id}"
                                        if render_key not in image_cache:
                                            try:
                                                scene_id = _find_episode_scene_id(
                                                    scene_data, ep_id_s,
                                                )
                                                if scene_id is not None:
                                                    sf = find_scene_file(scene_id, scene_root_path)
                                                    sim = sim_session.get(sf)
                                                    rgb = render_goat_image_goal(sim, img_params)
                                                    image_cache[render_key] = encode_image(rgb).tolist()
                                            except Exception as exc:
                                                print(f"    [WARN] render failed {render_key}: {exc}")
                                        if render_key in image_cache:
                                            gn["target_embedding"] = image_cache[render_key]
                                            gn["embedding_source"] = "rendered_clip"
                                            gn["match_status"] = "rendered"
                                            gn["image_cache_key"] = f"{scene_name}_{object_id}"
                                            gn["goal_image_id"] = goal_image_id
                                            img_render += 1
                                            embedded = True

                            # --- Priority 3: Text CLIP fallback (category name) ---
                            if not embedded:
                                text = (gn.get("target_object") or "").strip()
                                if text:
                                    encode_texts([text])
                                if text and text in text_cache:
                                    gn["target_embedding"] = text_cache[text]
                                    gn["embedding_source"] = "text_clip_fallback"
                                    gn["match_status"] = "text_fallback"
                                    img_text_fallback += 1
                                else:
                                    gn["embedding_source"] = "none"
                                    gn["match_status"] = "failed"
                                    img_failed += 1

                rp = gn.get("room_prior", [])
                if rp:
                    rp_emb = [text_cache[r] for r in rp if isinstance(r, str) and r in text_cache]
                    if rp_emb:
                        gn["room_prior_embeddings"] = rp_emb
                lms = gn.get("landmarks", [])
                if lms:
                    lm_emb = [text_cache[lm] for lm in lms if isinstance(lm, str) and lm in text_cache]
                    if lm_emb:
                        gn["landmark_embeddings"] = lm_emb

        sim_session.close()

        total_image = img_official_hit + img_render + img_text_fallback + img_skipped + img_failed
        if total_image > 0:
            print(
                f"    image embed: official_cache={img_official_hit} rendered={img_render} "
                f"text_fallback={img_text_fallback} skipped={img_skipped} failed={img_failed}",
                flush=True,
            )

        with open(jf, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"    saved with embeddings")
