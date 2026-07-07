"""Clean Agent1 implementation for GOAT smoke experiments.

This agent follows the project proposal directly:

perception -> memory update -> structure intent -> navigation target -> control.

The agent keeps the public ``ConfTopoGOATAgent`` entrypoint used by
``scripts/run_goat_agent1_smoke_v2.py`` but replaces the old mixed planning
flow with explicit structure and navigation stages.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.agents.goat_agent1_active_topo import ActiveTopoProjector
from conftopo.agents.goat_agent1_control import StateMachine
from conftopo.agents.goat_agent1_memory import (
    Agent1MemoryWriter,
    GoalRelevanceScorer,
)
from conftopo.agents.goat_agent1_planner import (
    NavigationPlanner,
    StructurePlanner,
    _goal_id,
    hypothesis_c_task,
    object_c_task,
)
from conftopo.agents.goat_agent1_types import (
    ActiveTopoView,
    AgentPose,
    MapAuditContext,
    NavigationHistoryEntry,
    NavigationTarget,
    PlanningContext,
    ProjectionConfig,
    RecoveryState,
    ScanContext,
    SearchIntent,
)
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.navigation.pathfinder_executor import PathfinderExecutor
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import Qwen3VLBackend
from conftopo.perception.vlm_perceiver import VLMPerceiver


NAV_REACH_RADIUS = 0.40

# Scale-fair arrival detection: the agent has physically reached / been blocked
# by the target when it keeps trying to move forward but its own pose stops
# advancing. This replaces object-size-dependent bbox proximity heuristics.
ARRIVAL_STALL_WINDOW = 4      # steps of pose history to inspect
ARRIVAL_STALL_EPS = 0.30      # meters: max horizontal spread that counts as stalled
ARRIVAL_MIN_FORWARD = 2       # min forward attempts in the window (not just turning)

# Minimum label relevance for treating a memory object as THE goal (for the
# metric arrival / commit gates). Kept strict (> alias-synonym score of 0.80) so
# semantic cousins -- e.g. a "shelf"/"bookcase" for goal "rack" -- do not trigger
# an early stop when the real target is still far away. Exact matches (1.0) and
# direct variants like "wooden rack"/"clothes rack" (0.85, contain the goal word)
# still qualify.
GOAL_MATCH_MIN_RELEVANCE = 0.85

# --- Sticky local commitment (exploit) --------------------------------------
# When a goal-relevant object's task-conditioned confidence c_task clears this
# bar, the agent LOCKS navigation onto its approach point and holds the lock
# across frames (even when the object is momentarily out of view), so per-frame
# hypotheses cannot hijack the target and spin the agent in place. Uncertain
# candidates (below the bar) still compete normally against exploration via
# c_task, preserving the explore/exploit arbitration.
LOCK_CTASK_THRESHOLD = 0.7

# While navigating (SEARCH) toward a locked candidate's OBSERVATION anchor, if
# the agent makes no measurable progress (its min horizontal distance to that
# anchor stops improving) for this many planning steps -- and the goal is not
# being actively servoed to (not seen recently) -- the detection is treated as
# unreachable / mislocalised / false: it is demoted (cooldown + decay) and the
# lock released so exploration resumes. This is the "can't push in" release that
# was missing, and prevents a single bad detection from pinning the whole
# episode. plan() early-returns during CONFIRM, so a genuine goal being visually
# approached never accumulates this count.
LOCK_NO_PROGRESS_STEPS = 15

# A demoted goal object is excluded from goal candidates for this many steps
# (a cooldown, not a permanent removal) and has its existence confidence decayed.
OBJECT_SEARCH_COOLDOWN = 60


def _normalize_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _crop_rgb_bbox(rgb: np.ndarray, bbox: Any, *, pad: float = 0.03) -> Optional[np.ndarray]:
    if rgb is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    height, width = int(arr.shape[0]), int(arr.shape[1])
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        px1, px2 = x1 * width, x2 * width
        py1, py2 = y1 * height, y2 * height
        pad_x = pad * width
        pad_y = pad * height
    else:
        px1, px2 = x1, x2
        py1, py2 = y1, y2
        pad_x = pad * max(width, 1)
        pad_y = pad * max(height, 1)
    if px2 - px1 < 4 or py2 - py1 < 4:
        return None
    if (px2 - px1) * (py2 - py1) < max(16, int(0.0001 * width * height)):
        return None
    ix1 = int(np.floor(max(0.0, px1 - pad_x)))
    iy1 = int(np.floor(max(0.0, py1 - pad_y)))
    ix2 = int(np.ceil(min(float(width), px2 + pad_x)))
    iy2 = int(np.ceil(min(float(height), py2 + pad_y)))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    if (ix2 - ix1) * (iy2 - iy1) < max(16, int(0.0001 * width * height)):
        return None
    crop = arr[iy1:iy2, ix1:ix2]
    if crop.size <= 0:
        return None
    return crop


class GoalManager:
    def __init__(self, agent: "GOATAgent"):
        self.agent = agent

    def current_goal(self) -> Optional[GoalNode]:
        if self.agent.instruction_graph is None:
            return None
        goal = self.agent.instruction_graph.get_current_goal()
        return goal if isinstance(goal, GoalNode) else None

    def set_goal(self, goal: GoalNode) -> None:
        if self.agent.instruction_graph is None:
            self.agent.instruction_graph = InstructionGraph(
                goal_type="object_goal",
                goal_nodes=[goal],
            )
        else:
            self.agent.instruction_graph.set_current_goal(goal)


class Agent1Perception:
    """CLIP every step, VLM only when triggered."""

    def __init__(self, config: ConfTopoConfig):
        self.config = config
        self.clip = LightPerceiver(room_labels=config.perception.room_labels)
        self._clip_encoder = None
        self.vlm: Optional[VLMPerceiver] = None
        if config.perception.backend == "vlm":
            self.vlm = VLMPerceiver(
                Qwen3VLBackend(
                    api_base=config.perception.vlm_api_base,
                    model=config.perception.vlm_model,
                    timeout=config.perception.vlm_timeout,
                    temperature=config.perception.vlm_temperature,
                    seed=config.perception.vlm_seed,
                )
            )
        self.last_vlm_step = -1
        self.last_vlm_reason = "init"

    def set_clip_encoder(self, encoder: Any) -> None:
        self._clip_encoder = encoder

    def configure_goal(self, goal: Optional[GoalNode]) -> None:
        if goal is None:
            return
        if goal.target_embedding is None and self._clip_encoder is not None:
            try:
                embeds = self._clip_encoder.encode_text([goal.target_object])
                if embeds is not None and len(embeds) > 0:
                    goal.target_embedding = _normalize_embedding(np.asarray(embeds[0], dtype=np.float32))
                    goal.embedding_source = "runtime_clip_text"
            except Exception:
                pass
        self.clip.set_goal_labels(
            labels=[goal.target_object],
            embeddings=(
                goal.target_embedding[np.newaxis, :]
                if goal.target_embedding is not None
                else None
            ),
        )
        if goal.landmarks:
            self.clip.set_landmark_labels(
                labels=goal.landmarks,
                embeddings=goal.landmark_embeddings,
        )

    def encode_object_crop(self, rgb: Optional[np.ndarray], bbox: Any) -> Optional[np.ndarray]:
        if self._clip_encoder is None or rgb is None:
            return None
        crop = _crop_rgb_bbox(rgb, bbox)
        if crop is None:
            return None
        try:
            emb = self._clip_encoder.encode_image(crop, goal_type="category")
        except TypeError:
            try:
                emb = self._clip_encoder.encode_image(crop)
            except Exception:
                return None
        except Exception:
            return None
        return _normalize_embedding(emb)

    def encode_object_text(self, label: str, description: str = "") -> Optional[np.ndarray]:
        if self._clip_encoder is None:
            return None
        text = " ".join(part for part in (str(label or "").strip(), str(description or "").strip()) if part)
        if not text:
            return None
        try:
            embeds = self._clip_encoder.encode_text([text])
        except Exception:
            return None
        if embeds is None or len(embeds) <= 0:
            return None
        return _normalize_embedding(np.asarray(embeds[0], dtype=np.float32))

    def run_clip(
        self,
        rgb_embed: Optional[np.ndarray],
        *,
        step_id: int,
    ) -> PerceptionReport:
        if rgb_embed is None:
            return PerceptionReport(step_id=step_id, source="none")
        raw = self.clip.perceive(rgb_embed)
        return PerceptionReport(
            room_label=str(raw.get("room_label", "unknown")),
            room_confidence=float(raw.get("room_confidence", 0.0)),
            room_scores=list(raw.get("room_scores", [])),
            goal_scores=list(raw.get("goal_scores", [])),
            landmark_scores=list(raw.get("landmark_scores", [])),
            best_goal_sim=float(raw.get("best_goal_sim", 0.0)),
            best_landmark_sim=float(raw.get("best_landmark_sim", 0.0)),
            visual_embed=rgb_embed,
            source="clip",
            is_full=False,
            step_id=step_id,
            raw=raw,
        )

    def should_run_vlm(
        self,
        *,
        clip_report: PerceptionReport,
        goal: Optional[GoalNode],
        state: str,
        step_id: int,
    ) -> bool:
        if self.vlm is None or goal is None:
            self.last_vlm_reason = "no_vlm"
            return False
        if state == "CONFIRM":
            self.last_vlm_reason = "state_refresh"
            return True
        if clip_report.best_goal_sim >= 0.20:
            self.last_vlm_reason = "clip_goal_candidate"
            return True
        if self.last_vlm_step < 0:
            self.last_vlm_reason = "initial_search_scan"
            return True
        interval = 2 if step_id <= 60 else 4
        if step_id - self.last_vlm_step >= interval:
            self.last_vlm_reason = "periodic_search"
            return True
        self.last_vlm_reason = "skip"
        return False

    def run_vlm(
        self,
        rgb: Optional[np.ndarray],
        *,
        goal: Optional[GoalNode],
        clip_report: PerceptionReport,
        state: str,
        step_id: int,
        scan_type: str = "",
    ) -> Optional[PerceptionReport]:
        if self.vlm is None or rgb is None or goal is None:
            return None
        confirm_scan_types = {"object_anchor", "hypothesis_anchor", "goal_visible_followup"}
        mode = (
            "confirm"
            if state == "CONFIRM" or scan_type in confirm_scan_types
            else "explore"
        )
        alias_text = _goal_alias_text(goal)
        context_parts = [
            f"state={state}",
            f"scan_type={scan_type or 'none'}",
            f"room={clip_report.room_label}",
            f"goal_sim={clip_report.best_goal_sim:.3f}",
            f"target_aliases={alias_text}",
        ]
        if mode == "explore":
            context_parts.append(
                "Explore openly: report salient navigation-useful objects, "
                "landmarks, fixtures, and portals, with the goal prioritized first."
            )
        else:
            context_parts.append("Confirm only the target or its aliases; ignore unrelated objects.")
        context = "; ".join(context_parts)
        try:
            report = self.vlm.perceive(
                np.asarray(rgb),
                _goal_text(goal),
                visual_embed=clip_report.visual_embed,
                step_id=step_id,
                context=context,
                mode=mode,
            )
        except Exception as exc:
            self.last_vlm_step = step_id
            return PerceptionReport(
                source="vlm",
                is_full=False,
                step_id=step_id,
                raw={"error": str(exc), "error_type": type(exc).__name__},
            )
        self.last_vlm_step = step_id
        return report


class GOATAgent(ConfTopoBaseAgent):
    """Agent1 clean orchestration agent."""

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        super().__init__(config)
        self.goal_manager = GoalManager(self)
        self.perception = Agent1Perception(self.config)
        self.memory = Agent1MemoryWriter(self.topo_map)
        self.structure_planner = StructurePlanner()
        self.navigation_planner = NavigationPlanner()
        self.active_projector = ActiveTopoProjector(
            ProjectionConfig(suppress_enabled=True, verify_enabled=True, fold_enabled=True)
        )
        self._active_view = ActiveTopoView()
        self._recovery = RecoveryState()
        self._active_view_signature: Optional[tuple] = None
        self._recovery_ref_pose: Optional[np.ndarray] = None
        self._recovery_ref_step: int = 0
        self._verify_success_count = 0
        self._verify_fail_count = 0
        self._last_needs_verify = False
        self.state_machine = StateMachine()
        self.step_id = 0
        self._pose: Optional[AgentPose] = None
        self._recent_positions: Deque[Tuple[float, float]] = deque(maxlen=ARRIVAL_STALL_WINDOW)
        self._recent_forward: Deque[bool] = deque(maxlen=ARRIVAL_STALL_WINDOW)
        self._rgb: Optional[np.ndarray] = None
        self._rgb_embed: Optional[np.ndarray] = None
        self._clip_report = PerceptionReport()
        self._vlm_report: Optional[PerceptionReport] = None
        self._last_intent = None
        self._last_nav_target = NavigationTarget(None, None)
        self._sticky_nav_target = NavigationTarget(None, None)
        self._sticky_nav_steps = 0
        self._locked_object_id: Optional[str] = None
        self._lock_best_dist: Optional[float] = None
        self._lock_stall_steps = 0
        self._pathfinder = None
        self._nav_sim = None
        self._nav_origin_world: Optional[np.ndarray] = None
        self._pathfinder_executor = PathfinderExecutor(
            reach_radius=NAV_REACH_RADIUS,
            waypoint_radius=NAV_REACH_RADIUS,
            turn_threshold=0.35,
        )
        self._cur_perception: Dict[str, Any] = {}
        self._last_debug: Dict[str, Any] = {}
        self._nav_history: List[NavigationHistoryEntry] = []
        self._target_progress: Dict[str, Dict[str, Any]] = {}
        self._stale_target_until_step: Dict[str, int] = {}
        self._last_planning_context = PlanningContext()
        self._force_vlm_scan_steps = 0
        self._goal_visible_vlm_steps = 0
        self._visual_scan_steps = 0
        self._scan_context = ScanContext()
        self._last_valid_vlm_report: Optional[PerceptionReport] = None
        self._last_valid_vlm_step = -1
        self._last_valid_vlm_pose: Optional[np.ndarray] = None
        self._last_reachability_probe: Dict[str, Any] = {}
        self.memory.set_frontier_probe(self._probe_reachable)

    def set_clip_encoder(self, encoder: Any) -> None:
        self.perception.set_clip_encoder(encoder)
        goal = self.goal_manager.current_goal()
        if goal is not None:
            self.perception.configure_goal(goal)

    def reset(self):
        super().reset()
        self.memory = Agent1MemoryWriter(self.topo_map)
        self.structure_planner = StructurePlanner()
        self.navigation_planner = NavigationPlanner()
        self.active_projector = ActiveTopoProjector(
            ProjectionConfig(suppress_enabled=True, verify_enabled=True, fold_enabled=True)
        )
        self._active_view = ActiveTopoView()
        self._recovery = RecoveryState()
        self._active_view_signature = None
        self._recovery_ref_pose = None
        self._recovery_ref_step = 0
        self._verify_success_count = 0
        self._verify_fail_count = 0
        self._last_needs_verify = False
        self.state_machine = StateMachine()
        self.step_id = 0
        self._recent_positions.clear()
        self._recent_forward.clear()
        self._last_nav_target = NavigationTarget(None, None)
        self._sticky_nav_target = NavigationTarget(None, None)
        self._sticky_nav_steps = 0
        self._locked_object_id = None
        self._lock_best_dist = None
        self._lock_stall_steps = 0
        self._nav_history = []
        self._target_progress = {}
        self._stale_target_until_step = {}
        self._last_planning_context = PlanningContext()
        self._force_vlm_scan_steps = 0
        self._goal_visible_vlm_steps = 0
        self._visual_scan_steps = 0
        self._scan_context = ScanContext()
        self._last_valid_vlm_report = None
        self._last_valid_vlm_step = -1
        self._last_valid_vlm_pose = None
        self._last_reachability_probe = {}
        self.perception.last_vlm_step = -1
        self.perception.last_vlm_reason = "reset"
        self.memory.set_frontier_probe(self._probe_reachable)

    def reset_keep_memory(self):
        super().reset_keep_memory()
        self.memory.reset_runtime()
        self.state_machine.reset_for_goal()
        self._sticky_nav_target = NavigationTarget(None, None)
        self._sticky_nav_steps = 0
        self._target_progress = {}
        self._stale_target_until_step = {}
        self._nav_history = []
        self._force_vlm_scan_steps = 0
        self._goal_visible_vlm_steps = 0
        self._visual_scan_steps = 0
        self._scan_context = ScanContext()
        self._last_valid_vlm_report = None
        self._last_valid_vlm_step = -1
        self._last_valid_vlm_pose = None
        self.perception.last_vlm_step = -1
        self.perception.last_vlm_reason = "reset_keep_memory"

    def set_pathfinder(self, pathfinder: Any) -> None:
        self._pathfinder = pathfinder
        self._nav_sim = pathfinder

    def set_navigation_context(self, sim: Any, origin_world: Any) -> None:
        self._pathfinder = sim
        self._nav_sim = sim
        self._nav_origin_world = np.asarray(origin_world, dtype=np.float32).copy()
        self.memory.set_frontier_probe(self._probe_reachable)

    def _probe_reachable(
        self,
        position_relative: np.ndarray,
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._nav_sim is None or self._nav_origin_world is None:
            return {
                "reachable": False,
                "reason": "reachability_unavailable",
                "target_position": np.asarray(position_relative, dtype=np.float32).round(4).tolist(),
                "target_node_id": node_id,
            }
        raw_position = np.asarray(position_relative, dtype=np.float32)
        try:
            probe = self._pathfinder_executor.probe(
                self._nav_sim,
                raw_position,
                self._nav_origin_world,
                target_node_id=node_id,
            )
            data = probe.to_dict()
        except Exception as exc:
            data = {
                "reachable": False,
                "reason": f"probe_error:{type(exc).__name__}",
                "target_position": raw_position.round(4).tolist(),
                "target_node_id": node_id,
            }
        target_position = data.get("target_position") or data.get("target_position_relative")
        data["candidate_position"] = raw_position.round(4).tolist()
        if target_position is not None:
            data["target_position"] = target_position
            if not np.allclose(raw_position, np.asarray(target_position, dtype=np.float32), atol=1e-3):
                data["snapped_from"] = raw_position.round(4).tolist()
        self._last_reachability_probe = dict(data)
        return data

    def set_new_goal(self, goal: GoalNode) -> None:
        self.goal_manager.set_goal(goal)
        self.perception.configure_goal(goal)
        self.state_machine.reset_for_goal()
        self._recent_positions.clear()
        self._recent_forward.clear()
        self._active_view = ActiveTopoView()
        self._recovery.reset()
        self._active_view_signature = None
        self._recovery_ref_pose = None
        self._recovery_ref_step = 0
        self._verify_success_count = 0
        self._verify_fail_count = 0
        self._last_needs_verify = False
        self._sticky_nav_target = NavigationTarget(None, None)
        self._sticky_nav_steps = 0
        self._target_progress = {}
        self._stale_target_until_step = {}
        self._nav_history = []
        self._force_vlm_scan_steps = 0
        self._goal_visible_vlm_steps = 0
        self._visual_scan_steps = 0
        self._scan_context = ScanContext()
        self._last_valid_vlm_report = None
        self._last_valid_vlm_step = -1
        self._last_valid_vlm_pose = None
        self.perception.last_vlm_step = -1
        self.perception.last_vlm_reason = "new_goal"

    def observe(self, obs: Dict[str, Any]) -> None:
        pos = np.asarray(obs.get("position", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        pos = pos.copy()
        pos[1] = 0.0
        self._pose = AgentPose(position=pos, heading=float(obs.get("heading", 0.0)))
        self._rgb = obs.get("rgb")
        self._rgb_embed = obs.get("rgb_embed")

        goal = self.goal_manager.current_goal()
        state_name = self.state_machine.state.mode.value
        self._clip_report = self.perception.run_clip(self._rgb_embed, step_id=self.step_id)
        need_vlm = self.perception.should_run_vlm(
            clip_report=self._clip_report,
            goal=goal,
            state=state_name,
            step_id=self.step_id,
        )
        vlm_scan_type = ""
        if (
            (self._scan_context.active() or self._force_vlm_scan_steps > 0 or self._goal_visible_vlm_steps > 0)
            and self.perception.vlm is not None
            and goal is not None
        ):
            need_vlm = True
            vlm_scan_type = (
                self._scan_context.scan_type
                if self._scan_context.active()
                else "frontier"
                if self._force_vlm_scan_steps > 0
                else "goal_visible_followup"
            )
            self.perception.last_vlm_reason = (
                f"{self._scan_context.scan_type}_scan"
                if self._scan_context.active()
                else "frontier_reached_scan"
                if self._force_vlm_scan_steps > 0
                else "goal_visible_followup"
            )
            if self._force_vlm_scan_steps > 0:
                self._force_vlm_scan_steps -= 1
            if self._goal_visible_vlm_steps > 0:
                self._goal_visible_vlm_steps -= 1
        self._vlm_report = (
            self.perception.run_vlm(
                self._rgb,
                goal=goal,
                clip_report=self._clip_report,
                state=state_name,
                step_id=self.step_id,
                scan_type=vlm_scan_type,
            )
            if need_vlm
            else None
        )
        if self._vlm_report is not None and bool(self._vlm_report.goal_visible):
            # Keep a short follow-up window after any positive sighting so
            # CONFIRM gets denser evidence instead of one-frame spikes.
            self._goal_visible_vlm_steps = max(self._goal_visible_vlm_steps, 5)
        if self._vlm_report is not None and self._vlm_report.is_full:
            self._last_valid_vlm_report = self._vlm_report
            self._last_valid_vlm_step = int(self.step_id)
            self._last_valid_vlm_pose = None if self._pose is None else self._pose.position.copy()
        self._cur_perception = self._perception_debug(need_vlm)

    def update_memory(self) -> None:
        if self._pose is None:
            return
        self.memory.update(
            pose=self._pose,
            rgb=self._rgb,
            rgb_embed=self._rgb_embed,
            clip_report=self._clip_report,
            vlm_report=self._vlm_report,
            goal=self.goal_manager.current_goal(),
            object_crop_encoder=self.perception.encode_object_crop,
            object_text_encoder=self.perception.encode_object_text,
            allow_frontier_generation=not self._scan_context.active(),
        )

    RECOVERY_BUDGET = 20
    RECOVERY_PROGRESS_RADIUS = 1.5

    def _update_recovery(self) -> None:
        """Escalate recovery when the active region is exhausted (no spatial
        search progress for a budget of steps). Escalation relaxes folding in
        the projector so the agent can reach next-relevant / previously folded
        regions instead of oscillating in a wrong room."""
        if self._pose is None:
            return
        pos = np.asarray(self._pose.position, dtype=np.float32)
        if self._recovery_ref_pose is None:
            self._recovery_ref_pose = pos.copy()
            self._recovery_ref_step = self.step_id
            return
        moved = float(np.linalg.norm(pos - self._recovery_ref_pose))
        if moved >= self.RECOVERY_PROGRESS_RADIUS:
            self._recovery_ref_pose = pos.copy()
            self._recovery_ref_step = self.step_id
            return
        no_progress = int(self.step_id - self._recovery_ref_step)
        current_region = ActiveTopoProjector._room_of_waypoint(
            self.topo_map, self._active_view.entry_waypoint
        )
        if self._recovery.maybe_escalate(
            no_progress_steps=no_progress,
            budget=self.RECOVERY_BUDGET,
            current_region=current_region,
        ):
            self._recovery_ref_step = self.step_id
            self._active_view_signature = None

    def _maybe_rebuild_active_view(self) -> None:
        """Event-driven ActiveTopoView refresh.

        Phase B: the view is observability-only (all nodes ACTIVE), so it never
        changes planner behavior. We still rebuild on meaningful events -- goal
        switch, node-count change, recovery-level change -- so the trace's
        rebuild_count and activity distribution are informative rather than
        rebuilding blindly every step.
        """
        goal = self.goal_manager.current_goal()
        signature = (
            _goal_id(goal),
            len(self.topo_map.structure_layer.nodes()),
            len(self.topo_map.navigation_layer.nodes()),
            self._recovery.level,
        )
        if signature == self._active_view_signature and self._active_view.node_infos:
            return
        self._active_view = self.active_projector.build(
            topo_map=self.topo_map,
            goal=goal,
            pose=self._pose,
            recovery_level=self._recovery.level,
            prev_view=self._active_view,
            goal_id=_goal_id(goal),
            step_id=self.step_id,
        )
        self._active_view_signature = signature

    def plan(self) -> NavigationTarget:
        if self._pose is None:
            self._last_nav_target = NavigationTarget(None, None, "idle")
            return self._last_nav_target
        self._update_recovery()
        self._maybe_rebuild_active_view()
        if (
            self.state_machine.state.mode.value == "CONFIRM"
            and self._last_nav_target.node_id is not None
            and self._last_nav_target.position is not None
        ):
            self._last_planning_context = self._build_planning_context()
            return self._last_nav_target
        if self._scan_context.active():
            planning_context = self._build_planning_context()
            # No position on a scan target: the scan is a pure in-place look-around
            # (handled by _scan_step -> _scan_sweep_action). Leaving position None
            # guarantees that even if the sweep is ever bypassed the executor turns
            # in place rather than re-navigating onto the already-reached anchor
            # (the v20 "target_reached" no-op spin).
            target = NavigationTarget(
                node_id=self._scan_context.anchor_node_id,
                position=None,
                target_type=self._scan_context.scan_type,
                reason=f"scan_active:{self._scan_context.scan_type}",
                structure_node_id=self._scan_context.structure_node_id,
            )
            self._last_nav_target = target
            self._last_planning_context = planning_context
            return target
        self._update_navigation_progress()
        planning_context = self._build_planning_context()
        goal = self.goal_manager.current_goal()
        committed = self._goal_object_commit_target(goal, planning_context)
        if committed is not None:
            committed = self._stabilize_navigation_target(committed, planning_context)
            self._last_intent = committed.source_intent
            self._last_nav_target = committed
            self._last_planning_context = planning_context
            self._record_navigation_selection(committed)
            return committed
        intents = self.structure_planner.build_intents(
            topo_map=self.topo_map,
            hypotheses=self.memory.hypothesis_pool,
            goal=goal,
            pose=self._pose,
            planning_context=planning_context,
            active_view=self._active_view,
        )
        top_intents = intents[:3] if intents else []
        target = self.navigation_planner.select_from_intents(
            topo_map=self.topo_map,
            intents=top_intents,
            pose=self._pose,
            planning_context=planning_context,
            active_view=self._active_view,
        )
        target = self._stabilize_navigation_target(target, planning_context)
        self._last_intent = target.source_intent or (top_intents[0] if top_intents else None)
        self._last_nav_target = target
        self._last_planning_context = planning_context
        self._record_navigation_selection(target)
        return target

    def _stabilize_navigation_target(
        self,
        target: NavigationTarget,
        planning_context: Optional[PlanningContext] = None,
    ) -> NavigationTarget:
        """Keep a still-valid navigation target briefly to avoid frontier oscillation."""
        if self._pose is None:
            return target
        sticky = self._sticky_nav_target
        sticky_intent = sticky.source_intent.intent_type if sticky.source_intent else None
        # Commit longer to an exploration frontier so we actually reach it instead
        # of re-picking among the ~5 near-equivalent local frontiers every step
        # (which manifests as circling in place).
        sticky_cap = 26 if sticky_intent in ("explore", "hypothesis") else 18
        keep = False
        if sticky.node_id and sticky.position is not None and self._sticky_nav_steps < sticky_cap:
            node = self.topo_map.get_node(str(sticky.node_id))
            dist = float(np.linalg.norm(sticky.position - self._pose.position))
            new_intent = target.source_intent.intent_type if target.source_intent else None
            stale = bool(planning_context and sticky.node_id in planning_context.stale_target_ids)
            same_task = (
                sticky_intent == new_intent
                or sticky_intent == "explore" and new_intent in {"explore", "hypothesis"}
                or sticky_intent == "hypothesis" and new_intent == "explore"
            )
            keep = (
                node is not None
                and node.node_type in self.topo_map._NAV_NODE_TYPES
                and not node.attributes.get("blocked", False)
                and not node.attributes.get("consumed", False)
                and dist > NAV_REACH_RADIUS
                and same_task
                and not stale
            )
        if keep:
            self._sticky_nav_steps += 1
            sticky.reason = f"{sticky.reason}; sticky={self._sticky_nav_steps}"
            return sticky
        self._sticky_nav_target = target
        self._sticky_nav_steps = 0
        return target

    def _record_history_entry(
        self,
        *,
        target: Optional[NavigationTarget] = None,
        node_id: Optional[str] = None,
        target_type: str = "",
        structure_node_id: Optional[str] = None,
        distance: Optional[float] = None,
        progress_delta: float = 0.0,
        event: str,
    ) -> None:
        if target is not None:
            node_id = target.node_id
            target_type = target.target_type
            structure_node_id = target.structure_node_id
            if distance is None:
                distance = self._target_distance(target)
        self._nav_history.append(NavigationHistoryEntry(
            step_id=int(self.step_id),
            node_id=node_id,
            target_type=str(target_type or ""),
            structure_node_id=structure_node_id,
            distance=None if distance is None else float(distance),
            progress_delta=float(progress_delta),
            event=event,
        ))
        self._nav_history = self._nav_history[-160:]

    def _target_distance(self, target: NavigationTarget) -> Optional[float]:
        if self._pose is None or target.position is None:
            return None
        delta = np.asarray(target.position, dtype=np.float32) - self._pose.position
        return float(np.linalg.norm(delta))

    def _record_navigation_selection(self, target: NavigationTarget) -> None:
        if not target.node_id:
            return
        self._record_history_entry(target=target, event="selected")

    def _update_navigation_progress(self) -> None:
        target = self._last_nav_target
        if self._pose is None or not target.node_id or target.position is None:
            return
        target_id = str(target.node_id)
        dist = self._target_distance(target)
        if dist is None:
            return
        state = self._target_progress.get(target_id)
        if state is None:
            self._target_progress[target_id] = {
                "best_distance": dist,
                "last_distance": dist,
                "stale_steps": 0,
            }
            return
        best = float(state.get("best_distance", dist))
        progress_delta = best - dist
        if progress_delta > 0.05:
            state["best_distance"] = dist
            state["stale_steps"] = 0
        else:
            state["stale_steps"] = int(state.get("stale_steps", 0) or 0) + 1
        state["last_distance"] = dist
        state["last_progress_delta"] = progress_delta
        stale_steps = int(state.get("stale_steps", 0) or 0)
        if stale_steps >= 12:
            until = int(self.step_id + 20)
            if int(self._stale_target_until_step.get(target_id, 0) or 0) < self.step_id:
                self._record_history_entry(
                    target=target,
                    distance=dist,
                    progress_delta=progress_delta,
                    event="stale_no_progress",
                )
            self._stale_target_until_step[target_id] = until
            state["stale_steps"] = 0

    def _build_planning_context(self) -> PlanningContext:
        current_step = int(self.step_id)
        self._stale_target_until_step = {
            str(node_id): int(until)
            for node_id, until in self._stale_target_until_step.items()
            if int(until) > current_step
        }
        recent_target_counts: Dict[str, int] = {}
        for entry in self._nav_history[-32:]:
            if entry.event != "selected" or not entry.node_id:
                continue
            node_id = str(entry.node_id)
            recent_target_counts[node_id] = recent_target_counts.get(node_id, 0) + 1
        current_waypoint = self._current_waypoint_id()
        return PlanningContext(
            recent_target_counts=recent_target_counts,
            stale_target_ids=dict(self._stale_target_until_step),
            current_waypoint_id=current_waypoint,
            trajectory=self._trajectory_ids(),
            recent_failures=list(self.topo_map.failed_target_history[-8:]),
        )

    def _current_waypoint_id(self) -> Optional[str]:
        if self._pose is None:
            return None
        best = None
        best_dist = float("inf")
        for node in self.topo_map.navigation_layer.nodes():
            if node.node_type != NodeType.WAYPOINT_VISITED:
                continue
            dist = float(np.linalg.norm(node.position - self._pose.position))
            if dist < best_dist:
                best = node
                best_dist = dist
        return best.node_id if best is not None else None

    def _trajectory_ids(self) -> List[str]:
        visited = [
            node for node in self.topo_map.navigation_layer.nodes()
            if node.node_type == NodeType.WAYPOINT_VISITED
        ]
        visited.sort(
            key=lambda node: (
                int(node.attributes.get("last_reached_step", node.step_id) or node.step_id),
                node.node_id,
            )
        )
        return [node.node_id for node in visited[-32:]]

    def _start_scan(
        self,
        scan_type: str,
        *,
        anchor_node_id: Optional[str],
        structure_node_id: Optional[str],
        steps: int = 8,
    ) -> None:
        self._scan_context = ScanContext(
            scan_type=scan_type,
            anchor_node_id=anchor_node_id,
            structure_node_id=structure_node_id,
            remaining_steps=int(steps),
            fresh_vlm_required=True,
            started_step=int(self.step_id),
        )
        self._force_vlm_scan_steps = max(self._force_vlm_scan_steps, int(steps))
        self._visual_scan_steps = max(self._visual_scan_steps, int(steps))

    def _scan_step(self, target: NavigationTarget) -> Optional[Dict[str, Any]]:
        """Return the next look-around sweep action, or None if no scan is active.

        Self-terminating: `_scan_sweep_action` decrements the remaining steps and
        finishes the scan at 0, so an active scan always emits a real turn and
        ends within `remaining_steps` frames. Aborts early (returns None) if the
        goal became visible or control took over (CONFIRM/STOP)."""
        if not self._scan_context.active():
            return None
        if self._vlm_report is not None and bool(self._vlm_report.goal_visible):
            self._scan_context = ScanContext()
            self._visual_scan_steps = 0
            return None
        if self.state_machine.state.mode.value in {"CONFIRM", "STOP"}:
            self._scan_context = ScanContext()
            self._visual_scan_steps = 0
            return None
        return self._scan_sweep_action(target)

    def _scan_sweep_action(self, target: NavigationTarget) -> Dict[str, Any]:
        total = 8
        elapsed = max(0, total - int(self._scan_context.remaining_steps))
        scan_type = self._scan_context.scan_type
        anchor_node_id = self._scan_context.anchor_node_id
        # Observe current heading first, then sweep left and right over several
        # simulator frames so side-facing objects can enter the camera.
        pattern = [
            "turn_left",
            "turn_left",
            "turn_left",
            "turn_right",
            "turn_right",
            "turn_right",
            "turn_right",
            "turn_left",
        ]
        low_action = pattern[min(elapsed, len(pattern) - 1)]
        self._scan_context.remaining_steps = max(0, int(self._scan_context.remaining_steps) - 1)
        self._visual_scan_steps = max(0, int(self._scan_context.remaining_steps))
        ctx_snapshot = self._scan_context.to_dict()
        if self._scan_context.remaining_steps <= 0:
            self._finish_scan_without_target()
        return {
            "action": low_action,
            "reason": f"{scan_type or 'visual'}_sweep",
            "target_node_id": anchor_node_id or target.node_id,
            "target_type": target.target_type,
            "scan_context": ctx_snapshot,
        }

    def _finish_scan_without_target(self) -> None:
        ctx = self._scan_context
        if not ctx.scan_type or ctx.failure_recorded:
            self._scan_context = ScanContext()
            return
        if ctx.scan_type in {"object_anchor", "hypothesis_anchor", "portal_anchor"}:
            structure_id = ctx.structure_node_id
            anchor_id = ctx.anchor_node_id
            if structure_id:
                node = self.topo_map.get_node(str(structure_id))
                if node is not None:
                    bad = [str(x) for x in node.attributes.get("bad_anchor_waypoint_ids", []) or []]
                    if anchor_id and str(anchor_id) not in bad:
                        bad.append(str(anchor_id))
                    node.attributes["bad_anchor_waypoint_ids"] = bad[-12:]
                    node.attributes["failed_count"] = int(node.attributes.get("failed_count", 0) or 0) + 1
                    node.attributes["last_failed_step"] = self.topo_map.current_step
                    node.attributes["failure_reason"] = f"{ctx.scan_type}_no_visual_confirmation"
                    # Arriving at the observation anchor and sweeping without ever
                    # re-acquiring the goal strongly implies the detection was
                    # mislocalised: demote the object (cooldown + decay) and drop
                    # any lock so the agent stops re-committing to it and resumes
                    # exploration instead of parking on the dead anchor.
                    if ctx.scan_type == "object_anchor":
                        self._demote_goal_object(node)
                        if str(self._locked_object_id or "") == str(structure_id):
                            self._release_object_lock()
            if anchor_id:
                self._stale_target_until_step[str(anchor_id)] = int(self.step_id + 20)
            self._record_history_entry(
                node_id=anchor_id,
                target_type=ctx.scan_type,
                structure_node_id=structure_id,
                event="scan_failed_no_target",
            )
        self._sticky_nav_target = NavigationTarget(None, None)
        self._sticky_nav_steps = 0
        ctx.failure_recorded = True
        self._scan_context = ScanContext()

    def _record_pose_for_arrival(self) -> None:
        if self._pose is not None:
            pos = self._pose.position
            self._recent_positions.append((float(pos[0]), float(pos[2])))

    def _compute_arrived(self) -> bool:
        """Scale-fair arrival: recent forward attempts but pose stopped advancing.

        Object-size independent (unlike absolute bbox area/height): it keys on the
        agent's own quantized pose no longer moving despite trying to advance. The
        bbox ground-contact corroboration lives in the StopVerifier.
        """
        if len(self._recent_positions) < ARRIVAL_STALL_WINDOW:
            return False
        if sum(1 for f in self._recent_forward if f) < ARRIVAL_MIN_FORWARD:
            return False
        pts = list(self._recent_positions)
        spread = max(
            math.dist(pts[i], pts[j])
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
        )
        return spread < ARRIVAL_STALL_EPS

    def _goal_relevant_objects(self, goal: Optional[GoalNode]) -> List[Any]:
        """Goal-relevant, non-folded OBJECT nodes in memory.

        Relevance is recomputed against the CURRENT goal (stored target_relevance
        is frozen at creation and would leak across GOAT goals).
        """
        out: List[Any] = []
        if goal is None:
            return out
        for node in self.topo_map.object_memory.get_all():
            if node.attributes.get("folded") or node.attributes.get("blocked"):
                continue
            if node.position is None:
                continue
            # Skip candidates demoted after an unconfirmable visit (cooldown).
            cooldown = int(node.attributes.get("goal_search_cooldown_until", -1) or -1)
            if cooldown >= int(self.step_id):
                continue
            if GoalRelevanceScorer.label_relevance(node.label, goal) < GOAL_MATCH_MIN_RELEVANCE:
                continue
            out.append(node)
        return out

    def _horizontal_distance(self, pos) -> float:
        return math.hypot(
            float(pos[0] - self._pose.position[0]),
            float(pos[2] - self._pose.position[2]),
        )

    def _release_object_lock(self) -> None:
        self._locked_object_id = None
        self._lock_best_dist = None
        self._lock_stall_steps = 0

    def _observation_anchor(self, node: Any) -> Optional[Any]:
        """Return the real waypoint the object was OBSERVED from (a reachable,
        already-visited pose), NOT the estimated stand-off approach point.

        Navigating to where we actually saw the object -- then closing the last
        gap with the live visual servo (CONFIRM) -- avoids trusting the monocular
        position estimate for path planning. Falls back to the generic structure
        anchor (which may include the approach point) only if no usable observed
        waypoint exists."""
        attrs = node.attributes
        bad = {str(x) for x in attrs.get("bad_anchor_waypoint_ids", []) or []}
        cand_ids: List[str] = []
        best = attrs.get("best_anchor_waypoint_id")
        if best:
            cand_ids.append(str(best))
        observed = attrs.get("observed_from_waypoint_ids")
        if isinstance(observed, list):
            cand_ids.extend(str(x) for x in reversed(observed))
        anchor = attrs.get("anchor_waypoint_id")
        if anchor:
            cand_ids.append(str(anchor))
        for cid in cand_ids:
            if cid in bad:
                continue
            c = self.topo_map.get_node(cid)
            if (
                c is not None
                and c.node_type in self.topo_map._NAV_NODE_TYPES
                and not c.attributes.get("blocked", False)
            ):
                return c
        return self.topo_map.navigation_target_for_structure_node(node.node_id)

    def _closer_object_anchor(self, node: Any, observation_anchor: Any) -> Optional[Any]:
        """The object's stand-off approach point (a navmesh pose set back from the
        estimated object position). Used ONLY after the trusted observation
        waypoint has been reached, so we keep edging closer for a confirming view
        instead of idling at the reached waypoint. If it is unreachable/bad the
        progress-based lock release will demote the candidate."""
        ap_id = node.attributes.get("approach_point_id")
        if not ap_id or str(ap_id) == str(observation_anchor.node_id):
            return None
        bad = {str(x) for x in node.attributes.get("bad_anchor_waypoint_ids", []) or []}
        if str(ap_id) in bad:
            return None
        ap = self.topo_map.get_node(str(ap_id))
        if (
            ap is None
            or ap.node_type not in self.topo_map._NAV_NODE_TYPES
            or ap.attributes.get("blocked", False)
            or ap.attributes.get("consumed", False)
        ):
            return None
        return ap

    def _demote_goal_object(self, node: Any) -> None:
        """General negative evidence: we reached this candidate but could not
        confirm it as the goal, so exclude it from goal candidates for a cooldown
        and decay its existence confidence. Not object/goal specific and not
        permanent -- after the cooldown it can be reconsidered."""
        node.attributes["goal_search_cooldown_until"] = int(self.step_id) + OBJECT_SEARCH_COOLDOWN
        node.confidence = max(0.0, float(node.confidence) * 0.5)

    def _lock_candidates(self, goal: GoalNode) -> List[Dict[str, Any]]:
        """Unified lock candidates: confirmed goal OBJECTS and goal HYPOTHESES,
        each scored by the same task-conditioned confidence. Only candidates
        clearing ``LOCK_CTASK_THRESHOLD`` are eligible for the sticky lock."""
        cands: List[Dict[str, Any]] = []
        for node in self._goal_relevant_objects(goal):
            ct = object_c_task(node, goal)
            if ct >= LOCK_CTASK_THRESHOLD:
                cands.append({"kind": "object", "key": node.node_id,
                              "pos": node.position, "c_task": ct, "node": node})
        pool = getattr(self.memory, "hypothesis_pool", None)
        if pool is not None:
            gid = _goal_id(goal)
            for hyp in pool.get_active(goal_id=gid, include_needs_verify=True):
                if hyp.anchor_node_id is None:
                    continue
                anchor = self.topo_map.get_node(hyp.anchor_node_id)
                if anchor is None:
                    continue
                ct = hypothesis_c_task(hyp, goal)
                if ct >= LOCK_CTASK_THRESHOLD:
                    cands.append({"kind": "hypothesis", "key": f"hyp:{hyp.id}",
                                  "pos": anchor.position, "c_task": ct,
                                  "hyp": hyp, "anchor": anchor})
        return cands

    def _demote_candidate(self, cand: Dict[str, Any]) -> None:
        if cand["kind"] == "object":
            self._demote_goal_object(cand["node"])
            return
        pool = getattr(self.memory, "hypothesis_pool", None)
        if pool is not None:
            try:
                pool.reject(cand["hyp"].id, "search_exhausted")
            except Exception:
                pass

    def _commit_intent_for(self, cand: Dict[str, Any]) -> Optional[SearchIntent]:
        if cand["kind"] == "object":
            node = cand["node"]
            nav = self._observation_anchor(node)
            if nav is None:
                return None
            # Fix 3: don't freeze the target at the reached observation waypoint.
            # Once we're essentially standing at the (trusted) observed pose but
            # still haven't confirmed the object, retarget to a closer stand-off
            # approach point so navigation keeps pulling us in for a better view
            # rather than idling. Until the observed pose is reached we keep the
            # observation waypoint (avoids trusting the position estimate early).
            anchor = nav
            if self._pose is not None and self._horizontal_distance(nav.position) <= NAV_REACH_RADIUS * 2.5:
                closer = self._closer_object_anchor(node, nav)
                if closer is not None:
                    anchor = closer
            return SearchIntent(
                intent_type="object",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=anchor.node_id,
                target_position=anchor.position.copy(),
                score=10.0,
                reason=(
                    f"commit_locked_object:{node.node_id}; anchor={anchor.node_id}; "
                    f"c_task={cand['c_task']:.2f}; stall={self._lock_stall_steps}"
                ),
                preferred_room=node.attributes.get("room_label"),
                evidence_refs=[node.node_id],
            )
        hyp = cand["hyp"]
        anchor = cand["anchor"]
        return SearchIntent(
            intent_type="hypothesis",
            target_label=hyp.label,
            preferred_anchor_waypoint_id=anchor.node_id,
            target_position=anchor.position.copy(),
            score=10.0,
            reason=(
                f"commit_locked_hypothesis:{hyp.id}; "
                f"c_task={cand['c_task']:.2f}; stall={self._lock_stall_steps}"
            ),
            evidence_refs=[hyp.id],
        )

    def _goal_object_commit_target(
        self,
        goal: Optional[GoalNode],
        planning_context: Optional[PlanningContext],
    ) -> Optional[NavigationTarget]:
        """Sticky local commitment to a confident goal candidate (general exploit).

        A candidate is either a confirmed goal OBJECT or a goal HYPOTHESIS whose
        task-conditioned confidence ``c_task`` clears ``LOCK_CTASK_THRESHOLD``.
        The agent LOCKS navigation onto the candidate's OBSERVATION anchor (the
        real waypoint it was seen from, not the estimated stand-off) and holds the
        lock across frames -- regardless of per-frame visibility -- so competing
        objects / hypotheses / frontiers cannot hijack the target and spin the
        agent in place. Confirmed objects take priority over hypotheses; among
        objects the nearest wins, among hypotheses the most confident. The last
        gap is then closed by the CONFIRM visual servo (walk toward the live
        bbox, correct each frame). If the agent cannot make progress toward the
        observation anchor for LOCK_NO_PROGRESS_STEPS while the goal is not being
        actively servoed to (mislocalised / unreachable / false detection), the
        candidate is demoted (object -> cooldown+decay, hypothesis -> reject) and
        the lock released so exploration resumes. Uncertain candidates below the
        bar fall through to normal intent competition, where c_task still
        arbitrates explore/exploit.
        """
        if goal is None or self._pose is None:
            self._release_object_lock()
            return None
        cands = self._lock_candidates(goal)
        if not cands:
            self._release_object_lock()
            return None
        locked = None
        if self._locked_object_id is not None:
            locked = next((c for c in cands if c["key"] == self._locked_object_id), None)
        if locked is None:
            objs = [c for c in cands if c["kind"] == "object"]
            if objs:
                locked = min(objs, key=lambda c: self._horizontal_distance(c["pos"]))
            else:
                locked = max(cands, key=lambda c: c["c_task"])
            self._locked_object_id = locked["key"]
            self._lock_best_dist = None
            self._lock_stall_steps = 0
        intent = self._commit_intent_for(locked)
        if intent is None:
            self._demote_candidate(locked)
            self._release_object_lock()
            return None
        # Progress-based exhaustion (observation-driven, not estimate-driven):
        # measure the agent's horizontal distance to the OBSERVATION anchor we are
        # navigating to. If we keep improving, hold the lock. If the goal was seen
        # recently we're actively servoing to it (CONFIRM), so never count that as
        # stalled. Otherwise, if the distance stops improving for
        # LOCK_NO_PROGRESS_STEPS, the anchor is unreachable / the detection is
        # false -> demote and resume exploration.
        last_seen = int(self.state_machine.state.last_goal_seen_step)
        seen_recently = last_seen >= 0 and int(self.step_id) - last_seen <= 5
        anchor_pos = intent.target_position
        d = self._horizontal_distance(anchor_pos) if anchor_pos is not None else float("inf")
        if seen_recently or self._lock_best_dist is None or d < self._lock_best_dist - 0.15:
            self._lock_best_dist = d
            self._lock_stall_steps = 0
        else:
            self._lock_stall_steps += 1
            if self._lock_stall_steps >= LOCK_NO_PROGRESS_STEPS:
                self._demote_candidate(locked)
                self._release_object_lock()
                return None
        target = self.navigation_planner.select_from_intents(
            topo_map=self.topo_map,
            intents=[intent],
            pose=self._pose,
            planning_context=planning_context,
            active_view=self._active_view,
        )
        if target.node_id is not None:
            return target
        # No navigable route to the locked candidate right now -> drop it for a
        # cooldown / reject so we don't keep re-selecting an unreachable one.
        self._demote_candidate(locked)
        self._release_object_lock()
        return None

    def act(self, plan_output: NavigationTarget) -> Dict[str, Any]:
        self._record_pose_for_arrival()
        # Deliberate look-around scan handled FIRST and self-terminating: it
        # decrements every step it is active and finishes at 0, so it can never
        # deadlock the agent parked on an already-reached anchor (the v20 rack
        # sat 60 steps "scan_active" but never actually turned). Running it before
        # the state machine also means no downstream transition can shadow it.
        scan_action = self._scan_step(plan_output)
        if scan_action is not None:
            self._recent_forward.append(False)
            scan_action["debug"] = self._debug_snapshot()
            scan_action.setdefault("target_node_id", plan_output.node_id)
            scan_action.setdefault("target_type", plan_output.target_type)
            return scan_action
        # Observation-based arrival: the agent kept trying to move forward toward
        # the (visible, centered) goal but its own pose stopped advancing -- it is
        # physically as close as the navmesh allows. This is decided from actual
        # motion, NOT from the estimated object position, so a mislocalised object
        # node no longer dictates where we stop. The STOP gate still requires the
        # goal to be visible + semantically relevant (see StopVerifier), so a
        # pose-stall while merely blocked far from the target never stops.
        arrived = self._compute_arrived()
        control_report = self._vlm_report
        fresh_report = self._vlm_report is not None and self._vlm_report.is_full
        if control_report is None and self._last_valid_vlm_report is not None:
            if int(self.step_id) - int(self._last_valid_vlm_step) <= 2:
                control_report = self._last_valid_vlm_report
        obj_node_id = (
            plan_output.structure_node_id
            if plan_output.source_intent and plan_output.source_intent.intent_type == "object"
            else None
        )
        needs_verify = bool(obj_node_id) and self._active_view.is_verify(obj_node_id)
        self._last_needs_verify = needs_verify
        detection_score = float(getattr(control_report, "goal_match_confidence", 0.0) or 0.0)

        def _on_verify_success() -> None:
            self.memory.record_object_verification(
                obj_node_id, success=True, detection_score=detection_score,
            )
            self._verify_success_count += 1
            self._active_view_signature = None  # force reprojection: C changed

        def _on_verify_failure() -> None:
            self.memory.record_object_verification(
                obj_node_id, success=False, detection_score=detection_score,
            )
            self._verify_fail_count += 1
            self._active_view_signature = None

        action = self.state_machine.act(
            step_id=self.step_id,
            target=plan_output,
            report=control_report,
            goal=self.goal_manager.current_goal(),
            fresh_report=fresh_report,
            needs_verify=needs_verify,
            on_verify_success=_on_verify_success,
            on_verify_failure=_on_verify_failure,
            arrived=arrived,
        )
        # (Scan sweep is handled at the top of act() now; see _scan_step.)
        # Arrival = "kept trying to step FORWARD but pose stopped advancing" (i.e.
        # blocked right at the target). Only genuine forward low-actions may count:
        # a `navigate` intent or an in-place `turn_*` must NOT, otherwise merely
        # spinning to centre at an already-reached waypoint reads as arrival (the
        # v18 rack regression stopped ~3 m short exactly this way).
        self._recent_forward.append(str(action.get("action", "")) == "move_forward")
        action["debug"] = self._debug_snapshot()
        if "target_node_id" not in action:
            action["target_node_id"] = plan_output.node_id
        if "target_type" not in action:
            action["target_type"] = plan_output.target_type
        return action

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_id += 1
        self._step_count += 1
        self.topo_map.step()
        self.observe(obs)
        self.update_memory()
        target = self.plan()
        return self.act(target)

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> bool:
        target = self._last_nav_target
        if target_node_id is not None and target.node_id != target_node_id:
            return False
        if event == "target_reached":
            self._sticky_nav_target = NavigationTarget(None, None)
            self._sticky_nav_steps = 0
            node = self.topo_map.get_node(str(target.node_id)) if target.node_id else None
            if node is not None:
                node.attributes["visit_count"] = int(node.attributes.get("visit_count", 0) or 0) + 1
                node.attributes["last_reached_step"] = self.topo_map.current_step
                if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                    # No turn-sweep at plain frontiers: they are dense (~2.3 m
                    # apart), so a full 8-step sweep at every hop reads as in-place
                    # spinning and stalls forward exploration. We already get a
                    # fresh VLM report each nav step; the sweep is reserved for
                    # portals / object / region anchors where a deliberate look
                    # around actually pays off.
                    self.topo_map.promote_frontier_to_visited(node.node_id)
                elif target.target_type in {"object_anchor", "object_anchor_offset"}:
                    self._start_scan(
                        "object_anchor",
                        anchor_node_id=node.node_id,
                        structure_node_id=target.structure_node_id,
                        steps=8,
                    )
                elif target.target_type in {"hypothesis_waypoint", "hypothesis_offset"}:
                    self._start_scan(
                        "hypothesis_anchor",
                        anchor_node_id=node.node_id,
                        structure_node_id=target.structure_node_id,
                        steps=8,
                    )
                elif target.target_type == "portal_waypoint":
                    self._start_scan(
                        "portal_anchor",
                        anchor_node_id=node.node_id,
                        structure_node_id=target.structure_node_id,
                        steps=8,
                    )
            if target.node_id:
                self._target_progress.pop(str(target.node_id), None)
                self._stale_target_until_step.pop(str(target.node_id), None)
            self._record_history_entry(target=target, event="target_reached")
            return True
        if event in {"navigation_failed", "unreachable"}:
            self._sticky_nav_target = NavigationTarget(None, None)
            self._sticky_nav_steps = 0
            return self.report_navigation_failed(target_node_id, unreachable=(event == "unreachable"))
        return False

    def report_navigation_failed(self, target_node_id: Optional[str] = None, *, unreachable: bool = False) -> bool:
        target = self._last_nav_target
        if target_node_id is not None and target.node_id != target_node_id:
            return False
        if target.source_intent and target.source_intent.intent_type == "object" and target.structure_node_id:
            obj = self.topo_map.get_node(target.structure_node_id)
            if obj is not None:
                bad = [str(x) for x in obj.attributes.get("bad_anchor_waypoint_ids", []) or []]
                if target.node_id and target.node_id not in bad:
                    bad.append(target.node_id)
                obj.attributes["bad_anchor_waypoint_ids"] = bad[-8:]
                obj.attributes["failed_count"] = int(obj.attributes.get("failed_count", 0) or 0) + 1
                if unreachable:
                    obj.attributes["unreachable_count"] = int(obj.attributes.get("unreachable_count", 0) or 0) + 1
                # Dead-target circuit breaker: repeated external nav failures on
                # the same object (harness escalates a reached-but-stuck target to
                # navigation_failed) mean we keep re-committing to a target we can
                # neither approach nor confirm. Demote it (cooldown + decay) and
                # release the lock so exploration resumes instead of spinning.
                if int(obj.attributes.get("failed_count", 0) or 0) >= 3:
                    self._demote_goal_object(obj)
                    if str(self._locked_object_id or "") == str(obj.node_id):
                        self._release_object_lock()
        if (
            target.target_type == "portal_waypoint"
            and target.source_intent
            and target.source_intent.intent_type == "portal"
            and target.structure_node_id
        ):
            self.topo_map.mark_structure_target_failed(
                target.structure_node_id,
                anchor_node_id=target.node_id,
                target_type=target.target_type,
                unreachable=unreachable,
                reason="external_navigation_failed",
            )
        marked = self.topo_map.mark_navigation_target_failed(
            target.node_id,
            target_type=target.target_type,
            unreachable=unreachable,
            reason="external_navigation_failed",
        )
        if target.node_id:
            self._target_progress.pop(str(target.node_id), None)
            self._stale_target_until_step[str(target.node_id)] = int(self.step_id + 20)
        self._record_history_entry(
            target=target,
            event="unreachable" if unreachable else "navigation_failed",
        )
        return marked

    def _debug_snapshot(self) -> Dict[str, Any]:
        state_debug = self.state_machine.debug()
        intents = [self.structure_planner.debug_intent(intent) for intent in self.structure_planner.last_intents[:8]]
        nav_props = [self.navigation_planner.debug_proposal(p) for p in self.navigation_planner.last_proposals[:8]]
        selected_intent = self.structure_planner.debug_intent(self._last_intent) if self._last_intent else {}
        selected_target = self.navigation_planner.debug_target(self._last_nav_target)
        map_audit = self._build_map_audit_context(nav_props)
        debug = {
            **state_debug,
            "step_id": self.step_id,
            "perception": dict(self._cur_perception or self._perception_debug(False)),
            "memory": self._memory_debug(),
            "structure_intents": intents,
            "selected_intent": selected_intent,
            "navigation_proposals": nav_props,
            "selected_navigation_target": selected_target,
            "navigation_sticky": {
                "node_id": self._sticky_nav_target.node_id,
                "steps": int(self._sticky_nav_steps),
            },
            "visual_sweep": {
                "remaining_steps": int(self._visual_scan_steps),
                "force_vlm_scan_steps": int(self._force_vlm_scan_steps),
                "goal_visible_vlm_steps": int(self._goal_visible_vlm_steps),
            },
            "scan_context": self._scan_context.to_dict(),
            "frontier_rejected": list(getattr(self.memory, "last_frontier_rejected", [])),
            "reachability_probe": dict(self._last_reachability_probe),
            "planning_context": {
                "recent_target_counts": dict(self._last_planning_context.recent_target_counts),
                "stale_target_ids": dict(self._last_planning_context.stale_target_ids),
                "current_waypoint_id": self._last_planning_context.current_waypoint_id,
                "trajectory": list(self._last_planning_context.trajectory),
                "recent_failures": list(self._last_planning_context.recent_failures),
            },
            "map_audit": map_audit,
            "stop_verifier": dict(self.state_machine.stop_verifier.last_debug),
            "active_topo": self._active_view.stats(),
            "active_topo_nodes": self._active_view.state_buckets(),
            "verify_events": {
                "needs_verify_now": bool(self._last_needs_verify),
                "success_count": int(self._verify_success_count),
                "fail_count": int(self._verify_fail_count),
            },
            "recovery": {
                "level": int(self._recovery.level),
                "exhausted_regions": len(self._recovery.exhausted_regions),
                "steps_since_progress": int(self.step_id - self._recovery_ref_step),
            },
            "target_node_id": self._last_nav_target.node_id,
            "target_type": self._last_nav_target.target_type,
        }
        self._last_debug = debug
        return debug

    def _perception_debug(self, need_vlm: bool) -> Dict[str, Any]:
        report = self._vlm_report or self._clip_report
        raw = report.raw if isinstance(report.raw, dict) else {}
        raw_objects = raw.get("objects", [])
        if not isinstance(raw_objects, list):
            raw_objects = []
        normalized_objects = [obj.to_dict() for obj in report.objects[:8]]
        raw_text = str(raw.get("_raw_text_truncated", ""))
        parse_status = {
            "query_ok": bool(raw.get("_query_ok", True)),
            "parse_ok": bool(raw.get("_parse_ok", True)),
            "error": str(raw.get("_error", "")),
            "raw_has_json": bool(raw_objects or "{" in raw_text or bool(raw.get("_parse_ok", False))),
            "raw_object_count": len(raw_objects),
            "normalized_object_count": len(report.objects),
            "objects_with_bbox": sum(1 for obj in normalized_objects if obj.get("bbox")),
            "objects_with_display_bbox": sum(
                1
                for obj in normalized_objects
                if obj.get("display_bbox") or (obj.get("attributes") or {}).get("display_bbox")
            ),
        }
        return {
            "room_label": self._clip_report.room_label,
            "room_confidence": float(self._clip_report.room_confidence),
            "best_goal_sim": float(self._clip_report.best_goal_sim),
            "best_landmark_sim": float(self._clip_report.best_landmark_sim),
            "room_scores": _score_items(self._clip_report.room_scores),
            "goal_scores": _score_items(self._clip_report.goal_scores),
            "landmark_scores": _score_items(self._clip_report.landmark_scores),
            "need_vlm": bool(need_vlm),
            "fresh_vlm": bool(self._vlm_report is not None and self._vlm_report.is_full),
            "vlm_reason": self.perception.last_vlm_reason,
            "vlm_goal_visible": bool(report.goal_visible),
            "vlm_object_count": len(report.objects),
            "vlm_objects": normalized_objects,
            "vlm_parse_status": parse_status,
            "vlm_goal_reason": str(raw.get("goal_reason", "")),
            "vlm_scene_summary": str(raw.get("scene_summary", "")),
            "vlm_recommended_action": str(raw.get("recommended_action", "")),
            "vlm_raw_text": raw_text,
        }

    def _memory_debug(self) -> Dict[str, Any]:
        structure_nodes = self.topo_map.structure_layer.nodes()
        navigation_nodes = self.topo_map.navigation_layer.nodes()
        base = self.memory.to_debug()
        base.update({
            "structure_layer": {
                "total": len(structure_nodes),
                "rooms": sum(1 for n in structure_nodes if n.node_type == NodeType.ROOM),
                "objects": sum(1 for n in structure_nodes if n.node_type == NodeType.OBJECT),
                "landmarks": sum(1 for n in structure_nodes if n.node_type == NodeType.LANDMARK),
                "portals": sum(1 for n in structure_nodes if n.node_type == NodeType.PORTAL),
            },
            "navigation_layer": {
                "total": len(navigation_nodes),
                "visited_waypoints": sum(1 for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_VISITED),
                "frontiers": sum(1 for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_FRONTIER),
                "candidate_waypoints": sum(1 for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_CANDIDATE),
                "approach_waypoints": sum(1 for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_APPROACH),
            },
            "cross_layer_links": len(self.topo_map.cross_layer.edges()),
            "frontier_rejected": list(getattr(self.memory, "last_frontier_rejected", [])),
            "frontier_probes": list(getattr(self.memory, "last_frontier_probes", [])),
        })
        return base

    def _build_map_audit_context(self, current_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        option_by_id = {
            str(item.get("target_node_id")): item
            for item in current_options
            if item.get("target_node_id") is not None
        }
        map_edges = []
        for src, dst, data in self.topo_map.navigation_layer.edges():
            map_edges.append({
                "source": src,
                "target": dst,
                "type": data.get("edge_type", ""),
                "edge_types": list(data.get("edge_types", [])),
                "blocked": bool(data.get("blocked", False)),
                "weight": float(data.get("weight", 1.0)),
            })
        supplementary = []
        for node in self.topo_map.navigation_layer.nodes():
            if node.node_type not in (
                NodeType.WAYPOINT_FRONTIER,
                NodeType.WAYPOINT_CANDIDATE,
                NodeType.WAYPOINT_APPROACH,
            ):
                continue
            if node.attributes.get("blocked", False) or node.attributes.get("consumed", False):
                continue
            option = option_by_id.get(node.node_id, {})
            stale_until = self._stale_target_until_step.get(node.node_id)
            supplementary.append({
                "node_id": node.node_id,
                "type": node.node_type.value,
                "position": node.position.round(4).tolist(),
                "score": float(option.get("score", node.confidence)),
                "confidence": float(node.confidence),
                "failed_count": int(node.attributes.get("failed_count", 0) or 0),
                "unreachable_count": int(node.attributes.get("unreachable_count", 0) or 0),
                "visit_count": int(node.attributes.get("visit_count", node.visit_count) or 0),
                "stale_until_step": stale_until,
                "reachability": node.attributes.get("reachability", ""),
                "geodesic_distance": node.attributes.get("geodesic_distance"),
                "frontier_source": node.attributes.get("frontier_source", ""),
                "direction_label": node.attributes.get("direction_label", ""),
                "semantic_value": float(node.attributes.get("semantic_value", 0.0) or 0.0),
                "expected_room_probs": dict(node.attributes.get("expected_room_probs", {}) or {}),
                "visual_semantic_reason": node.attributes.get("visual_semantic_reason", ""),
                "reason": str(option.get("reason", node.attributes.get("source", ""))),
            })
        supplementary.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        audit = MapAuditContext(
            trajectory=self._trajectory_ids(),
            map_edges=map_edges[-96:],
            supplementary_candidates=supplementary[:16],
            current_options=current_options[:12],
            recent_failures=list(self.topo_map.failed_target_history[-8:]),
        )
        return {
            "trajectory": audit.trajectory,
            "map_edges": audit.map_edges,
            "supplementary_candidates": audit.supplementary_candidates,
            "current_options": audit.current_options,
            "recent_failures": audit.recent_failures,
        }


def _goal_text(goal: GoalNode) -> str:
    return " ".join(
        part
        for part in [
            str(goal.description or "").strip(),
            str(goal.target_object or "").strip(),
            " ".join(str(attr) for attr in goal.attributes or []),
        ]
        if part
    )


def _goal_alias_text(goal: Optional[GoalNode]) -> str:
    if goal is None:
        return ""
    key = str(goal.target_object or "").lower().strip().replace("_", " ")
    aliases = {
        "rack": [
            "rack", "clothes rack", "coat rack", "towel rack", "shelf",
            "shelving", "storage shelf", "wooden shelf", "shelf rack",
            "bookcase",
        ],
        "bed": ["bed", "mattress", "bunk bed"],
        "heater": ["heater", "radiator", "baseboard heater", "space heater"],
        "sofa": ["sofa", "couch", "loveseat"],
        "tv": ["tv", "television", "monitor", "screen"],
        "sink": ["sink", "basin", "washbasin"],
    }
    values = aliases.get(key, [key] if key else [])
    return ", ".join(values)


def _score_items(scores: Any) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for item in scores or []:
        if isinstance(item, dict):
            label = item.get("label", item.get("name", "n/a"))
            score = item.get("score", item.get("confidence", 0.0))
        else:
            try:
                label, score = item
            except (TypeError, ValueError):
                label, score = str(item), 0.0
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        out.append({"label": str(label), "score": score_f})
    return out


ConfTopoGOATAgent = GOATAgent


__all__ = [
    "Agent1MemoryWriter",
    "Agent1Perception",
    "ConfTopoGOATAgent",
    "GOATAgent",
    "GoalManager",
    "NavigationPlanner",
    "StructurePlanner",
]
