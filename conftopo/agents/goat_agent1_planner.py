"""Two-stage structure/navigation planner for clean Agent1."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from conftopo.agents.goat_agent1_memory import GoalRelevanceScorer, _goal_id
from conftopo.agents.goat_agent1_types import (
    ActiveTopoView,
    AgentPose,
    NavigationProposal,
    NavigationTarget,
    PlanningContext,
    SearchIntent,
)
from conftopo.core.confidence import RelationScorer
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.hypothesis_pool import HypothesisPool
from conftopo.core.instruction_graph import GoalNode


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def frontier_novelty(topo_map: Optional["DynamicTopoMap"], node: "SemanticNode", pose: "AgentPose") -> float:
    """Genuine unexplored-ness: how far this frontier sits from the NEAREST place
    we've already visited (rewards opening up new area), NOT merely how far it is
    from the agent's current spot. Every ~2.3 m local frontier is trivially far
    from the agent, so the old (dist-from-agent) novelty could not distinguish a
    frontier pushing into new space from one hugging already-explored area -- a
    key reason exploration circled inside one room."""
    if topo_map is not None:
        nearest = topo_map.find_nearest_node(node.position, node_type=NodeType.WAYPOINT_VISITED)
        if nearest is not None:
            d = distance(node.position, nearest.position)
            return float(max(0.0, min(1.0, d / 3.0)))
    from_agent = distance(pose.position, node.position)
    return float(max(0.0, min(1.0, (from_agent - 1.2) / 4.0)))


def heading_alignment(pose: "AgentPose", position: np.ndarray) -> float:
    """cos(angle) between the agent's forward heading and the direction to
    `position` (xz plane): +1 straight ahead, 0 to the side, -1 directly behind.
    Used to discourage picking an exploration frontier that requires turning back
    the way we came ("刚出门就掉头")."""
    dx = float(position[0] - pose.position[0])
    dz = float(position[2] - pose.position[2])
    norm = math.hypot(dx, dz)
    if norm < 1e-6:
        return 1.0
    fwd_x = -math.sin(float(pose.heading))
    fwd_z = -math.cos(float(pose.heading))
    return float((dx * fwd_x + dz * fwd_z) / norm)


def cosine_similarity(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    try:
        va = np.asarray(a, dtype=np.float32).reshape(-1)
        vb = np.asarray(b, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if va.size <= 0 or vb.size <= 0 or va.size != vb.size:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / denom)


PORTAL_NEAR_VISITED_RADIUS = 0.85

# Floor of the reliability discount in the task-conditioned confidence fusion
# (see ``task_confidence``). A semantically-matching but only-glimpsed candidate
# (high R, low existence e) must not collapse to ~0 -- it should stay medium so
# the agent goes to verify it. The floor sets how much a perfect semantic match
# is worth on its own before any corroboration.
TASK_CONF_RELIABILITY_FLOOR = 0.45


def _view_excludes(active_view: Optional[ActiveTopoView], node_id: Optional[str]) -> bool:
    """True when the ActiveTopoView says this node must not enter candidates.

    In Phase C the view only suppresses nodes the planner already skips
    (blocked/consumed navigation nodes, blocked portals), so gating on this is
    behavior-neutral while routing all candidate generation through the view.
    """
    return active_view is not None and not active_view.is_candidate(node_id)


class StructurePlanner:
    """Select semantic search intent from structure-layer memory."""

    def __init__(self):
        self.last_intents: List[SearchIntent] = []

    def build_intents(
        self,
        *,
        topo_map: DynamicTopoMap,
        hypotheses: HypothesisPool,
        goal: Optional[GoalNode],
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        if goal is not None:
            intents.extend(self._object_intents(topo_map, goal, pose, active_view))
            intents.extend(self._room_intents(topo_map, goal, pose, active_view))
            intents.extend(self._landmark_intents(topo_map, goal, pose, active_view))
            intents.extend(self._portal_intents(topo_map, goal, pose, planning_context, active_view))
            intents.extend(self._hypothesis_intents(topo_map, hypotheses, goal, pose))
        intents.append(self._explore_intent(topo_map, goal, pose, planning_context, active_view))
        intents.sort(key=lambda item: item.score, reverse=True)
        self.last_intents = intents
        return intents

    def select(
        self,
        *,
        topo_map: DynamicTopoMap,
        hypotheses: HypothesisPool,
        goal: Optional[GoalNode],
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> SearchIntent:
        intents = self.build_intents(
            topo_map=topo_map,
            hypotheses=hypotheses,
            goal=goal,
            pose=pose,
            planning_context=planning_context,
            active_view=active_view,
        )
        return intents[0] if intents else SearchIntent(intent_type="explore", target_label="frontier")

    def _object_intents(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalNode,
        pose: AgentPose,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[SearchIntent]:
        out: List[SearchIntent] = []
        for node in topo_map.structure_layer.nodes():
            if node.node_type != NodeType.OBJECT:
                continue
            if _view_excludes(active_view, node.node_id):
                continue
            # Task-conditioned confidence drives the ranking: R (semantic match to
            # the goal) is fused *multiplicatively* with e (existence/corroboration
            # confidence). A semantic mismatch collapses c_task regardless of how
            # sure we are the object exists, so unrelated objects no longer
            # out-score exploration and the agent is free to leave the room; a
            # strong-but-under-observed match stays medium and is worth verifying.
            relevance = object_goal_relevance(node, goal)
            if relevance <= 0.0:
                continue
            nav = topo_map.navigation_target_for_structure_node(node.node_id)
            if nav is None:
                continue
            failed = int(node.attributes.get("failed_count", 0) or 0)
            unreachable = int(node.attributes.get("unreachable_count", 0) or 0)
            dist = distance(pose.position, nav.position)
            existence = float(node.confidence)
            c_task = object_c_task(node, goal)
            reach_penalty = object_reachability_penalty(node, nav.node_id)
            score = c_task - 0.035 * dist - reach_penalty
            out.append(SearchIntent(
                intent_type="object",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav.node_id,
                target_position=nav.position.copy(),
                score=score,
                reason=(
                    f"object_memory:{node.label}; c_task={c_task:.2f}; R={relevance:.2f}; e={existence:.2f}; "
                    f"failed={failed}; unreachable={unreachable}"
                ),
                preferred_room=node.attributes.get("room_label"),
                evidence_refs=[node.node_id],
            ))
        return out

    def _room_intents(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalNode,
        pose: AgentPose,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[SearchIntent]:
        priors = {str(room).lower().strip() for room in goal.room_prior or []}
        if not priors:
            return []
        out: List[SearchIntent] = []
        for node in topo_map.structure_layer.nodes():
            if node.node_type != NodeType.ROOM:
                continue
            if _view_excludes(active_view, node.node_id):
                continue
            probs = _score_dict(node.attributes.get("room_probs", {}))
            prob = max((float(probs.get(room, 0.0)) for room in priors), default=0.0)
            label_bonus = 0.35 if str(node.label).lower().strip() in priors else 0.0
            if prob <= 0.0 and label_bonus <= 0.0:
                continue
            nav = topo_map.navigation_target_for_structure_node(node.node_id)
            dist = distance(pose.position, nav.position) if nav is not None else 0.0
            relevance = 0.35 + label_bonus + 0.65 * prob
            map_conf = room_map_confidence(node)
            score = room_intent_score(relevance, map_conf, dist)
            out.append(SearchIntent(
                intent_type="room",
                target_label=str(node.label or ""),
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav.node_id if nav is not None else None,
                target_position=nav.position.copy() if nav is not None else None,
                score=score,
                reason=f"room_prior:{','.join(sorted(priors))}; prob={prob:.2f}",
                preferred_room=next(iter(priors), None),
                evidence_refs=[node.node_id],
            ))
        return out

    def _landmark_intents(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalNode,
        pose: AgentPose,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[SearchIntent]:
        labels = {str(label).lower().strip() for label in goal.landmarks or []}
        if not labels:
            return []
        out: List[SearchIntent] = []
        for node in topo_map.structure_layer.nodes():
            if node.node_type != NodeType.LANDMARK:
                continue
            if _view_excludes(active_view, node.node_id):
                continue
            label = str(node.label or "").lower().strip()
            if label not in labels:
                continue
            nav = topo_map.navigation_target_for_structure_node(node.node_id)
            if nav is None:
                continue
            dist = distance(pose.position, nav.position)
            relevance = landmark_task_relevance(node, goal)
            map_conf = landmark_map_confidence(node)
            score = landmark_intent_score(relevance, map_conf, dist)
            out.append(SearchIntent(
                intent_type="landmark",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav.node_id,
                target_position=nav.position.copy(),
                score=score,
                reason=f"landmark:{node.label}; conf={node.confidence:.2f}",
                related_landmarks=[node.label],
                evidence_refs=[node.node_id],
            ))
        return out

    def _portal_intents(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalNode,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[SearchIntent]:
        priors = {str(room).lower().strip() for room in goal.room_prior or []}
        stale_ids = set((planning_context.stale_target_ids or {}).keys()) if planning_context else set()
        out: List[SearchIntent] = []
        for node in topo_map.structure_layer.nodes():
            if node.node_type != NodeType.PORTAL or node.attributes.get("blocked", False):
                continue
            if _view_excludes(active_view, node.node_id):
                continue
            bad_anchor_ids = {str(x) for x in node.attributes.get("bad_anchor_waypoint_ids", []) or []}
            usable_links: List[SemanticNode] = []
            failed_links = 0
            near_links = 0
            visit_penalty = 0.0
            failed_penalty = 0.0
            for wp_id in node.attributes.get("connected_waypoints", []) or []:
                wp = topo_map.get_node(str(wp_id))
                if wp is None or wp.node_type not in topo_map._NAV_NODE_TYPES:
                    continue
                is_bad = wp.node_id in bad_anchor_ids
                blocked = bool(wp.attributes.get("blocked", False) or wp.attributes.get("consumed", False))
                failed = int(wp.attributes.get("failed_count", 0) or 0)
                unreachable = int(wp.attributes.get("unreachable_count", 0) or 0)
                visits = int(wp.attributes.get("visit_count", wp.visit_count) or 0)
                stale = wp.node_id in stale_ids
                near = (
                    wp.node_type == NodeType.WAYPOINT_VISITED
                    and distance(pose.position, wp.position) < PORTAL_NEAR_VISITED_RADIUS
                )
                failed_penalty += 0.18 * failed + 0.65 * unreachable + (0.45 if is_bad else 0.0) + (0.50 if stale else 0.0)
                visit_penalty += 0.04 * min(8, visits)
                if failed or unreachable or is_bad or blocked or stale:
                    failed_links += 1
                if near:
                    near_links += 1
                if not blocked and not is_bad and not near and not stale:
                    usable_links.append(wp)
            if not usable_links:
                continue
            connected = {str(x).lower().strip() for x in node.attributes.get("connected_rooms", []) or []}
            room_match = 0.25 if priors and connected.intersection(priors) else 0.0
            nav = min(usable_links, key=lambda wp: distance(pose.position, wp.position))
            dist = distance(pose.position, nav.position)
            portal_failed = int(node.attributes.get("failed_count", 0) or 0)
            portal_unreachable = int(node.attributes.get("unreachable_count", 0) or 0)
            relevance = 0.25 + room_match
            map_conf = portal_map_confidence(node)
            reach_penalty = portal_reachability_penalty(
                portal_failed, portal_unreachable, failed_penalty + visit_penalty
            )
            score = portal_intent_score(relevance, map_conf, reach_penalty, dist)
            out.append(SearchIntent(
                intent_type="portal",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav.node_id,
                target_position=nav.position.copy(),
                score=score,
                reason=(
                    f"portal:{node.label}; room_match={room_match:.2f}; "
                    f"usable_links={len(usable_links)}; failed_links={failed_links}; near_links={near_links}"
                ),
                preferred_room=next(iter(connected), None),
                evidence_refs=[node.node_id],
            ))
        return out

    def _hypothesis_intents(
        self,
        topo_map: DynamicTopoMap,
        hypotheses: HypothesisPool,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        out: List[SearchIntent] = []
        goal_id = _goal_id(goal)
        for hyp in hypotheses.get_active(goal_id=goal_id, include_needs_verify=True):
            if hyp.anchor_node_id is None:
                continue
            anchor = topo_map.get_node(hyp.anchor_node_id)
            if anchor is None or anchor.node_type not in topo_map._NAV_NODE_TYPES:
                continue
            hyp_relevance = GoalRelevanceScorer.label_relevance(hyp.label, goal)
            if hyp.kind == "object" and hyp_relevance < 0.15:
                continue
            dist = distance(pose.position, anchor.position)
            if hyp.kind == "object":
                # Same task-conditioned confidence as confirmed objects, but on a
                # low-existence guess: a strong semantic match still reads as a
                # medium c_task worth going to verify.
                c_task = hypothesis_c_task(hyp, goal)
                score = c_task + min(0.12, 0.03 * hyp.seen_count) - 0.03 * dist
            else:
                score = 0.20 + 0.40 * float(hyp.confidence) + min(0.12, 0.03 * hyp.seen_count) - 0.03 * dist
            out.append(SearchIntent(
                intent_type="hypothesis",
                target_label=hyp.label,
                preferred_anchor_waypoint_id=anchor.node_id,
                target_position=anchor.position.copy(),
                score=score,
                reason=f"hypothesis:{hyp.kind}:{hyp.label}; conf={hyp.confidence:.2f}; seen={hyp.seen_count}",
                evidence_refs=[hyp.id],
            ))
        return out

    def _explore_intent(
        self,
        topo_map: DynamicTopoMap,
        goal: Optional[GoalNode],
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> SearchIntent:
        best_node: Optional[SemanticNode] = None
        best_score = 0.05
        best_reason = "memory_insufficient_explore"
        preferred_room = None
        recent_counts = planning_context.recent_target_counts if planning_context else {}
        stale_ids = set((planning_context.stale_target_ids or {}).keys()) if planning_context else set()
        for node in topo_map.navigation_layer.nodes():
            if node.node_type != NodeType.WAYPOINT_FRONTIER:
                continue
            if node.attributes.get("blocked", False) or node.attributes.get("consumed", False):
                continue
            if _view_excludes(active_view, node.node_id):
                continue
            dist = distance(pose.position, node.position)
            frontier_conf = float(node.confidence)
            semantic = float(node.attributes.get("semantic_value", 0.0) or 0.0)
            expected = _score_dict(node.attributes.get("expected_room_probs", {}))
            room_match = 0.0
            if goal is not None:
                priors = {str(room).lower().strip() for room in goal.room_prior or []}
                room_match = max((float(expected.get(room, 0.0)) for room in priors), default=0.0)
                semantic = max(semantic, room_match)
            failed = float(node.attributes.get("failed_count", 0) or 0)
            unreachable = float(node.attributes.get("unreachable_count", 0) or 0)
            visits = float(node.attributes.get("visit_count", node.visit_count) or 0)
            novelty = frontier_novelty(topo_map, node, pose)
            heading_align = heading_alignment(pose, node.position)
            recent = float(recent_counts.get(node.node_id, 0))
            stale_penalty = 0.35 if node.node_id in stale_ids else 0.0
            risk = frontier_reachability_penalty(failed, unreachable, visits, recent, stale_penalty)
            relevance = frontier_task_relevance(semantic)
            map_conf = frontier_map_confidence(frontier_conf)
            score = frontier_intent_score(relevance, map_conf, novelty, dist, risk, heading_align)
            if score > best_score:
                best_score = score
                best_node = node
                if room_match > 0.0 and goal is not None:
                    preferred_room = max(
                        (str(room).lower().strip() for room in goal.room_prior or []),
                        key=lambda room: float(expected.get(room, 0.0)),
                    )
                best_reason = (
                    f"explore_frontier:{node.node_id}; conf={frontier_conf:.2f}; "
                    f"semantic={semantic:.2f}; novelty={novelty:.2f}; heading={heading_align:.2f}; "
                    f"risk={risk:.2f}; recent={recent:.0f}"
                )
        return SearchIntent(
            intent_type="explore",
            target_label="frontier",
            structure_node_id=None,
            preferred_anchor_waypoint_id=best_node.node_id if best_node is not None else None,
            target_position=best_node.position.copy() if best_node is not None else None,
            score=best_score,
            reason=best_reason,
            preferred_room=preferred_room,
            evidence_refs=[best_node.node_id] if best_node is not None else [],
        )

    @staticmethod
    def debug_intent(intent: SearchIntent) -> Dict[str, Any]:
        return {
            "intent_type": intent.intent_type,
            "target_label": intent.target_label,
            "structure_node_id": intent.structure_node_id,
            "preferred_anchor_waypoint_id": intent.preferred_anchor_waypoint_id,
            "score": round(float(intent.score), 4),
            "reason": intent.reason,
            "preferred_room": intent.preferred_room,
            "evidence_refs": list(intent.evidence_refs),
        }


class NavigationPlanner:
    """Resolve SearchIntent into an executable navigation-layer target."""

    def __init__(self):
        self.last_proposals: List[NavigationProposal] = []

    def build_proposals(
        self,
        *,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> List[NavigationProposal]:
        proposals: List[NavigationProposal] = []
        if intent.intent_type in {"object", "landmark", "hypothesis"}:
            proposals.extend(self._anchor_proposals(topo_map, intent, pose, planning_context))
            if not proposals:
                proposals.extend(self._frontier_proposals(topo_map, intent, pose, planning_context, base_alignment=0.08))
        elif intent.intent_type == "room":
            proposals.extend(self._room_proposals(topo_map, intent, pose, planning_context))
        elif intent.intent_type == "portal":
            proposals.extend(self._portal_proposals(topo_map, intent, pose, planning_context))
            if not proposals:
                proposals.extend(self._frontier_proposals(topo_map, intent, pose, planning_context, base_alignment=0.10))
        else:
            proposals.extend(self._frontier_proposals(topo_map, intent, pose, planning_context))
        proposals = [
            p for p in proposals
            if p.target_node_id is not None
            and self._available(
                topo_map.get_node(str(p.target_node_id)),
                pose=pose,
                target_type=p.target_type,
                planning_context=planning_context,
                active_view=active_view,
            )
        ]
        proposals.sort(key=lambda p: p.score, reverse=True)
        self.last_proposals = proposals
        return proposals

    def select(
        self,
        *,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> NavigationTarget:
        return self.select_from_intents(
            topo_map=topo_map,
            intents=[intent],
            pose=pose,
            planning_context=planning_context,
            active_view=active_view,
        )

    def select_from_intents(
        self,
        *,
        topo_map: DynamicTopoMap,
        intents: List[SearchIntent],
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> NavigationTarget:
        """Resolve Top-K structure intents into one globally ranked nav target."""
        proposals: List[NavigationProposal] = []
        for intent in intents:
            proposals.extend(self.build_proposals(
                topo_map=topo_map,
                intent=intent,
                pose=pose,
                planning_context=planning_context,
                active_view=active_view,
            ))
        proposals.sort(key=lambda item: item.score, reverse=True)
        self.last_proposals = proposals
        if not proposals:
            fallback_intent = intents[0] if intents else SearchIntent(intent_type="explore", target_label="frontier")
            return NavigationTarget(
                node_id=None,
                position=None,
                target_type="idle",
                reason=f"no_navigation_target:{fallback_intent.intent_type}",
                source_intent=fallback_intent,
                structure_node_id=fallback_intent.structure_node_id,
                evidence_refs=list(fallback_intent.evidence_refs),
            )
        best: Optional[NavigationProposal] = None
        for proposal in proposals:
            routed, route_blocked = self._route_global_target(topo_map, proposal, pose, planning_context)
            if route_blocked:
                continue
            best = routed if routed is not None else proposal
            break
        if best is None:
            fallback_intent = intents[0] if intents else SearchIntent(intent_type="explore", target_label="frontier")
            return NavigationTarget(
                node_id=None,
                position=None,
                target_type="idle",
                reason=f"all_routes_blocked:{fallback_intent.intent_type}",
                source_intent=fallback_intent,
                structure_node_id=fallback_intent.structure_node_id,
                evidence_refs=list(fallback_intent.evidence_refs),
            )
        return NavigationTarget(
            node_id=best.target_node_id,
            position=best.target_position,
            target_type=best.target_type,
            reason=best.reason,
            source_intent=best.source_intent,
            structure_node_id=best.structure_node_id,
            evidence_refs=list(best.evidence_refs),
            score=best.score,
        )

    def _route_global_target(
        self,
        topo_map: DynamicTopoMap,
        proposal: NavigationProposal,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
    ) -> tuple[Optional[NavigationProposal], bool]:
        """DUET-style global action: select an unvisited node, execute the topo next hop."""
        target = topo_map.get_node(str(proposal.target_node_id)) if proposal.target_node_id else None
        if target is None or target.node_type not in topo_map._NAV_NODE_TYPES:
            return None, False
        if planning_context and target.node_id in planning_context.stale_target_ids:
            return None, True
        current = self._nearest_current_waypoint(topo_map, pose)
        if current is None or current.node_id == target.node_id:
            return None, False
        path = topo_map.shortest_path(current.node_id, target.node_id, EdgeType.NAVIGABLE)
        if not path or len(path) < 2:
            return None, False
        next_hop = topo_map.get_node(str(path[1]))
        if next_hop is None or next_hop.node_type not in topo_map._NAV_NODE_TYPES:
            return None, False
        if planning_context and next_hop.node_id in planning_context.stale_target_ids:
            return None, True
        if next_hop.attributes.get("blocked", False) or next_hop.attributes.get("consumed", False):
            return None, True
        if next_hop.node_id == target.node_id:
            return None, False
        routed_type = proposal.target_type
        if next_hop.node_type == NodeType.WAYPOINT_VISITED:
            routed_type = "transit_waypoint"
        elif next_hop.node_type == NodeType.WAYPOINT_FRONTIER:
            routed_type = "frontier"
        return NavigationProposal(
            target_node_id=next_hop.node_id,
            target_position=next_hop.position.copy(),
            target_type=routed_type,
            score=proposal.score,
            reason=(
                f"duet_next_hop:{next_hop.node_id}->global:{target.node_id}; "
                f"path_len={len(path)}; {proposal.reason}"
            ),
            source_intent=proposal.source_intent,
            structure_node_id=proposal.structure_node_id,
            evidence_refs=[target.node_id, *list(proposal.evidence_refs)],
        ), False

    def _nearest_current_waypoint(self, topo_map: DynamicTopoMap, pose: AgentPose) -> Optional[SemanticNode]:
        best: Optional[SemanticNode] = None
        best_dist = float("inf")
        for node in topo_map.navigation_layer.nodes():
            if node.node_type != NodeType.WAYPOINT_VISITED:
                continue
            dist = distance(pose.position, node.position)
            if dist < best_dist:
                best = node
                best_dist = dist
        return best

    def _anchor_proposals(
        self,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
    ) -> List[NavigationProposal]:
        candidates: List[SemanticNode] = []
        if intent.preferred_anchor_waypoint_id:
            node = topo_map.get_node(intent.preferred_anchor_waypoint_id)
            if node is not None:
                candidates.append(node)
        if intent.structure_node_id:
            fallback = topo_map.navigation_target_for_structure_node(intent.structure_node_id)
            if fallback is not None:
                candidates.append(fallback)
            structure = topo_map.get_node(intent.structure_node_id)
            if structure is not None and intent.intent_type == "object":
                bad = {str(x) for x in structure.attributes.get("bad_anchor_waypoint_ids", []) or []}
                approach_id = structure.attributes.get("approach_point_id")
                if approach_id and str(approach_id) not in bad:
                    approach = topo_map.get_node(str(approach_id))
                    if approach is not None:
                        candidates.append(approach)
                for wp_id in structure.attributes.get("observed_from_waypoint_ids", []) or []:
                    if str(wp_id) in bad:
                        continue
                    wp = topo_map.get_node(str(wp_id))
                    if wp is not None:
                        candidates.append(wp)
        dedup: Dict[str, SemanticNode] = {}
        for node in candidates:
            if node.node_type in topo_map._NAV_NODE_TYPES:
                dedup.setdefault(node.node_id, node)
        near_anchor_offsets: Dict[str, SemanticNode] = {}
        for node in dedup.values():
            if distance(pose.position, node.position) >= 0.55:
                continue
            for neighbor_id in topo_map.get_neighbors(node.node_id, EdgeType.NAVIGABLE):
                neighbor = topo_map.get_node(str(neighbor_id))
                if neighbor is None or neighbor.node_type not in topo_map._NAV_NODE_TYPES:
                    continue
                if neighbor.attributes.get("blocked", False) or neighbor.attributes.get("consumed", False):
                    continue
                if distance(pose.position, neighbor.position) < 0.50:
                    continue
                near_anchor_offsets.setdefault(neighbor.node_id, neighbor)
        target_type = "object_anchor" if intent.intent_type == "object" else f"{intent.intent_type}_waypoint"
        # Prefer the intent's chosen anchor (for committed objects this is the
        # real OBSERVATION waypoint) over other structure anchors like the
        # estimated approach point, so navigation heads to where we actually saw
        # the object rather than a fabricated stand-off.
        proposals = [
            self._proposal(
                node,
                intent,
                pose,
                planning_context,
                target_type,
                1.0 if node.node_id == intent.preferred_anchor_waypoint_id else 0.82,
                f"{intent.intent_type}_anchor:{node.node_id}",
                topo_map=topo_map,
            )
            for node in dedup.values()
        ]
        offset_type = "object_anchor_offset" if intent.intent_type == "object" else f"{intent.intent_type}_offset"
        proposals.extend([
            self._proposal(
                node,
                intent,
                pose,
                planning_context,
                offset_type,
                0.92,
                f"{intent.intent_type}_anchor_offset:{node.node_id}",
                topo_map=topo_map,
            )
            for node in near_anchor_offsets.values()
            if node.node_id not in dedup
        ])
        return proposals

    def _room_proposals(
        self,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
    ) -> List[NavigationProposal]:
        room = topo_map.get_node(intent.structure_node_id) if intent.structure_node_id else None
        out: List[NavigationProposal] = []
        if room is not None:
            for wp_id in room.attributes.get("member_waypoints", []) or []:
                wp = topo_map.get_node(str(wp_id))
                if wp is not None and wp.node_type in topo_map._NAV_NODE_TYPES:
                    out.append(self._proposal(wp, intent, pose, planning_context, "room_waypoint", 0.75, f"room_member:{wp.node_id}", topo_map=topo_map))
        out.extend(self._frontier_proposals(topo_map, intent, pose, planning_context, preferred_room=intent.preferred_room, base_alignment=0.45))
        return out

    def _portal_proposals(
        self,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
    ) -> List[NavigationProposal]:
        portal = topo_map.get_node(intent.structure_node_id) if intent.structure_node_id else None
        out: List[NavigationProposal] = []
        stale_ids = set((planning_context.stale_target_ids or {}).keys()) if planning_context else set()
        if portal is not None:
            bad_anchor_ids = {str(x) for x in portal.attributes.get("bad_anchor_waypoint_ids", []) or []}
            for wp_id in portal.attributes.get("connected_waypoints", []) or []:
                wp = topo_map.get_node(str(wp_id))
                if wp is None or wp.node_type not in topo_map._NAV_NODE_TYPES:
                    continue
                if wp.node_id in stale_ids:
                    continue
                if wp.node_id in bad_anchor_ids:
                    continue
                if wp.attributes.get("blocked", False) or wp.attributes.get("consumed", False):
                    continue
                is_near_visited = (
                    wp.node_type == NodeType.WAYPOINT_VISITED
                    and distance(pose.position, wp.position) < PORTAL_NEAR_VISITED_RADIUS
                )
                if is_near_visited:
                    continue
                if wp.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE, NodeType.WAYPOINT_APPROACH):
                    alignment = 0.95
                else:
                    alignment = 0.55
                out.append(self._proposal(wp, intent, pose, planning_context, "portal_waypoint", alignment, f"portal_connection:{wp.node_id}", topo_map=topo_map))
        return out

    def _frontier_proposals(
        self,
        topo_map: DynamicTopoMap,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext] = None,
        *,
        preferred_room: Optional[str] = None,
        base_alignment: float = 0.20,
    ) -> List[NavigationProposal]:
        out: List[NavigationProposal] = []
        for node in topo_map.navigation_layer.nodes():
            if node.node_type != NodeType.WAYPOINT_FRONTIER:
                continue
            expected = _score_dict(node.attributes.get("expected_room_probs", {}))
            room_alignment = float(expected.get(str(preferred_room or intent.preferred_room or "").lower().strip(), 0.0))
            semantic = float(node.attributes.get("semantic_value", 0.0) or 0.0)
            alignment = max(base_alignment if intent.intent_type == "explore" else 0.05, room_alignment, semantic)
            reason = (
                f"frontier:{node.node_id}; dir={node.attributes.get('direction_label', '')}; "
                f"source={node.attributes.get('frontier_source', '')}"
            )
            if intent.intent_type == "explore" and node.node_id == intent.preferred_anchor_waypoint_id:
                alignment = max(alignment, 0.85)
                reason = f"frontier_preferred:{node.node_id}"
            out.append(self._proposal(node, intent, pose, planning_context, "frontier", alignment, reason, topo_map=topo_map))
        return out

    def _proposal(
        self,
        node: SemanticNode,
        intent: SearchIntent,
        pose: AgentPose,
        planning_context: Optional[PlanningContext],
        target_type: str,
        alignment: float,
        reason: str,
        *,
        topo_map: Optional[DynamicTopoMap] = None,
    ) -> NavigationProposal:
        dist = distance(pose.position, node.position)
        failed = float(node.attributes.get("failed_count", 0) or 0)
        unreachable = float(node.attributes.get("unreachable_count", 0) or 0)
        blocked = 1.0 if node.attributes.get("blocked", False) else 0.0
        visits = float(node.attributes.get("visit_count", node.visit_count) or 0)
        frontier_value = float(node.confidence) if node.node_type == NodeType.WAYPOINT_FRONTIER else 0.0
        is_frontier = node.node_type == NodeType.WAYPOINT_FRONTIER
        novelty = frontier_novelty(topo_map, node, pose) if is_frontier else 0.0
        # Direction continuity applies to exploration frontiers only: goal / region
        # targets (tier 1) must always be pursued regardless of heading.
        heading_align = heading_alignment(pose, node.position) if (is_frontier and pose is not None) else 0.0
        recent = float((planning_context.recent_target_counts or {}).get(node.node_id, 0)) if planning_context else 0.0
        stale = bool(planning_context and node.node_id in planning_context.stale_target_ids)
        current_wp = planning_context.current_waypoint_id if planning_context else None
        close_current_visited = (
            node.node_type == NodeType.WAYPOINT_VISITED
            and node.node_id == current_wp
            and target_type != "transit_waypoint"
            and dist < 0.45
        )
        unvisited_bonus = self._unvisited_neighbor_bonus(topo_map, node) if topo_map is not None else 0.0
        history_penalty = (
            0.07 * min(6.0, recent)
            + (0.75 if stale else 0.0)
            + (1.0 if close_current_visited else 0.0)
        )
        score = (
            0.35 * alignment
            + 0.22 * float(intent.score)
            + 0.20 * frontier_value
            + 0.16 * novelty
            + 0.20 * heading_align
            + unvisited_bonus
            + 0.35 * max(0.0, 1.0 - blocked - 0.2 * unreachable)
            - 0.035 * dist
            - 0.20 * failed
            - 0.45 * unreachable
            - 0.12 * min(4.0, visits)
            - history_penalty
            - blocked
        )
        return NavigationProposal(
            target_node_id=node.node_id,
            target_position=node.position.copy(),
            target_type=target_type,
            score=score,
            reason=(
                f"{reason}; score={score:.2f}; dist={dist:.2f}; novelty={novelty:.2f}; "
                f"heading={heading_align:.2f}; visits={visits:.0f}; failed={failed:.0f}; "
                f"recent={recent:.0f}; stale={int(stale)}; unvisited_bonus={unvisited_bonus:.2f}"
            ),
            source_intent=intent,
            structure_node_id=intent.structure_node_id,
            evidence_refs=list(intent.evidence_refs),
        )

    def _available(
        self,
        node: Optional[SemanticNode],
        *,
        pose: Optional[AgentPose] = None,
        target_type: str = "",
        planning_context: Optional[PlanningContext] = None,
        active_view: Optional[ActiveTopoView] = None,
    ) -> bool:
        if node is None:
            return False
        if node.attributes.get("blocked", False) or node.attributes.get("consumed", False):
            return False
        if _view_excludes(active_view, node.node_id):
            return False
        if planning_context and node.node_id in planning_context.stale_target_ids:
            return False
        if (
            pose is not None
            and node.node_type == NodeType.WAYPOINT_VISITED
            and target_type not in {"transit_waypoint", "object_anchor", "hypothesis_waypoint"}
            and distance(pose.position, node.position) < 0.45
        ):
            return False
        return True

    def _unvisited_neighbor_bonus(self, topo_map: Optional[DynamicTopoMap], node: SemanticNode) -> float:
        if topo_map is None:
            return 0.0
        bonus = 0.0
        if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE, NodeType.WAYPOINT_APPROACH):
            bonus += 0.18
        for neighbor_id in topo_map.get_neighbors(node.node_id, EdgeType.NAVIGABLE):
            neighbor = topo_map.get_node(str(neighbor_id))
            if neighbor is None:
                continue
            if neighbor.node_type not in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE, NodeType.WAYPOINT_APPROACH):
                continue
            if neighbor.attributes.get("blocked", False) or neighbor.attributes.get("consumed", False):
                continue
            bonus += 0.08
        return float(min(0.30, bonus))

    @staticmethod
    def debug_proposal(proposal: NavigationProposal) -> Dict[str, Any]:
        return {
            "target_node_id": proposal.target_node_id,
            "target_position": None if proposal.target_position is None else np.asarray(proposal.target_position, dtype=np.float32).round(4).tolist(),
            "target_type": proposal.target_type,
            "score": round(float(proposal.score), 4),
            "reason": proposal.reason,
            "structure_node_id": proposal.structure_node_id,
            "evidence_refs": list(proposal.evidence_refs),
            "source_intent": proposal.source_intent.intent_type,
        }

    @staticmethod
    def debug_target(target: NavigationTarget) -> Dict[str, Any]:
        return {
            "node_id": target.node_id,
            "target_position": None if target.position is None else np.asarray(target.position, dtype=np.float32).round(4).tolist(),
            "target_type": target.target_type,
            "score": round(float(target.score), 4),
            "reason": target.reason,
            "structure_node_id": target.structure_node_id,
            "evidence_refs": list(target.evidence_refs),
            "source_intent": target.source_intent.intent_type if target.source_intent else None,
        }


def _score_dict(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k).lower().strip(): float(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out: Dict[str, float] = {}
        for item in value:
            try:
                key, score = item
            except (TypeError, ValueError):
                continue
            out[str(key).lower().strip()] = float(score)
        return out
    return {}


# --- Phase G: multimodal task-relevance weight profiles ---------------------
# The same object signals (visual / text / attribute / relation / room) are
# blended with different weights depending on the goal modality. GOAT goals come
# as category (a noun), description (attributes + spatial relations), or image (a
# reference photo). The ``image`` profile is deliberately identical to the legacy
# fixed blend so image/instance goals stay byte-for-byte unchanged.
_RELEVANCE_WEIGHTS = {
    "image": {"visual": 0.60, "text": 0.20, "attribute": 0.10, "relation": 0.00, "room": 0.10},
    "category": {"visual": 0.50, "text": 0.15, "attribute": 0.15, "relation": 0.00, "room": 0.20},
    "description": {"visual": 0.35, "text": 0.15, "attribute": 0.25, "relation": 0.15, "room": 0.10},
}


def _relevance_weights(goal: GoalNode) -> Dict[str, float]:
    goal_type = str(getattr(goal, "goal_type", "category") or "category").lower().strip()
    return _RELEVANCE_WEIGHTS.get(goal_type, _RELEVANCE_WEIGHTS["category"])


def _object_relation_similarity(node: SemanticNode, goal: GoalNode) -> float:
    """R-relation: how well the object's observed context matches goal relations
    and landmarks (reuses the shared RelationScorer). 0 when no evidence."""
    relations = getattr(goal, "relations", None)
    landmarks = getattr(goal, "landmarks", None)
    if not relations and not landmarks:
        return 0.0
    attrs = node.attributes if isinstance(node.attributes, dict) else {}
    observed_relations = attrs.get("spatial_relation") or attrs.get("relations") or []
    observed_room = attrs.get("room_context") or attrs.get("room_label") or ""
    return float(RelationScorer.score(
        goal_relations=relations,
        goal_landmarks=landmarks,
        observed_relations=observed_relations,
        observed_room=str(observed_room),
        room_prior=goal.room_prior,
    ))


def _object_retrieval_score(node: SemanticNode, goal: GoalNode) -> Dict[str, float]:
    goal_embedding = getattr(goal, "target_embedding", None)
    visual_similarity = max(0.0, cosine_similarity(node.embedding, goal_embedding))
    attrs = node.attributes if isinstance(node.attributes, dict) else {}
    object_attrs = attrs.get("object_attributes", {})
    text_embedding = None
    if isinstance(object_attrs, dict):
        text_embedding = object_attrs.get("text_embedding")
    text_similarity = max(0.0, cosine_similarity(text_embedding, goal_embedding))
    attribute_similarity = float(GoalRelevanceScorer.object_score(node, goal))
    room_consistency = _object_room_consistency(attrs, goal)
    label_bonus = _label_bonus(node.label, goal)
    w = _relevance_weights(goal)
    relation_similarity = (
        _object_relation_similarity(node, goal) if w["relation"] > 0.0 else 0.0
    )
    score = (
        w["visual"] * visual_similarity
        + w["text"] * text_similarity
        + w["attribute"] * attribute_similarity
        + w["relation"] * relation_similarity
        + w["room"] * room_consistency
        + label_bonus
    )
    # Backward-compatible fallback for old object nodes without embeddings.
    if visual_similarity <= 0.0 and text_similarity <= 0.0 and attribute_similarity >= 0.65:
        score = max(score, 0.35 + 0.25 * attribute_similarity)
    return {
        "score": float(max(0.0, min(1.0, score))),
        "visual_similarity": float(visual_similarity),
        "text_similarity": float(text_similarity),
        "attribute_similarity": float(attribute_similarity),
        "relation_similarity": float(relation_similarity),
        "room_consistency": float(room_consistency),
        "label_bonus": float(label_bonus),
    }


def _object_room_consistency(attrs: Dict[str, Any], goal: GoalNode) -> float:
    priors = {str(room).lower().strip() for room in goal.room_prior or [] if str(room).strip()}
    if not priors:
        return 0.0
    values: List[str] = []
    for key in ("room_context", "room_label", "preferred_room"):
        value = attrs.get(key)
        if value is not None:
            values.append(str(value).lower().strip())
    contexts = attrs.get("room_contexts")
    if isinstance(contexts, (list, tuple, set)):
        values.extend(str(item).lower().strip() for item in contexts)
    return 1.0 if any(value in priors for value in values) else 0.0


def _label_bonus(label: Any, goal: GoalNode) -> float:
    relevance = GoalRelevanceScorer.label_relevance(str(label or ""), goal)
    if relevance >= 1.0:
        return 0.05
    if relevance >= 0.78:
        return 0.035
    if relevance >= 0.65:
        return 0.02
    return 0.0


# --- Explicit C / R / A decomposition for the object intent ------------------
# These expose the object signals as separate axes so ActiveTopo can read each
# independently instead of re-deriving a blend:
#   R (task relevance)       -> object_task_relevance
#   C (map confidence)       -> object_map_confidence
#   A (reachability penalty) -> object_reachability_penalty
# The planner ranks objects by task-conditioned confidence (``object_c_task`` /
# ``_object_intents``), not an additive blend.

def object_task_relevance(node: SemanticNode, goal: GoalNode) -> float:
    """R: relevance of this object node to the current goal (0..1)."""
    return float(_object_retrieval_score(node, goal)["score"])


# FOV-weighted confidence saturation constant (ApexNav / VLFM style). Larger K
# means more central observations are needed to reach the corroboration cap.
OBJECT_FOV_MASS_K = 2.0


def object_map_confidence(node: SemanticNode) -> float:
    """C: map-confidence contribution.

    ApexNav / VLFM-style fusion: instead of counting every re-detection equally
    (legacy ``0.04 * multi_view``), we read a FOV/center-weighted value ``V``
    (running mean of detection scores weighted by how centered each observation
    was) and an accumulated view mass ``M`` (central views add ~1, edge glimpses
    add ~0). The corroboration bonus saturates in ``M`` so repeated *central*
    views drive C up while glancing edge detections barely count.

    Falls back to the legacy formula (ULP-identical) for nodes that predate the
    FOV accumulator so old maps / tests stay stable.
    """
    mass = node.attributes.get("fov_view_mass")
    value = node.attributes.get("fov_value_v")
    if mass is None or value is None:
        multi_view = int(node.attributes.get("multi_view_count", node.visit_count or 1) or 1)
        return 0.30 * float(node.confidence) + min(0.20, 0.04 * multi_view)
    m = float(mass)
    return 0.30 * float(value) + min(0.20, 0.20 * (m / (m + OBJECT_FOV_MASS_K)))


def object_reachability_penalty(node: SemanticNode, anchor_node_id: Optional[str]) -> float:
    """A: penalty from failed / unreachable history and bad anchor waypoints."""
    failed = int(node.attributes.get("failed_count", 0) or 0)
    unreachable = int(node.attributes.get("unreachable_count", 0) or 0)
    bad_anchor_ids = {str(x) for x in node.attributes.get("bad_anchor_waypoint_ids", []) or []}
    anchor_penalty = 0.45 if anchor_node_id in bad_anchor_ids else 0.0
    return 0.25 * failed + 0.75 * unreachable + anchor_penalty


def task_confidence(relevance: float, existence: float) -> float:
    """Task-conditioned confidence c = P(this node is the goal | evidence).

    Fuses the two signals that used to be *added* (``R + C``) into a single
    number, the way the project's premise ("task-driven confidence navigation")
    actually intends:

      * ``relevance`` (R): how well the node matches the goal semantically. It is
        *multiplicative* -- a semantic mismatch (R -> 0) drives c to ~0 no matter
        how certain we are that the thing exists, so an unrelated-but-well-seen
        object can no longer out-score exploration.
      * ``existence`` (e): how reliable the detection is (corroboration). It only
        *discounts* c, bounded below by ``TASK_CONF_RELIABILITY_FLOOR`` so a
        strong semantic match that is merely under-observed stays medium -- worth
        approaching to verify rather than being killed.

    Additive R + C could not express either behaviour (a high-C chair and a
    low-C bed ended up with similar totals). c = R * (floor + (1-floor)*e) does.
    """
    r = max(0.0, min(1.0, float(relevance)))
    e = max(0.0, min(1.0, float(existence)))
    reliability = TASK_CONF_RELIABILITY_FLOOR + (1.0 - TASK_CONF_RELIABILITY_FLOOR) * e
    return r * reliability


def object_goal_relevance(node: SemanticNode, goal: GoalNode) -> float:
    """R for the task-confidence fusion, chosen by goal modality.

    * category (a noun like "bed"): use the clean, discrete label relevance
      (bed=1.0, mattress=0.8, unrelated furniture=0.0). Embedding-only similarity
      gave generic objects ~0.5 and hijacked navigation, so it must not drive R
      for noun goals.
    * image / description: no reliable text label to match, so fall back to the
      multimodal embedding retrieval blend (``object_task_relevance``).
    """
    goal_type = str(getattr(goal, "goal_type", "category") or "category").lower().strip()
    if goal_type in ("image", "description"):
        return object_task_relevance(node, goal)
    return float(GoalRelevanceScorer.label_relevance(str(node.label or ""), goal))


def object_c_task(node: SemanticNode, goal: GoalNode) -> float:
    """Single source of truth for a confirmed object's task-conditioned
    confidence: goal relevance (modality-aware) fused with existence confidence.
    Used by both the planner ranking and the agent's sticky lock."""
    return float(task_confidence(object_goal_relevance(node, goal), float(node.confidence)))


def hypothesis_c_task(hyp: Any, goal: GoalNode) -> float:
    """Task-conditioned confidence for a goal hypothesis. Object-kind guesses use
    their label relevance to the goal; goal-directed (e.g. goal_view) guesses are
    inherently relevant (R=1). Shared by the planner and the agent's sticky lock."""
    r = (
        GoalRelevanceScorer.label_relevance(str(getattr(hyp, "label", "") or ""), goal)
        if getattr(hyp, "kind", "object") == "object"
        else 1.0
    )
    return float(task_confidence(r, float(getattr(hyp, "confidence", 0.0))))


# --- Explicit C / R / A decomposition for room / landmark / portal / frontier -
# Same philosophy as the object axes above: each intent's scalar score is split
# into task relevance (R), map confidence (C) and a reachability penalty (A),
# then re-blended by an ``*_intent_score`` that reproduces the legacy arithmetic.
# ActiveTopo reads R (and C/A) per node from these instead of the entangled
# scalar, but the planner ranking is byte-for-byte (ULP-level) unchanged.


def room_task_relevance(node: SemanticNode, goal: GoalNode) -> float:
    """R: how well this room matches the goal room priors (base + label + prob)."""
    priors = {str(room).lower().strip() for room in goal.room_prior or []}
    probs = _score_dict(node.attributes.get("room_probs", {}))
    prob = max((float(probs.get(room, 0.0)) for room in priors), default=0.0)
    label_bonus = 0.35 if str(node.label).lower().strip() in priors else 0.0
    return 0.35 + label_bonus + 0.65 * prob


def room_map_confidence(node: SemanticNode) -> float:
    """C: evidence accumulated for this room (distinct observed objects)."""
    objects = len(node.attributes.get("observed_objects", {}) or {})
    return min(0.15, 0.03 * objects)


def room_intent_score(relevance: float, map_confidence: float, dist: float) -> float:
    return relevance + map_confidence - 0.03 * dist


def landmark_task_relevance(node: SemanticNode, goal: GoalNode) -> float:
    """R: landmark relevance. Label match is a hard gate upstream, so flat base."""
    return 0.40


def landmark_map_confidence(node: SemanticNode) -> float:
    """C: detection confidence for the landmark node."""
    return 0.35 * float(node.confidence)


def landmark_intent_score(relevance: float, map_confidence: float, dist: float) -> float:
    return relevance + map_confidence - 0.03 * dist


def portal_task_relevance(node: SemanticNode, goal: GoalNode) -> float:
    """R: base traversal value plus a bonus when the portal links a prior room."""
    priors = {str(room).lower().strip() for room in goal.room_prior or []}
    connected = {str(x).lower().strip() for x in node.attributes.get("connected_rooms", []) or []}
    room_match = 0.25 if priors and connected.intersection(priors) else 0.0
    return 0.25 + room_match


def portal_map_confidence(node: SemanticNode) -> float:
    """C: confidence that this portal exists / is a real opening."""
    return 0.25 * float(node.confidence)


def portal_reachability_penalty(
    portal_failed: float,
    portal_unreachable: float,
    link_penalty: float,
) -> float:
    """A: portal-level failure history plus accumulated per-link penalties."""
    return 0.30 * portal_failed + 0.80 * portal_unreachable + link_penalty


def portal_intent_score(
    relevance: float,
    map_confidence: float,
    reachability_penalty: float,
    dist: float,
) -> float:
    return relevance + map_confidence - 0.03 * dist - reachability_penalty


def frontier_task_relevance(semantic: float) -> float:
    """R: goal relevance of a frontier (semantic value, already max'd w/ room prob)."""
    return 0.32 * float(semantic)


def frontier_map_confidence(frontier_conf: float) -> float:
    """C: the map's confidence in the frontier being a real, useful opening."""
    return 0.38 * float(frontier_conf)


def frontier_reachability_penalty(
    failed: float,
    unreachable: float,
    visits: float,
    recent: float,
    stale_penalty: float,
) -> float:
    """A: risk from failed / unreachable / over-visited / recently-tried history."""
    return min(
        1.35,
        0.20 * failed
        + 0.45 * unreachable
        + 0.08 * min(4.0, visits)
        + 0.05 * min(6.0, recent)
        + stale_penalty,
    )


def frontier_intent_score(
    relevance: float,
    map_confidence: float,
    novelty: float,
    dist: float,
    reachability_penalty: float,
    heading_align: float = 0.0,
) -> float:
    return (
        0.18
        + map_confidence
        + relevance
        + 0.18 * novelty
        + 0.20 * heading_align
        - 0.025 * dist
        - reachability_penalty
    )
