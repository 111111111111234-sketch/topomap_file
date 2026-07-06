"""Phase A regression: R/C/A decomposition must reproduce the legacy intent
scores (ULP-level) for room / landmark / portal / frontier.

Run directly (``python conftopo/tests/test_planner_rca_split.py``) or via pytest.
"""

from __future__ import annotations

import random

import numpy as np

from conftopo.agents.goat_agent1_planner import (
    frontier_intent_score,
    frontier_map_confidence,
    frontier_reachability_penalty,
    frontier_task_relevance,
    landmark_intent_score,
    landmark_map_confidence,
    landmark_task_relevance,
    portal_intent_score,
    portal_map_confidence,
    portal_reachability_penalty,
    room_intent_score,
    room_map_confidence,
)
from conftopo.core.dynamic_topo_map import NodeType, SemanticNode

TOL = 1e-12


def _node(node_type, *, confidence=0.5, label="", attributes=None):
    return SemanticNode(
        node_id="n",
        node_type=node_type,
        position=np.zeros(3, dtype=np.float32),
        confidence=confidence,
        label=label,
        attributes=dict(attributes or {}),
    )


def test_room_score_matches_legacy():
    rng = random.Random(0)
    for _ in range(2000):
        label_bonus = rng.choice([0.0, 0.35])
        prob = rng.random()
        objects = rng.randint(0, 12)
        dist = rng.uniform(0.0, 10.0)
        legacy = 0.35 + label_bonus + 0.65 * prob + min(0.15, 0.03 * objects) - 0.03 * dist
        node = _node(NodeType.ROOM, attributes={"observed_objects": {str(i): 1 for i in range(objects)}})
        relevance = 0.35 + label_bonus + 0.65 * prob
        new = room_intent_score(relevance, room_map_confidence(node), dist)
        assert abs(new - legacy) <= TOL, (new, legacy)


def test_landmark_score_matches_legacy():
    rng = random.Random(1)
    for _ in range(2000):
        conf = rng.random()
        dist = rng.uniform(0.0, 10.0)
        legacy = 0.40 + 0.35 * conf - 0.03 * dist
        node = _node(NodeType.LANDMARK, confidence=conf)
        new = landmark_intent_score(landmark_task_relevance(node, None), landmark_map_confidence(node), dist)
        assert abs(new - legacy) <= TOL, (new, legacy)


def test_portal_score_matches_legacy_ulp():
    rng = random.Random(2)
    for _ in range(2000):
        room_match = rng.choice([0.0, 0.25])
        conf = rng.random()
        dist = rng.uniform(0.0, 10.0)
        pf = rng.randint(0, 4)
        pu = rng.randint(0, 3)
        failed_penalty = rng.uniform(0.0, 2.0)
        visit_penalty = rng.uniform(0.0, 1.0)
        legacy = (
            0.25 + room_match + 0.25 * conf - 0.03 * dist
            - 0.30 * pf - 0.80 * pu - failed_penalty - visit_penalty
        )
        node = _node(NodeType.PORTAL, confidence=conf)
        relevance = 0.25 + room_match
        reach = portal_reachability_penalty(pf, pu, failed_penalty + visit_penalty)
        new = portal_intent_score(relevance, portal_map_confidence(node), reach, dist)
        # lumped subtraction -> ULP-level, not bit-exact.
        assert abs(new - legacy) <= 1e-9, (new, legacy)


def test_frontier_score_matches_legacy():
    rng = random.Random(3)
    for _ in range(2000):
        frontier_conf = rng.random()
        semantic = rng.random()
        novelty = rng.random()
        dist = rng.uniform(0.0, 10.0)
        failed = rng.randint(0, 5)
        unreachable = rng.randint(0, 4)
        visits = rng.randint(0, 10)
        recent = rng.randint(0, 12)
        stale_penalty = rng.choice([0.0, 0.35])
        risk = min(
            1.35,
            0.20 * failed + 0.45 * unreachable + 0.08 * min(4.0, visits)
            + 0.05 * min(6.0, recent) + stale_penalty,
        )
        legacy = (
            0.18 + 0.38 * frontier_conf + 0.32 * semantic + 0.18 * novelty
            - 0.025 * dist - risk
        )
        new = frontier_intent_score(
            frontier_task_relevance(semantic),
            frontier_map_confidence(frontier_conf),
            novelty,
            dist,
            frontier_reachability_penalty(failed, unreachable, visits, recent, stale_penalty),
        )
        assert abs(new - legacy) <= TOL, (new, legacy)


if __name__ == "__main__":
    test_room_score_matches_legacy()
    test_landmark_score_matches_legacy()
    test_portal_score_matches_legacy_ulp()
    test_frontier_score_matches_legacy()
    print("PASS: all R/C/A splits reproduce legacy intent scores")
