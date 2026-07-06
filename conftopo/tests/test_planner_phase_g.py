"""Phase G: multimodal task relevance.

object_task_relevance dispatches its blend weights by goal.goal_type:
  - image       : visual-heavy (identical to the legacy fixed blend)
  - category    : label/room emphasised
  - description : attribute + spatial-relation emphasised (reuses RelationScorer)
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_planner import (
    _object_retrieval_score,
    _relevance_weights,
)
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode


def _node():
    m = DynamicTopoMap()
    emb = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
    m.add_node(
        NodeType.OBJECT, np.array([1.0, 0.0, 0.0]), embedding=emb, confidence=0.8,
        label="mug", node_id="obj",
        attributes={"room_context": "kitchen", "spatial_relation": ["next to table"]},
    )
    return m.get_node("obj")


def _goal(goal_type):
    g = GoalNode(target_object="mug", room_prior=["kitchen"], landmarks=["table"],
                 goal_type=goal_type)
    g.target_embedding = np.array([0.9, 0.6, 0.0, 0.0], dtype=np.float32)
    return g


def test_weight_profiles_differ():
    wi = _relevance_weights(_goal("image"))
    wc = _relevance_weights(_goal("category"))
    wd = _relevance_weights(_goal("description"))
    assert wi["visual"] > wc["visual"], "image is more visual than category"
    assert wd["relation"] > 0.0 and wi["relation"] == 0.0
    assert wc["room"] > wi["room"], "category leans on room prior"
    # unknown type falls back to category
    assert _relevance_weights(_goal("route")) == wc


def test_image_profile_matches_legacy_blend():
    node = _node()
    r = _object_retrieval_score(node, _goal("image"))
    legacy = (
        0.60 * r["visual_similarity"]
        + 0.20 * r["text_similarity"]
        + 0.10 * r["attribute_similarity"]
        + 0.10 * r["room_consistency"]
        + r["label_bonus"]
    )
    legacy = max(0.0, min(1.0, legacy))
    assert abs(r["score"] - legacy) < 1e-12, (r["score"], legacy)
    assert r["relation_similarity"] == 0.0  # image ignores relation


def test_description_uses_relation_evidence():
    node = _node()
    r = _object_retrieval_score(node, _goal("description"))
    # the observed "next to table" matches the goal landmark "table"
    assert r["relation_similarity"] > 0.0, r
    # blend reproduces the description weights + relation term
    expected = (
        0.35 * r["visual_similarity"]
        + 0.15 * r["text_similarity"]
        + 0.25 * r["attribute_similarity"]
        + 0.15 * r["relation_similarity"]
        + 0.10 * r["room_consistency"]
        + r["label_bonus"]
    )
    expected = max(0.0, min(1.0, expected))
    assert abs(r["score"] - expected) < 1e-12, (r["score"], expected)


if __name__ == "__main__":
    test_weight_profiles_differ()
    test_image_profile_matches_legacy_blend()
    test_description_uses_relation_evidence()
    print("PASS: Phase G multimodal relevance (per-goal-type weight dispatch)")
