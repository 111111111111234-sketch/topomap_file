"""FOV/center-weighted object confidence fusion (ApexNav / VLFM style).

``object_map_confidence`` now reads a FOV-weighted value (V) and view mass (M)
accumulated by ``accumulate_object_fov_confidence``. Central, repeated
observations build confidence while glancing edge detections barely count.
Nodes without the accumulator fall back to the legacy formula (ULP-identical).

Run directly (``python conftopo/tests/test_object_fov_confidence.py``) or via
pytest.
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_memory import (
    accumulate_object_fov_confidence,
    object_fov_weight,
)
from conftopo.agents.goat_agent1_planner import (
    OBJECT_FOV_MASS_K,
    object_map_confidence,
)
from conftopo.core.dynamic_topo_map import NodeType, SemanticNode

TOL = 1e-12


def _node(*, confidence=0.5, attributes=None):
    return SemanticNode(
        node_id="obj",
        node_type=NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        confidence=confidence,
        label="chair",
        attributes=dict(attributes or {}),
    )


def _bbox_for_center(cx: float, width: float = 0.1):
    return [cx - width / 2.0, 0.4, cx + width / 2.0, 0.6]


def _fov_weight_ref(cx):
    if cx is None:
        return 0.5
    theta = float(np.clip((cx - 0.5) / 0.5, -1.0, 1.0))
    return max(float(np.cos(theta * (np.pi / 2.0)) ** 2), 1e-3)


def test_fov_weight_center_edge():
    assert abs(object_fov_weight(_bbox_for_center(0.5)) - 1.0) <= 1e-9
    # A near-edge detection contributes almost no mass.
    assert object_fov_weight(_bbox_for_center(0.02, width=0.04)) < 0.05
    # Unknown bbox -> neutral 0.5.
    assert abs(object_fov_weight(None) - 0.5) <= TOL


def test_accumulation_matches_recurrence():
    node = _node()
    observations = [(0.50, 0.80), (0.30, 0.60), (0.95, 0.90), (0.50, 0.70)]
    m_ref, v_ref = 0.0, 0.0
    for cx, score in observations:
        accumulate_object_fov_confidence(node, bbox=_bbox_for_center(cx), detection_score=score)
        w = _fov_weight_ref(cx)
        m_new = m_ref + w
        v_ref = (w * score + m_ref * v_ref) / m_new
        m_ref = m_new
        assert abs(node.attributes["fov_view_mass"] - m_ref) <= TOL
        assert abs(node.attributes["fov_value_v"] - v_ref) <= TOL
        expected = 0.30 * v_ref + min(0.20, 0.20 * (m_ref / (m_ref + OBJECT_FOV_MASS_K)))
        assert abs(object_map_confidence(node) - expected) <= TOL


def test_central_repeats_beat_edge_glimpses():
    central = _node()
    for _ in range(4):
        accumulate_object_fov_confidence(central, bbox=_bbox_for_center(0.5), detection_score=0.7)
    edge = _node()
    for _ in range(4):
        accumulate_object_fov_confidence(edge, bbox=_bbox_for_center(0.03, width=0.06), detection_score=0.7)
    assert object_map_confidence(central) > object_map_confidence(edge)


def test_legacy_fallback_is_ulp_identical():
    node = _node(confidence=0.62, attributes={"multi_view_count": 3})
    legacy = 0.30 * 0.62 + min(0.20, 0.04 * 3)
    assert "fov_view_mass" not in node.attributes
    assert abs(object_map_confidence(node) - legacy) <= TOL


if __name__ == "__main__":
    test_fov_weight_center_edge()
    test_accumulation_matches_recurrence()
    test_central_repeats_beat_edge_glimpses()
    test_legacy_fallback_is_ulp_identical()
    print("ok")
