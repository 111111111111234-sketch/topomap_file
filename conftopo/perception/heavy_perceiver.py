"""Object-level heavy perception interface.

Heavy perception is optional at runtime. The default GroundingDINO backend is
loaded lazily so Phase 2 CLIP-only execution and tests do not require model
weights or the GroundingDINO package.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
import hashlib
import os

import numpy as np


def normalize_bbox(raw: Any) -> Optional[List[float]]:
    """Validate and normalise a raw bbox to ``[x1, y1, x2, y2]``.

    Returns ``None`` when the input cannot be interpreted as a valid bbox
    (wrong type, fewer than 4 numbers, degenerate box, etc.).
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


@dataclass
class ObjectObservation:
    """Normalized object-level detection result."""

    label: str
    bbox: Optional[List[float]]
    confidence: float
    embedding: Optional[np.ndarray] = None
    source: str = "heavy"
    view_heading: float = 0.0
    step_id: int = 0
    visible: Optional[bool] = None
    visibility: str = "unknown"
    bearing: str = "unknown"
    range_bin: str = "unknown"
    spatial_relation: List[str] = None
    bbox_confidence: str = "medium"
    room_context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spatial_relation is None:
            self.spatial_relation = []
        elif isinstance(self.spatial_relation, str):
            self.spatial_relation = [self.spatial_relation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bbox": [float(v) for v in self.bbox] if self.bbox is not None else None,
            "confidence": float(self.confidence),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "source": self.source,
            "view_heading": float(self.view_heading),
            "step_id": int(self.step_id),
            "visible": self.visible,
            "visibility": self.visibility,
            "bearing": self.bearing,
            "range_bin": self.range_bin,
            "bbox_confidence": self.bbox_confidence,
            "spatial_relation": list(self.spatial_relation),
            "room_context": self.room_context,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectObservation":
        embedding = data.get("embedding")
        relation = data.get("spatial_relation", data.get("relation", []))
        if isinstance(relation, str):
            relation = [relation]
        if relation is None:
            relation = []
        return cls(
            label=str(data.get("label", "")),
            bbox=normalize_bbox(data.get("bbox")),
            confidence=float(data.get("confidence", 0.0)),
            embedding=np.array(embedding, dtype=np.float32) if embedding is not None else None,
            source=str(data.get("source", "heavy")),
            view_heading=float(data.get("view_heading", 0.0)),
            step_id=int(data.get("step_id", 0)),
            visible=data.get("visible"),
            visibility=str(data.get("visibility", "unknown")),
            bearing=str(data.get("bearing", "unknown")),
            range_bin=str(data.get("range_bin", data.get("range", "unknown"))),
            spatial_relation=[str(x) for x in relation],
            bbox_confidence=str(data.get("bbox_confidence", "medium")),
            room_context=data.get("room_context"),
            attributes=dict(data.get("attributes") or {}),
        )


class GroundingDINOBackend:
    """Lazy GroundingDINO adapter.

    This class intentionally keeps imports out of module import time. A caller
    may provide an already constructed backend with the same `detect` method.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.model = model
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._predict_fn = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        config_path = self.config_path or os.environ.get("GROUNDINGDINO_CONFIG")
        checkpoint_path = self.checkpoint_path or os.environ.get("GROUNDINGDINO_CHECKPOINT")
        if not config_path or not checkpoint_path:
            raise RuntimeError(
                "GroundingDINO backend requires config/checkpoint paths "
                "or GROUNDINGDINO_CONFIG/GROUNDINGDINO_CHECKPOINT"
            )
        try:
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise RuntimeError("GroundingDINO package is not installed") from exc
        self.model = load_model(config_path, checkpoint_path, device=self.device)
        self._predict_fn = predict

    def detect(self, image: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
        self._ensure_model()
        if hasattr(self.model, "detect"):
            return self.model.detect(image=image, labels=list(labels))
        if self._predict_fn is None:
            from groundingdino.util.inference import predict
            self._predict_fn = predict
        caption = " . ".join(str(label) for label in labels)
        image_array = np.asarray(image)
        image_tensor = self._prepare_image(image_array)
        boxes, logits, phrases = self._predict_fn(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes_np = _as_numpy(boxes)
        logits_np = _as_numpy(logits)
        height, width = image_array.shape[:2]
        rows = []
        for box, score, phrase in zip(boxes_np, logits_np, phrases):
            rows.append({
                "label": _clean_phrase_label(str(phrase), labels),
                "bbox": _cxcywh_to_xyxy(box, width, height),
                "confidence": float(score),
                "source": "groundingdino",
            })
        return rows

    def _prepare_image(self, image: Any) -> Any:
        """Convert RGB arrays to the normalized tensor GroundingDINO expects."""
        try:
            from importlib import import_module
            import torch
            from PIL import Image
            T = import_module("groundingdino.datasets.transforms")
        except ImportError as exc:
            raise RuntimeError("GroundingDINO image preprocessing dependencies are not installed") from exc

        if isinstance(image, torch.Tensor):
            return image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if arr.size > 0 and float(arr.max()) <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected image with shape [H,W,C], got {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB/RGBA image, got shape {arr.shape}")

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pil = Image.fromarray(arr).convert("RGB")
        image_tensor, _ = transform(image_pil, None)
        return image_tensor



def normalize_bbox_flexible(raw: Any) -> Optional[List[float]]:
    """Like `normalize_bbox` but handles VLM-common output quirks:

    - string-encoded lists (e.g. ``"[0.1, 0.2, 0.3, 0.4]"``)
    - dicts with various key names
    - lists of strings

    Also adds strict validation that the original lacks:
    - rejects NaN / inf
    - clamps values slightly outside [0, 1] (logs a warning)
    - rejects grossly out-of-range values
    - rejects degenerate (zero-area) boxes
    """
    if raw is None:
        return None
    # String-encoded list
    if isinstance(raw, str):
        import ast, json
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
                    raw = parsed
                    break
            except Exception:
                continue
        else:
            return None
    # Dict format  (try common key-name conventions)
    if isinstance(raw, dict):
        for x1k, x2k in (("x1","x2"),("xmin","xmax"),("min_x","max_x"),
                         ("left","right"),("bbox_x1","bbox_x2")):
            x1 = raw.get(x1k, raw.get("bbox_"+x1k))
            x2 = raw.get(x2k, raw.get("bbox_"+x2k))
            if x1 is not None and x2 is not None:
                for y1k, y2k in (("y1","y2"),("ymin","ymax"),("min_y","max_y"),
                                 ("top","bottom"),("bbox_y1","bbox_y2")):
                    y1 = raw.get(y1k, raw.get("bbox_"+y1k))
                    y2 = raw.get(y2k, raw.get("bbox_"+y2k))
                    if y1 is not None and y2 is not None:
                        raw = [x1, y1, x2, y2]
                        break
                break
        else:
            return None
    # List of strings → floats
    if isinstance(raw, (list, tuple)):
        try:
            raw = [float(v) for v in raw[:4]]
        except (TypeError, ValueError):
            import logging
            logging.getLogger("conftopo.perception").debug(
                "normalize_bbox_flexible: non-numeric values in %r", raw)
            return None
    else:
        return None

    # Auto-detect coordinate scale. Qwen-VL family natively emits 0-1000 boxes;
    # some models use 0-100. Rescale into [0, 1] instead of rejecting the whole
    # box (which would drop the detection entirely for that frame).
    import math
    _finite = [v for v in raw[:4] if math.isfinite(v)]
    if _finite:
        _m = max(abs(v) for v in _finite)
        if _m > 1.05:
            _divisor = 1000.0 if _m > 100.0 else 100.0
            raw = [v / _divisor for v in raw[:4]]

    # --- strict validation ---
    clamped = False
    out = []
    for v in raw[:4]:
        if not math.isfinite(v):
            import logging
            logging.getLogger("conftopo.perception").debug(
                "normalize_bbox_flexible: NaN/inf in %r", raw)
            return None
        if v < -0.05 or v > 1.05:
            # Grossly out of range – reject
            import logging
            logging.getLogger("conftopo.perception").debug(
                "normalize_bbox_flexible: value %s out of range in %r", v, raw)
            return None
        if v < 0.0 or v > 1.0:
            clamped = True
        out.append(min(1.0, max(0.0, v)))

    if out[2] <= out[0] or out[3] <= out[1]:
        import logging
        logging.getLogger("conftopo.perception").debug(
            "normalize_bbox_flexible: degenerate box %r", out)
        return None

    area = (out[2] - out[0]) * (out[3] - out[1])
    if area < 1e-4:
        import logging
        logging.getLogger("conftopo.perception").debug(
            "normalize_bbox_flexible: near-zero area %r", out)
        return None

    if clamped:
        import logging
        logging.getLogger("conftopo.perception").debug(
            "normalize_bbox_flexible: clamped %r → %r", raw, out)
    return out


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> List[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
        cx *= width
        bw *= width
        cy *= height
        bh *= height
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]


def _clean_phrase_label(phrase: str, labels: Sequence[str]) -> str:
    lower = phrase.lower()
    for label in labels:
        if str(label).lower() in lower:
            return str(label)
    return phrase.split(",")[0].strip()


class FakeGroundingDINOBackend:
    """Deterministic test backend that does not require model weights."""

    def __init__(self, detections: Optional[Iterable[Dict[str, Any]]] = None):
        self._detections = list(detections) if detections is not None else None
        self.calls = 0

    def detect(self, image: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
        self.calls += 1
        if self._detections is not None:
            return [dict(row) for row in self._detections]
        rows = []
        for idx, label in enumerate(labels):
            digest = hashlib.sha1(str(label).encode("utf-8")).digest()
            x1 = 5.0 + float(digest[0] % 20)
            y1 = 6.0 + float(digest[1] % 20)
            rows.append({
                "label": str(label),
                "bbox": [x1, y1, x1 + 24.0 + idx, y1 + 18.0 + idx],
                "confidence": 0.75,
                "source": "fake_groundingdino",
            })
        return rows


class HeavyPerceiver:
    """Normalize object-level detections from a heavy backend."""

    def __init__(self, backend: Optional[Any] = None, min_confidence: float = 0.25):
        self.backend = backend if backend is not None else GroundingDINOBackend()
        self.min_confidence = float(min_confidence)

    def perceive(
        self,
        rgb: Any,
        labels: Sequence[str],
        *,
        visual_embedding: Optional[np.ndarray] = None,
        view_heading: float = 0.0,
        step_id: int = 0,
    ) -> List[ObjectObservation]:
        if rgb is None or not labels:
            return []
        raw_rows = self.backend.detect(rgb, labels)
        observations = []
        for row in raw_rows:
            label = str(row.get("label", "")).strip()
            conf = float(row.get("confidence", row.get("score", 0.0)))
            if not label or conf < self.min_confidence:
                continue
            embedding = row.get("embedding")
            if embedding is None:
                embedding = visual_embedding
            observations.append(ObjectObservation(
                label=label,
                bbox=normalize_bbox(row.get("bbox")),
                confidence=conf,
                embedding=np.array(embedding, dtype=np.float32) if embedding is not None else None,
                source=str(row.get("source", "groundingdino")),
                view_heading=float(row.get("view_heading", view_heading)),
                step_id=int(row.get("step_id", step_id)),
            ))
        return observations
