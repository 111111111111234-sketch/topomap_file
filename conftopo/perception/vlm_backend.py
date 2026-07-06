"""VLM backend abstraction, fake backend for testing, and Qwen3-VL backend."""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from conftopo.perception.vlm_prompts import (
    build_user_prompt,
    get_system_prompt,
    parse_vlm_json,
)

logger = logging.getLogger(__name__)


class VLMBackendBase(ABC):
    """Abstract interface for Vision-Language Model backends."""

    supports_multi_image: bool = True
    @abstractmethod
    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
        mode: str = "explore",
        previous_rgb: Optional[np.ndarray] = None,
        action_history: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Send an image + goal description to the VLM and return parsed JSON.

        *mode* selects the system prompt: ``"explore"`` for discovery,
        ``"confirm"`` for strict stop verification.  When *previous_rgb*
        is provided, backends that support multi-image may use it for
        temporal comparison (e.g. relative_progress).

        Returns a dict with the canonical schema::

            {
                "room": {"label": str, "confidence": float},
                "objects": [{"label": str, "bbox": [x1,y1,x2,y2], "attributes": {"color": {"value": str, "confidence": float}, "shape": {"value": str, "confidence": float}}, "confidence": float}, ...],
                "goal_visible": bool,
                "goal_match_confidence": float,
                "target_direction": str,
                "target_visibility": str,
                "apparent_scale": str,
                "relative_progress": str,
                "stop_candidate": bool,
                "recommended_action": str,
                "goal_reason": str,
                "portals": [
                    {"type": str, "state": str, "bearing": str, "traversable": bool},
                    ...
                ],
                "scene_summary": str,
                "uncertainty": float,
            }
        """


class FakeVLMBackend(VLMBackendBase):
    """Deterministic fake backend for unit / integration tests.

    Always returns a fixed response.  Optionally accepts a ``response``
    dict to customise the output.
    """

    def __init__(self, response: Optional[Dict[str, Any]] = None):
        self._response = response or {
            "room": {"label": "unknown", "confidence": 0.5},
            "objects": [],
            "goal_visible": False,
            "goal_reason": "",
            "portals": [],
            "scene_summary": "",
            "uncertainty": 0.5,
        }
        self.calls: int = 0

    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
        mode: str = "explore",
        previous_rgb: Optional[np.ndarray] = None,
        action_history: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self.calls += 1
        return dict(self._response)


def _rgb_to_base64_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    """Encode an HWC uint8 RGB array to a base64 JPEG string."""
    from PIL import Image  # lazy import to avoid hard dependency

    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Qwen3VLBackend(VLMBackendBase):
    """Qwen3-VL backend via OpenAI-compatible chat API (vLLM / SGLang).

    Expects the VLM server to be running at ``api_base`` and serving the
    model named ``model``.  Uses the ``openai`` Python SDK under the hood.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: float = 5.0,
        max_tokens: int = 1536,
        temperature: float = 0.1,
        seed: Optional[int] = None,
    ):
        self._api_base = api_base
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._temperature = float(temperature)
        self._seed = seed
        self.supports_multi_image = False
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Qwen3VLBackend. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(
            base_url=self._api_base,
            api_key="EMPTY",
            timeout=self._timeout,
        )

    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
        mode: str = "explore",
        previous_rgb: Optional[np.ndarray] = None,
        action_history: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_client()

        # Single-image backends: query current and previous separately, merge.
        if previous_rgb is not None and not self.supports_multi_image:
            primary = self._query_single_image(
                rgb, goal_text, context, mode, action_history,
            )
            secondary = self._query_single_image(
                previous_rgb, goal_text, context, mode, action_history,
            )
            sec_progress = str(secondary.get("relative_progress", "uncertain")).lower()
            if sec_progress != "uncertain":
                primary["relative_progress"] = sec_progress
            return primary

        b64_img = _rgb_to_base64_jpeg(rgb)
        has_previous = bool(previous_rgb is not None and self.supports_multi_image)
        user_text = build_user_prompt(
            goal_text, context, mode=mode,
            has_previous_image=has_previous,
            action_history=list(action_history or []),
        )

        messages = [
            {"role": "system", "content": get_system_prompt(mode)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        return self._query_single_image_raw(messages)


    def _query_single_image_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single VLM query and return the parsed JSON response."""
        query_ok = True
        error = ""
        try:
            create_kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if self._seed is not None:
                create_kwargs["seed"] = int(self._seed)
            response = self._client.chat.completions.create(**create_kwargs)
            raw_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("VLM query failed: %s", exc)
            query_ok = False
            error = str(exc)
            raw_text = ""
        parsed = parse_vlm_json(raw_text)
        parsed["_query_ok"] = query_ok
        if error:
            parsed["_error"] = error
        parsed["_raw_text"] = raw_text
        parsed["_raw_text_truncated"] = raw_text[:2000]
        return parsed

    def _query_single_image(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
        mode: str = "explore",
        action_history: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Build a single-image message and query the VLM."""
        b64_img = _rgb_to_base64_jpeg(rgb)
        user_text = build_user_prompt(
            goal_text, context, mode=mode,
            has_previous_image=False,
            action_history=list(action_history or []),
        )
        messages = [
            {"role": "system", "content": get_system_prompt(mode)},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return self._query_single_image_raw(messages)
