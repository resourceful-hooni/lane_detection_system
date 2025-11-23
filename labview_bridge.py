"""
labview_bridge.py
LabVIEW과의 연동을 돕기 위한 경량 브릿지 모듈.
- 최신 차선/경로 상태를 JSON 파일로 기록
- (옵션) 오버레이 프레임을 이미지로 저장하여 LabVIEW에서 로드 가능
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np


class LabViewBridge:
    """LabVIEW 연동을 위한 파일 기반 브릿지."""

    def __init__(
        self,
        state_path: str = "labview_bridge/state.json",
        frame_path: Optional[str] = "labview_bridge/overlay.jpg",
        write_frame: bool = False
    ):
        self.state_path = state_path
        self.frame_path = frame_path
        self.write_frame = write_frame and frame_path is not None
        self._lock = threading.Lock()

        if self.state_path:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        if self.write_frame and self.frame_path:
            os.makedirs(os.path.dirname(self.frame_path), exist_ok=True)

    def update(
        self,
        lane_result: Dict[str, Any],
        path_result: Dict[str, Any],
        fps: float,
        frame: Optional[np.ndarray] = None
    ):
        """최신 상태를 JSON/이미지 파일로 기록."""
        if not self.state_path:
            return

        def _to_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, np.generic):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        gap_flags = lane_result.get("gap_flags", {}) or {}
        gap_flags = {key: bool(val) for key, val in gap_flags.items()}

        payload = {
            "timestamp": time.time(),
            "fps": fps,
            "frame": lane_result.get("frame_count", 0),
            "lane": {
                "detected": lane_result.get("detected", False),
                "gap_flags": gap_flags,
                "had_gaps": any(gap_flags.values()) if gap_flags else False
            },
            "path": {
                "valid": path_result.get("valid", False),
                "center_offset": _to_float(path_result.get("center_offset")),
                "steering_angle": _to_float(path_result.get("steering_angle")),
                "left_curvature": _to_float(path_result.get("left_curvature")),
                "right_curvature": _to_float(path_result.get("right_curvature")),
                "lane_departure_warning": bool(path_result.get("lane_departure_warning", False))
            }
        }

        with self._lock:
            with open(self.state_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)

            if self.write_frame and self.frame_path and frame is not None:
                cv2.imwrite(self.frame_path, frame)