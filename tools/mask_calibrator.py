"""Interactive hood mask polygon calibrator.

Usage:
    python tools/mask_calibrator.py [--camera-index N] [--frame path]

Controls:
    Left click : add vertex (in order)
    U          : undo last vertex
    C          : clear all vertices
    S          : save vertices to calibration/hood_mask.json
    ENTER      : print normalized coordinates to console
    Q / ESC    : quit

If --frame is provided, the script loads that image instead of grabbing
from the camera. Saved JSON contains both pixel coordinates and normalized
coordinates (0~1) so it can be pasted into config.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config


def load_reference_frame(camera_index: int, frame_path: str | None) -> np.ndarray:
    if frame_path:
        image = cv2.imread(frame_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {frame_path}")
        return image

    cfg = get_config()
    if camera_index is None:
        camera_index = cfg.camera.camera_index

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera_index}를 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("카메라 프레임을 읽을 수 없습니다.")
    return frame


def normalize_polygon(points: List[Tuple[int, int]], width: int, height: int) -> List[Tuple[float, float]]:
    return [(round(x / width, 4), round(y / height, 4)) for x, y in points]


def draw_overlay(frame: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    display = frame.copy()
    if len(points) >= 2:
        cv2.polylines(display, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
    for idx, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    instructions = [
        "[Left Click] add vertex",
        "[U] undo", "[C] clear", "[S] save JSON", "[Enter] print coords", "[Q] quit"
    ]
    y = 25
    for line in instructions:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    return display


def save_polygon(points: List[Tuple[int, int]], size: Tuple[int, int], output: Path) -> None:
    width, height = size
    normalized = normalize_polygon(points, width, height)
    payload = {
        "width": width,
        "height": height,
        "polygon_pixels": points,
        "polygon_normalized": normalized,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 완료: {output}")
    print("polygon_normalized =", normalized)


def main():
    parser = argparse.ArgumentParser(description="차량 보닛 마스크 폴리곤 보정 도구")
    parser.add_argument("--camera-index", type=int, default=None, help="캡처에 사용할 카메라 인덱스")
    parser.add_argument("--frame", type=str, help="이미지 경로 (지정 시 카메라 대신 사용)")
    parser.add_argument("--output", type=str, default="calibration/hood_mask.json", help="결과 JSON 경로")
    args = parser.parse_args()

    frame = load_reference_frame(args.camera_index, args.frame)
    height, width = frame.shape[:2]
    points: List[Tuple[int, int]] = []

    window_name = "Hood Mask Calibrator"
    cv2.namedWindow(window_name)

    def mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(window_name, mouse)

    while True:
        display = draw_overlay(frame, points)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key in (ord('q'), ord('Q')):
            break
        elif key in (ord('u'), ord('U')):
            if points:
                removed = points.pop()
                print(f"[INFO] Undo: {removed}")
        elif key in (ord('c'), ord('C')):
            points.clear()
            print("[INFO] Cleared all points")
        elif key in (ord('\r'), ord('\n')):  # Enter
            if points:
                print("normalized polygon:", normalize_polygon(points, width, height))
            else:
                print("[WARN] 포인트가 없습니다.")
        elif key in (ord('s'), ord('S')):
            if len(points) < 3:
                print("[WARN] 최소 3개의 포인트가 필요합니다.")
                continue
            save_polygon(points, (width, height), Path(args.output))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
