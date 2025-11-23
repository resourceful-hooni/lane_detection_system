"""
main.py
메인 실행 파일
모든 모듈을 통합하여 실시간 차선 검출 시스템 실행
"""

import argparse
import os
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# 커스텀 모듈
from typing import Optional

from config import get_config
from lane_detector import LaneDetector
from path_planner import PathPlanner
from gui_controller import GUIController
from data_logger import DataLogger
from labview_bridge import LabViewBridge


class LaneDetectionSystem:
    """차선 검출 시스템 메인 클래스"""
    
    def __init__(
        self,
        camera_index: Optional[int] = None,
        enable_gui: bool = False,
        display_window: bool = False,
        labview_bridge: Optional[LabViewBridge] = None,
        max_frames: Optional[int] = None
    ):
        """초기화"""
        print("="*60)
        print("차선 검출 시스템 초기화 중...")
        print("="*60)
        
        # 설정 로드
        self.config = get_config()

        if camera_index is not None:
            print(f"[INFO] 카메라 인덱스를 {camera_index}으로 오버라이드합니다.")
            self.config.camera.camera_index = camera_index
        
        # 모듈 초기화
        self.lane_detector = LaneDetector()
        self.path_planner = PathPlanner()
        self.data_logger = DataLogger()
        
        self.enable_gui = enable_gui and GUIController is not None
        self.display_window = display_window
        self.bridge = labview_bridge
        self.gui = GUIController(update_callback=self._on_parameter_change) if self.enable_gui else None
        self.max_frames = max_frames if max_frames and max_frames > 0 else None
        self.frame_size = (self.config.camera.height, self.config.camera.width)
        
        # 카메라 초기화
        self.camera = self._initialize_camera()
        self.lane_detector.set_hood_mask(self.frame_size)
        
        # 실행 상태
        self.running = True
        
        # FPS 계산용
        self.prev_time = time.time()
        self.fps = 0.0
        
        # 프레임 카운터
        self.frame_number = 0
        
        print("[✓] 모든 모듈 초기화 완료!")
        print("="*60)
    
    def _backend_to_string(self, backend: int) -> str:
        mapping = {
            cv2.CAP_ANY: "CAP_ANY",
            cv2.CAP_DSHOW: "CAP_DSHOW",
            cv2.CAP_MSMF: "CAP_MSMF",
            cv2.CAP_VFW: "CAP_VFW",
        }
        return mapping.get(backend, str(backend))

    def _open_camera_with_backends(self, index: int) -> cv2.VideoCapture:
        """다양한 백엔드로 카메라 연결을 시도합니다."""
        backend_candidates = []
        if os.name == "nt":
            backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backend_candidates = [cv2.CAP_ANY]

        last_error = None
        for backend in backend_candidates:
            print(f"[INFO] 백엔드 {self._backend_to_string(backend)}로 카메라 연결 시도...")
            try:
                cap = cv2.VideoCapture(index, backend)
            except Exception as exc:
                last_error = exc
                continue

            if cap.isOpened():
                print(f"[✓] 카메라 백엔드 선택: {self._backend_to_string(backend)}")
                return cap

            cap.release()

        message = f"카메라 인덱스 {index}를 어떤 백엔드로도 열 수 없습니다."
        if last_error:
            message += f" 마지막 오류: {last_error}"
        raise RuntimeError(message)

    def _initialize_camera(self) -> cv2.VideoCapture:
        """
        카메라 초기화
        
        Returns:
            VideoCapture 객체
        """
        print(f"[INFO] 카메라 초기화 중... (인덱스: {self.config.camera.camera_index})")
        
        cap = self._open_camera_with_backends(self.config.camera.camera_index)
        
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        # 실제 설정된 값 확인
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[✓] 카메라 설정: {actual_width}x{actual_height} @ {actual_fps}fps")
        self.frame_size = (actual_height, actual_width)
        
        # 자동 노출 설정
        if self.config.camera.auto_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 자동
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 수동
            cap.set(cv2.CAP_PROP_EXPOSURE, self.config.camera.exposure)
        
        return cap
    
    def _on_parameter_change(self, section: str, param: str, value):
        """
        파라미터 변경 콜백
        
        Args:
            section: 설정 섹션
            param: 파라미터 이름
            value: 새 값
        """
        print(f"[INFO] 파라미터 변경: {section}.{param} = {value}")
        
        # PID 리셋이 필요한 경우
        if section == "path_planning" and param.startswith("pid_"):
            self.path_planner.reset_pid()
            print("[INFO] PID 제어 변수 리셋됨")
    
    def draw_lane_overlay(
        self,
        frame: np.ndarray,
        lane_result: dict,
        path_result: dict
    ) -> np.ndarray:
        """
        차선 및 정보 오버레이
        
        Args:
            frame: 원본 프레임
            lane_result: 차선 검출 결과
            path_result: 경로 계획 결과
            
        Returns:
            오버레이된 프레임
        """
        overlay = frame.copy()
        
        # 차선이 검출된 경우
        if lane_result['detected']:
            # Bird's eye view에서 차선 영역 그리기
            binary_warped = lane_result['binary_warped']
            height, width = binary_warped.shape
            
            # y 좌표 생성
            ploty = np.linspace(0, height - 1, height)
            
            # 다항식으로부터 x 좌표 계산
            left_fit = lane_result['left_fit']
            right_fit = lane_result['right_fit']
            
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            
            # 차선 영역 폴리곤 생성
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            # 빈 이미지에 차선 영역 그리기
            lane_image = np.zeros_like(frame)
            cv2.fillPoly(lane_image, np.int32([pts]), (0, 255, 0))
            
            # 원근 역변환
            lane_warped = cv2.warpPerspective(
                lane_image,
                self.lane_detector.M_inv,
                (frame.shape[1], frame.shape[0])
            )
            
            # 원본 프레임과 합성
            overlay = cv2.addWeighted(overlay, 1, lane_warped, 0.3, 0)
            
            # 차선 경계선 그리기
            left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            
            left_line_warped = cv2.perspectiveTransform(
                left_line.astype(np.float32),
                self.lane_detector.M_inv
            )
            right_line_warped = cv2.perspectiveTransform(
                right_line.astype(np.float32),
                self.lane_detector.M_inv
            )
            
            cv2.polylines(
                overlay,
                np.int32([left_line_warped]),
                False,
                self.config.gui.color_left_lane,
                3
            )
            cv2.polylines(
                overlay,
                np.int32([right_line_warped]),
                False,
                self.config.gui.color_right_lane,
                3
            )

            gap_mask = lane_result.get('gap_mask')
            if gap_mask is not None and np.count_nonzero(gap_mask) > 0:
                gap_color = np.zeros((gap_mask.shape[0], gap_mask.shape[1], 3), dtype=np.uint8)
                gap_color[:, :, 2] = gap_mask  # Red channel
                gap_warped = cv2.warpPerspective(
                    gap_color,
                    self.lane_detector.M_inv,
                    (frame.shape[1], frame.shape[0])
                )
                overlay = cv2.addWeighted(overlay, 1, gap_warped, 0.4, 0)
        # ROI 및 마스크 시각화
        self._draw_roi_overlay(overlay)

        # 텍스트 정보 추가
        self._draw_text_info(overlay, lane_result, path_result)
        
        return overlay
    
    def _draw_text_info(
        self,
        frame: np.ndarray,
        lane_result: dict,
        path_result: dict
    ):
        """
        텍스트 정보 오버레이
        
        Args:
            frame: 프레임 (in-place 수정)
            lane_result: 차선 검출 결과
            path_result: 경로 계획 결과
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.gui.font_scale
        thickness = self.config.gui.font_thickness
        color = self.config.gui.color_text
        
        y_offset = 30
        line_height = 35
        
        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, y_offset),
            font, font_scale, color, thickness
        )
        y_offset += line_height
        
        # 프레임 번호
        cv2.putText(
            frame,
            f"Frame: {self.frame_number}",
            (10, y_offset),
            font, font_scale, color, thickness
        )
        y_offset += line_height
        
        # 검출 상태
        if lane_result['detected']:
            status_text = "Lane: DETECTED"
            status_color = (0, 255, 0)  # 초록색
        else:
            status_text = "Lane: LOST"
            status_color = (0, 0, 255)  # 빨간색
        
        cv2.putText(
            frame,
            status_text,
            (10, y_offset),
            font, font_scale, status_color, thickness
        )
        y_offset += line_height
        
        # 경로 정보
        if path_result['valid']:
            # 오프셋
            offset = path_result['center_offset']
            cv2.putText(
                frame,
                f"Offset: {offset:.3f} m",
                (10, y_offset),
                font, font_scale, color, thickness
            )
            y_offset += line_height
            
            # 조향각
            steering = path_result['steering_angle']
            cv2.putText(
                frame,
                f"Steering: {steering:.1f} deg",
                (10, y_offset),
                font, font_scale, color, thickness
            )
            y_offset += line_height
            
            # 곡률
            left_curv = path_result['left_curvature']
            right_curv = path_result['right_curvature']
            avg_curv = (left_curv + right_curv) / 2
            
            cv2.putText(
                frame,
                f"Curvature: {avg_curv:.1f} m",
                (10, y_offset),
                font, font_scale, color, thickness
            )
            y_offset += line_height
            
            # 차선 이탈 경고
            if path_result['lane_departure_warning']:
                cv2.putText(
                    frame,
                    "WARNING: Lane Departure!",
                    (10, y_offset),
                    font, font_scale, (0, 255, 255), thickness
                )

    def _draw_roi_overlay(self, frame: np.ndarray):
        """ROI 비주얼과 후드 마스크 영역을 오버레이한다."""
        lane_cfg = self.config.lane_detection
        h, w = frame.shape[:2]

        roi_top_ratio = float(np.clip(lane_cfg.roi_top_ratio, 0.0, 1.0))
        roi_bottom_ratio = float(np.clip(lane_cfg.roi_bottom_ratio, 0.0, 1.0))

        roi_top = int(h * roi_top_ratio)
        roi_bottom = int(h * roi_bottom_ratio)
        if roi_bottom <= roi_top:
            roi_bottom = h
        roi_bottom = max(roi_top + 1, min(h, roi_bottom))
        trimmed_ratio = max(0.0, (h - roi_bottom) / max(h, 1))

        overlay = frame.copy()
        if trimmed_ratio > 1e-3:
            cv2.rectangle(overlay, (0, roi_bottom), (w, h), (0, 0, 0), -1)
            frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.line(frame, (0, roi_top), (w, roi_top), (0, 200, 0), 2)
        cv2.line(frame, (0, roi_bottom), (w, roi_bottom), (0, 165, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "ROI Start",
            (10, max(30, roi_top + 20)),
            font,
            0.6,
            (0, 200, 0),
            2,
        )
        if trimmed_ratio > 1e-3:
            cv2.putText(
                frame,
                f"Trimmed {trimmed_ratio * 100:.0f}%",
                (10, min(h - 10, roi_bottom + 30)),
                font,
                0.6,
                (0, 165, 255),
                2,
            )

        polygon = lane_cfg.hood_mask_polygon
        if polygon is not None and len(polygon) >= 3:
            pts = np.array([
                [int(point[0] * w), int(point[1] * h)]
                for point in polygon
            ], dtype=np.int32)

            hood_overlay = frame.copy()
            cv2.fillPoly(hood_overlay, [pts], (0, 0, 255))
            frame[:] = cv2.addWeighted(hood_overlay, 0.25, frame, 0.75, 0)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            label_pos = tuple(pts[np.argmin(pts[:, 1])])  # 가장 위쪽 포인트
            cv2.putText(
                frame,
                "Hood Mask",
                (label_pos[0], max(20, label_pos[1] - 10)),
                font,
                0.6,
                (0, 0, 255),
                2,
            )

    def run(self):
        """메인 루프 실행"""
        print("[INFO] 시스템 시작!")
        print("[INFO] 종료하려면 'q' 키 또는 Ctrl+C를 누르세요.")
        if self.enable_gui:
            print("[INFO] Tk GUI 활성화: 패널에서 일시정지/종료 가능")
        if self.display_window:
            print("[INFO] OpenCV 미리보기 창이 활성화되었습니다.")
        print("="*60)
        
        try:
            while self.running and (self.gui.running if self.gui else True):
                if self.gui and self.gui.paused:
                    self.gui.update()
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.camera.read()
                if not ret:
                    print("[ERROR] 프레임을 읽을 수 없습니다!")
                    break
                
                self.frame_number += 1
                if self.max_frames and self.frame_number >= self.max_frames:
                    self.running = False
                    print(f"[INFO] 최대 프레임 {self.max_frames}에 도달하여 종료합니다.")
                    break
                
                lane_result = self.lane_detector.detect_lanes(frame)
                path_result = self.path_planner.plan_path(
                    lane_result,
                    self.config.camera.width,
                    self.config.camera.height,
                    use_pure_pursuit=False,
                    fps=self.fps if self.fps > 0 else 30.0
                )
                overlay_frame = self.draw_lane_overlay(frame, lane_result, path_result)
                
                current_time = time.time()
                self.fps = 1.0 / max(current_time - self.prev_time, 1e-6)
                self.prev_time = current_time
                
                if self.gui:
                    self.gui.update_video(overlay_frame)
                    self.gui.update_status(self.fps, lane_result, path_result)
                    self.gui.update()
                
                if self.display_window:
                    cv2.imshow('Lane Detection (Press Q to quit)', overlay_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                if self.bridge:
                    self.bridge.update(lane_result, path_result, self.fps, overlay_frame)
                
                self.data_logger.log_frame_data(
                    self.frame_number,
                    lane_result,
                    path_result,
                    self.fps
                )
                self.data_logger.log_video_frame(overlay_frame)
        
        except KeyboardInterrupt:
            print("\n[INFO] 사용자에 의해 중단됨")
        
        except Exception as e:
            print(f"\n[ERROR] 예외 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리 작업"""
        print("\n" + "="*60)
        print("시스템 종료 중...")
        print("="*60)
        
        # 카메라 해제
        if self.camera is not None:
            self.camera.release()
            print("[✓] 카메라 해제됨")
        
        # OpenCV 창 닫기
        if self.display_window:
            cv2.destroyAllWindows()
            print("[✓] OpenCV 창 닫힘")
        
        # 데이터 로거 종료
        self.data_logger.close()
        print("[✓] 로그 파일 저장 완료")
        
        # GUI 종료
        if self.gui:
            if self.gui.running:
                self.gui.on_closing()
            print("[✓] GUI 종료됨")
        
        print("="*60)
        print("시스템이 안전하게 종료되었습니다.")
        print("="*60)


def list_available_cameras(max_index: int = 5) -> list[int]:
    """시스템에 연결된 카메라 인덱스를 탐색합니다."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
        else:
            cap.release()
    return available


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="자율주행 차선 검출 시스템을 실행합니다."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        help="사용할 카메라 인덱스를 지정합니다 (기본 설정을 덮어씁니다).",
        default=None
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        help="카메라 프레임 너비를 강제로 설정합니다.",
        default=None
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        help="카메라 프레임 높이를 강제로 설정합니다.",
        default=None
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="사용 가능한 카메라 인덱스를 확인하고 종료합니다."
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=5,
        help="카메라 탐색 시 검사할 최대 인덱스입니다."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Tkinter GUI 패널을 비활성화합니다."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="OpenCV 미리보기 창을 표시합니다."
    )
    parser.add_argument(
        "--labview-state-path",
        type=str,
        default="labview_bridge/state.json",
        help="LabVIEW에서 읽을 상태 JSON 파일 경로 (비우면 비활성)."
    )
    parser.add_argument(
        "--labview-frame-path",
        type=str,
        default="labview_bridge/overlay.jpg",
        help="LabVIEW에서 읽을 오버레이 이미지 경로."
    )
    parser.add_argument(
        "--labview-write-frame",
        action="store_true",
        help="LabVIEW용 오버레이 이미지를 저장합니다."
    )
    parser.add_argument(
        "--enable-labview-bridge",
        action="store_true",
        help="LabVIEW 브릿지를 활성화합니다."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="(사용 중단) 시스템은 항상 무제한으로 실행됩니다."
    )
    parser.add_argument(
        "--dev-max-frames",
        type=int,
        default=0,
        help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_available_cameras(args.max_camera_index)
        if cameras:
            print(f"사용 가능한 카메라 인덱스: {cameras}")
        else:
            print("사용 가능한 카메라를 찾지 못했습니다. 인덱스를 늘려 다시 시도하세요.")
        return 0

    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  차선 검출 시스템 (Lane Detection System)  ".center(58) + "║")
    print("║" + "  자율주행 모빌리티 레이싱 대회  ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")
    
    enable_gui = not args.no_gui
    display_window = args.preview
    max_frames = None

    if args.dev_max_frames and args.dev_max_frames > 0:
        max_frames = args.dev_max_frames
        print(f"[INFO] 개발자 전용 프레임 제한 활성화: {max_frames}프레임 후 종료 예정")
    elif args.max_frames and args.max_frames > 0:
        print(
            "[WARN] --max-frames 옵션은 더 이상 사용되지 않으며 무시됩니다. "
            "시스템은 수동으로 중지할 때까지 계속 실행됩니다."
        )

    config = get_config()
    if args.frame_width and args.frame_width > 0:
        config.camera.width = args.frame_width
    if args.frame_height and args.frame_height > 0:
        config.camera.height = args.frame_height

    bridge = None
    if args.enable_labview_bridge and args.labview_state_path:
        bridge = LabViewBridge(
            state_path=args.labview_state_path,
            frame_path=args.labview_frame_path,
            write_frame=args.labview_write_frame
        )

    try:
        # 시스템 생성 및 실행
        system = LaneDetectionSystem(
            camera_index=args.camera_index,
            enable_gui=enable_gui,
            display_window=display_window,
            labview_bridge=bridge,
            max_frames=max_frames
        )
        system.run()
    
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
