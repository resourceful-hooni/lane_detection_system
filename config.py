import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np

@dataclass
class CameraConfig:
    """카메라 관련 설정"""
    # [Task 11] 640x360 @ 30fps로 최적화 (LattePanda 성능 대응)
    width: int = 640
    height: int = 360
    fps: int = 30
    camera_index: int = 1
    camera_height_mm: float = 450.0
    camera_angle_deg: float = 23.0
    auto_exposure: bool = True
    exposure: int = -4
    auto_white_balance: bool = True
    
    # [Task 11] 해상도 프리셋
    # PRESET_LOW: 640x360 @ 30fps (LattePanda 권장)
    # PRESET_MED: 848x480 @ 30fps
    # PRESET_HIGH: 848x480 @ 60fps (고성능 PC)
    resolution_preset: str = "PRESET_LOW"

@dataclass
class LaneDetectionConfig:
    """차선 검출 관련 설정"""
    roi_top_ratio: float = 0.10         # [Indoor] 0.4 -> 0.1 (바닥을 보므로 상단 영역도 활용)
    roi_bottom_ratio: float = 1.0
    roi_left_ratio: float = 0.0
    roi_right_ratio: float = 1.0
    roi_trapezoid_top_width_ratio: float = 0.85  # 사다리꼴 상단 너비 비율 (0.0 ~ 1.0)
    enable_joint_fitting: bool = True  # 양쪽 차선을 평행하게 강제 맞춤 (꼬임 방지)
    # [LattePanda Optimized] Strict Noise Filtering Parameters
    enable_blob_filter: bool = True    # 덩어리 필터링 (차량 등 비차선 객체 제거)
    blob_min_height: int = 30          # 594 -> 30 (안전한 기본값으로 복구)
    blob_min_width: int = 5            # 10 -> 5 (얇은 차선 허용)
    blob_max_width: int = 200          # 123 -> 200 (여유 있게)
    white_threshold: int = 140         # 100 -> 140 (노이즈 방지)
    gray_threshold: int = 120          # 100 -> 120 (노이즈 방지)
    
    # Morphology
    morph_kernel_open: int = 5         # 5 (유지 - 노이즈 제거)
    morph_kernel_close: int = 15       # 15 (유지 - 끊김 보완)
    morph_iterations: int = 1          # 반복 횟수
    
    # Blob Filter
    blob_min_aspect_ratio: float = 0.8 # 1.0 -> 0.8 (카메라 각도로 인해 납작해질 수 있음)
    
    # ROI Mask (BEV)
    roi_mask_top_ratio: float = 0.0    # BEV 상단 마스킹 비율
    roi_mask_bottom_ratio: float = 1.0 # BEV 하단 마스킹 비율
    roi_mask_side_margin: int = 71     # BEV 좌우 마스킹 픽셀
    
    # [GUI Tunable] Strict ROI Mask (BEV)
    roi_mask_top_ratio: float = 0.0    # BEV 상단 마스킹 비율 (0.0 ~ 1.0) - 전체 영역 탐색
    roi_mask_side_margin: int = 71      # BEV 좌우 마스킹 픽셀 - 전체 영역 탐색
    
    # [Legacy] Pattern Validation (GUI 호환성 유지용)
    pattern_min_white_len: int = 5     # 최소 흰색 픽셀 길이
    pattern_min_black_len: int = 5     # 최소 검은색 픽셀 길이
    pattern_max_gap_len: int = 20      # 최대 끊김 허용 길이
    pattern_min_segments: int = 2      # 최소 세그먼트 수
    
    # [Legacy] Canny Edge (Logging 호환성 유지용)
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    # [complete-fix-guide] 추가 파라미터
    use_row_anchor: bool = True  # Row-Anchor Detection 사용
    line_iou_threshold: float = 0.5  # Line IoU 임계값
    enable_strict_validation: bool = True  # 엄격한 기하학적 검증
    
    # [Task: Force Straight] 끊긴 구간 직선 강제
    force_straight_on_gap: bool = True      # 끊긴 구간에서 직선 강제 활성화
    straight_force_threshold: int = 100     # 이 픽셀 수보다 적으면 직선으로 간주 (점선 대응)
    
    # [Task: Wide Lane] 폭넓은 차선 대응
    lane_width_tolerance: float = 0.8       # 0.6 -> 0.8 (차선 폭 변화 허용 범위 대폭 확대)
    max_lane_width_pixel: int = 500         # 400 -> 500 (Blob Filter와 동기화)

    # [Task 4] Adaptive Threshold 설정
    enable_adaptive_threshold: bool = True  # 적응적 threshold 활성화
    threshold_low_light: int = 100          # 어두운 환경 threshold
    threshold_normal: int = 150             # 보통 환경 threshold
    threshold_bright: int = 180             # 밝은 환경 threshold
    white_saturation_max: int = 80          # 흰색 채도 최대값

    # [Dynamic Tuning] 차선 폭 기반 자동 튜닝
    enable_dynamic_tuning: bool = True
    dynamic_margin_ratio: float = 0.25      # 차선 폭의 25%를 마진으로 설정 (예: 폭 600px -> 마진 150px)
    dynamic_blob_min_width_ratio: float = 0.05 # 차선 폭의 5% (예: 30px)
    dynamic_blob_max_width_ratio: float = 0.4  # 차선 폭의 40% (예: 240px)

    # Perspective/본네트 mask 관련 값 (차선 검출 정확도 높이기 위해 추가됨)
    # [Task 11] 640x360 기준으로 스케일링됨
    # [Fix] 좌우 끝 차선 인식을 위해 시야 범위 확장 (20 ~ 620)
    lane_left_x: int = 212    # 90 -> 20 (좌측 끝까지 포함)
    lane_right_x: int = 417  # 550 -> 620 (우측 끝까지 포함)
    horizon_y: int = 100      # 120 * (360/480) = 90
    hood_bottom_y: int = 360  # 480 * (360/480) = 360
    hood_top_y: int = 315     # 420 * (360/480) = 315
    perspective_src_points: np.ndarray = None
    perspective_dst_points: np.ndarray = None
    auto_scale_perspective: bool = True
    perspective_reference_resolution: Tuple[int, int] = (640, 360)  # [Task 11] 기준 해상도
    enable_hood_mask: bool = True
    hood_mask_polygon: np.ndarray = None
    hood_mask_path: str = "calibration/hood_mask.json"
    bottom_trim_ratio: float = 0.0

    def __post_init__(self):
        # Perspective 포인트 및 본네트 polygon 초기화
        if self.perspective_src_points is None:
            # [Task 11] 640x360 해상도 기준
            img_w, img_h = 640, 360
            center_x = img_w // 2
            
            # 하단: 보닛 바로 앞 (차선 너비 + 여유)
            # lane_left_x(120) ~ lane_right_x(728) -> 폭 608
            bottom_width = self.lane_right_x - self.lane_left_x
            
            # 상단: 소실점 근처 (원근감으로 인해 좁아짐)
            # 보통 하단 폭의 20~40% 정도로 설정
            top_width = int(bottom_width * 0.35)
            
            self.perspective_src_points = np.float32([
                [center_x - top_width // 2, self.horizon_y],      # Top Left
                [center_x + top_width // 2, self.horizon_y],      # Top Right
                [center_x + bottom_width // 2, self.hood_bottom_y], # Bottom Right
                [center_x - bottom_width // 2, self.hood_bottom_y], # Bottom Left
            ])
            
            # Destination: Bird's Eye View (직사각형)
            # 이미지 좌우에 여백을 두어 차선이 휘어질 공간 확보
            margin_x = 150
            self.perspective_dst_points = np.float32([
                [margin_x, 0],
                [img_w - margin_x, 0],
                [img_w - margin_x, img_h],
                [margin_x, img_h]
            ])
            
        if self.hood_mask_polygon is None:
            px = [
                [292,716],[305,655],[311,615],[324,579],[337,555],[354,534],[370,517],[384,502],
                [401,484],[422,468],[438,455],[457,450],[476,442],[497,436],[510,431],[536,428],
                [549,423],[566,420],[586,421],[606,421],[627,419],[645,420],[665,419],[681,422],
                [695,424],[714,428],[734,435],[751,439],[765,443],[786,450],[800,457],[818,463],
                [835,475],[849,489],[857,503],[866,515],[877,520],[890,530],[896,543],[906,555],
                [917,575],[933,607],[939,635],[937,654],[945,685],[949,718]
            ]
            # 1280x720 polygon을 848x480 환경에 맞게 정규화
            width_, height_ = 1280, 720
            self.hood_mask_polygon = np.float32([[x/width_, y/height_] for x, y in px])
        self._load_hood_mask_from_file()

    def _load_hood_mask_from_file(self):
        if not self.enable_hood_mask or not self.hood_mask_path:
            return
        path = Path(self.hood_mask_path)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            polygon = data.get("polygon_normalized")
            if polygon and len(polygon) >= 3:
                self.hood_mask_polygon = np.float32(polygon)
                print(f"[INFO] Hood mask loaded from {path}")
        except Exception as exc:
            print(f"[WARN] Failed to load hood mask from {path}: {exc}")

@dataclass
class SlidingWindowConfig:
    n_windows: int = 9       # 12 -> 9 (속도 향상)
    margin: int = 100        # 71 -> 100 (여유 있게)
    min_pixels: int = 50     # 1188 -> 50 (안전한 기본값)
    histogram_start_ratio: float = 0.0
    search_y_start_ratio: float = 0.0  # 검색 시작 Y 비율 (0.0 = Top)
    search_y_end_ratio: float = 1.0    # 검색 종료 Y 비율 (1.0 = Bottom)

@dataclass
class RowAnchorConfig:
    """[Task 6] Row-Anchor Detection 설정"""
    enabled: bool = False       # [Fix] False로 변경 (사용자 피드백: 이전 코드가 더 좋음)
    num_rows: int = 36          # 샘플링 row 개수 (72보다 36이 2배 빠름)
    search_range: int = 50      # anchor 기준 탐색 범위 (±픽셀)
    min_pixels: int = 50        # 1188 -> 50 (안전한 기본값)
    min_points_for_fit: int = 10  # 다항식 피팅에 필요한 최소 포인트 수
    fallback_to_sliding: bool = True  # Row-Anchor 실패 시 Sliding Window로 fallback

@dataclass
class PathPlanningConfig:
    lane_width_m: float = 1.0  # 트랙 차선폭 (미터). 실제 환경에 맞게 수정 필요 (예: 일반도로 3.7, 모형 1.0)
    pid_kp: float = 25.0       # 0.8 -> 25.0 (단위 보정: m -> deg)
    pid_ki: float = 0.05
    pid_kd: float = 10.0       # 0.1 -> 10.0
    max_steering_angle_deg: float = 30.0
    lookahead_distance_m: float = 10.0
    # [Task 11] 640x360 기준 픽셀-미터 변환
    # Y축: 30m 가시거리 / 360px = 0.0833 m/px
    ym_per_pix: float = 30.0 / 360
    # X축: 차선폭(1.0m) / BEV차선폭(약 410px) = 0.00244 m/px
    # BEV 폭 = 640 - (115*2) = 410px (margin 비례 축소)
    xm_per_pix: float = 1.0 / 410

@dataclass
class GUIConfig:
    window_title: str = "Lane Detection System - 자율주행 차선 인식"
    display_width: int = 1280
    display_height: int = 720
    update_interval_ms: int = 16
    font_scale: float = 0.6
    font_thickness: int = 2
    color_left_lane: Tuple[int, int, int] = (255, 0, 0)
    color_right_lane: Tuple[int, int, int] = (0, 0, 255)
    color_center_line: Tuple[int, int, int] = (0, 255, 0)
    color_text: Tuple[int, int, int] = (255, 255, 255)
    color_warning: Tuple[int, int, int] = (0, 255, 255)

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    video_dir: str = "videos"
    save_video: bool = True
    video_codec: str = "XVID"
    video_extension: str = ".avi"
    video_fps: int = 30
    save_csv: bool = True
    csv_columns: list = None
    def __post_init__(self):
        if self.csv_columns is None:
            self.csv_columns = [
                "timestamp", "frame_number", "left_curve_rad", "right_curve_rad",
                "center_offset_m", "steering_angle_deg", "lane_detected", "fps",
                "white_threshold", "canny_low", "canny_high",
                "roi_top", "roi_bottom", "pixel_min_y", "pixel_max_y", "pixel_mean_y"
            ]
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

@dataclass
class SystemConfig:
    camera: CameraConfig = None
    lane_detection: LaneDetectionConfig = None
    sliding_window: SlidingWindowConfig = None
    row_anchor: RowAnchorConfig = None  # [Task 6] Row-Anchor 설정 추가
    path_planning: PathPlanningConfig = None
    gui: GUIConfig = None
    logging: LoggingConfig = None
    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.lane_detection is None:
            self.lane_detection = LaneDetectionConfig()
        if self.sliding_window is None:
            self.sliding_window = SlidingWindowConfig()
        if self.row_anchor is None:
            self.row_anchor = RowAnchorConfig()  # [Task 6]
        if self.path_planning is None:
            self.path_planning = PathPlanningConfig()
        if self.gui is None:
            self.gui = GUIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()

# 전역 설정 인스턴스
config = SystemConfig()

def get_config() -> SystemConfig:
    return config

def update_config(section: str, param: str, value):
    global config
    if hasattr(config, section):
        section_obj = getattr(config, section)
        if hasattr(section_obj, param):
            setattr(section_obj, param, value)
            print(f"[INFO] 파라미터 변경: {section}.{param} = {value}")
            return True
    return False

if __name__ == "__main__":
    cfg = get_config()
    print("=== 카메라 설정 ===")
    print(f"해상도: {cfg.camera.width}x{cfg.camera.height}")
    print(f"FPS: {cfg.camera.fps}")
    print(f"카메라 높이: {cfg.camera.camera_height_mm}mm")
    print(f"카메라 각도: {cfg.camera.camera_angle_deg}도")
    print("\n=== 차선 검출 설정 ===")
    print(f"ROI: 상단 {cfg.lane_detection.roi_top_ratio*100:.0f}% ~ 하단 {cfg.lane_detection.roi_bottom_ratio*100:.0f}%")
    print(f"사용 영역: {int(cfg.camera.height * cfg.lane_detection.roi_bottom_ratio)}픽셀")
    print(f"흰색 임계값: {cfg.lane_detection.white_threshold}")
    print("\n=== 경로 계획 설정 ===")
    print(f"PID 게인: Kp={cfg.path_planning.pid_kp}, Ki={cfg.path_planning.pid_ki}, Kd={cfg.path_planning.pid_kd}")
    print(f"최대 조향각: {cfg.path_planning.max_steering_angle_deg}도")
    print(f"픽셀-미터 변환: Y={cfg.path_planning.ym_per_pix:.6f}, X={cfg.path_planning.xm_per_pix:.6f}")
    print("\n=== 로깅 설정 ===")
    print(f"영상 저장: {cfg.logging.save_video}")
    print(f"CSV 저장: {cfg.logging.save_csv}")
    print(f"로그 디렉토리: {cfg.logging.log_dir}")
