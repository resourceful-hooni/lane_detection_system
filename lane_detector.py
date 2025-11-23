"""
lane_detector.py
차선 검출 핵심 알고리즘 모듈
- 이미지 전처리
- 차선 픽셀 검출
- Sliding Window 알고리즘
- 다항식 피팅
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from config import get_config


class LaneTracker:
    """Kalman Filter를 이용한 차선 추적 클래스"""
    def __init__(self):
        # State: [a, b, c, da, db, dc] (2차 곡선 계수 및 변화율)
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        # dt = 1 (프레임 단위)
        self.kf.transitionMatrix[0, 3] = 1.0
        self.kf.transitionMatrix[1, 4] = 1.0
        self.kf.transitionMatrix[2, 5] = 1.0
        
        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        
        # Process Noise (시스템 노이즈) - 작을수록 모델 예측을 더 신뢰 (더 부드러움)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 5e-5
        
        # Measurement Noise (측정 노이즈) - 클수록 측정값 노이즈를 무시 (더 부드러움)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-1
        
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        self.initialized = False

    def update(self, fit: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        측정값으로 필터를 업데이트하고 최적 추정값을 반환한다.
        fit이 None이면 예측값만 반환한다.
        """
        if fit is None:
            if not self.initialized:
                return None
            # 측정값이 없으면 예측만 수행
            pred = self.kf.predict()
            return pred[:3].flatten()
        
        fit = np.array(fit, dtype=np.float32)
        
        if not self.initialized:
            # 초기화
            self.kf.statePost = np.array([
                [fit[0]], [fit[1]], [fit[2]],
                [0], [0], [0]
            ], dtype=np.float32)
            self.initialized = True
            return fit
            
        # 예측 및 보정
        self.kf.predict()
        corrected = self.kf.correct(fit)
        return corrected[:3].flatten()
        
    def reset(self):
        self.initialized = False


class LaneDetector:
    """차선 검출 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = get_config()
        
        # 원근 변환 행렬 및 맵
        self.M = None
        self.M_inv = None
        self.map_x = None
        self.map_y = None
        
        self.hood_mask = None
        self._hood_mask_shape = None
        
        # Kalman Filter Trackers
        self.left_tracker = LaneTracker()
        self.right_tracker = LaneTracker()
        
        # 이전 프레임 정보 (호환성 유지용)
        self.left_fit = None
        self.right_fit = None
        
        # Hood Mask Warped Bounds
        self.hood_warped_left_x = None
        self.hood_warped_right_x = None
        
        # 검출 상태
        self.detected = False
        
        # 프레임 카운터
        self.frame_count = 0
        self.detection_failure_count = 0
        
        # 픽셀 통계
        self.pixel_stats = {
            'min_y': 0,
            'max_y': 0,
            'mean_y': 0.0
        }
        
        # 학습된 차선 폭
        self.avg_lane_width = 548.0

        self._last_roi_bounds: Tuple[int, int, int] = (0, self.config.camera.height, self.config.camera.height)
        
        self._initialize_perspective_transform()
    
    def _initialize_perspective_transform(self):
        """원근 변환 행렬 초기화"""
        lane_cfg = self.config.lane_detection
        src = lane_cfg.perspective_src_points
        dst = lane_cfg.perspective_dst_points

        if lane_cfg.auto_scale_perspective and lane_cfg.perspective_reference_resolution:
            ref_w, ref_h = lane_cfg.perspective_reference_resolution
            scale_x = self.config.camera.width / ref_w
            scale_y = self.config.camera.height / ref_h
            src = self._scale_points(src, scale_x, scale_y)
            dst = self._scale_points(dst, scale_x, scale_y)
        else:
            src = np.array(src, dtype=np.float32)
            dst = np.array(dst, dtype=np.float32)
        
        # 변환 행렬 계산
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        self._scaled_src = src
        self._scaled_dst = dst
        
        # [FPS 최적화] Remap용 맵 생성 (warpPerspective 대체)
        self._init_remap_maps()
        
        # 차량 마스크 기준 탐색 범위 계산
        self._calculate_warped_hood_bounds()

    def _init_remap_maps(self):
        """warpPerspective 속도 향상을 위한 Remap Map 생성"""
        h, w = self.config.camera.height, self.config.camera.width
        
        # 목적지 이미지의 좌표 그리드 생성
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # 좌표 평탄화 및 동차 좌표계 변환 [x, y, 1]
        # shape: (3, N)
        coords = np.stack([map_x.flatten(), map_y.flatten(), np.ones(w*h)])
        
        # 역변환 행렬 적용: dst -> src 좌표 계산
        # src_coords = M_inv * dst_coords
        src_coords = self.M_inv @ coords
        
        # 동차 좌표 정규화 (w로 나누기)
        src_coords = src_coords / (src_coords[2, :] + 1e-6)
        
        # 맵 형태로 다시 변환
        self.map_x = src_coords[0, :].reshape(h, w).astype(np.float32)
        self.map_y = src_coords[1, :].reshape(h, w).astype(np.float32)
        # print("[INFO] Perspective Remap Maps Initialized")
        
        # print(f"[INFO] Perspective Transform Initialized:")
        # print(f"  - Source Y Range: {src[0][1]:.1f} ~ {src[3][1]:.1f}")

    def _calculate_warped_hood_bounds(self):
        """차량 마스크(보닛)의 하단 경계를 BEV 좌표로 변환하여 차선 탐색 범위를 제한한다."""
        polygon = self.config.lane_detection.hood_mask_polygon
        if polygon is None or len(polygon) == 0:
            return

        # 하단부(y > 0.9) 포인트만 추출 (정규화 좌표 기준)
        bottom_points = [p for p in polygon if p[1] > 0.9]
        if not bottom_points:
            bottom_points = polygon
        
        bottom_points = np.array(bottom_points)
        
        # 좌측 끝(최소 x)과 우측 끝(최대 x) 찾기
        min_x_idx = np.argmin(bottom_points[:, 0])
        max_x_idx = np.argmax(bottom_points[:, 0])
        
        left_pt_norm = bottom_points[min_x_idx]
        right_pt_norm = bottom_points[max_x_idx]
        
        w, h = self.config.camera.width, self.config.camera.height
        
        # 원본 좌표계로 변환
        src_pts = np.array([
            [left_pt_norm[0] * w, left_pt_norm[1] * h],
            [right_pt_norm[0] * w, right_pt_norm[1] * h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # BEV 좌표계로 변환
        dst_pts = cv2.perspectiveTransform(src_pts, self.M)
        
        self.hood_warped_left_x = int(dst_pts[0][0][0])
        self.hood_warped_right_x = int(dst_pts[1][0][0])
        # print(f"[INFO] Warped Hood Bounds: Left={self.hood_warped_left_x}, Right={self.hood_warped_right_x}")

    def _scale_points(self, points: np.ndarray, sx: float, sy: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        pts[:, 0] = pts[:, 0] * sx
        pts[:, 1] = pts[:, 1] * sy
        return pts

    def set_hood_mask(
        self,
        image_shape: Tuple[int, int],
        polygon_points: Optional[np.ndarray] = None
    ):
        """차량 본체 영역 마스크를 설정한다."""
        if not self.config.lane_detection.enable_hood_mask:
            self.hood_mask = None
            self._hood_mask_shape = None
            return

        height, width = image_shape
        polygon = polygon_points
        if polygon is None:
            polygon = self.config.lane_detection.hood_mask_polygon
        if polygon is None or len(polygon) < 3:
            polygon = np.array([
                [0.2, 1.0],
                [0.8, 1.0],
                [0.65, 0.75],
                [0.35, 0.75]
            ], dtype=np.float32)
        if polygon is None:
            self.hood_mask = None
            self._hood_mask_shape = None
            return

        pts = np.array(polygon, dtype=np.float32).copy()
        if pts.max() <= 1.0:
            pts[:, 0] = pts[:, 0] * width
            pts[:, 1] = pts[:, 1] * height

        pts = pts.astype(np.int32)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        cv2.fillPoly(mask, [pts], 0)
        self.hood_mask = mask
        self._hood_mask_shape = (height, width)
        # print(f"[INFO] Hood mask initialized for resolution {width}x{height}")
    
    def _apply_hood_mask(self, binary: np.ndarray) -> np.ndarray:
        if not self.config.lane_detection.enable_hood_mask:
            return binary

        height, width = binary.shape[:2]
        if self.hood_mask is None or self._hood_mask_shape != (height, width):
            self.set_hood_mask((height, width))

        if self.hood_mask is None:
            return binary

        return cv2.bitwise_and(binary, self.hood_mask)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 전처리
        
        Args:
            frame: 입력 프레임 (BGR)
            
        Returns:
            전처리된 이진화 이미지
        """
        # 1. ROI 설정 (관심 영역만 추출)
        height, width = frame.shape[:2]
        
        # [User Request] 초반에는 ROI Top을 0.6으로 설정하여 가까운 곳만 보다가(안정화),
        # 일정 프레임(60) 이후 0.3으로 변경하여 먼 곳까지 보도록 함
        if self.frame_count < 60:
            roi_top_ratio = 0.6
        else:
            roi_top_ratio = 0.3
            
        # roi_top_ratio = np.clip(self.config.lane_detection.roi_top_ratio, 0.0, 1.0)
        roi_bottom_ratio = np.clip(self.config.lane_detection.roi_bottom_ratio, 0.0, 1.0)
        roi_left_ratio = np.clip(self.config.lane_detection.roi_left_ratio, 0.0, 1.0)
        roi_right_ratio = np.clip(self.config.lane_detection.roi_right_ratio, 0.0, 1.0)

        roi_top = int(height * roi_top_ratio)
        roi_bottom = int(height * roi_bottom_ratio)
        roi_left = int(width * roi_left_ratio)
        roi_right = int(width * roi_right_ratio)

        if roi_bottom <= roi_top:
            roi_bottom = height
        
        if roi_right <= roi_left:
            roi_right = width
            roi_left = 0
        
        self._last_roi_bounds = (roi_top, roi_bottom, height)
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # [추가] 사다리꼴 ROI 마스크 적용
        # 직사각형 ROI 내부에서 다시 사다리꼴로 마스킹하여 상단 좌우 노이즈 제거
        if self.config.lane_detection.roi_trapezoid_top_width_ratio < 1.0:
            roi_h, roi_w = roi.shape[:2]
            top_w_ratio = self.config.lane_detection.roi_trapezoid_top_width_ratio
            top_w = int(roi_w * top_w_ratio)
            
            # 상단 중앙 정렬
            top_left_x = (roi_w - top_w) // 2
            top_right_x = top_left_x + top_w
            
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            pts = np.array([
                [0, roi_h],                 # Bottom Left
                [top_left_x, 0],            # Top Left
                [top_right_x, 0],           # Top Right
                [roi_w, roi_h]              # Bottom Right
            ], dtype=np.int32)
            
            cv2.fillPoly(mask, [pts], 255)
            roi = cv2.bitwise_and(roi, roi, mask=mask)

        # 2. 색상 기반 차선 검출
        white_mask = self._detect_white_lane(roi)
        black_mask = self._detect_black_lane(roi)
        if self.config.lane_detection.enable_vehicle_color_suppression:
            vehicle_mask = self._suppress_vehicle_colors(roi)
        else:
            vehicle_mask = np.full_like(white_mask, 255)
        
        # 3. 마스크 결합 (흰색 또는 검정색)
        combined_mask = cv2.bitwise_or(white_mask, black_mask)
        combined_mask = cv2.bitwise_and(combined_mask, vehicle_mask)
        
        # 4. Canny Edge Detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # [추가] Top-Hat 변환으로 검-흰-검 패턴(밝은 선) 강조
        # 차선 폭보다 약간 큰 커널을 사용하여 밝은 선만 추출
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
        _, tophat_mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
        
        blurred = cv2.GaussianBlur(
            gray, 
            (self.config.lane_detection.gaussian_kernel_size, 
             self.config.lane_detection.gaussian_kernel_size), 
            0
        )
        edges = cv2.Canny(
            blurred,
            self.config.lane_detection.canny_low_threshold,
            self.config.lane_detection.canny_high_threshold
        )
        triplet_mask = np.zeros_like(white_mask)
        if self.config.lane_detection.enable_triplet_detection:
            triplet_mask = self._detect_white_black_white_pattern(roi)
        
        # 5. 색상 마스크와 에지 결합
        combined = cv2.bitwise_or(combined_mask, edges)
        combined = cv2.bitwise_or(combined, triplet_mask)
        combined = cv2.bitwise_or(combined, tophat_mask)  # Top-Hat 결과 추가
        
        # 6. 전체 이미지 크기로 복원 (상단은 0으로 채움)
        full_binary = np.zeros((height, width), dtype=np.uint8)
        full_binary[roi_top:roi_bottom, roi_left:roi_right] = combined

        full_binary = self._apply_hood_mask(full_binary)
        
        # [추가] Blob Filtering (차량 등 비차선 객체 제거)
        full_binary = self._filter_false_positives(full_binary)

        bottom_trim = self.config.lane_detection.bottom_trim_ratio
        if bottom_trim > 0:
            trim_pixels = int(height * bottom_trim)
            if trim_pixels > 0:
                full_binary[-trim_pixels:, :] = 0

        # --- 분석: Y축 픽셀 분포 (Mask Y-axis Histogram) ---
        # 흰색 픽셀(255)의 Y좌표 분포 확인
        nonzero_y, nonzero_x = full_binary.nonzero()
        
        if len(nonzero_y) > 0:
            min_y = np.min(nonzero_y)
            max_y = np.max(nonzero_y)
            mean_y = np.mean(nonzero_y)
            
            self.pixel_stats = {
                'min_y': int(min_y),
                'max_y': int(max_y),
                'mean_y': float(mean_y)
            }
        else:
            self.pixel_stats = {'min_y': 0, 'max_y': 0, 'mean_y': 0.0}
        
        return full_binary

    def _suppress_vehicle_colors(self, image: np.ndarray) -> np.ndarray:
        """노란/녹색 차량 색상을 제거하는 마스크"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([20, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([85, 255, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        vehicle_mask = cv2.bitwise_or(yellow_mask, green_mask)
        vehicle_mask = cv2.bitwise_not(vehicle_mask)
        return vehicle_mask

    def _detect_white_black_white_pattern(self, image: np.ndarray) -> np.ndarray:
        """강한 좌우 경계가 있는 흰-검-흰 패턴을 강조한다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_abs = cv2.convertScaleAbs(sobelx)
        _, thresh = cv2.threshold(
            sobel_abs,
            self.config.lane_detection.triplet_gradient_threshold,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.config.lane_detection.triplet_morph_kernel
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        return dilated
    
    def _detect_white_lane(self, image: np.ndarray) -> np.ndarray:
        """
        흰색 차선 검출 (Adaptive Threshold 추가)
        
        Args:
            image: BGR 이미지
            
        Returns:
            흰색 차선 이진 마스크
        """
        # 1. HLS L-channel Threshold
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        # Config 값 사용하되, 너무 높으면(200 이상) 조금 낮춰서 잡음
        thresh = min(self.config.lane_detection.white_threshold, 170)
        white_mask_hls = cv2.inRange(l_channel, thresh, 255)
        
        # 2. Adaptive Threshold (조명 변화에 강함)
        # 그레이스케일 변환 후 적응형 이진화
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_mask_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # Block size (홀수)
            -5   # C constant (음수면 밝은 영역 강조)
        )
        
        # 두 결과 결합 (OR)
        # HLS는 밝은 흰색을 잘 잡고, Adaptive는 대비가 있는 선을 잘 잡음
        white_mask = cv2.bitwise_or(white_mask_hls, white_mask_adaptive)
        
        # 노이즈 제거 (Morphology Open)
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask
    
    def _detect_black_lane(self, image: np.ndarray) -> np.ndarray:
        """
        검정색 차선 검출 (양쪽 테이프)
        
        Args:
            image: BGR 이미지
            
        Returns:
            검정색 차선 이진 마스크
        """
        upper = np.array([
            self.config.lane_detection.black_threshold,
            self.config.lane_detection.black_threshold,
            self.config.lane_detection.black_threshold
        ])
        lower = np.array([0, 0, 0])
        black_mask = cv2.inRange(image, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return black_mask
    
    def warp_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Bird's Eye View로 원근 변환 (Optimized with remap)
        
        Args:
            image: 입력 이미지
            
        Returns:
            변환된 이미지
        """
        if self.map_x is not None and self.map_y is not None:
            # [FPS 최적화] 미리 계산된 맵 사용 (약 2~3배 빠름)
            warped = cv2.remap(
                image,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        else:
            # Fallback
            height, width = image.shape[:2]
            warped = cv2.warpPerspective(
                image,
                self.M,
                (width, height),
                flags=cv2.INTER_LINEAR
            )
        
        return warped
    
    def find_lane_pixels_sliding_window(
        self, 
        binary_warped: np.ndarray,
        visualize: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Sliding Window 알고리즘으로 차선 픽셀 검출
        """
        roi_top, roi_bottom, roi_height = self._last_roi_bounds

        # 히스토그램으로 차선 시작점 찾기 (전체 영역 사용)
        histogram = np.sum(binary_warped, axis=0)
        
        # 시각화용 컬러 이미지 생성 (필요 시에만)
        out_img = None
        if visualize:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # 좌/우 차선 시작점 (히스토그램 피크)
        midpoint = len(histogram) // 2
        
        # [수정] 차량 마스크 기준 탐색 범위 제한
        # 차량 폭(보닛)을 기준으로 일정 범위 내에서만 차선 시작점을 찾음
        search_margin = 100  # 탐색 범위 완화 (60 -> 100)
        
        if self.hood_warped_left_x is not None and self.hood_warped_right_x is not None:
            # Left Lane: Hood 왼쪽 끝 주변 탐색
            l_center = self.hood_warped_left_x
            l_min = max(0, l_center - search_margin)
            l_max = min(midpoint, l_center + search_margin)
            
            hist_slice_l = histogram[l_min:l_max]
            if len(hist_slice_l) > 0:
                leftx_base = np.argmax(hist_slice_l) + l_min
            else:
                # 범위 내에 없으면 전체 좌측 영역에서 찾음 (Fallback)
                leftx_base = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0

            # Right Lane: Hood 오른쪽 끝 주변 탐색
            r_center = self.hood_warped_right_x
            r_min = max(midpoint, r_center - search_margin)
            r_max = min(binary_warped.shape[1], r_center + search_margin)
            
            hist_slice_r = histogram[r_min:r_max]
            if len(hist_slice_r) > 0:
                rightx_base = np.argmax(hist_slice_r) + r_min
            else:
                # 범위 내에 없으면 전체 우측 영역에서 찾음 (Fallback)
                right_slice = histogram[midpoint:] if midpoint > 0 else histogram
                rightx_base = (np.argmax(right_slice) + midpoint) if right_slice.size > 0 else 0
        else:
            # 기존 로직 (전체 탐색)
            leftx_base = np.argmax(histogram[:midpoint]) if midpoint > 0 else 0
            right_slice = histogram[midpoint:] if midpoint > 0 else histogram
            rightx_base = (
                np.argmax(right_slice) + midpoint
                if right_slice.size > 0 else 0
            )

        # 윈도우 설정
        n_windows = max(1, self.config.sliding_window.n_windows)
        window_height = int(np.ceil(binary_warped.shape[0] / n_windows))
        
        # 0이 아닌 픽셀 좌표 추출
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 현재 윈도우 중심점
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # 차선 픽셀 인덱스 저장
        left_lane_inds = []
        right_lane_inds = []
        
        margin = self.config.sliding_window.margin
        min_pixels = self.config.sliding_window.min_pixels
        
        # 각 윈도우별로 처리
        for window in range(n_windows):
            # 윈도우 경계 계산
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            if window == n_windows - 1:
                win_y_low = 0
            win_y_low = max(0, win_y_low)
            win_y_high = min(binary_warped.shape[0], win_y_high)
            
            # 좌측 윈도우
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            
            # 우측 윈도우
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # 윈도우 그리기 (시각화)
            if visualize and out_img is not None:
                cv2.rectangle(
                    out_img,
                    (win_xleft_low, win_y_low),
                    (win_xleft_high, win_y_high),
                    self.config.gui.color_left_lane,
                    2
                )
                cv2.rectangle(
                    out_img,
                    (win_xright_low, win_y_low),
                    (win_xright_high, win_y_high),
                    self.config.gui.color_right_lane,
                    2
                )
            
            # 윈도우 내 픽셀 찾기
            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            
            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]
            
            # 인덱스 추가
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # 충분한 픽셀이 있으면 윈도우 중심 업데이트
            if len(good_left_inds) > min_pixels:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            
            if len(good_right_inds) > min_pixels:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # 인덱스 병합
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        return out_img, left_lane_inds, right_lane_inds
    
    def find_lane_pixels_using_prior(
        self,
        binary_warped: np.ndarray,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        margin: Optional[int] = None,
        visualize: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """이전 프레임의 다항식을 활용해 빠르게 차선 픽셀을 찾는다."""
        if margin is None:
            margin = self.config.sliding_window.margin

        out_img = None
        if visualize:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = np.array([], dtype=np.int64)
        right_lane_inds = np.array([], dtype=np.int64)

        if left_fit is not None:
            left_fitx = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2]
            left_lane_inds = (
                (nonzerox >= (left_fitx - margin)) &
                (nonzerox <= (left_fitx + margin))
            ).nonzero()[0]
            if visualize and out_img is not None:
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = self.config.gui.color_left_lane

        if right_fit is not None:
            right_fitx = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2]
            right_lane_inds = (
                (nonzerox >= (right_fitx - margin)) &
                (nonzerox <= (right_fitx + margin))
            ).nonzero()[0]
            if visualize and out_img is not None:
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = self.config.gui.color_right_lane

        return out_img, left_lane_inds, right_lane_inds
    
    def fit_polynomial(
        self,
        binary_warped: np.ndarray,
        left_lane_inds: np.ndarray,
        right_lane_inds: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        다항식 피팅 (2차)
        
        Args:
            binary_warped: Bird's eye view 이진 이미지
            left_lane_inds: 좌측 차선 픽셀 인덱스
            right_lane_inds: 우측 차선 픽셀 인덱스
            
        Returns:
            left_fit: 좌측 차선 다항식 계수 [a, b, c] (ax^2 + bx + c)
            right_fit: 우측 차선 다항식 계수
        """
        # 0이 아닌 픽셀 좌표
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_fit = None
        right_fit = None
        
        # 좌측 차선 좌표 추출
        leftx, lefty = [], []
        if len(left_lane_inds) > 0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            if len(leftx) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
        
        # 우측 차선 좌표 추출
        rightx, righty = [], []
        if len(right_lane_inds) > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            if len(rightx) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
        
        # [추가] Joint Fitting (평행 제약 조건 적용)
        # 양쪽 차선이 모두 검출되었고, 설정이 켜져있다면 평행하게 보정
        if (self.config.lane_detection.enable_joint_fitting and 
            left_fit is not None and right_fit is not None):
            
            # 가중치 계산 (픽셀 수가 많을수록 신뢰도 높음)
            n_left = len(leftx)
            n_right = len(rightx)
            
            if n_left > 0 and n_right > 0:
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)
                
                # 곡률(a)과 기울기(b)를 가중 평균
                avg_a = left_fit[0] * w_left + right_fit[0] * w_right
                avg_b = left_fit[1] * w_left + right_fit[1] * w_right
                
                # 절편(c) 재계산: x - (ay^2 + by)의 평균
                # 왼쪽
                left_residuals = leftx - (avg_a * lefty**2 + avg_b * lefty)
                new_c_left = np.mean(left_residuals)
                
                # 오른쪽
                right_residuals = rightx - (avg_a * righty**2 + avg_b * righty)
                new_c_right = np.mean(right_residuals)
                
                left_fit = np.array([avg_a, avg_b, new_c_left])
                right_fit = np.array([avg_a, avg_b, new_c_right])
        
        return left_fit, right_fit
    
    def detect_lanes(self, frame: np.ndarray, visualize: bool = True) -> dict:
        """
        차선 검출 메인 함수
        
        Args:
            frame: 입력 프레임 (BGR)
            visualize: 시각화 이미지 생성 여부 (False일 경우 FPS 향상)
            
        Returns:
            검출 결과 딕셔너리
        """
        self.frame_count += 1
        
        # 1. 전처리
        binary = self.preprocess_frame(frame)
        
        # 2. 원근 변환
        binary_warped = self.warp_perspective(binary)
        
        # 3. 이전 프레임 폴리라인 기반 탐색 → 필요 시 슬라이딩 윈도우
        use_prior = (
            self.detected and
            self.left_fit is not None and
            self.right_fit is not None
        )

        out_img = None
        left_lane_inds: np.ndarray
        right_lane_inds: np.ndarray

        if use_prior:
            # Prior 탐색은 시각화가 필요 없으면 이미지 생성을 건너뛰도록 내부 수정 필요하지만,
            # 여기서는 반환값만 무시하는 형태로 처리 (함수 내부 최적화는 별도)
            _img, left_lane_inds, right_lane_inds = self.find_lane_pixels_using_prior(
                binary_warped,
                self.left_fit,
                self.right_fit,
                visualize=visualize
            )
            if visualize:
                out_img = _img

            min_points_prior = self.config.sliding_window.min_pixels * max(
                4,
                self.config.sliding_window.n_windows // 2
            )
            if (
                len(left_lane_inds) < min_points_prior or
                len(right_lane_inds) < min_points_prior
            ):
                _img, left_lane_inds, right_lane_inds = \
                    self.find_lane_pixels_sliding_window(binary_warped, visualize=visualize)
                if visualize:
                    out_img = _img
        else:
            _img, left_lane_inds, right_lane_inds = \
                self.find_lane_pixels_sliding_window(binary_warped, visualize=visualize)
            if visualize:
                out_img = _img
        
        # 4. 다항식 피팅
        new_left_fit, new_right_fit = self.fit_polynomial(
            binary_warped,
            left_lane_inds,
            right_lane_inds
        )
        
        # 5. 유효성 검사 및 단일 차선 복구
        # 예상 차선 폭 (픽셀) - BEV 변환 후의 실제 폭을 사용
        if self.config.lane_detection.perspective_dst_points is not None:
            dst = self.config.lane_detection.perspective_dst_points
            default_width = dst[1][0] - dst[0][0]
        else:
            default_width = 548  # Fallback
            
        # 현재 프레임에서 사용할 폭 (학습된 값 우선 사용)
        # 단, 학습된 값이 너무 이상하면(초기값 대비 ±30% 이상) 초기값으로 리셋
        if abs(self.avg_lane_width - default_width) > (default_width * 0.3):
            self.avg_lane_width = default_width
        
        use_width = self.avg_lane_width
        
        # Case 1: 둘 다 잡혔지만 Sanity Check 실패 -> 신뢰도 높은 쪽 기준으로 재생성
        if new_left_fit is not None and new_right_fit is not None:
            # Sanity Check 수행
            is_sane = self._sanity_check(new_left_fit, new_right_fit, binary_warped.shape)
            
            if is_sane:
                # 유효하면 차선 폭 학습 (EMA)
                # 이미지 하단(차량 근처)에서의 폭을 기준으로 함
                y_eval = binary_warped.shape[0] - 1
                lx = new_left_fit[0]*y_eval**2 + new_left_fit[1]*y_eval + new_left_fit[2]
                rx = new_right_fit[0]*y_eval**2 + new_right_fit[1]*y_eval + new_right_fit[2]
                current_width = rx - lx
                
                # 폭이 합리적인 범위 내에 있을 때만 학습
                if 0.7 * default_width < current_width < 1.3 * default_width:
                    self.avg_lane_width = 0.9 * self.avg_lane_width + 0.1 * current_width
            else:
                # Sanity Check 실패 시: 더 신뢰할 수 있는 차선 하나만 남기고 나머지 재생성
                min_pixels = self.config.sliding_window.min_pixels
                left_count = len(left_lane_inds)
                right_count = len(right_lane_inds)
                
                # 픽셀 수가 현저히 적은 쪽은 버림
                if left_count > right_count * 1.5:
                    new_right_fit = np.copy(new_left_fit)
                    new_right_fit[2] += use_width
                elif right_count > left_count * 1.5:
                    new_left_fit = np.copy(new_right_fit)
                    new_left_fit[2] -= use_width
                else:
                    # 둘 다 비슷하면 이전 프레임 정보를 따라가거나, 그냥 둠 (이번 프레임 스킵)
                    # 여기서는 안전하게 이전 프레임 유지 시도 (detected=False로 처리됨)
                    pass
        
        # Case 2: 하나만 잡힘 -> 학습된 폭으로 나머지 생성
        elif new_left_fit is not None and new_right_fit is None:
            new_right_fit = np.copy(new_left_fit)
            new_right_fit[2] += use_width
        elif new_left_fit is None and new_right_fit is not None:
            new_left_fit = np.copy(new_right_fit)
            new_left_fit[2] -= use_width

        # 최종 확인 (복구된 값으로 다시 Sanity Check)
        is_valid = False
        if new_left_fit is not None and new_right_fit is not None:
            # 복구된 경우 width는 강제로 맞췄으므로 통과 가능성 높음
            is_valid = self._sanity_check(new_left_fit, new_right_fit, binary_warped.shape)

        if is_valid:
            # 유효하면 Kalman Filter 업데이트
            self.left_fit = self.left_tracker.update(new_left_fit)
            self.right_fit = self.right_tracker.update(new_right_fit)
            self.detected = True
            self.detection_failure_count = 0
        else:
            # 검출 실패 또는 유효성 검사 탈락 시
            self.detection_failure_count += 1
            
            # 예측값만 사용하여 업데이트 (Kalman Filter Prediction)
            self.left_fit = self.left_tracker.update(None)
            self.right_fit = self.right_tracker.update(None)
            
            # 5프레임 연속 실패 시 리셋 (완전히 놓침)
            if self.detection_failure_count > 5:
                self.detected = False
                self.left_fit = None
                self.right_fit = None
                self.left_tracker.reset()
                self.right_tracker.reset()
            # 5프레임 이내라면 예측값(self.left_fit)을 사용하여 깜빡임 방지
        
        # 6. 끊긴 차선 보간 (필요 시)
        binary_filled = binary_warped
        gap_mask = np.zeros_like(binary_warped)
        gap_flags = {'left': False, 'right': False}
        if self.left_fit is not None or self.right_fit is not None:
            binary_filled, gap_mask, gap_flags = self._interpolate_lane_gaps(
                binary_warped,
                self.left_fit,
                self.right_fit
            )

        # 7. 결과 반환
        result = {
            'detected': self.detected,
            'left_fit': self.left_fit,
            'right_fit': self.right_fit,
            'binary': binary,
            'binary_warped': binary_warped,
            'binary_filled': binary_filled,
            'gap_mask': gap_mask,
            'gap_flags': gap_flags,
            'out_img': out_img,
            'frame_count': self.frame_count,
            'pixel_stats': self.pixel_stats  # 픽셀 통계 추가
        }
        
        return result

    def _interpolate_lane_gaps(
        self,
        binary_warped: np.ndarray,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        thickness: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """예측된 차선 폴리라인을 이용해 끊긴 구간을 보간한다."""
        height, width = binary_warped.shape
        ploty = np.linspace(0, height - 1, height)

        def build_mask(fit: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
            mask = np.zeros_like(binary_warped)
            if fit is None:
                return mask, False
            fitx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
            valid = (fitx >= 0) & (fitx < width)
            if np.count_nonzero(valid) < 5:
                return mask, False

            pts = np.array([
                np.transpose(np.vstack([fitx[valid], ploty[valid]]))
            ], dtype=np.int32)
            cv2.polylines(mask, pts, False, 255, thickness)

            gap_mask = cv2.bitwise_and(mask, cv2.bitwise_not(binary_warped))
            gap_ratio = np.count_nonzero(gap_mask) / max(np.count_nonzero(mask), 1)
            had_gap = gap_ratio > 0.05
            return mask, had_gap

        left_mask, left_gap = build_mask(left_fit)
        right_mask, right_gap = build_mask(right_fit)

        combined_mask = cv2.bitwise_or(left_mask, right_mask)
        filled_binary = cv2.bitwise_or(binary_warped, combined_mask)
        filled_binary = np.clip(filled_binary, 0, 255)

        gap_mask_total = cv2.bitwise_and(combined_mask, cv2.bitwise_not(binary_warped))

        return filled_binary, gap_mask_total, {'left': left_gap, 'right': right_gap}
    
    def calculate_curvature(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        y_eval: float
    ) -> Tuple[float, float]:
        """
        차선 곡률 계산 (미터 단위)
        
        Args:
            left_fit: 좌측 차선 다항식 계수
            right_fit: 우측 차선 다항식 계수
            y_eval: 평가 지점 (픽셀, 일반적으로 이미지 하단)
            
        Returns:
            left_curverad: 좌측 곡률 반지름 (m)
            right_curverad: 우측 곡률 반지름 (m)
        """
        # 픽셀 → 미터 변환
        ym_per_pix = self.config.path_planning.ym_per_pix
        xm_per_pix = self.config.path_planning.xm_per_pix
        
        # 실제 좌표 변환된 다항식 계수 계산
        left_fit_cr = np.array([
            left_fit[0] * xm_per_pix / (ym_per_pix ** 2),
            left_fit[1] * xm_per_pix / ym_per_pix,
            left_fit[2] * xm_per_pix
        ])
        
        right_fit_cr = np.array([
            right_fit[0] * xm_per_pix / (ym_per_pix ** 2),
            right_fit[1] * xm_per_pix / ym_per_pix,
            right_fit[2] * xm_per_pix
        ])
        
        # 곡률 반지름 계산 공식
        # R = (1 + (dy/dx)^2)^(3/2) / |d2y/dx2|
        y_eval_m = y_eval * ym_per_pix
        
        left_curverad = (
            (1 + (2 * left_fit_cr[0] * y_eval_m + left_fit_cr[1]) ** 2) ** 1.5
        ) / abs(2 * left_fit_cr[0])
        
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1]) ** 2) ** 1.5
        ) / abs(2 * right_fit_cr[0])
        
        return left_curverad, right_curverad

    def _sanity_check(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> bool:
        """
        검출된 차선이 물리적으로 타당한지 검사한다.
        1. 교차 여부 확인
        2. 차선 폭 적절성 확인
        """
        height, width = img_shape
        ploty = np.linspace(0, height - 1, num=10)  # 10개 포인트만 샘플링
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # 1. 교차 검사 (모든 구간에서 오른쪽이 더 커야 함)
        diff = right_fitx - left_fitx
        if np.any(diff <= 0):
            # print("[Sanity] Lanes crossed!")
            return False
            
        # 2. 차선 폭 검사
        # BEV 변환 후의 예상 폭
        if self.config.lane_detection.perspective_dst_points is not None:
            dst = self.config.lane_detection.perspective_dst_points
            default_width = dst[1][0] - dst[0][0]
        else:
            default_width = 548
            
        # 학습된 폭이 있으면 그것을 기준으로 검사하되, 너무 벗어나지 않도록 함
        check_width = self.avg_lane_width if hasattr(self, 'avg_lane_width') else default_width
        
        # 허용 오차 (±30%)
        min_width = check_width * 0.70
        max_width = check_width * 1.30
        
        mean_width = np.mean(diff)
        if mean_width < min_width or mean_width > max_width:
            # print(f"[Sanity] Width out of range: {mean_width:.1f} (Expected: {expected_width})")
            return False
            
        return True

    def _smooth_fit(
        self,
        old_fit: Optional[np.ndarray],
        new_fit: np.ndarray,
        alpha: float = 0.2
    ) -> np.ndarray:
        """
        (Deprecated) 지수 이동 평균(EMA)을 이용한 스무딩
        현재는 Kalman Filter로 대체됨.
        """
        if old_fit is None:
            return new_fit
        return alpha * new_fit + (1 - alpha) * old_fit

    def _filter_false_positives(self, binary: np.ndarray) -> np.ndarray:
        """
        연결된 구성 요소 분석(CCA)을 통해 차선이 아닌 객체(차량, 횡단보도 등)를 제거한다.
        - 너비가 너무 넓은 객체 제거 (차량)
        - 높이가 너무 낮은 객체 제거 (자잘한 노이즈)
        - 종횡비가 가로로 긴 객체 제거
        """
        if not self.config.lane_detection.enable_blob_filter:
            return binary
            
        # 레이블링 (8-connectivity)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # stats: [x, y, width, height, area]
        # 배경(0)은 제외하고 처리
        
        min_h = self.config.lane_detection.blob_min_height
        max_w = self.config.lane_detection.blob_max_width
        
        # 유효한 레이블만 마스킹
        mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 1. 너무 넓은 물체 (차량, 횡단보도 등) 제거
            if w > max_w:
                continue
                
            # 2. 너무 납작한 물체 (가로선 노이즈) 제거
            if h < min_h:
                continue
                
            # 3. 종횡비 체크 (차선은 세로로 길거나, 점선이라도 정사각형에 가까움)
            # 가로로 너무 긴 물체(범퍼 등) 제거
            if h / w < 0.5:
                continue
                
            # 조건 통과한 객체만 유지
            mask[labels == i] = 255
            
        return mask


# 테스트 코드
if __name__ == "__main__":
    print("LaneDetector 모듈 로드 완료")
    print("카메라로 테스트하려면 main.py를 실행하세요.")
