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


class LaneDetector:
    """차선 검출 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = get_config()
        
        # 원근 변환 행렬 (캘리브레이션 후 설정)
        self.M = None  # 변환 행렬
        self.M_inv = None  # 역변환 행렬
        self.hood_mask = None
        self._hood_mask_shape = None
        
        # 이전 프레임 정보 (시간적 일관성 유지)
        self.left_fit = None   # 좌측 차선 다항식 계수
        self.right_fit = None  # 우측 차선 다항식 계수
        
        # 검출 상태
        self.detected = False
        
        # 프레임 카운터 (연속 검출 실패 추적)
        self.frame_count = 0
        self.detection_failure_count = 0
        
        # 픽셀 통계
        self.pixel_stats = {
            'min_y': 0,
            'max_y': 0,
            'mean_y': 0.0
        }

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
        
        # print(f"[INFO] Perspective Transform Initialized:")
        # print(f"  - Source Y Range: {src[0][1]:.1f} ~ {src[3][1]:.1f}")
        # print(f"  - Dest Y Range: {dst[0][1]:.1f} ~ {dst[3][1]:.1f}")
        # print(f"  - Note: This range defines the visible area in Bird's Eye View.")

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

        roi_top = int(height * roi_top_ratio)
        roi_bottom = int(height * roi_bottom_ratio)
        if roi_bottom <= roi_top:
            roi_bottom = height
        
        self._last_roi_bounds = (roi_top, roi_bottom, height)
        roi = frame[roi_top:roi_bottom, :]
        
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
        
        # 6. 전체 이미지 크기로 복원 (상단은 0으로 채움)
        full_binary = np.zeros((height, width), dtype=np.uint8)
        full_binary[roi_top:roi_bottom, :] = combined

        full_binary = self._apply_hood_mask(full_binary)

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
        흰색 차선 검출
        
        Args:
            image: BGR 이미지
            
        Returns:
            흰색 차선 이진 마스크
        """
        # HLS 색 공간 변환 (밝기 변화에 강함)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # L 채널(밝기)과 S 채널(채도) 사용
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # 흰색: 높은 밝기, 낮은 채도
        white_mask = cv2.inRange(
            l_channel,
            self.config.lane_detection.white_threshold,
            255
        )
        
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
        Bird's Eye View로 원근 변환
        
        Args:
            image: 입력 이미지
            
        Returns:
            변환된 이미지
        """
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
        binary_warped: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sliding Window 알고리즘으로 차선 픽셀 검출
        
        Args:
            binary_warped: Bird's eye view 이진 이미지
            
        Returns:
            out_img: 시각화 이미지
            left_lane_inds: 좌측 차선 픽셀 인덱스
            right_lane_inds: 우측 차선 픽셀 인덱스
        """
        roi_top, roi_bottom, roi_height = self._last_roi_bounds

        # 히스토그램으로 차선 시작점 찾기 (전체 영역 사용)
        histogram = np.sum(binary_warped, axis=0)
        
        # 시각화용 컬러 이미지 생성
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # 좌/우 차선 시작점 (히스토그램 피크)
        midpoint = len(histogram) // 2
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
        margin: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이전 프레임의 다항식을 활용해 빠르게 차선 픽셀을 찾는다."""
        if margin is None:
            margin = self.config.sliding_window.margin

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
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = self.config.gui.color_left_lane

        if right_fit is not None:
            right_fitx = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2]
            right_lane_inds = (
                (nonzerox >= (right_fitx - margin)) &
                (nonzerox <= (right_fitx + margin))
            ).nonzero()[0]
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
        
        # 좌측 차선 좌표 추출
        if len(left_lane_inds) > 0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            
            # 2차 다항식 피팅
            if len(leftx) > 0:
                left_fit = np.polyfit(lefty, leftx, 2)
            else:
                left_fit = None
        else:
            left_fit = None
        
        # 우측 차선 좌표 추출
        if len(right_lane_inds) > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # 2차 다항식 피팅
            if len(rightx) > 0:
                right_fit = np.polyfit(righty, rightx, 2)
            else:
                right_fit = None
        else:
            right_fit = None
        
        return left_fit, right_fit
    
    def detect_lanes(self, frame: np.ndarray) -> dict:
        """
        차선 검출 메인 함수
        
        Args:
            frame: 입력 프레임 (BGR)
            
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

        out_img: np.ndarray
        left_lane_inds: np.ndarray
        right_lane_inds: np.ndarray

        if use_prior:
            out_img, left_lane_inds, right_lane_inds = self.find_lane_pixels_using_prior(
                binary_warped,
                self.left_fit,
                self.right_fit
            )

            min_points_prior = self.config.sliding_window.min_pixels * max(
                4,
                self.config.sliding_window.n_windows // 2
            )
            if (
                len(left_lane_inds) < min_points_prior or
                len(right_lane_inds) < min_points_prior
            ):
                out_img, left_lane_inds, right_lane_inds = \
                    self.find_lane_pixels_sliding_window(binary_warped)
        else:
            out_img, left_lane_inds, right_lane_inds = \
                self.find_lane_pixels_sliding_window(binary_warped)
        
        # 4. 다항식 피팅
        left_fit, right_fit = self.fit_polynomial(
            binary_warped,
            left_lane_inds,
            right_lane_inds
        )
        
        # 5. 검출 성공 여부 판단
        if left_fit is not None and right_fit is not None:
            self.left_fit = left_fit
            self.right_fit = right_fit
            self.detected = True
            self.detection_failure_count = 0
        else:
            # 검출 실패 시 이전 값 유지
            self.detection_failure_count += 1
            
            # 5프레임 연속 실패 시 리셋
            if self.detection_failure_count > 5:
                self.detected = False
                self.left_fit = None
                self.right_fit = None
        
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


# 테스트 코드
if __name__ == "__main__":
    print("LaneDetector 모듈 로드 완료")
    print("카메라로 테스트하려면 main.py를 실행하세요.")
