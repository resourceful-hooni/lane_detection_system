import cv2
import numpy as np
import sys
import os
import time
import argparse
import re
from dataclasses import dataclass

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from lane_detector import LaneDetector

@dataclass
class SearchResult:
    score: float
    white_thresh: int
    gray_thresh: int
    lane_width: float
    left_x: float
    right_x: float
    avg_pixel_density: float  # Window당 평균 픽셀 수

class AdvancedCalibrationTool:
    def __init__(self, camera_index=1):
        self.config = get_config()
        self.cap = self._init_camera(camera_index)
        self.detector = LaneDetector()
        
        # 초기 상태: Config 값 로드
        self.src_points = self.config.lane_detection.perspective_src_points.copy()
        # [Fix] Start with full view (0.0) to ensure far lanes are visible
        self.roi_top = 0.0 
        self.roi_trap_ratio = self.config.lane_detection.roi_trapezoid_top_width_ratio
        
        # 튜닝할 파라미터
        self.white_thresh = self.config.lane_detection.white_threshold
        self.gray_thresh = self.config.lane_detection.gray_threshold
        
        # UI 상태
        self.mode = "GEOMETRY" # GEOMETRY, AUTO_SEARCH, RESULT, MANUAL_TUNING
        self.selected_point = 0
        self.point_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.auto_result = None
        self.derived_params = {}
        self.best_exposure = self.config.camera.exposure
        self.trackbar_window = "Manual Tuning"

    def _create_trackbars(self):
        cv2.namedWindow(self.trackbar_window)
        cv2.resizeWindow(self.trackbar_window, 400, 600)
        
        # 1. Thresholds
        cv2.createTrackbar("White Thresh", self.trackbar_window, self.white_thresh, 255, lambda x: None)
        cv2.createTrackbar("Gray Thresh", self.trackbar_window, self.gray_thresh, 255, lambda x: None)
        
        # 2. Exposure
        current_exp = self.config.camera.exposure
        slider_val = max(0, min(13, current_exp + 13))
        cv2.createTrackbar("Exposure (+13)", self.trackbar_window, slider_val, 13, lambda x: None)
        
        # 3. Blob Filter (Noise Removal)
        # Min Width: 0 ~ 50
        cv2.createTrackbar("Blob Min Width", self.trackbar_window, self.config.lane_detection.blob_min_width, 50, lambda x: None)
        # Min Height: 0 ~ 200
        cv2.createTrackbar("Blob Min Height", self.trackbar_window, self.config.lane_detection.blob_min_height, 200, lambda x: None)
        
        # 4. ROI Geometry
        # Top Ratio: 0 ~ 100 (float 0.0 ~ 1.0)
        top_ratio_int = int(self.roi_top * 100)
        cv2.createTrackbar("ROI Top (%)", self.trackbar_window, top_ratio_int, 100, lambda x: None)
        
        # Perspective Height (Look Distance): 10 ~ 90% (from bottom)
        # Default around 40%
        cv2.createTrackbar("Look Dist (%)", self.trackbar_window, 40, 90, lambda x: None)
        
        # Trapezoid Top Width: 5 ~ 100 (float 0.05 ~ 1.0)
        trap_ratio_int = int(self.roi_trap_ratio * 100)
        cv2.createTrackbar("Trap Top (%)", self.trackbar_window, trap_ratio_int, 100, lambda x: None)

    def _update_from_trackbars(self):
        try:
            # Read Trackbars
            w = cv2.getTrackbarPos("White Thresh", self.trackbar_window)
            g = cv2.getTrackbarPos("Gray Thresh", self.trackbar_window)
            e_slider = cv2.getTrackbarPos("Exposure (+13)", self.trackbar_window)
            b_w = cv2.getTrackbarPos("Blob Min Width", self.trackbar_window)
            b_h = cv2.getTrackbarPos("Blob Min Height", self.trackbar_window)
            r_top = cv2.getTrackbarPos("ROI Top (%)", self.trackbar_window) / 100.0
            
            # Safety: Prevent 0 values for geometry
            look_dist = max(10, cv2.getTrackbarPos("Look Dist (%)", self.trackbar_window)) / 100.0
            t_top = max(5, cv2.getTrackbarPos("Trap Top (%)", self.trackbar_window)) / 100.0
            
            # Apply to State
            self.white_thresh = w
            self.gray_thresh = g
            self.roi_top = r_top
            self.roi_trap_ratio = t_top
            
            # Apply to Config (Live Update)
            self.detector.config.lane_detection.white_threshold = w
            self.detector.config.lane_detection.gray_threshold = g
            self.detector.config.lane_detection.blob_min_width = b_w
            self.detector.config.lane_detection.blob_min_height = b_h
            self.detector.config.lane_detection.roi_top_ratio = r_top
            self.detector.config.lane_detection.roi_trapezoid_top_width_ratio = t_top
            
            # Update Perspective Points based on Look Dist & Trap Top
            self.detector.recalculate_perspective_points(
                top_width_ratio=t_top * 0.5, # 0.0 ~ 0.5 range
                bottom_width_ratio=0.85,
                height_ratio=look_dist
            )
            # Sync src_points for Geometry mode
            self.src_points = self.detector.config.lane_detection.perspective_src_points
            try:
                self.detector.M = cv2.getPerspectiveTransform(self.src_points, self.detector.config.lane_detection.perspective_dst_points)
            except cv2.error:
                pass # Ignore invalid transform
            
            # Exposure Update
            e = e_slider - 13
            if e != self.best_exposure:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, e)
                self.best_exposure = e
                
            return {
                "white": w, "gray": g, "exp": e,
                "blob_w": b_w, "blob_h": b_h,
                "roi_top": r_top, "trap": t_top, "look": look_dist
            }
        except Exception as e:
            print(f"[Trackbar Error] {e}")
            return {}

    def _draw_guide_overlay(self, img, params):
        """
        현재 조정 중인 파라미터에 대한 설명을 화면에 오버레이합니다.
        """
        h, w = img.shape[:2]
        # 반투명 배경 박스
        overlay = img.copy()
        cv2.rectangle(overlay, (w-350, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        x = w - 340
        y = 30
        gap = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        color = (200, 200, 200)
        
        cv2.putText(img, "[ Parameter Guide ]", (x, y), font, 0.6, (0, 255, 255), 2)
        y += 30
        
        guides = [
            ("White/Gray Thresh", "Lower = More Sensitive (See faint lines)"),
            ("", "Higher = Less Noise (Ignore reflections)"),
            ("Exposure", "Adjust brightness. Lines must be visible."),
            ("Blob Min Width", f"Current: {params.get('blob_w', 0)}px"),
            ("", "Increase to remove small noise dots."),
            ("Blob Min Height", f"Current: {params.get('blob_h', 0)}px"),
            ("", "Increase to ignore short horizontal lines."),
            ("ROI Top (%)", "Cut off top part of image (ignore far bg)."),
            ("Trap Top (%)", "Adjust trapezoid shape for perspective."),
        ]
        
        for title, desc in guides:
            if title:
                cv2.putText(img, f"- {title}", (x, y), font, scale, (0, 255, 0), 1)
                y += 20
            cv2.putText(img, f"  {desc}", (x, y), font, scale-0.1, color, 1)
            y += gap

    def _init_camera(self, index):
        print(f"[INFO] 카메라 {index}번 연결 중...")
        
        cap = None
        if os.name == 'nt':
            # Windows: DSHOW -> MSMF -> ANY 순서로 시도
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                print(f"  Trying backend: {backend}...")
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    print(f"  [SUCCESS] Connected with backend {backend}")
                    break
                cap.release()
        else:
            cap = cv2.VideoCapture(index)
            
        if cap is None or not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다. 인덱스를 확인하거나 다른 프로그램을 종료하세요.")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        
        if os.name == 'nt':
            # DSHOW에서만 동작할 수 있음
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Manual Mode
            cap.set(cv2.CAP_PROP_EXPOSURE, self.config.camera.exposure)
            
        return cap

    def tune_exposure(self):
        """
        자동 노출 튜닝: 이미지의 히스토그램을 분석하여 최적의 노출값을 찾습니다.
        너무 어둡거나(0에 몰림) 너무 밝은(255에 몰림) 상태를 피하고 대비를 최대화합니다.
        """
        print("\n[Auto Exposure] 노출 최적화 시작...")
        best_score = -1
        best_exp = -6
        
        # Windows DSHOW 기준 음수 값 범위 (-10 ~ -3 정도가 보통)
        # 환경에 따라 다르므로 넓게 탐색
        test_range = range(-10, 0) if os.name == 'nt' else range(0, 100)
        
        for exp in test_range:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            # 안정화를 위해 몇 프레임 대기
            for _ in range(5): self.cap.read()
            
            ret, frame = self.cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 점수 계산: 표준편차 (대비) + 적절한 밝기 평균 (100~150 사이 선호)
            std_dev = np.std(gray)
            mean_val = np.mean(gray)
            
            # 너무 어둡거나 밝으면 감점
            penalty = 0
            if mean_val < 50 or mean_val > 200:
                penalty = 50
                
            score = std_dev - penalty
            
            if score > best_score:
                best_score = score
                best_exp = exp
                print(f"  Exp: {exp} | Mean: {mean_val:.1f} | Std: {std_dev:.1f} -> Score: {score:.1f} (New Best)")
            else:
                print(f"  Exp: {exp} | Mean: {mean_val:.1f} | Std: {std_dev:.1f} -> Score: {score:.1f}")
                
        print(f"[Auto Exposure] 최적 노출값 선정: {best_exp}")
        self.cap.set(cv2.CAP_PROP_EXPOSURE, best_exp)
        self.best_exposure = best_exp
        return best_exp

    def evaluate_frame(self, binary, warped_shape):
        """
        이진화된 이미지의 품질을 평가하고 차선 정보를 추출합니다.
        """
        h, w = warped_shape
        
        # 1. Histogram으로 차선 위치 파악
        histogram = np.sum(binary[h//2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # 피크 강도 확인 (노이즈 판별)
        left_peak = histogram[leftx_base]
        right_peak = histogram[rightx_base]
        
        # [Debug] Peak Threshold 완화 (500 -> 100)
        # 픽셀값 255 기준, 100은 약 0.4픽셀... 너무 낮나?
        # 하지만 binary가 0/1일 수도 있으므로 안전하게 확인 필요
        # 보통 cv2.threshold는 255를 리턴함.
        if left_peak < 100 or right_peak < 100: 
            return 0.0, 0, 0, 0, "Low Peak Signal"
            
        # 2. Sliding Window 시뮬레이션 (간소화)
        # Connected Components로 덩어리 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        lane_candidates = []
        noise_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            width = stats[i, cv2.CC_STAT_WIDTH]
            ratio = height / width
            
            # 차선 후보 조건: 세로로 길거나(ratio > 2), 면적이 큼
            if ratio > 1.5 and height > 20: # Height 30 -> 20 완화
                lane_candidates.append(stats[i])
            else:
                noise_area += area
                
        if len(lane_candidates) < 2:
            return 0.0, 0, 0, 0, f"Not Enough Blobs ({len(lane_candidates)})"
            
        # 가장 큰 두 덩어리 선택 (좌/우 차선 가정)
        lane_candidates.sort(key=lambda x: x[cv2.CC_STAT_AREA], reverse=True)
        c1 = lane_candidates[0]
        c2 = lane_candidates[1]
        
        # 점수 계산
        # 1. 노이즈 비율 (낮을수록 좋음)
        total_lane_area = c1[cv2.CC_STAT_AREA] + c2[cv2.CC_STAT_AREA]
        snr = total_lane_area / (noise_area + 1)
        score_snr = min(snr, 5.0) / 5.0 # 0~1
        
        # 2. 차선 길이 (길수록 좋음 - 멀리까지 보임)
        avg_height = (c1[cv2.CC_STAT_HEIGHT] + c2[cv2.CC_STAT_HEIGHT]) / 2
        score_height = min(avg_height / h, 1.0)
        
        # 3. 차선 폭 (너무 얇거나 두꺼우면 감점)
        avg_width = (c1[cv2.CC_STAT_WIDTH] + c2[cv2.CC_STAT_WIDTH]) / 2
        score_width = 1.0
        if avg_width < 3 or avg_width > 150: # 5~100 -> 3~150 완화
            score_width = 0.5
            
        final_score = (score_snr * 0.4) + (score_height * 0.4) + (score_width * 0.2)
        
        # 차선 위치 반환 (Centroid X)
        # c1, c2 중 왼쪽/오른쪽 구분
        c1_x = c1[cv2.CC_STAT_LEFT] + c1[cv2.CC_STAT_WIDTH]//2
        c2_x = c2[cv2.CC_STAT_LEFT] + c2[cv2.CC_STAT_WIDTH]//2
        
        if c1_x < c2_x:
            lx, rx = c1_x, c2_x
        else:
            lx, rx = c2_x, c1_x
            
        # 평균 픽셀 밀도 (Sliding Window minpix 계산용)
        # 높이 100px 당 픽셀 수 근사
        avg_density = (total_lane_area / (avg_height + 1)) * 100
            
        return final_score, lx, rx, avg_density, "Success"

    def run_auto_search(self, frame):
        # 1. 노출 튜닝 먼저 수행
        self.tune_exposure()
        
        print("\n[Auto Search] 최적 파라미터 탐색 시작...")
        h, w = frame.shape[:2]
        
        # 탐색 범위
        white_range = range(100, 240, 10)
        gray_range = range(100, 240, 10)
        
        best_result = SearchResult(score=-1, white_thresh=0, gray_thresh=0, lane_width=0, left_x=0, right_x=0, avg_pixel_density=0)
        
        total_steps = len(white_range) * len(gray_range)
        curr_step = 0
        
        # 실패 원인 분석용
        fail_reasons = {}
        
        # Perspective Transform 행렬 미리 계산 (Geometry는 고정)
        M = cv2.getPerspectiveTransform(self.src_points, self.detector.config.lane_detection.perspective_dst_points)
        
        for w_th in white_range:
            for g_th in gray_range:
                # 설정 적용
                self.detector.config.lane_detection.white_threshold = w_th
                self.detector.config.lane_detection.gray_threshold = g_th
                
                # 전처리
                binary, _ = self.detector.preprocess_frame(frame)
                warped = cv2.warpPerspective(binary, M, (w, h))
                
                # 평가
                score, lx, rx, density, reason = self.evaluate_frame(warped, (h, w))
                
                if score > best_result.score:
                    width = rx - lx
                    # 차선 폭이 너무 좁거나(붙어있음) 너무 넓으면(오인식) 제외
                    # [Debug] 폭 조건 완화 (100~90% -> 50~95%)
                    if 50 < width < w * 0.95:
                        best_result = SearchResult(score, w_th, g_th, width, lx, rx, density)
                        print(f"  New Best! Score:{score:.2f} | W:{w_th} G:{g_th} | Width:{width} | Density:{density:.0f}")
                    else:
                        r = f"Width Out of Range ({width})"
                        fail_reasons[r] = fail_reasons.get(r, 0) + 1
                else:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                
                curr_step += 1
                if curr_step % 50 == 0:
                    print(f"  Progress: {curr_step}/{total_steps}")
        
        if best_result.score <= 0:
            print("\n[FAIL Analysis] 실패 원인 분석:")
            for reason, count in fail_reasons.items():
                print(f"  - {reason}: {count}회")
            print("  -> 힌트: 'Low Peak'는 조명/Threshold 문제, 'Not Enough Blobs'는 끊긴 차선/ROI 문제, 'Width'는 ROI 크기 문제입니다.")
                    
        return best_result

    def derive_parameters(self, result: SearchResult):
        """
        검출된 차선 정보를 바탕으로 최적 파라미터를 역산합니다.
        """
        lane_width = result.lane_width
        
        params = {}
        params['white_threshold'] = result.white_thresh
        params['gray_threshold'] = result.gray_thresh
        params['exposure'] = self.best_exposure
        
        # 1. Sliding Window Margin (차선 폭의 30% 정도 여유)
        # 차선이 휘어질 때를 대비해 충분히 넓게 잡되, 옆 차선을 침범하지 않도록
        params['window_margin'] = int(lane_width * 0.35)
        
        # 2. Blob Filter (차선 폭 기준)
        # 최소 너비: 차선 폭의 5% (끊긴 차선이나 얇은 부분 고려)
        # 최대 너비: 차선 폭의 50% (그림자 등 덩어리 제거)
        params['blob_min_width'] = max(5, int(lane_width * 0.05))
        params['blob_max_width'] = int(lane_width * 0.6)
        
        # 3. Lane Position (Perspective Source 보정용 참고값)
        params['lane_left_x'] = int(result.left_x)
        params['lane_right_x'] = int(result.right_x)
        
        # 4. Lane Width Tolerance (차선 폭 변화 허용 범위)
        # 코너링 시 차선 폭이 달라 보일 수 있으므로 여유 있게
        params['lane_width_tolerance'] = 0.8 # 80% 변화 허용
        
        # 5. Sliding Window Min Pixels (밀도 기반)
        # 평균 밀도의 20% 수준으로 설정하여 끊긴 차선도 잡되 노이즈는 무시
        # [Safety] 너무 큰 값 방지 (Max 100)
        params['min_pixels'] = min(100, max(10, int(result.avg_pixel_density * 0.2)))
        
        # 6. Blob Min Height (밀도 기반)
        # 픽셀 밀도가 높으면(선명하면) 기준을 높여 노이즈 제거
        # [Safety] 너무 큰 값 방지 (Max 60)
        params['blob_min_height'] = min(60, max(20, int(result.avg_pixel_density * 0.1)))
        
        # 7. Trapezoid Ratio (자동 맞춤)
        # 차선이 11자라면 상단 폭도 하단 폭과 비슷해야 함
        # 하지만 원근 왜곡 보정이 완벽하지 않을 수 있으므로 0.8 정도로 설정
        params['roi_trapezoid_top_width_ratio'] = 0.85
        
        return params

    def save_config(self):
        if not self.derived_params:
            print("[WARN] 저장할 파라미터가 없습니다. Auto Search를 먼저 수행하세요.")
            return

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        p = self.derived_params
        
        # Regex Replacement
        replacements = [
            (r'white_threshold: int = \d+', f'white_threshold: int = {p["white_threshold"]}'),
            (r'gray_threshold: int = \d+', f'gray_threshold: int = {p["gray_threshold"]}'),
            (r'margin: int = \d+', f'margin: int = {p["window_margin"]}'),
            (r'blob_min_width: int = \d+', f'blob_min_width: int = {p["blob_min_width"]}'),
            (r'blob_max_width: int = \d+', f'blob_max_width: int = {p["blob_max_width"]}'),
            (r'lane_width_tolerance: float = [\d\.]+', f'lane_width_tolerance: float = {p["lane_width_tolerance"]}'),
            # ROI Top Ratio
            (r'roi_top_ratio: float = [\d\.]+', f'roi_top_ratio: float = {self.roi_top:.2f}'),
            # Lane Positions (참고용)
            (r'lane_left_x: int = \d+', f'lane_left_x: int = {p["lane_left_x"]}'),
            (r'lane_right_x: int = \d+', f'lane_right_x: int = {p["lane_right_x"]}'),
            # Exposure
            (r'exposure: int = -?\d+', f'exposure: int = {p["exposure"]}'),
            # New Params
            (r'min_pixels: int = \d+', f'min_pixels: int = {p["min_pixels"]}'),
            (r'blob_min_height: int = \d+', f'blob_min_height: int = {p["blob_min_height"]}'),
            (r'roi_trapezoid_top_width_ratio: float = [\d\.]+', f'roi_trapezoid_top_width_ratio: float = {p["roi_trapezoid_top_width_ratio"]}'),
        ]
        
        # Perspective Points (역산)
        tl, tr, br, bl = self.src_points
        horizon_y = int((tl[1] + tr[1]) / 2)
        hood_bottom_y = int((bl[1] + br[1]) / 2)
        
        replacements.append((r'horizon_y: int = \d+', f'horizon_y: int = {horizon_y}'))
        replacements.append((r'hood_bottom_y: int = \d+', f'hood_bottom_y: int = {hood_bottom_y}'))
        
        for pattern, repl in replacements:
            content = re.sub(pattern, repl, content)
            
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n[저장 완료] {config_path} 업데이트 됨.")
        print("적용된 값:", p)

    def run(self):
        print("="*60)
        print("       Ultimate Auto Calibration Tool")
        print("="*60)
        print("1. [GEOMETRY] 초록색 박스(ROI)를 조절하여 차선이 11자가 되게 맞추세요.")
        print("   - 멀리 있는 차선이 안 보이면 ROI 상단을 위로 올리세요 (T 키).")
        print("2. [SPACE] 키를 누르면 '자동 탐색'이 시작됩니다.")
        print("   - 노출(밝기) 자동 조절 -> Threshold 탐색 -> 파라미터 계산")
        print("3. [M] 키를 누르면 '수동 튜닝(Manual)' 모드로 진입합니다.")
        print("   - 슬라이더로 직접 Threshold와 노출을 조절할 수 있습니다.")
        print("4. [S] 키를 눌러 저장하고 종료합니다.")
        print("-" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다. (ret=False)")
                break
            
            display = frame.copy()
            h, w = frame.shape[:2]
            
            # 현재 설정 적용
            self.detector.config.lane_detection.perspective_src_points = self.src_points
            self.detector.M = cv2.getPerspectiveTransform(self.src_points, self.detector.config.lane_detection.perspective_dst_points)
            
            # 시각화
            if self.mode == "GEOMETRY":
                # ROI Box
                pts = self.src_points.astype(np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                
                # Selected Point
                cx, cy = pts[self.selected_point]
                cv2.circle(display, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(display, self.point_names[self.selected_point], (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Warped Preview
                warped = cv2.warpPerspective(frame, self.detector.M, (w, h))
                
                # Guide Lines
                cv2.line(warped, (w//4, 0), (w//4, h), (0, 255, 255), 1)
                cv2.line(warped, (w*3//4, 0), (w*3//4, h), (0, 255, 255), 1)
                
                final_view = np.hstack((display, warped))
                cv2.putText(final_view, "MODE: GEOMETRY (Adjust ROI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(final_view, "Keys: 1-4(Select), I/J/K/L(Move), T/G(Top), SPACE(Auto), M(Manual)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            elif self.mode == "MANUAL_TUNING":
                params = self._update_from_trackbars()
                
                # 전처리 결과 확인
                binary, _ = self.detector.preprocess_frame(frame)
                warped_bin = cv2.warpPerspective(binary, self.detector.M, (w, h))
                warped_color = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)
                
                # 차선 검출 시도 (시각화용)
                score, lx, rx, _, _ = self.evaluate_frame(warped_bin, (h, w))
                if score > 0:
                    cv2.line(warped_color, (lx, 0), (lx, h), (0, 255, 0), 2)
                    cv2.line(warped_color, (rx, 0), (rx, h), (0, 255, 0), 2)
                    status = f"DETECTED (Width: {rx-lx})"
                    color = (0, 255, 0)
                else:
                    status = "NOT DETECTED"
                    color = (0, 0, 255)
                
                final_view = np.hstack((display, warped_color))
                
                # 가이드 오버레이 그리기
                self._draw_guide_overlay(final_view, params)
                
                cv2.putText(final_view, f"MODE: MANUAL TUNING - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(final_view, "Adjust Sliders in 'Manual Tuning' window. Press 'S' to Save.", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            elif self.mode == "RESULT":
                # 결과 보여주기
                p = self.derived_params
                
                # Apply found params for preview
                self.detector.config.lane_detection.white_threshold = p['white_threshold']
                self.detector.config.lane_detection.gray_threshold = p['gray_threshold']
                
                binary, _ = self.detector.preprocess_frame(frame)
                warped_bin = cv2.warpPerspective(binary, self.detector.M, (w, h))
                warped_color = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)
                
                # Draw detected lane width
                lx = p['lane_left_x']
                rx = p['lane_right_x']
                cv2.line(warped_color, (lx, 0), (lx, h), (0, 0, 255), 2)
                cv2.line(warped_color, (rx, 0), (rx, h), (0, 0, 255), 2)
                
                final_view = np.hstack((display, warped_color))
                
                lines = [
                    f"MODE: RESULT (Score: {self.auto_result.score:.2f})",
                    f"Exp: {p['exposure']}, White: {p['white_threshold']}, Gray: {p['gray_threshold']}",
                    f"Lane Width: {self.auto_result.lane_width:.0f}px",
                    f"-> Margin: {p['window_margin']}px, MinPix: {p['min_pixels']}",
                    f"-> Blob Min: {p['blob_min_width']}px, Height: {p['blob_min_height']}",
                    "Press 'S' to Save & Quit, 'R' to Retry"
                ]
                
                for i, line in enumerate(lines):
                    cv2.putText(final_view, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Calibration Tool", final_view)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.mode == "RESULT":
                    self.save_config()
                    break
                elif self.mode == "MANUAL_TUNING":
                    # 수동 모드에서 저장 시 현재 값으로 파라미터 생성
                    # 차선이 검출되지 않았어도 강제 저장 가능하게 함
                    # 단, derived_params를 채워야 save_config가 동작함
                    
                    # 현재 상태에서 한 번 더 평가
                    binary, _ = self.detector.preprocess_frame(frame)
                    warped = cv2.warpPerspective(binary, self.detector.M, (w, h))
                    score, lx, rx, density, _ = self.evaluate_frame(warped, (h, w))
                    
                    width = rx - lx if score > 0 else 400 # 기본값
                    
                    # 가짜 결과 생성
                    dummy_result = SearchResult(score, self.white_thresh, self.gray_thresh, width, lx, rx, density)
                    self.derived_params = self.derive_parameters(dummy_result)
                    self.save_config()
                    break

            elif key == ord('r') and self.mode == "RESULT":
                self.mode = "GEOMETRY"
            elif key == ord('m'): # Manual Mode Toggle
                if self.mode != "MANUAL_TUNING":
                    self.mode = "MANUAL_TUNING"
                    self._create_trackbars()
                else:
                    self.mode = "GEOMETRY"
                    cv2.destroyWindow(self.trackbar_window)

            elif key == 32: # SPACE
                if self.mode == "GEOMETRY":
                    # 10프레임 평균 이미지 사용 (노이즈 감소)
                    print("Capturing frames...")
                    frames = []
                    for _ in range(10):
                        ret, f = self.cap.read()
                        if ret: frames.append(f)
                        time.sleep(0.05)
                    avg_frame = frames[len(frames)//2]
                    
                    self.auto_result = self.run_auto_search(avg_frame)
                    if self.auto_result.score > 0:
                        self.derived_params = self.derive_parameters(self.auto_result)
                        self.mode = "RESULT"
                    else:
                        print("[FAIL] 차선을 찾지 못했습니다. ROI를 조정하거나 조명을 확인하세요.")
            
            # Geometry Controls
            if self.mode == "GEOMETRY":
                if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    self.selected_point = int(chr(key)) - 1
                
                step = 2
                if key == ord('i'): self.src_points[self.selected_point][1] -= step
                if key == ord('k'): self.src_points[self.selected_point][1] += step
                if key == ord('j'): self.src_points[self.selected_point][0] -= step
                if key == ord('l'): self.src_points[self.selected_point][0] += step
                
                if key == ord('t'): self.roi_top = max(0.0, self.roi_top - 0.01)
                if key == ord('g'): self.roi_top = min(1.0, self.roi_top + 0.01)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=1)
    args = parser.parse_args()
    
    tool = AdvancedCalibrationTool(args.camera_index)
    tool.run()
