"""
gui_controller.py
GUI 컨트롤러 모듈 (Tkinter 기반)
- 실시간 영상 표시
- 파라미터 실시간 조정
- 상태 정보 표시
"""

import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL
import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Callable, Optional
from config import get_config, update_config


class GUIController:
    """GUI 컨트롤러 클래스"""
    
    def __init__(self, update_callback: Optional[Callable] = None):
        """
        초기화
        
        Args:
            update_callback: 파라미터 변경 시 호출될 콜백 함수
        """
        self.config = get_config()
        self.update_callback = update_callback
        
        # Tkinter 윈도우 생성
        self.root = tk.Tk()
        self.root.title(self.config.gui.window_title)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 상태 변수
        self.running = True
        self.paused = False
        
        # UI 구성
        self._create_ui()
    
    def _create_ui(self):
        """UI 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 좌측: 비디오 디스플레이
        video_frame = ttk.LabelFrame(main_frame, text="실시간 영상", padding="5")
        video_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky=(tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # 우측: 제어 패널 (스크롤 가능하도록 Canvas + Scrollbar 사용)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Canvas 생성
        canvas = tk.Canvas(right_frame, width=350)  # 너비 고정
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 제어 패널 내용을 scrollable_frame에 추가
        control_frame = ttk.LabelFrame(scrollable_frame, text="제어 패널", padding="10")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self._create_control_panel(control_frame)
        
        # 상태 정보 패널
        status_frame = ttk.LabelFrame(scrollable_frame, text="상태 정보", padding="10")
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self._create_status_panel(status_frame)
        
        # 버튼 패널
        button_frame = ttk.Frame(scrollable_frame, padding="5")
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self._create_buttons(button_frame)

        # 마우스 휠 스크롤 바인딩
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_control_panel(self, parent):
        """제어 패널 생성"""
        # Grid 가중치 설정
        parent.columnconfigure(0, weight=1)
        
        row = 0
        
        # 1. PID 제어 그룹
        pid_group = ttk.LabelFrame(parent, text="PID 제어", padding="5")
        pid_group.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        pid_group.columnconfigure(1, weight=1)
        row += 1
        
        self.kp_scale = self._add_slider(pid_group, 0, "Kp (비례)", 0.0, 100.0, 1.0, 
                        self.config.path_planning.pid_kp,
                        lambda v: self._update_param("path_planning", "pid_kp", float(v)))
        
        self.ki_scale = self._add_slider(pid_group, 1, "Ki (적분)", 0.0, 10.0, 0.01,
                        self.config.path_planning.pid_ki,
                        lambda v: self._update_param("path_planning", "pid_ki", float(v)))
        
        self.kd_scale = self._add_slider(pid_group, 2, "Kd (미분)", 0.0, 50.0, 0.5,
                        self.config.path_planning.pid_kd,
                        lambda v: self._update_param("path_planning", "pid_kd", float(v)))

        # 2. 차선 검출 파라미터 그룹
        lane_group = ttk.LabelFrame(parent, text="차선 검출 파라미터", padding="5")
        lane_group.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        lane_group.columnconfigure(1, weight=1)
        row += 1
        
        # White Threshold
        self.white_thresh_scale = self._add_slider(lane_group, 0, "White Thresh", 50, 255, 1,
                        self.config.lane_detection.white_threshold,
                        lambda v: self._update_param("lane_detection", "white_threshold", int(float(v))))
        
        # Black Threshold
        self.black_thresh_scale = self._add_slider(lane_group, 1, "Black Thresh", 10, 150, 1,
                        self.config.lane_detection.black_threshold,
                        lambda v: self._update_param("lane_detection", "black_threshold", int(float(v))))

        # Canny Low
        self.canny_low_scale = self._add_slider(lane_group, 2, "Canny Low", 0, 100, 1,
                        self.config.lane_detection.canny_low_threshold,
                        lambda v: self._update_param("lane_detection", "canny_low_threshold", int(float(v))))

        # Canny High
        self.canny_high_scale = self._add_slider(lane_group, 3, "Canny High", 50, 255, 1,
                        self.config.lane_detection.canny_high_threshold,
                        lambda v: self._update_param("lane_detection", "canny_high_threshold", int(float(v))))

        # Sliding Window Margin
        self.margin_scale = self._add_slider(lane_group, 4, "SW Margin", 30, 180, 5,
                        self.config.sliding_window.margin,
                        lambda v: self._update_param("sliding_window", "margin", int(float(v))))

        # Sliding Window Min Pixels
        self.min_pix_scale = self._add_slider(lane_group, 5, "SW Min Pix", 5, 60, 1,
                        self.config.sliding_window.min_pixels,
                        lambda v: self._update_param("sliding_window", "min_pixels", int(float(v))))

        # 3. ROI 설정 그룹
        roi_group = ttk.LabelFrame(parent, text="ROI 설정", padding="5")
        roi_group.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        roi_group.columnconfigure(1, weight=1)
        row += 1
        
        self.roi_top_scale = self._add_slider(roi_group, 0, "Top Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_top_ratio,
                        lambda v: self._update_param("lane_detection", "roi_top_ratio", float(v)))
        
        self.roi_bottom_scale = self._add_slider(roi_group, 1, "Bottom Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_bottom_ratio,
                        lambda v: self._update_param("lane_detection", "roi_bottom_ratio", float(v)))

        self.roi_left_scale = self._add_slider(roi_group, 2, "Left Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_left_ratio,
                        lambda v: self._update_param("lane_detection", "roi_left_ratio", float(v)))

        self.roi_right_scale = self._add_slider(roi_group, 3, "Right Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_right_ratio,
                        lambda v: self._update_param("lane_detection", "roi_right_ratio", float(v)))

        self.roi_trap_scale = self._add_slider(roi_group, 4, "Trap Top Width", 0.0, 1.0, 0.05,
                        self.config.lane_detection.roi_trapezoid_top_width_ratio,
                        lambda v: self._update_param("lane_detection", "roi_trapezoid_top_width_ratio", float(v)))

        # Parallel Lock Checkbox
        self.parallel_var = tk.BooleanVar(value=self.config.lane_detection.enable_joint_fitting)
        ttk.Checkbutton(roi_group, text="Parallel Lock (꼬임 방지)", variable=self.parallel_var,
                        command=lambda: self._update_param("lane_detection", "enable_joint_fitting", self.parallel_var.get())
                        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Blob Filter Checkbox
        self.blob_var = tk.BooleanVar(value=self.config.lane_detection.enable_blob_filter)
        ttk.Checkbutton(roi_group, text="Blob Filter (차량 제거)", variable=self.blob_var,
                        command=lambda: self._update_param("lane_detection", "enable_blob_filter", self.blob_var.get())
                        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        self.blob_width_scale = self._add_slider(roi_group, 7, "Max Blob Width", 10, 200, 5,
                        self.config.lane_detection.blob_max_width,
                        lambda v: self._update_param("lane_detection", "blob_max_width", int(float(v))))

    def _add_slider(self, parent, row, label_text, from_, to, resolution, init_val, command):
        """슬라이더 추가 헬퍼 메서드"""
        ttk.Label(parent, text=label_text, width=12).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        scale = Scale(parent, from_=from_, to=to, resolution=resolution, orient=HORIZONTAL, command=command)
        scale.set(init_val)
        scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        return scale
    
    def _create_status_panel(self, parent):
        """상태 정보 패널 생성"""
        # 상태 라벨들
        self.status_labels = {}
        
        labels = [
            ("fps", "FPS:"),
            ("offset", "중앙 오프셋:"),
            ("steering", "조향각:"),
            ("curvature", "곡률:"),
            ("detection", "차선 검출:"),
            ("warning", "경고:"),
            ("last_param", "최근 변경:")
        ]
        
        for idx, (key, text) in enumerate(labels):
            ttk.Label(parent, text=text, font=("", 9, "bold")).grid(
                row=idx, column=0, sticky=tk.W, pady=2
            )
            
            label = ttk.Label(parent, text="N/A", font=("", 9))
            label.grid(row=idx, column=1, sticky=tk.W, padx=10, pady=2)
            self.status_labels[key] = label
    
    def _create_buttons(self, parent):
        """버튼 생성"""
        # 일시정지/재개 버튼
        self.pause_button = ttk.Button(
            parent, text="일시정지", command=self.toggle_pause
        )
        self.pause_button.grid(row=0, column=0, padx=5)
        
        # 리셋 버튼
        ttk.Button(parent, text="파라미터 리셋", command=self.reset_parameters).grid(
            row=0, column=1, padx=5
        )
        
        # 종료 버튼
        ttk.Button(parent, text="종료", command=self.on_closing).grid(
            row=0, column=2, padx=5
        )
    
    def _update_param(self, section: str, param: str, value):
        """
        파라미터 업데이트
        
        Args:
            section: 설정 섹션
            param: 파라미터 이름
            value: 새 값
        """
        success = update_config(section, param, value)
        
        if success:
            # 상태 패널에 변경된 값 표시
            if "last_param" in self.status_labels:
                self.status_labels["last_param"].config(
                    text=f"{param} = {value}", foreground="blue"
                )
            
            if self.update_callback is not None:
                self.update_callback(section, param, value)
    
    def update_video(self, frame: np.ndarray):
        """
        비디오 프레임 업데이트
        
        Args:
            frame: BGR 이미지
        """
        # BGR → RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(rgb_frame)
        
        # Tkinter PhotoImage로 변환
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # 라벨 업데이트
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # 참조 유지 (가비지 컬렉션 방지)
    
    def update_status(
        self,
        fps: float,
        lane_result: dict,
        path_result: dict
    ):
        """
        상태 정보 업데이트
        
        Args:
            fps: 현재 FPS
            lane_result: 차선 검출 결과
            path_result: 경로 계획 결과
        """
        # FPS
        self.status_labels["fps"].config(text=f"{fps:.1f}")
        
        # 검출 상태
        if lane_result['detected']:
            self.status_labels["detection"].config(text="성공 ✓", foreground="green")
        else:
            self.status_labels["detection"].config(text="실패 ✗", foreground="red")
        
        # 경로 정보
        if path_result['valid']:
            # 오프셋
            offset = path_result['center_offset']
            self.status_labels["offset"].config(
                text=f"{offset:.3f} m {'(좌)' if offset < 0 else '(우)'}"
            )
            
            # 조향각
            steering = path_result['steering_angle']
            self.status_labels["steering"].config(
                text=f"{steering:.1f}°"
            )
            
            # 곡률
            left_curv = path_result['left_curvature']
            right_curv = path_result['right_curvature']
            avg_curv = (left_curv + right_curv) / 2
            self.status_labels["curvature"].config(
                text=f"{avg_curv:.1f} m"
            )
            
            # 경고
            if path_result['lane_departure_warning']:
                self.status_labels["warning"].config(
                    text="차선 이탈 위험!", foreground="red", font=("", 9, "bold")
                )
            else:
                self.status_labels["warning"].config(
                    text="정상", foreground="green", font=("", 9)
                )
        else:
            # 유효하지 않은 경우
            self.status_labels["offset"].config(text="N/A")
            self.status_labels["steering"].config(text="N/A")
            self.status_labels["curvature"].config(text="N/A")
            self.status_labels["warning"].config(text="N/A", foreground="black")
    
    def toggle_pause(self):
        """일시정지/재개 토글"""
        self.paused = not self.paused
        
        if self.paused:
            self.pause_button.config(text="재개")
        else:
            self.pause_button.config(text="일시정지")
    
    def reset_parameters(self):
        """파라미터 리셋"""
        # 기본값으로 리셋
        from config import PathPlanningConfig, LaneDetectionConfig, SlidingWindowConfig
        
        default_path = PathPlanningConfig()
        default_lane = LaneDetectionConfig()
        default_sw = SlidingWindowConfig()
        
        # 슬라이더 업데이트
        self.kp_scale.set(default_path.pid_kp)
        self.ki_scale.set(default_path.pid_ki)
        self.kd_scale.set(default_path.pid_kd)
        
        self.white_thresh_scale.set(default_lane.white_threshold)
        self.black_thresh_scale.set(default_lane.black_threshold)
        self.canny_low_scale.set(default_lane.canny_low_threshold)
        self.canny_high_scale.set(default_lane.canny_high_threshold)
        
        self.margin_scale.set(default_sw.margin)
        self.min_pix_scale.set(default_sw.min_pixels)
        
        self.roi_top_scale.set(default_lane.roi_top_ratio)
        self.roi_bottom_scale.set(default_lane.roi_bottom_ratio)
        self.roi_left_scale.set(default_lane.roi_left_ratio)
        self.roi_right_scale.set(default_lane.roi_right_ratio)
        self.roi_trap_scale.set(default_lane.roi_trapezoid_top_width_ratio)
        
        print("[INFO] 파라미터가 기본값으로 리셋되었습니다.")
    
    def on_closing(self):
        """윈도우 닫기 이벤트"""
        self.running = False
        self.root.quit()
        self.root.destroy()
    
    def update(self):
        """Tkinter 이벤트 루프 업데이트"""
        if self.running:
            self.root.update_idletasks()
            self.root.update()


# 테스트 코드
if __name__ == "__main__":
    print("GUIController 모듈 로드 완료")
    
    # 간단한 GUI 테스트
    def param_callback(section, param, value):
        print(f"파라미터 변경: {section}.{param} = {value}")
    
    gui = GUIController(update_callback=param_callback)
    
    # 테스트용 빈 프레임
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Frame", (500, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    gui.update_video(test_frame)
    
    try:
        while gui.running:
            gui.update()
    except KeyboardInterrupt:
        pass

    def setup_debug_windows(self):
        """디버깅 창 초기화"""
        cv2.namedWindow("1. Combined Binary", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1. Combined Binary", 640, 360)
        
        cv2.namedWindow("2. Binary Warped (BEV)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2. Binary Warped (BEV)", 640, 360)
        
        cv2.namedWindow("3. Histogram", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3. Histogram", 800, 300)
        
        cv2.namedWindow("4. Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("4. Detection", 640, 360)
        
        cv2.namedWindow("5. Color Masks", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("5. Color Masks", 640, 360)
        
        self.debug_windows_enabled = True
        print("[GUI] Debug windows created")
    
    def close_debug_windows(self):
        """디버깅 창 닫기"""
        for name in ["1. Combined Binary", "2. Binary Warped (BEV)", 
                     "3. Histogram", "4. Detection", "5. Color Masks"]:
            try:
                cv2.destroyWindow(name)
            except:
                pass
        self.debug_windows_enabled = False
    
    def update_debug_windows(self, result: dict):
        """디버깅 정보 실시간 업데이트"""
        if not hasattr(self, 'debug_windows_enabled') or not self.debug_windows_enabled:
            return
        
        # 1. Combined Binary
        if 'preprocess_debug' in result and 'combined_binary' in result['preprocess_debug']:
            cv2.imshow("1. Combined Binary", result['preprocess_debug']['combined_binary'])
        
        # 2. Binary Warped
        if 'binary_warped' in result:
            cv2.imshow("2. Binary Warped (BEV)", result['binary_warped'])
        
        # 3. Histogram
        if 'binary_warped' in result:
            binary_warped = result['binary_warped']
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
            hist_img = self.draw_histogram_graph(histogram)
            cv2.imshow("3. Histogram", hist_img)
        
        # 4. Detection (sliding window visualization)
        if 'out_img' in result:
            cv2.imshow("4. Detection", result['out_img'].astype(np.uint8))
        
        # 5. Color Masks
        if 'preprocess_debug' in result:
            debug = result['preprocess_debug']
            if 'white_combined' in debug and 'yellow_combined' in debug:
                white = debug['white_combined']
                yellow = debug['yellow_combined']
                
                # 3채널로 변환
                h, w = white.shape
                color_masks = np.zeros((h, w, 3), dtype=np.uint8)
                color_masks[white > 0] = [255, 255, 255]  # 흰색
                color_masks[yellow > 0] = [0, 255, 255]   # 노랑
                
                cv2.imshow("5. Color Masks", color_masks)
    
    def draw_histogram_graph(self, histogram: np.ndarray) -> np.ndarray:
        """히스토그램 그래프 그리기"""
        hist_height = 300
        hist_width = len(histogram)
        hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        # 정규화
        if np.max(histogram) > 0:
            norm_hist = histogram / np.max(histogram) * (hist_height - 10)
        else:
            norm_hist = np.zeros_like(histogram)
        
        # 히스토그램 그리기
        for i in range(len(histogram)):
            cv2.line(hist_img,
                     (i, hist_height),
                     (i, hist_height - int(norm_hist[i])),
                     (255, 255, 255), 1)
        
        # 중앙선
        midpoint = hist_width // 2
        cv2.line(hist_img, (midpoint, 0), (midpoint, hist_height), (0, 255, 0), 2)
        
        # Hood bounds 표시
        # (optional: 나중에 추가)
        
        return hist_img
    
    def draw_debug_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """프레임에 디버깅 정보 오버레이"""
        frame = frame.copy()
        
        # 검출 상태
        if result.get('detected', False):
            status_text = "DETECTED"
            status_color = (0, 255, 0)
        else:
            reason = result.get('validation_reason', 'unknown')
            status_text = f"LOST: {reason}"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS
        if hasattr(self, 'current_fps'):
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Line IoU (있으면)
        if 'line_iou_left' in result:
            iou_text = f"IoU L:{result['line_iou_left']:.2f} R:{result['line_iou_right']:.2f}"
            cv2.putText(frame, iou_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Lane Width 학습 상태
        if 'lane_width_learning_complete' in result:
            if result['lane_width_learning_complete']:
                width_text = f"Width: {result.get('avg_lane_width', 0):.0f}px (Learned)"
                width_color = (0, 255, 0)
            else:
                history_len = len(result.get('lane_width_history', []))
                width_text = f"Width: Learning... ({history_len}/30)"
                width_color = (0, 255, 255)
            
            cv2.putText(frame, width_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, width_color, 2)
        
        return frame

