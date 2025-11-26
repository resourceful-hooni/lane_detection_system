"""
gui_controller.py
GUI 컨트롤러 모듈 (Tkinter 기반)
- 실시간 영상 표시
- 파라미터 실시간 조정
- 상태 정보 표시
- 디버그 윈도우 제어
"""

import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL, VERTICAL
import cv2
from PIL import Image, ImageTk
import numpy as np
from typing import Callable, Optional
from config import get_config, update_config


class GUIController:
    """GUI 컨트롤러 클래스"""
    
    def __init__(self, update_callback: Optional[Callable] = None, record_callback: Optional[Callable] = None):
        """
        초기화
        
        Args:
            update_callback: 파라미터 변경 시 호출될 콜백 함수
            record_callback: 녹화 토글 시 호출될 콜백 함수
        """
        self.config = get_config()
        self.update_callback = update_callback
        self.record_callback = record_callback
        
        # Tkinter 윈도우 생성
        self.root = tk.Tk()
        self.root.title(f"{self.config.gui.window_title} (v2.0)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 스타일 설정
        self._setup_styles()
        
        # 상태 변수
        self.running = True
        self.paused = False
        self.debug_windows_enabled = False  # 디버그 창 상태
        
        # UI 구성
        self._create_ui()
        
        # 키보드 바인딩
        self._setup_keyboard_bindings()
        
        # 초기 디버그 창 설정 (Config에 따라)
        # 기본적으로는 닫아두고 사용자가 버튼으로 열게 함
        self.setup_debug_windows()
        print("[GUI] GUI v2.0 초기화 완료 - 디버그 창 자동 실행")

    def _setup_styles(self):
        """GUI 스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')  # 조금 더 현대적인 테마
        
        # 프레임 스타일
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0", relief="groove")
        style.configure("TLabelframe.Label", font=("맑은 고딕", 10, "bold"), background="#f0f0f0", foreground="#333")
        
        # 라벨 스타일
        style.configure("TLabel", background="#f0f0f0", font=("맑은 고딕", 9))
        style.configure("Status.TLabel", background="#f0f0f0", font=("맑은 고딕", 11))
        style.configure("Value.TLabel", background="#ffffff", font=("Consolas", 11, "bold"), relief="sunken", padding=2)
        
        # 버튼 스타일
        style.configure("TButton", font=("맑은 고딕", 9))
        style.configure("Action.TButton", font=("맑은 고딕", 10, "bold"), padding=5)
        
        # 탭 스타일
        style.configure("TNotebook", background="#e0e0e0")
        style.configure("TNotebook.Tab", padding=[10, 5], font=("맑은 고딕", 10))

    def _create_ui(self):
        """UI 구성"""
        # 메인 컨테이너
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # === 좌측: 비디오 디스플레이 ===
        video_frame = ttk.LabelFrame(main_container, text="🎥 실시간 모니터링", padding="5")
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.pack(fill="both", expand=True)
        
        # === 우측: 제어 패널 (탭 구조) ===
        control_panel = ttk.Frame(main_container, width=400)
        control_panel.pack(side="right", fill="y")
        
        # 탭 컨트롤 생성
        notebook = ttk.Notebook(control_panel)
        notebook.pack(fill="both", expand=True)
        
        # 탭 1: 대시보드 (상태 정보)
        tab_dashboard = ttk.Frame(notebook, padding=10)
        notebook.add(tab_dashboard, text="📊 대시보드")
        self._create_dashboard_tab(tab_dashboard)
        
        # 탭 2: 튜닝 (파라미터 조절)
        tab_tuning = ttk.Frame(notebook, padding=10)
        notebook.add(tab_tuning, text="⚙️ 튜닝")
        self._create_tuning_tab(tab_tuning)
        
        # 탭 3: 시스템 (디버그 및 설정)
        tab_system = ttk.Frame(notebook, padding=10)
        notebook.add(tab_system, text="🔧 시스템")
        self._create_system_tab(tab_system)
        
        # 하단: 공통 액션 버튼
        action_frame = ttk.Frame(control_panel, padding="5")
        action_frame.pack(fill="x", pady=10)
        
        self.record_button = ttk.Button(action_frame, text="⏺ 녹화 시작", style="Action.TButton", command=self.toggle_record)
        self.record_button.pack(side="left", fill="x", expand=True, padx=2)
        
        self.pause_button = ttk.Button(action_frame, text="⏯ 일시정지", style="Action.TButton", command=self.toggle_pause)
        self.pause_button.pack(side="left", fill="x", expand=True, padx=2)
        
        ttk.Button(action_frame, text="❌ 종료", style="Action.TButton", command=self.on_closing).pack(side="left", fill="x", expand=True, padx=2)

    def _create_dashboard_tab(self, parent):
        """대시보드 탭 생성"""
        # 1. 주요 상태 (FPS, 조향각)
        status_group = ttk.LabelFrame(parent, text="주행 상태", padding=10)
        status_group.pack(fill="x", pady=5)
        
        self.status_labels = {}
        
        # Grid Layout
        # FPS
        ttk.Label(status_group, text="FPS:", style="Status.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.status_labels["fps"] = ttk.Label(status_group, text="0.0", style="Value.TLabel", width=10)
        self.status_labels["fps"].grid(row=0, column=1, sticky="e", pady=5)
        
        # 조향각
        ttk.Label(status_group, text="조향각:", style="Status.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.status_labels["steering"] = ttk.Label(status_group, text="0.0°", style="Value.TLabel", width=10)
        self.status_labels["steering"].grid(row=1, column=1, sticky="e", pady=5)
        
        # 중앙 오프셋
        ttk.Label(status_group, text="오프셋:", style="Status.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.status_labels["offset"] = ttk.Label(status_group, text="0.00m", style="Value.TLabel", width=10)
        self.status_labels["offset"].grid(row=2, column=1, sticky="e", pady=5)
        
        # 곡률
        ttk.Label(status_group, text="곡률반경:", style="Status.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        self.status_labels["curvature"] = ttk.Label(status_group, text="0.0m", style="Value.TLabel", width=10)
        self.status_labels["curvature"].grid(row=3, column=1, sticky="e", pady=5)

        status_group.columnconfigure(1, weight=1)
        
        # 2. 감지 상태
        detect_group = ttk.LabelFrame(parent, text="감지 정보", padding=10)
        detect_group.pack(fill="x", pady=5)
        
        # 차선 감지 여부
        ttk.Label(detect_group, text="차선 감지:", style="Status.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.status_labels["detection"] = ttk.Label(detect_group, text="대기중", style="Value.TLabel", width=15)
        self.status_labels["detection"].grid(row=0, column=1, sticky="e", pady=5)
        
        # 경고 메시지
        ttk.Label(detect_group, text="시스템 경고:", style="Status.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.status_labels["warning"] = ttk.Label(detect_group, text="정상", style="Value.TLabel", width=15, foreground="green")
        self.status_labels["warning"].grid(row=1, column=1, sticky="e", pady=5)
        
        detect_group.columnconfigure(1, weight=1)

    def _create_tuning_tab(self, parent):
        """튜닝 탭 생성 (스크롤 가능)"""
        # Canvas & Scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 마우스 휠 스크롤
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 1. PID 제어
        pid_group = ttk.LabelFrame(scroll_frame, text="PID 제어 (조향)", padding=5)
        pid_group.pack(fill="x", pady=5, padx=5)
        
        self.kp_scale = self._add_slider(pid_group, "Kp (비례)", 0.0, 100.0, 1.0, 
                        self.config.path_planning.pid_kp,
                        lambda v: self._update_param("path_planning", "pid_kp", float(v)))
        
        self.ki_scale = self._add_slider(pid_group, "Ki (적분)", 0.0, 10.0, 0.01,
                        self.config.path_planning.pid_ki,
                        lambda v: self._update_param("path_planning", "pid_ki", float(v)))
        
        self.kd_scale = self._add_slider(pid_group, "Kd (미분)", 0.0, 50.0, 0.5,
                        self.config.path_planning.pid_kd,
                        lambda v: self._update_param("path_planning", "pid_kd", float(v)))

        # 2. 차선 검출 (Threshold)
        lane_group = ttk.LabelFrame(scroll_frame, text="차선 검출 (Threshold)", padding=5)
        lane_group.pack(fill="x", pady=5, padx=5)
        
        self.white_thresh_scale = self._add_slider(lane_group, "White Thresh", 50, 255, 1,
                        self.config.lane_detection.white_threshold,
                        lambda v: self._update_param("lane_detection", "white_threshold", int(float(v))))
        
        self.gray_thresh_scale = self._add_slider(lane_group, "Gray Thresh", 50, 255, 1,
                        self.config.lane_detection.gray_threshold,
                        lambda v: self._update_param("lane_detection", "gray_threshold", int(float(v))))
        
        # 3. ROI 설정
        roi_group = ttk.LabelFrame(scroll_frame, text="ROI (관심 영역)", padding=5)
        roi_group.pack(fill="x", pady=5, padx=5)
        
        self.roi_top_scale = self._add_slider(roi_group, "Top Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_top_ratio,
                        lambda v: self._update_param("lane_detection", "roi_top_ratio", float(v)))
        
        self.roi_bottom_scale = self._add_slider(roi_group, "Bottom Ratio", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_bottom_ratio,
                        lambda v: self._update_param("lane_detection", "roi_bottom_ratio", float(v)))

        self.roi_left_scale = self._add_slider(roi_group, "Left Ratio", 0.0, 0.5, 0.01,
                        self.config.lane_detection.roi_left_ratio,
                        lambda v: self._update_param("lane_detection", "roi_left_ratio", float(v)))

        self.roi_right_scale = self._add_slider(roi_group, "Right Ratio", 0.5, 1.0, 0.01,
                        self.config.lane_detection.roi_right_ratio,
                        lambda v: self._update_param("lane_detection", "roi_right_ratio", float(v)))

        self.roi_trap_width_scale = self._add_slider(roi_group, "Trap Top Width", 0.0, 1.0, 0.01,
                        self.config.lane_detection.roi_trapezoid_top_width_ratio,
                        lambda v: self._update_param("lane_detection", "roi_trapezoid_top_width_ratio", float(v)))

        self.roi_mask_top_scale = self._add_slider(roi_group, "BEV Mask Top", 0.0, 0.9, 0.05,
                        self.config.lane_detection.roi_mask_top_ratio,
                        lambda v: self._update_param("lane_detection", "roi_mask_top_ratio", float(v)))

        self.roi_mask_side_scale = self._add_slider(roi_group, "BEV Mask Side", 0, 200, 10,
                        self.config.lane_detection.roi_mask_side_margin,
                        lambda v: self._update_param("lane_detection", "roi_mask_side_margin", int(float(v))))

        # 4. Sliding Window 설정
        sw_group = ttk.LabelFrame(scroll_frame, text="Sliding Window", padding=5)
        sw_group.pack(fill="x", pady=5, padx=5)

        self.n_windows_scale = self._add_slider(sw_group, "Windows (개수)", 1, 30, 1,
                        self.config.sliding_window.n_windows,
                        lambda v: self._update_param("sliding_window", "n_windows", int(float(v))))

        self.margin_scale = self._add_slider(sw_group, "Margin (폭)", 10, 300, 5,
                        self.config.sliding_window.margin,
                        lambda v: self._update_param("sliding_window", "margin", int(float(v))))

        self.min_pixels_scale = self._add_slider(sw_group, "Min Pixels", 10, 200, 5,
                        self.config.sliding_window.min_pixels,
                        lambda v: self._update_param("sliding_window", "min_pixels", int(float(v))))
        
        self.hist_ratio_scale = self._add_slider(sw_group, "Hist Start Ratio", 0.0, 0.8, 0.05,
                        self.config.sliding_window.histogram_start_ratio,
                        lambda v: self._update_param("sliding_window", "histogram_start_ratio", float(v)))

        # 5. Advanced Filtering (Morphology & Pattern)
        adv_group = ttk.LabelFrame(scroll_frame, text="고급 필터링 (Morphology & Pattern)", padding=5)
        adv_group.pack(fill="x", pady=5, padx=5)

        self.morph_open_scale = self._add_slider(adv_group, "Morph Open (Noise)", 3, 21, 2,
                        self.config.lane_detection.morph_kernel_open,
                        lambda v: self._update_param("lane_detection", "morph_kernel_open", int(float(v))))

        self.morph_close_scale = self._add_slider(adv_group, "Morph Close (Fill)", 3, 21, 2,
                        self.config.lane_detection.morph_kernel_close,
                        lambda v: self._update_param("lane_detection", "morph_kernel_close", int(float(v))))

        self.blob_min_w_scale = self._add_slider(adv_group, "Blob Min Width", 5, 100, 5,
                        self.config.lane_detection.blob_min_width,
                        lambda v: self._update_param("lane_detection", "blob_min_width", int(float(v))))

        self.blob_max_w_scale = self._add_slider(adv_group, "Blob Max Width", 50, 300, 10,
                        self.config.lane_detection.blob_max_width,
                        lambda v: self._update_param("lane_detection", "blob_max_width", int(float(v))))

        self.pat_white_scale = self._add_slider(adv_group, "Pattern Min White", 1, 50, 1,
                        self.config.lane_detection.pattern_min_white_len,
                        lambda v: self._update_param("lane_detection", "pattern_min_white_len", int(float(v))))

        self.pat_black_scale = self._add_slider(adv_group, "Pattern Min Black", 1, 50, 1,
                        self.config.lane_detection.pattern_min_black_len,
                        lambda v: self._update_param("lane_detection", "pattern_min_black_len", int(float(v))))

        self.pat_seg_scale = self._add_slider(adv_group, "Pattern Min Segs", 1, 10, 1,
                        self.config.lane_detection.pattern_min_segments,
                        lambda v: self._update_param("lane_detection", "pattern_min_segments", int(float(v))))

        self.morph_iter_scale = self._add_slider(adv_group, "Morph Iterations", 1, 5, 1,
                        self.config.lane_detection.morph_iterations,
                        lambda v: self._update_param("lane_detection", "morph_iterations", int(float(v))))

        self.blob_ar_scale = self._add_slider(adv_group, "Blob Min AR", 0.1, 5.0, 0.1,
                        self.config.lane_detection.blob_min_aspect_ratio,
                        lambda v: self._update_param("lane_detection", "blob_min_aspect_ratio", float(v)))

        self.roi_mask_bottom_scale = self._add_slider(roi_group, "BEV Mask Bottom", 0.0, 1.0, 0.05,
                        self.config.lane_detection.roi_mask_bottom_ratio,
                        lambda v: self._update_param("lane_detection", "roi_mask_bottom_ratio", float(v)))

        self.sw_ystart_scale = self._add_slider(sw_group, "Search Y Start", 0.0, 1.0, 0.05,
                        self.config.sliding_window.search_y_start_ratio,
                        lambda v: self._update_param("sliding_window", "search_y_start_ratio", float(v)))

        self.sw_yend_scale = self._add_slider(sw_group, "Search Y End", 0.0, 1.0, 0.05,
                        self.config.sliding_window.search_y_end_ratio,
                        lambda v: self._update_param("sliding_window", "search_y_end_ratio", float(v)))

    def _create_system_tab(self, parent):
        """시스템 탭 생성"""
        # 1. 디버그 윈도우 제어
        debug_group = ttk.LabelFrame(parent, text="🐞 디버그 도구", padding=10)
        debug_group.pack(fill="x", pady=5)
        
        # 디버그 윈도우 토글 버튼 (Checkbutton)
        self.debug_var = tk.BooleanVar(value=False)
        
        def toggle_debug():
            if self.debug_var.get():
                self.setup_debug_windows()
                self.debug_btn.config(text="디버그 창 닫기 (ON)", style="Action.TButton")
            else:
                self.close_debug_windows()
                self.debug_btn.config(text="디버그 창 열기 (OFF)")
        
        self.debug_btn = ttk.Checkbutton(
            debug_group, 
            text="디버그 창 열기 (OFF)", 
            variable=self.debug_var,
            style="TButton",
            command=toggle_debug
        )
        self.debug_btn.pack(fill="x", pady=5)
        
        ttk.Label(debug_group, text="※ 5개의 상세 분석 창이 열립니다.", foreground="gray").pack(anchor="w")
        
        # 2. 카메라 제어
        cam_group = ttk.LabelFrame(parent, text="📷 카메라 설정", padding=10)
        cam_group.pack(fill="x", pady=5)
        
        self.exposure_scale = self._add_slider(cam_group, "노출 (Exposure)", -13, 0, 1,
                        self.config.camera.exposure,
                        lambda v: self._update_param("camera", "exposure", int(float(v))))
        
        # 3. 초기화
        reset_group = ttk.LabelFrame(parent, text="초기화", padding=10)
        reset_group.pack(fill="x", pady=5)
        
        ttk.Button(reset_group, text="모든 파라미터 기본값으로 리셋", command=self.reset_parameters).pack(fill="x")

    def _add_slider(self, parent, label_text, from_, to, resolution, init_val, command):
        """슬라이더 추가 헬퍼"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=label_text, width=15).pack(side="left")
        
        scale = Scale(frame, from_=from_, to=to, resolution=resolution, orient=HORIZONTAL, command=command)
        scale.set(init_val)
        scale.pack(side="right", fill="x", expand=True)
        return scale

    # =========================================================================
    # 기능 메서드
    # =========================================================================
    
    def setup_debug_windows(self):
        """디버그 시각화 창 생성"""
        if self.debug_windows_enabled:
            return
            
        self.debug_windows_enabled = True
        self.debug_var.set(True)
        
        # 5개 디버그 창 생성
        cv2.namedWindow("1. Combined Binary", cv2.WINDOW_NORMAL)
        cv2.namedWindow("2. Binary Warped (BEV)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3. Histogram", cv2.WINDOW_NORMAL)
        cv2.namedWindow("4. Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("5. Color Masks", cv2.WINDOW_NORMAL)
        cv2.namedWindow("6. BWB Pattern", cv2.WINDOW_NORMAL)  # [Task] BWB 시각화 추가
        
        # 창 크기 및 위치 설정 (화면 하단에 배치)
        screen_h = self.root.winfo_screenheight()
        win_w, win_h = 350, 250  # 크기 약간 축소
        y_pos = screen_h - win_h - 50
        
        cv2.resizeWindow("1. Combined Binary", win_w, win_h)
        cv2.moveWindow("1. Combined Binary", 0, y_pos)
        
        cv2.resizeWindow("2. Binary Warped (BEV)", win_w, win_h)
        cv2.moveWindow("2. Binary Warped (BEV)", win_w, y_pos)
        
        cv2.resizeWindow("3. Histogram", win_w, 200)
        cv2.moveWindow("3. Histogram", win_w*2, y_pos)
        
        cv2.resizeWindow("4. Detection", win_w, win_h)
        cv2.moveWindow("4. Detection", win_w*3, y_pos)
        
        cv2.resizeWindow("5. Color Masks", win_w, win_h)
        cv2.moveWindow("5. Color Masks", win_w*4, y_pos)

        cv2.resizeWindow("6. BWB Pattern", win_w, win_h)
        cv2.moveWindow("6. BWB Pattern", win_w*5, y_pos)
        
        print("[GUI] 디버그 창 6개 생성 완료")

    def close_debug_windows(self):
        """디버그 창 닫기"""
        self.debug_windows_enabled = False
        self.debug_var.set(False)
        
        try:
            cv2.destroyWindow("1. Combined Binary")
            cv2.destroyWindow("2. Binary Warped (BEV)")
            cv2.destroyWindow("3. Histogram")
            cv2.destroyWindow("4. Detection")
            cv2.destroyWindow("5. Color Masks")
            cv2.destroyWindow("6. BWB Pattern")
        except:
            pass
        print("[GUI] 디버그 창 닫힘")

    def update_debug_windows(self, result: dict):
        """디버그 정보 시각화 업데이트"""
        if not self.debug_windows_enabled:
            return
        
        try:
            # 1. Combined Binary
            if 'preprocess_debug' in result and 'combined_binary' in result['preprocess_debug']:
                cv2.imshow("1. Combined Binary", result['preprocess_debug']['combined_binary'])
            
            # 2. Binary Warped
            if 'binary_warped' in result and result['binary_warped'] is not None:
                cv2.imshow("2. Binary Warped (BEV)", result['binary_warped'])
            
            # 3. Histogram
            if 'binary_warped' in result and result['binary_warped'] is not None:
                binary_warped = result['binary_warped']
                histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
                hist_img = self._draw_histogram_graph(histogram)
                cv2.imshow("3. Histogram", hist_img)
            
            # 4. Detection (sliding window visualization)
            if 'out_img' in result and result['out_img'] is not None:
                cv2.imshow("4. Detection", result['out_img'])
            
            # 5. Color Masks (white + yellow combined)
            if 'preprocess_debug' in result:
                debug = result['preprocess_debug']
                white_mask = debug.get('white_combined')
                yellow_mask = debug.get('yellow_combined')
                
                if white_mask is not None and yellow_mask is not None:
                    # 컬러 시각화: 흰색=흰색, 노란색=노란색
                    h, w = white_mask.shape
                    color_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    color_vis[white_mask > 0] = [255, 255, 255]  # 흰색
                    color_vis[yellow_mask > 0] = [0, 255, 255]   # 노란색 (BGR)
                    cv2.imshow("5. Color Masks", color_vis)
            
            # 6. BWB Pattern
            if 'preprocess_debug' in result and 'bwb_mask' in result['preprocess_debug']:
                cv2.imshow("6. BWB Pattern", result['preprocess_debug']['bwb_mask'])

        except Exception as e:
            print(f"[WARN] 디버그 창 업데이트 오류: {e}")

    def _draw_histogram_graph(self, histogram: np.ndarray) -> np.ndarray:
        """히스토그램을 그래프 이미지로 변환"""
        h, w = 200, len(histogram)
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if histogram.max() > 0:
            normalized = (histogram / histogram.max() * (h-1)).astype(np.int32)
            # 벡터화된 연산 대신 루프 사용 (안전성)
            for x, val in enumerate(normalized):
                cv2.line(hist_img, (x, h-1), (x, h-1 - val), (255, 255, 255), 1)
            
            # 중심선
            midpoint = w // 2
            cv2.line(hist_img, (midpoint, 0), (midpoint, h-1), (0, 0, 255), 1)
        
        return hist_img

    def update_video(self, frame: np.ndarray):
        """비디오 프레임 업데이트"""
        # BGR → RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 크기 조정 (GUI에 맞게)
        h, w = rgb_frame.shape[:2]
        target_w = 640
        target_h = int(h * (target_w / w))
        rgb_frame = cv2.resize(rgb_frame, (target_w, target_h))
        
        # PIL Image로 변환
        pil_image = Image.fromarray(rgb_frame)
        
        # Tkinter PhotoImage로 변환
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # 라벨 업데이트
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def update_status(self, fps: float, lane_result: dict, path_result: dict):
        """상태 정보 업데이트"""
        # FPS
        self.status_labels["fps"].config(text=f"{fps:.1f}")
        
        # 검출 상태
        if lane_result['detected']:
            self.status_labels["detection"].config(text="성공 ✓", foreground="green")
        else:
            self.status_labels["detection"].config(text="실패 ✗", foreground="red")
        
        # 경로 정보
        if path_result['valid']:
            offset = path_result['center_offset']
            self.status_labels["offset"].config(text=f"{offset:.3f} m")
            
            steering = path_result['steering_angle']
            self.status_labels["steering"].config(text=f"{steering:.1f}°")
            
            left_curv = path_result['left_curvature']
            right_curv = path_result['right_curvature']
            avg_curv = (left_curv + right_curv) / 2
            self.status_labels["curvature"].config(text=f"{avg_curv:.1f} m")
            
            if path_result['lane_departure_warning']:
                self.status_labels["warning"].config(text="이탈 위험!", foreground="red")
            else:
                self.status_labels["warning"].config(text="정상", foreground="green")
        else:
            self.status_labels["offset"].config(text="N/A")
            self.status_labels["steering"].config(text="N/A")
            self.status_labels["curvature"].config(text="N/A")
            self.status_labels["warning"].config(text="N/A", foreground="gray")

    def _update_param(self, section: str, param: str, value):
        """파라미터 업데이트"""
        success = update_config(section, param, value)
        if success and self.update_callback:
            self.update_callback(section, param, value)

    def toggle_record(self):
        """녹화 토글"""
        if self.record_callback:
            is_recording = self.record_callback()
            if is_recording:
                self.record_button.config(text="⏹ 녹화 중지")
            else:
                self.record_button.config(text="⏺ 녹화 시작")

    def toggle_pause(self):
        """일시정지 토글"""
        self.paused = not self.paused
        self.pause_button.config(text="▶ 재개" if self.paused else "⏯ 일시정지")

    def reset_parameters(self):
        """파라미터 리셋"""
        # Config 파일 다시 로드 또는 기본값 설정 (여기서는 생략, 필요시 구현)
        print("[INFO] 파라미터 리셋 요청됨")

    def on_closing(self):
        """종료 처리"""
        self.running = False
        self.close_debug_windows()
        self.root.quit()
        self.root.destroy()

    def update(self):
        """이벤트 루프"""
        if self.running:
            self.root.update_idletasks()
            self.root.update()
            cv2.waitKey(1)

    def _setup_keyboard_bindings(self):
        """키보드 단축키"""
        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('<d>', lambda e: self.setup_debug_windows())
