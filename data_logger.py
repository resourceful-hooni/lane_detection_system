"""
data_logger.py
데이터 로깅 및 영상 저장 모듈
- CSV 로깅 (프레임별 데이터)
- AVI 영상 저장 (오버레이)
- 파일명 자동 생성 (타임스탬프)
"""

import cv2
import csv
import os
from datetime import datetime
from typing import Optional, List
import numpy as np
from config import get_config


class DataLogger:
    """데이터 로깅 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = get_config()
        
        # 파일명 (타임스탬프 기반)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV 파일
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        
        # 비디오 파일
        self.video_writer = None
        self.video_path = None
        self.is_recording = False  # 녹화 상태 플래그
        
        # 로깅 시작 시간
        self.start_time = None
        
        # 통계
        self.frame_count = 0
        self.logged_frames = 0
        
        self._initialize_logging()
    
    def _initialize_logging(self):
        """로깅 초기화"""
        # CSV 로깅 초기화
        if self.config.logging.save_csv:
            self.csv_path = os.path.join(
                self.config.logging.log_dir,
                f"lane_data_{self.timestamp}.csv"
            )
            
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=self.config.logging.csv_columns
            )
            self.csv_writer.writeheader()
            
            print(f"[INFO] CSV 로그 생성: {self.csv_path}")
        
        # 비디오 로깅은 이제 수동으로 시작합니다.
        # if self.config.logging.save_video: ... (Removed)
        
        # 시작 시간 기록
        self.start_time = datetime.now()

    def start_recording(self):
        """비디오 녹화 시작"""
        if self.is_recording:
            return

        self.video_path = os.path.join(
            self.config.logging.video_dir,
            f"lane_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.config.logging.video_extension}"
        )
        
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*self.config.logging.video_codec)
        
        # VideoWriter 생성
        self.video_writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.config.logging.video_fps,
            (self.config.camera.width, self.config.camera.height)
        )
        
        self.is_recording = True
        print(f"[INFO] 비디오 녹화 시작: {self.video_path}")

    def stop_recording(self):
        """비디오 녹화 중지"""
        if not self.is_recording:
            return

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        print(f"[INFO] 비디오 녹화 중지 (저장됨): {self.video_path}")

    def toggle_recording(self) -> bool:
        """녹화 상태 토글 (Returns: 현재 녹화 중 여부)"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
        return self.is_recording
    
    def log_frame_data(
        self,
        frame_number: int,
        lane_result: dict,
        path_result: dict,
        fps: float
    ):
        """
        프레임 데이터 로깅 (CSV)
        
        Args:
            frame_number: 프레임 번호
            lane_result: 차선 검출 결과
            path_result: 경로 계획 결과
            fps: 현재 FPS
        """
        if not self.config.logging.save_csv or self.csv_writer is None:
            return
        
        # 타임스탬프 계산 (초 단위)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # 데이터 딕셔너리 생성
        pixel_stats = lane_result.get('pixel_stats', {})
        
        data = {
            "timestamp": f"{elapsed:.3f}",
            "frame_number": frame_number,
            "left_curve_rad": f"{path_result.get('left_curvature', 0):.2f}" if path_result['valid'] else "N/A",
            "right_curve_rad": f"{path_result.get('right_curvature', 0):.2f}" if path_result['valid'] else "N/A",
            "center_offset_m": f"{path_result.get('center_offset', 0):.4f}" if path_result['valid'] else "N/A",
            "steering_angle_deg": f"{path_result.get('steering_angle', 0):.2f}" if path_result['valid'] else "N/A",
            "lane_detected": "True" if lane_result['detected'] else "False",
            "fps": f"{fps:.1f}",
            # 파라미터 로깅
            "white_threshold": self.config.lane_detection.white_threshold,
            "canny_low": self.config.lane_detection.canny_low_threshold,
            "canny_high": self.config.lane_detection.canny_high_threshold,
            "roi_top": f"{self.config.lane_detection.roi_top_ratio:.2f}",
            "roi_bottom": f"{self.config.lane_detection.roi_bottom_ratio:.2f}",
            # 픽셀 분포 통계
            "pixel_min_y": pixel_stats.get('min_y', 0),
            "pixel_max_y": pixel_stats.get('max_y', 0),
            "pixel_mean_y": f"{pixel_stats.get('mean_y', 0):.1f}"
        }
        
        # CSV 쓰기
        self.csv_writer.writerow(data)
        self.logged_frames += 1
    
    def log_video_frame(self, frame: np.ndarray):
        """
        비디오 프레임 저장
        
        Args:
            frame: 저장할 프레임 (오버레이 포함)
        """
        if not self.is_recording or self.video_writer is None:
            return
        
        # 프레임 쓰기
        self.video_writer.write(frame)
        self.frame_count += 1
    
    def close(self):
        """로깅 종료 및 파일 닫기"""
        # CSV 파일 닫기
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[INFO] CSV 로그 저장 완료: {self.logged_frames} 프레임")
            print(f"       파일: {self.csv_path}")
        
        # 비디오 파일 닫기
        self.stop_recording()
    
    def __del__(self):
        """소멸자 - 자동으로 파일 닫기"""
        self.close()


# 테스트 코드
if __name__ == "__main__":
    print("DataLogger 모듈 로드 완료")
    
    # 테스트
    logger = DataLogger()
    print(f"CSV 경로: {logger.csv_path}")
    print(f"비디오 경로: {logger.video_path}")
    logger.close()
