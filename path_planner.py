"""
path_planner.py
경로 계획 및 조향 제어 모듈
- 차선 중앙 오프셋 계산
- PID 제어 기반 조향각 계산
- 차선 이탈 경고
"""

import numpy as np
from typing import Tuple
from config import get_config


class PathPlanner:
    """경로 계획 및 조향 제어 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = get_config()
        
        # PID 제어 변수
        self.previous_error = 0.0
        self.integral_error = 0.0
        
        # 차선 이탈 경고
        self.lane_departure_warning = False
        
    def calculate_center_offset(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        image_width: int,
        image_height: int,
        y_eval: float = None
    ) -> float:
        """
        차량의 차선 중앙으로부터 오프셋 계산
        
        Args:
            left_fit: 좌측 차선 다항식 계수
            right_fit: 우측 차선 다항식 계수
            image_width: 이미지 너비 (픽셀)
            image_height: 이미지 높이 (픽셀)
            y_eval: 평가할 y 좌표 (None이면 이미지 하단)
            
        Returns:
            center_offset: 중앙 오프셋 (미터, 양수=우측, 음수=좌측)
        """
        # 이미지 하단에서 차선 위치 계산
        if y_eval is None:
            y_eval = image_height - 1
        
        # 좌측/우측 차선의 x 좌표 계산
        left_x = left_fit[0] * (y_eval ** 2) + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * (y_eval ** 2) + right_fit[1] * y_eval + right_fit[2]
        
        # 차선 중앙 계산
        lane_center = (left_x + right_x) / 2
        
        # 차량 중앙 (이미지 중앙으로 가정)
        vehicle_center = image_width / 2
        
        # 오프셋 (픽셀)
        offset_pixels = vehicle_center - lane_center
        
        # 픽셀 → 미터 변환
        xm_per_pix = self.config.path_planning.xm_per_pix
        center_offset = offset_pixels * xm_per_pix
        
        return center_offset
    
    def calculate_steering_angle_pid(
        self,
        center_offset: float,
        dt: float = 0.033  # 약 30fps 기준
    ) -> float:
        """
        PID 제어로 조향각 계산
        
        Args:
            center_offset: 차선 중앙 오프셋 (m)
            dt: 시간 간격 (초)
            
        Returns:
            steering_angle: 조향각 (도, 양수=우회전, 음수=좌회전)
        """
        # PID 게인
        kp = self.config.path_planning.pid_kp
        ki = self.config.path_planning.pid_ki
        kd = self.config.path_planning.pid_kd
        
        # 오차 (음수=좌측, 양수=우측)
        error = -center_offset  # 오프셋과 반대 방향으로 조향
        
        # 비례 항 (P)
        p_term = kp * error
        
        # 적분 항 (I)
        self.integral_error += error * dt
        i_term = ki * self.integral_error
        
        # 미분 항 (D)
        derivative_error = (error - self.previous_error) / dt
        d_term = kd * derivative_error
        
        # PID 출력
        steering_angle = p_term + i_term + d_term
        
        # 이전 오차 저장
        self.previous_error = error
        
        # 조향각 제한
        max_angle = self.config.path_planning.max_steering_angle_deg
        steering_angle = np.clip(steering_angle, -max_angle, max_angle)
        
        return steering_angle
    
    def calculate_steering_angle_pure_pursuit(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        image_width: int,
        image_height: int,
        vehicle_speed_mps: float = 1.0  # 차량 속도 (m/s)
    ) -> float:
        """
        Pure Pursuit 알고리즘으로 조향각 계산
        (곡선 도로에 더 효과적)
        
        Args:
            left_fit: 좌측 차선 다항식 계수
            right_fit: 우측 차선 다항식 계수
            image_width: 이미지 너비 (픽셀)
            image_height: 이미지 높이 (픽셀)
            vehicle_speed_mps: 차량 속도 (m/s)
            
        Returns:
            steering_angle: 조향각 (도)
        """
        # Look-ahead distance (속도에 비례)
        lookahead = self.config.path_planning.lookahead_distance_m
        
        # Look-ahead distance를 픽셀로 변환
        ym_per_pix = self.config.path_planning.ym_per_pix
        lookahead_pixels = lookahead / ym_per_pix
        
        # Look-ahead 지점의 y 좌표
        y_lookahead = image_height - lookahead_pixels
        y_lookahead = max(0, y_lookahead)  # 범위 제한
        
        # 차선 중앙의 x 좌표 계산
        left_x = left_fit[0] * (y_lookahead ** 2) + left_fit[1] * y_lookahead + left_fit[2]
        right_x = right_fit[0] * (y_lookahead ** 2) + right_fit[1] * y_lookahead + right_fit[2]
        target_x = (left_x + right_x) / 2
        
        # 차량 위치 (이미지 하단 중앙)
        vehicle_x = image_width / 2
        
        # 횡방향 오차
        lateral_error_pixels = target_x - vehicle_x
        lateral_error = lateral_error_pixels * self.config.path_planning.xm_per_pix
        
        # 조향각 계산 (tan(θ) = lateral_error / lookahead)
        steering_angle_rad = np.arctan2(lateral_error, lookahead)
        steering_angle_deg = np.degrees(steering_angle_rad)
        
        # 조향각 제한
        max_angle = self.config.path_planning.max_steering_angle_deg
        steering_angle_deg = np.clip(steering_angle_deg, -max_angle, max_angle)
        
        return steering_angle_deg

    def calculate_lane_curvature(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        y_eval: float
    ) -> Tuple[float, float]:
        """좌우 차선 곡률 반지름 계산 (미터)."""
        ym_per_pix = self.config.path_planning.ym_per_pix
        xm_per_pix = self.config.path_planning.xm_per_pix
        
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
        
        y_eval_m = y_eval * ym_per_pix
        
        left_curverad = (
            (1 + (2 * left_fit_cr[0] * y_eval_m + left_fit_cr[1]) ** 2) ** 1.5
        ) / abs(2 * left_fit_cr[0])
        
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval_m + right_fit_cr[1]) ** 2) ** 1.5
        ) / abs(2 * right_fit_cr[0])
        
        return left_curverad, right_curverad
    
    def check_lane_departure(
        self,
        center_offset: float,
        threshold: float = 0.3  # 30cm
    ) -> bool:
        """
        차선 이탈 경고 체크
        
        Args:
            center_offset: 차선 중앙 오프셋 (m)
            threshold: 경고 임계값 (m)
            
        Returns:
            warning: True if 차선 이탈 위험
        """
        self.lane_departure_warning = abs(center_offset) > threshold
        return self.lane_departure_warning
    
    def plan_path(
        self,
        lane_result: dict,
        image_width: int,
        image_height: int,
        use_pure_pursuit: bool = False,
        fps: float = 30.0
    ) -> dict:
        """
        경로 계획 메인 함수
        
        Args:
            lane_result: 차선 검출 결과
            image_width: 이미지 너비
            image_height: 이미지 높이
            use_pure_pursuit: Pure Pursuit 사용 여부
            fps: 현재 프레임 속도 (dt 계산용)
            
        Returns:
            경로 계획 결과 딕셔너리
        """
        result = {
            'center_offset': 0.0,
            'steering_angle': 0.0,
            'left_curvature': 0.0,
            'right_curvature': 0.0,
            'lane_departure_warning': False,
            'valid': False
        }
        
        # 차선 검출 실패 시
        if not lane_result['detected']:
            return result
        
        left_fit = lane_result['left_fit']
        right_fit = lane_result['right_fit']
        
        # 1. 차선 중앙 오프셋 계산 (현재 위치 - 하단)
        center_offset = self.calculate_center_offset(
            left_fit,
            right_fit,
            image_width,
            image_height
        )
        
        # 2. 조향각 계산
        dt = 1.0 / max(fps, 1.0)
        if use_pure_pursuit:
            steering_angle = self.calculate_steering_angle_pure_pursuit(
                left_fit,
                right_fit,
                image_width,
                image_height
            )
        else:
            # PID 제어를 위한 Lookahead 오프셋 계산
            # 곡선 도로에서 미리 반응하기 위해 전방을 주시
            lookahead_m = self.config.path_planning.lookahead_distance_m
            ym_per_pix = self.config.path_planning.ym_per_pix
            lookahead_px = lookahead_m / ym_per_pix
            
            y_lookahead = max(0, image_height - lookahead_px)
            
            steering_offset = self.calculate_center_offset(
                left_fit,
                right_fit,
                image_width,
                image_height,
                y_eval=y_lookahead
            )
            
            steering_angle = self.calculate_steering_angle_pid(steering_offset, dt=dt)
        
        # 3. 곡률 계산
        left_curv, right_curv = self.calculate_lane_curvature(
            left_fit,
            right_fit,
            image_height - 1
        )
        
        # 4. 차선 이탈 경고
        lane_departure = self.check_lane_departure(center_offset)
        
        # 5. 결과 반환
        result = {
            'center_offset': center_offset,
            'steering_angle': steering_angle,
            'left_curvature': left_curv,
            'right_curvature': right_curv,
            'lane_departure_warning': lane_departure,
            'valid': True
        }
        
        return result
    
    def reset_pid(self):
        """PID 제어 변수 리셋"""
        self.previous_error = 0.0
        self.integral_error = 0.0


# 테스트 코드
if __name__ == "__main__":
    print("PathPlanner 모듈 로드 완료")
