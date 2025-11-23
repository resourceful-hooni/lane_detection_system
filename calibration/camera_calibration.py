"""
calibration/camera_calibration.py
카메라 캘리브레이션 모듈
- 체스보드 패턴을 사용한 카메라 내부 파라미터 계산
- 왜곡 보정 계수 계산
- 원근 변환 포인트 설정 도구
"""

import cv2
import numpy as np
import glob
import os
import pickle
from typing import Tuple, Optional, List


class CameraCalibration:
    """카메라 캘리브레이션 클래스"""
    
    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (9, 6),  # 내부 코너 수 (가로, 세로)
        square_size: float = 25.0  # 체스보드 정사각형 크기 (mm)
    ):
        """
        초기화
        
        Args:
            chessboard_size: 체스보드 내부 코너 개수 (가로, 세로)
            square_size: 한 칸의 실제 크기 (mm)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 캘리브레이션 결과
        self.camera_matrix = None  # 카메라 내부 파라미터
        self.dist_coeffs = None    # 왜곡 계수
        self.calibrated = False
        
        # 3D 포인트 준비 (실제 월드 좌표)
        self.objp = self._prepare_object_points()
    
    def _prepare_object_points(self) -> np.ndarray:
        """
        체스보드 3D 좌표 생성
        
        Returns:
            객체 포인트 배열
        """
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self.chessboard_size[0],
            0:self.chessboard_size[1]
        ].T.reshape(-1, 2)
        
        # 실제 크기로 스케일링 (mm)
        objp *= self.square_size
        
        return objp
    
    def calibrate_from_images(
        self,
        image_folder: str,
        image_pattern: str = "*.jpg"
    ) -> bool:
        """
        이미지 폴더에서 체스보드 이미지를 읽어 캘리브레이션 수행
        
        Args:
            image_folder: 체스보드 이미지가 있는 폴더
            image_pattern: 이미지 파일 패턴
            
        Returns:
            성공 여부
        """
        # 이미지 파일 목록
        image_paths = glob.glob(os.path.join(image_folder, image_pattern))
        
        if len(image_paths) == 0:
            print(f"[ERROR] {image_folder}에서 이미지를 찾을 수 없습니다!")
            return False
        
        print(f"[INFO] {len(image_paths)}개의 이미지를 찾았습니다.")
        
        # 객체 포인트와 이미지 포인트 저장
        objpoints = []  # 3D 포인트 (실제 월드 좌표)
        imgpoints = []  # 2D 포인트 (이미지 좌표)
        
        image_size = None
        successful_images = 0
        
        for idx, image_path in enumerate(image_paths):
            # 이미지 읽기
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"[WARNING] 이미지를 읽을 수 없습니다: {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # 체스보드 코너 찾기
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.chessboard_size,
                None
            )
            
            if ret:
                # 서브픽셀 정확도로 코너 위치 개선
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )
                
                objpoints.append(self.objp)
                imgpoints.append(corners_refined)
                successful_images += 1
                
                print(f"[✓] {idx+1}/{len(image_paths)}: 코너 검출 성공 - {os.path.basename(image_path)}")
                
                # 시각화 (옵션)
                # cv2.drawChessboardCorners(img, self.chessboard_size, corners_refined, ret)
                # cv2.imshow('Chessboard', img)
                # cv2.waitKey(500)
            else:
                print(f"[✗] {idx+1}/{len(image_paths)}: 코너 검출 실패 - {os.path.basename(image_path)}")
        
        # cv2.destroyAllWindows()
        
        if successful_images < 3:
            print(f"[ERROR] 캘리브레이션을 위한 이미지가 부족합니다 (최소 3장 필요, 현재 {successful_images}장)")
            return False
        
        print(f"\n[INFO] {successful_images}개 이미지로 캘리브레이션 수행 중...")
        
        # 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.calibrated = True
            
            print("[✓] 캘리브레이션 성공!")
            print(f"\n카메라 매트릭스:\n{mtx}")
            print(f"\n왜곡 계수:\n{dist}")
            
            # 재투영 오차 계산
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    objpoints[i],
                    rvecs[i],
                    tvecs[i],
                    mtx,
                    dist
                )
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            mean_error /= len(objpoints)
            print(f"\n재투영 평균 오차: {mean_error:.4f} 픽셀")
            
            return True
        else:
            print("[ERROR] 캘리브레이션 실패!")
            return False
    
    def calibrate_from_camera(
        self,
        camera_index: int = 0,
        num_images: int = 20
    ) -> bool:
        """
        실시간 카메라에서 체스보드 이미지를 캡처하여 캘리브레이션
        
        Args:
            camera_index: 카메라 인덱스
            num_images: 캡처할 이미지 수
            
        Returns:
            성공 여부
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다!")
            return False
        
        objpoints = []
        imgpoints = []
        
        captured_count = 0
        image_size = None
        
        print(f"[INFO] {num_images}장의 체스보드 이미지를 캡처합니다.")
        print("[INFO] 스페이스바를 눌러 캡처, ESC를 눌러 종료")
        
        while captured_count < num_images:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다!")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # 체스보드 코너 찾기
            ret_corners, corners = cv2.findChessboardCorners(
                gray,
                self.chessboard_size,
                None
            )
            
            # 화면에 표시
            display_frame = frame.copy()
            
            if ret_corners:
                # 코너 그리기
                cv2.drawChessboardCorners(
                    display_frame,
                    self.chessboard_size,
                    corners,
                    ret_corners
                )
                
                # 상태 표시
                cv2.putText(
                    display_frame,
                    "Chessboard Detected! Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    "Chessboard NOT detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # 캡처 카운트 표시
            cv2.putText(
                display_frame,
                f"Captured: {captured_count}/{num_images}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 스페이스바: 캡처
            if key == ord(' ') and ret_corners:
                # 서브픽셀 정확도 개선
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )
                
                objpoints.append(self.objp)
                imgpoints.append(corners_refined)
                captured_count += 1
                
                print(f"[✓] 이미지 {captured_count}/{num_images} 캡처됨")
            
            # ESC: 종료
            elif key == 27:
                print("[INFO] 사용자에 의해 중단됨")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count < 3:
            print(f"[ERROR] 캘리브레이션을 위한 이미지가 부족합니다 (최소 3장 필요, 현재 {captured_count}장)")
            return False
        
        print(f"\n[INFO] {captured_count}개 이미지로 캘리브레이션 수행 중...")
        
        # 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )
        
        if ret:
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.calibrated = True
            
            print("[✓] 캘리브레이션 성공!")
            print(f"\n카메라 매트릭스:\n{mtx}")
            print(f"\n왜곡 계수:\n{dist}")
            
            return True
        else:
            print("[ERROR] 캘리브레이션 실패!")
            return False
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 왜곡 보정
        
        Args:
            image: 입력 이미지
            
        Returns:
            왜곡 보정된 이미지
        """
        if not self.calibrated:
            print("[WARNING] 캘리브레이션이 수행되지 않았습니다!")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, None)
    
    def save_calibration(self, filepath: str):
        """
        캘리브레이션 결과 저장
        
        Args:
            filepath: 저장할 파일 경로 (.pkl)
        """
        if not self.calibrated:
            print("[ERROR] 저장할 캘리브레이션 데이터가 없습니다!")
            return
        
        data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[✓] 캘리브레이션 데이터 저장됨: {filepath}")
    
    def load_calibration(self, filepath: str) -> bool:
        """
        캘리브레이션 결과 로드
        
        Args:
            filepath: 로드할 파일 경로 (.pkl)
            
        Returns:
            성공 여부
        """
        if not os.path.exists(filepath):
            print(f"[ERROR] 파일을 찾을 수 없습니다: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.chessboard_size = data.get('chessboard_size', self.chessboard_size)
            self.square_size = data.get('square_size', self.square_size)
            self.calibrated = True
            
            print(f"[✓] 캘리브레이션 데이터 로드됨: {filepath}")
            print(f"\n카메라 매트릭스:\n{self.camera_matrix}")
            print(f"\n왜곡 계수:\n{self.dist_coeffs}")
            
            return True
        
        except Exception as e:
            print(f"[ERROR] 파일 로드 실패: {e}")
            return False


class PerspectiveCalibration:
    """원근 변환 포인트 설정 도구"""
    
    def __init__(self):
        """초기화"""
        self.src_points = []
        self.image = None
        self.window_name = "Perspective Calibration"
    
    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_points) < 4:
                self.src_points.append([x, y])
                print(f"포인트 {len(self.src_points)}: ({x}, {y})")
                
                # 포인트 표시
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    self.image,
                    str(len(self.src_points)),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                
                # 선 그리기 (2개 이상 포인트가 있을 때)
                if len(self.src_points) > 1:
                    cv2.line(
                        self.image,
                        tuple(self.src_points[-2]),
                        tuple(self.src_points[-1]),
                        (0, 255, 0),
                        2
                    )
                
                # 4개 포인트가 모두 선택되면 닫기
                if len(self.src_points) == 4:
                    cv2.line(
                        self.image,
                        tuple(self.src_points[-1]),
                        tuple(self.src_points[0]),
                        (0, 255, 0),
                        2
                    )
                
                cv2.imshow(self.window_name, self.image)
    
    def select_points_from_camera(
        self,
        camera_index: int = 0
    ) -> Optional[np.ndarray]:
        """
        카메라에서 원근 변환 포인트 선택
        
        Args:
            camera_index: 카메라 인덱스
            
        Returns:
            선택된 4개의 소스 포인트 [좌상단, 우상단, 우하단, 좌하단]
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다!")
            return None
        
        print("[INFO] 스페이스바를 눌러 프레임 캡처")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다!")
                break
            
            cv2.putText(
                frame,
                "Press SPACE to capture frame",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Select Frame', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                self.image = frame.copy()
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyWindow('Select Frame')
        
        return self.select_points_from_image(self.image)
    
    def select_points_from_image(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        이미지에서 원근 변환 포인트 선택
        
        Args:
            image: 입력 이미지
            
        Returns:
            선택된 4개의 소스 포인트
        """
        self.image = image.copy()
        self.src_points = []
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n[INFO] 마우스 클릭으로 4개 포인트를 선택하세요:")
        print("  1. 좌상단 (차선 소실점 근처)")
        print("  2. 우상단 (차선 소실점 근처)")
        print("  3. 우하단 (이미지 하단)")
        print("  4. 좌하단 (이미지 하단)")
        print("[INFO] 완료 후 아무 키나 누르세요")
        
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(self.src_points) == 4:
            points = np.float32(self.src_points)
            print(f"\n[✓] 선택된 포인트:\n{points}")
            return points
        else:
            print(f"[ERROR] 4개 포인트가 필요합니다 (현재 {len(self.src_points)}개)")
            return None
    
    @staticmethod
    def calculate_dst_points(
        src_points: np.ndarray,
        image_width: int,
        image_height: int,
        margin: int = 300
    ) -> np.ndarray:
        """
        목적지 포인트 자동 계산 (Bird's Eye View)
        
        Args:
            src_points: 소스 포인트
            image_width: 이미지 너비
            image_height: 이미지 높이
            margin: 좌우 마진
            
        Returns:
            목적지 포인트
        """
        dst_points = np.float32([
            [margin, 0],                    # 좌상단
            [image_width - margin, 0],      # 우상단
            [image_width - margin, image_height],  # 우하단
            [margin, image_height]          # 좌하단
        ])
        
        return dst_points


# 테스트 및 실행
if __name__ == "__main__":
    print("="*60)
    print("카메라 캘리브레이션 도구")
    print("="*60)
    
    print("\n1. 카메라 내부 파라미터 캘리브레이션")
    print("2. 원근 변환 포인트 설정")
    print("3. 종료")
    
    choice = input("\n선택: ")
    
    if choice == "1":
        # 카메라 캘리브레이션
        calibrator = CameraCalibration(chessboard_size=(9, 6))
        
        print("\n캘리브레이션 방법 선택:")
        print("1. 실시간 카메라에서 캡처")
        print("2. 저장된 이미지 폴더에서 로드")
        
        method = input("선택: ")
        
        if method == "1":
            success = calibrator.calibrate_from_camera(camera_index=0, num_images=15)
        elif method == "2":
            folder = input("이미지 폴더 경로: ")
            success = calibrator.calibrate_from_images(folder)
        else:
            print("[ERROR] 잘못된 선택")
            exit()
        
        if success:
            save_path = "calibration/camera_calibration.pkl"
            os.makedirs("calibration", exist_ok=True)
            calibrator.save_calibration(save_path)
    
    elif choice == "2":
        # 원근 변환 포인트 설정
        perspective = PerspectiveCalibration()
        src_points = perspective.select_points_from_camera(camera_index=0)
        
        if src_points is not None:
            # config.py에 복사할 수 있도록 출력
            print("\n" + "="*60)
            print("config.py에 다음 코드를 복사하세요:")
            print("="*60)
            print(f"\nself.perspective_src_points = np.float32({src_points.tolist()})")
            
            # 목적지 포인트 계산 (예시)
            dst_points = PerspectiveCalibration.calculate_dst_points(
                src_points, 1280, 720, margin=300
            )
            print(f"\nself.perspective_dst_points = np.float32({dst_points.tolist()})")
            print("\n" + "="*60)
    
    else:
        print("종료합니다.")
