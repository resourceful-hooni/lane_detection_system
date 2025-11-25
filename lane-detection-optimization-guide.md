# 차선 인식 시스템 최적화 가이드
**LattePanda 환경에서 실시간 차선 인식 성능 개선**

---

## 목차
1. [현재 문제점 진단](#1-현재-문제점-진단)
2. [LattePanda 최적화 전략](#2-lattepanda-최적화-전략)
3. [차선 인식 정확도 개선 방법](#3-차선-인식-정확도-개선-방법)
4. [구체적 구현 방법](#4-구체적-구현-방법)
5. [테스트 및 튜닝 가이드](#5-테스트-및-튜닝-가이드)

---

## 1. 현재 문제점 진단

### 1.1 Binary Mask에 차선 픽셀이 거의 안 남는 문제
**원인:**
- White/Black Threshold가 너무 높아 실제 차선(밝기 150~180) 픽셀이 조건(200+) 미달
- 단일 threshold만 사용해 조명/그림자/오염에 취약
- Canny Edge 기준이 너무 강해 희미한 차선 경계 검출 실패

### 1.2 성능 문제 (LattePanda)
- 고해상도(848x480) + 복잡한 연산으로 FPS 저하
- 실시간 처리 요구사항 충족 어려움

### 1.3 환경 적응성 부족
- 멀리/가까이 차선에 동일 파라미터 적용으로 검출 불안정
- 조명 변화, 바닥 오염, 반사광에 민감

---

## 2. LattePanda 최적화 전략

### 2.1 해상도 최적화
```python
# config.py 수정
class CameraConfig:
    # 해상도 낮춤: 640x360 또는 512x288
    width: int = 640    # 848 → 640
    height: int = 360   # 480 → 360
    fps: int = 30       # 60 → 30 (안정성 우선)
```

**효과:** 픽셀 수 30% 감소 → 연산량 대폭 감소

### 2.2 ROI 영역 제한 강화
```python
# 실제 차선이 존재하는 영역만 처리
roi_top_ratio: float = 0.25    # 상위 25%부터
roi_bottom_ratio: float = 0.9  # 하위 10% 제외(본네트)
```

### 2.3 불필요한 연산 제거
```python
# lane_detector.py 최적화 포인트

# 1. 프레임 스킵 (매 프레임 처리 대신 2프레임마다)
if frame_count % 2 == 0:
    # 차선 검출 수행
    pass
else:
    # 이전 결과 재사용
    pass

# 2. Gaussian Blur 커널 축소
gaussian_kernel_size: int = 3  # 5 → 3

# 3. Sliding Window 개수 축소
n_windows: int = 9  # 12 → 9
```

### 2.4 NumPy/OpenCV 최적화
```python
# 벡터화 연산 활용
# BAD: 반복문
for i in range(height):
    for j in range(width):
        if pixel[i, j] > threshold:
            mask[i, j] = 255

# GOOD: 벡터 연산
mask = np.where(image > threshold, 255, 0).astype(np.uint8)
```

---

## 3. 차선 인식 정확도 개선 방법

### 3.1 다중 색상 공간 활용 (CRITICAL!)

**현재 문제:** BGR만 사용 → 조명/그림자에 취약

**개선 방법:**
```python
def extract_lane_pixels_advanced(self, frame):
    """
    HLS, HSV, LAB 등 다중 색상 공간 조합으로 차선 추출
    """
    # 1. HLS 색상 공간 (조명 변화에 강함)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]  # Lightness
    s_channel = hls[:, :, 2]  # Saturation
    
    # 2. 흰색 차선: Lightness 높음 + Saturation 낮음
    white_mask = cv2.inRange(hls, 
                             np.array([0, 200, 0]),    # 하한
                             np.array([255, 255, 50]))  # 상한
    
    # 3. HSV 색상 공간 (노란색 차선용)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv,
                              np.array([20, 100, 100]),
                              np.array([30, 255, 255]))
    
    # 4. LAB 색상 공간 (밝기 독립적)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel_lab = lab[:, :, 0]
    white_mask_lab = cv2.inRange(l_channel_lab, 215, 255)
    
    # 5. 모든 마스크 결합
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_mask_lab)
    
    return combined_mask
```

### 3.2 Adaptive Thresholding (환경 적응)

**현재 문제:** 고정 threshold → 조명 변화 시 실패

**개선 방법:**
```python
def adaptive_threshold_lane(self, gray_image):
    """
    지역별 밝기에 따라 동적으로 threshold 조정
    """
    # Adaptive Gaussian Threshold
    adaptive = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,  # 지역 크기
        C=2            # 상수 보정값
    )
    
    # OTSU 자동 threshold (전역)
    _, otsu = cv2.threshold(
        gray_image, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 두 방법 결합
    combined = cv2.bitwise_or(adaptive, otsu)
    return combined
```

### 3.3 Morphological Operations (노이즈 제거)

**현재 문제:** 잡음, 끊김, 얼룩이 많음

**개선 방법:**
```python
def enhance_lane_mask(self, binary_mask):
    """
    형태학적 연산으로 차선 마스크 강화
    """
    # 1. Closing: 끊어진 차선 연결
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Opening: 작은 노이즈 제거
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 3. Dilation: 차선 두껍게 (검출 용이)
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    return dilated
```

### 3.4 Edge Detection 강화

**현재 문제:** Canny만 사용 → 약한 경계 놓침

**개선 방법:**
```python
def multi_edge_detection(self, gray_image):
    """
    여러 Edge 검출 방법 조합
    """
    # 1. Canny (기존)
    canny = cv2.Canny(gray_image, 30, 100)
    
    # 2. Sobel (수직/수평 경계)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.hypot(sobelx, sobely)
    sobel_normalized = np.uint8(sobel_combined / sobel_combined.max() * 255)
    _, sobel_binary = cv2.threshold(sobel_normalized, 50, 255, cv2.THRESH_BINARY)
    
    # 3. Laplacian (전방향 경계)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_normalized = np.uint8(np.absolute(laplacian))
    _, laplacian_binary = cv2.threshold(laplacian_normalized, 30, 255, cv2.THRESH_BINARY)
    
    # 4. 결합
    combined_edges = cv2.bitwise_or(canny, sobel_binary)
    combined_edges = cv2.bitwise_or(combined_edges, laplacian_binary)
    
    return combined_edges
```

### 3.5 Histogram Equalization (밝기 균일화)

```python
def preprocess_image_advanced(self, frame):
    """
    영상 전처리 강화
    """
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    enhanced_lab = cv2.merge([l_clahe, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr
```

---

## 4. 구체적 구현 방법

### 4.1 새로운 Lane Detector 구조

```python
# lane_detector_v2.py

class AdvancedLaneDetector:
    def __init__(self, config):
        self.config = config
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frame_buffer = []  # 프레임 평활화용
        
    def detect_lanes(self, frame):
        """
        메인 차선 검출 파이프라인
        """
        # Step 1: 전처리 (CLAHE)
        enhanced = self.preprocess_image_advanced(frame)
        
        # Step 2: 다중 색상 공간 마스크
        color_mask = self.extract_lane_pixels_advanced(enhanced)
        
        # Step 3: Adaptive Threshold
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        adaptive_mask = self.adaptive_threshold_lane(gray)
        
        # Step 4: Edge Detection
        edge_mask = self.multi_edge_detection(gray)
        
        # Step 5: 마스크 결합
        combined_mask = cv2.bitwise_or(color_mask, adaptive_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edge_mask)
        
        # Step 6: Morphology 강화
        enhanced_mask = self.enhance_lane_mask(combined_mask)
        
        # Step 7: ROI 적용
        roi_mask = self.apply_roi(enhanced_mask)
        
        # Step 8: Perspective Transform
        warped = self.perspective_transform(roi_mask)
        
        # Step 9: Sliding Window (기존)
        left_fit, right_fit = self.sliding_window_search(warped)
        
        # Step 10: 프레임 평활화 (이전 프레임과 평균)
        left_fit, right_fit = self.smooth_lane_fits(left_fit, right_fit)
        
        return left_fit, right_fit, enhanced_mask
    
    def smooth_lane_fits(self, left_fit, right_fit):
        """
        이전 프레임과 결합해 부드러운 차선 추적
        """
        alpha = 0.3  # 현재 프레임 가중치
        
        if self.prev_left_fit is not None:
            left_fit = alpha * left_fit + (1 - alpha) * self.prev_left_fit
            right_fit = alpha * right_fit + (1 - alpha) * self.prev_right_fit
        
        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit
        
        return left_fit, right_fit
```

### 4.2 Config 파라미터 최적화

```python
# config_optimized.py

@dataclass
class LaneDetectionConfig:
    # === 기본 파라미터 (관대하게) ===
    white_threshold: int = 120       # 200 → 120
    black_threshold: int = 60        # 100 → 60
    
    # === Canny (완화) ===
    canny_low_threshold: int = 30    # 50 → 30
    canny_high_threshold: int = 90   # 100 → 90
    
    # === Sliding Window (넓고 관대하게) ===
    margin: int = 100                # 30 → 100
    min_pixels: int = 15             # 50 → 15
    
    # === 새로운 파라미터 추가 ===
    use_multi_colorspace: bool = True       # HLS/HSV/LAB 사용
    use_adaptive_threshold: bool = True     # Adaptive Threshold
    use_morphology: bool = True             # 형태학 연산
    use_clahe: bool = True                  # 밝기 균일화
    use_multi_edge: bool = True             # 다중 Edge 검출
    use_frame_smoothing: bool = True        # 프레임 평활화
    
    # === CLAHE 설정 ===
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8
    
    # === Morphology 설정 ===
    morph_close_kernel: int = 5
    morph_open_kernel: int = 3
    morph_dilate_iter: int = 1
```

### 4.3 메인 루프 수정

```python
# main.py 수정

def main():
    # 새로운 Detector 사용
    from lane_detector_v2 import AdvancedLaneDetector
    
    detector = AdvancedLaneDetector(config)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 스킵 (성능 최적화)
        if frame_count % 2 == 0:
            left_fit, right_fit, debug_mask = detector.detect_lanes(frame)
            
            # 디버그 이미지 저장 (주기적으로)
            if frame_count % 30 == 0:
                cv2.imwrite(f'debug/mask_{frame_count}.png', debug_mask)
        
        frame_count += 1
        
        # ... 나머지 로직
```

---

## 5. 테스트 및 튜닝 가이드

### 5.1 디버그 이미지 확인 순서

1. **원본 영상** → 실제 차선 육안 확인
2. **CLAHE 전처리** → 밝기 균일화 효과
3. **Color Mask (HLS/HSV/LAB)** → 차선 색상 추출
4. **Adaptive Threshold** → 지역별 밝기 적응
5. **Edge Mask (Canny/Sobel/Laplacian)** → 경계 검출
6. **Combined Mask** → 모든 마스크 결합
7. **Morphology Enhanced** → 노이즈 제거/연결
8. **Final Binary** → 최종 차선 픽셀

### 5.2 파라미터 튜닝 전략

```python
# 실험적 튜닝 스크립트
def tune_parameters(video_path, output_dir):
    """
    다양한 파라미터 조합 자동 테스트
    """
    white_thresholds = [80, 100, 120, 140, 160]
    canny_lows = [20, 30, 40, 50]
    margins = [60, 80, 100, 120]
    
    for wt in white_thresholds:
        for cl in canny_lows:
            for mg in margins:
                config.white_threshold = wt
                config.canny_low_threshold = cl
                config.margin = mg
                
                # 테스트 실행
                result = test_detection(video_path, config)
                
                # 결과 저장
                save_result(result, f'{output_dir}/wt{wt}_cl{cl}_mg{mg}.png')
```

### 5.3 환경별 최적 설정 예시

**실내 환경 (조명 일정):**
```python
white_threshold = 120
black_threshold = 60
canny_low = 30
use_clahe = False  # 필요없음
```

**실외 환경 (햇빛 변화):**
```python
white_threshold = 100
black_threshold = 40
canny_low = 20
use_clahe = True   # 필수!
```

**멀리 있는 차선 중심:**
```python
white_threshold = 80
margin = 120
min_pixels = 10
roi_top_ratio = 0.0  # 상단부터 검출
```

---

## 6. 성능 벤치마크

### LattePanda 예상 성능

| 최적화 단계 | FPS | CPU 사용률 | 차선 검출률 |
|------------|-----|-----------|------------|
| 기존 (848x480, 복잡 연산) | 5-8 | 90%+ | 60% |
| 해상도 축소 (640x360) | 12-15 | 70% | 60% |
| + 프레임 스킵 | 20-25 | 60% | 60% |
| + 연산 최적화 | 25-30 | 50% | 75% |
| + 다중 색상/Adaptive | 20-25 | 60% | **90%+** |

**목표:** 20 FPS 이상 + 85% 이상 차선 검출률

---

## 7. Copilot 실행 명령어

```bash
# 이 가이드를 Copilot에게 전달하여 코드 생성 요청

"위 Markdown 가이드에 따라 다음을 구현해줘:

1. lane_detector_v2.py 파일 생성
   - AdvancedLaneDetector 클래스 구현
   - 다중 색상 공간(HLS/HSV/LAB) 차선 추출
   - Adaptive Thresholding
   - 다중 Edge Detection (Canny/Sobel/Laplacian)
   - Morphological Operations
   - CLAHE 전처리
   - 프레임 평활화 (Temporal Smoothing)

2. config_optimized.py 파일 생성
   - 최적화된 파라미터 기본값
   - 새로운 파라미터 추가 (use_multi_colorspace 등)

3. main.py 수정
   - 프레임 스킵 로직 추가
   - 디버그 이미지 주기적 저장

4. tune_parameters.py 파일 생성
   - 자동 파라미터 튜닝 스크립트

모든 코드는 LattePanda에서 실시간(20+ FPS) 동작하도록 최적화하고,
기존 코드와 호환되도록 작성해줘."
```

---

## 8. 체크리스트

### 구현 완료 체크
- [ ] lane_detector_v2.py 생성
- [ ] 다중 색상 공간 구현
- [ ] Adaptive Threshold 구현
- [ ] Morphology 연산 구현
- [ ] CLAHE 전처리 구현
- [ ] 프레임 평활화 구현
- [ ] config_optimized.py 생성
- [ ] 프레임 스킵 로직 추가
- [ ] 디버그 이미지 저장 기능

### 테스트 완료 체크
- [ ] 실내 환경 테스트
- [ ] 다양한 조명 테스트
- [ ] 멀리/가까이 차선 테스트
- [ ] FPS 20+ 달성
- [ ] 차선 검출률 85%+ 달성
- [ ] 디버그 이미지 확인

---

## 9. 문제 해결 FAQ

**Q: 여전히 binary mask에 픽셀이 안 남아요**
A: 
1. use_multi_colorspace = True 확인
2. white_threshold를 80까지 낮춰보기
3. 디버그 이미지에서 Color Mask 단계 확인

**Q: FPS가 여전히 낮아요**
A:
1. 해상도를 512x288로 더 낮추기
2. 프레임 스킵을 3프레임마다로 변경
3. use_multi_edge = False로 설정

**Q: 차선이 흔들려요**
A:
1. use_frame_smoothing = True 확인
2. smooth_lane_fits의 alpha 값 낮추기 (0.2로)
3. 프레임 버퍼 크기 늘리기

---

## 결론

이 가이드를 따라 구현하면:
- **LattePanda에서 실시간 동작 (20+ FPS)**
- **다양한 환경에서 견고한 차선 인식 (85%+ 검출률)**
- **조명/그림자/오염에 강한 시스템**

핵심은 **단일 threshold → 다중 필터 조합**입니다!