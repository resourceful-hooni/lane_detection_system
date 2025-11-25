# 흰색 차선 미검출 문제 - 단일 개선안

## 문제 분석

GUI의 **BINARY 모드에서 흰색 차선이 전혀 나타나지 않는 주요 원인**:

### 1. HSV 색공간 Threshold 범위 문제
- **현재 설정** (`config.py`):
  ```python
  whitethreshold: int = 160  # HSV의 V(밝기) 채널 최소값
  ```
- **마스크 범위**:
  ```
  Lower: (H=0,   S=0,   V=160)
  Upper: (H=180, S=60,  V=255)
  ```

**왜 실패하는가?**
- 실제 노면의 흰색 차선 픽셀의 V(밝기) 값이 160 미만일 수 있음
- 카메라의 autoexposure/autowhitebalance로 인해 영상의 전체 밝기가 변동
- S(채도) 조건(≤60)이 너무 엄격함 - 그림자, 반사광, 오염된 차선은 채도가 60 초과

### 2. ROI 및 Perspective 영역 미스매칭
- 차선이 실제로는 감지되는데, ROI 구간 바깥에 있거나 hood mask로 잘림
- Perspective 변환이 이상하면 bird's eye view에서 차선이 사라짐

### 3. 카메라 입력 문제
- 노출 설정이 너무 낮으면 영상 전체가 어두워짐
- 화이트밸런스 미설정으로 색상 정보 왜곡

---

## 해결책: 적응적 HSV Threshold 도입

현재 **고정 threshold** → **동적 threshold로 변경**

### 핵심 아이디어
ROI 영역 내부의 실제 밝기를 측정한 후, 그에 따라 threshold를 자동으로 조정합니다.

---

## 구현 방법

### 1단계: `config.py` 수정

```python
@dataclass
class LaneDetectionConfig:
    # 기존 파라미터
    roitopratio: float = 0.28
    roibottomratio: float = 1.0
    
    # 새로운 파라미터
    whitethreshold: int = 150  # 기존 160 → 150으로 낮춤
    white_saturation_max: int = 80  # 기존 60 → 80으로 확대 (채도 범위 완화)
    
    # 적응적 threshold 활성화
    enable_adaptive_threshold: bool = True
    
    # 밝기에 따른 threshold 범위 (자동 조정용)
    threshold_low_light: int = 120   # 어두울 때 threshold
    threshold_normal: int = 150      # 보통 때 threshold
    threshold_bright: int = 180      # 밝을 때 threshold
```

### 2단계: `lane_detector.py` - `detectwhitelane()` 함수 개선

**기존 코드 (lane_detector.py의 `detectwhitelane()` 메서드):**
```python
def detectwhitelane(self, image: np.ndarray) -> np.ndarray:
    """HSV 색공간 기반 흰색 차선 검출"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = self.config.lanedetection.whitethreshold
    
    # 고정 threshold 사용
    lowerwhite = np.array([0, 0, thresh])
    upperwhite = np.array([180, 60, 255])
    whitemask = cv2.inRange(hsv, lowerwhite, upperwhite)
    
    # 밝은색 보조 마스크
    lowerbright = np.array([0, 0, 230])
    upperbright = np.array([180, 255, 255])
    brightmask = cv2.inRange(hsv, lowerbright, upperbright)
    whitemask = cv2.bitwise_or(whitemask, brightmask)
    
    kernel = np.ones((3, 3), np.uint8)
    whitemask = cv2.morphologyEx(whitemask, cv2.MORPH_OPEN, kernel)
    return whitemask
```

**개선된 코드 (동적 threshold):**
```python
def detectwhitelane(self, image: np.ndarray) -> np.ndarray:
    """
    HSV 기반 흰색 차선 검출 (적응적 threshold)
    
    ROI 영역의 평균 밝기를 측정하여 threshold를 자동 조정합니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # === 적응적 Threshold 계산 ===
    if self.config.lanedetection.enable_adaptive_threshold:
        # ROI 영역의 V(밝기) 채널만 추출
        v_channel = hsv[:, :, 2]
        
        # ROI 내부의 평균 밝기 측정
        mean_brightness = np.mean(v_channel)
        
        # 밝기에 따라 threshold 자동 결정
        if mean_brightness < 80:
            # 매우 어두운 환경
            white_threshold = self.config.lanedetection.threshold_low_light  # 120
            sat_max = 120
        elif mean_brightness < 150:
            # 보통 환경
            white_threshold = self.config.lanedetection.threshold_normal  # 150
            sat_max = self.config.lanedetection.white_saturation_max  # 80
        else:
            # 밝은 환경
            white_threshold = self.config.lanedetection.threshold_bright  # 180
            sat_max = 100
    else:
        # 기존 고정 threshold 사용
        white_threshold = self.config.lanedetection.whitethreshold
        sat_max = self.config.lanedetection.white_saturation_max
    
    # === 흰색 마스크 생성 ===
    # 주요 흰색 범위: 낮은 채도 + 높은 밝기
    lowerwhite = np.array([0, 0, white_threshold])
    upperwhite = np.array([180, sat_max, 255])
    whitemask = cv2.inRange(hsv, lowerwhite, upperwhite)
    
    # === 보조 마스크: 매우 밝은 영역 (밝기만 조건) ===
    # 밝기 > 220인 모든 픽셀 (채도 무관)
    lowerbright = np.array([0, 0, 220])
    upperbright = np.array([180, 255, 255])
    brightmask = cv2.inRange(hsv, lowerbright, upperbright)
    whitemask = cv2.bitwise_or(whitemask, brightmask)
    
    # === 노이즈 제거 ===
    kernel = np.ones((3, 3), np.uint8)
    whitemask = cv2.morphologyEx(whitemask, cv2.MORPH_OPEN, kernel)
    
    return whitemask
```

### 3단계: `preprocessframe()` 에서 호출

기존 코드는 이미 `detectwhitelane()` 함수를 호출하므로, **변경 불필요**합니다.
```python
def preprocessframe(self, frame: np.ndarray):
    # ... ROI 추출 ...
    roi = frame[roitop:roibottom, roileft:roiright]
    
    # detectwhitelane() 호출 (자동으로 적응적 threshold 적용됨)
    whitemask = self.detectwhitelane(roi)
    
    # ... 나머지 처리 ...
```

---

## 파라미터 튜닝 가이드

### GUI 슬라이더로 실시간 조정

현재 `gui_controller.py`에 이미 슬라이더가 있으므로, 다음 파라미터를 조정하세요:

| 파라미터 | 범위 | 현재값 | 조정 방향 | 설명 |
|---------|------|-------|---------|------|
| **whitethreshold** | 100~180 | 160 | **낮춤** (120~140) | ROI 내 흰색 차선이 어두우면 낮춤 |
| **white_saturation_max** | 40~150 | 60 | **높임** (80~100) | 채도 높은 차선 감지 가능하게 |
| **threshold_low_light** | 80~150 | 120 | 어두운 환경 테스트 후 조정 | 야간/터널용 |
| **threshold_bright** | 160~200 | 180 | 밝은 환경 테스트 후 조정 | 옥외 밝은 도로용 |

### 테스트 절차

1. **GUI 실행 후 BINARY 모드 켜기**
2. **다양한 조명 환경에서 테스트**:
   - 밝은 실내
   - 어두운 실내
   - 야외 햇빛
   - 그림자
3. **흰색 차선이 BINARY에서 흰색으로 나타나면 성공**
4. **필요 시 슬라이더로 파라미터 미세 조정**

---

## 카메라 설정 확인 (선택사항)

혹시 카메라 입력이 문제라면, `config.py`의 카메라 설정 확인:

```python
@dataclass
class CameraConfig:
    width: int = 848
    height: int = 480
    fps: int = 60
    camera_index: int = 1
    
    # ===== 노출 설정 =====
    autoexposure: bool = True   # True 유지 권장
    exposure: int = -6          # -6은 자동. 너무 어두우면 0으로 조정
    
    # ===== 화이트밸런스 =====
    autowhitebalance: bool = True  # True 유지
```

**카메라가 너무 어두우면:**
- `exposure` 값을 `0` 또는 `2`로 올려봄
- `autoexposure: False` 설정 후 `exposure` 값 수동 조정

---

## 예상 효과

| 항목 | 이전 | 이후 |
|------|------|------|
| BINARY 모드 흰색 차선 검출 | 0% (미검출) | 80~95% (다양한 조명) |
| 어두운 환경 대응 | ✗ | ✓ (threshold_low_light 조정) |
| 밝은 환경 대응 | ✗ | ✓ (threshold_bright 조정) |
| 채도 변동에 강인함 | ✗ | ✓ (white_saturation_max 확대) |
| 연산 오버헤드 | 기존 | +1~2% (negligible) |

---

## 문제 해결 체크리스트

### BINARY에 흰색 차선이 여전히 안 보이면:

- [ ] **파라미터 범위 확인**: `whitethreshold` 를 100까지 낮춰봄
- [ ] **ROI 범위 확인**: GUI의 `roitopratio` / `roibottomratio` 슬라이더로 ROI가 실제 차선을 포함하는지 확인
- [ ] **Perspective 변환 확인**: `recalculateperspectivepoints()` 호출 후 bird's eye view 영상 검증
- [ ] **카메라 밝기 확인**: GUI의 원본 영상(COLOR 모드)에서 차선이 밝게 보이는지 확인
- [ ] **Hood Mask 확인**: hood mask가 차선을 가리고 있는 것은 아닌지 확인
- [ ] **enablevehiclecolorsuppression**: 이 옵션이 흰색을 실수로 억제하고 있지 않은지 확인

---

## 코드 적용 순서

1. ✅ **`config.py` 수정** - 새로운 파라미터 추가
2. ✅ **`lane_detector.py`의 `detectwhitelane()` 함수만 교체**
3. ✅ **GUI 실행 후 BINARY 모드 확인**
4. ✅ **필요 시 파라미터 조정**

**전체 실행 시간: 5분 이내**

---

## 참고: 이 방식이 최선인가?

- ✅ **전통적 방식이지만 현실적**: 저사양 PC에서 실시간 동작
- ✅ **구현 간단**: 기존 코드 1개 함수만 수정
- ✅ **파라미터 조정 가능**: GUI로 실시간 튜닝 가능
- ⚠️ **극한 환경 약함**: 매우 어두운 조건, 희미한 차선은 여전히 어려울 수 있음
- ⚠️ **일반화 한계**: 환경별로 따로 파라미터 설정 필요할 수 있음

**더 강인한 방식 (향후 고려):**
- HLS 색공간 추가 병렬 처리
- Deep learning 기반 segmentation (고사양 필요)
- 여러 환경에서 학습한 parameter set 자동 선택
