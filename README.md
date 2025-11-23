# Lane Detection System

실시간 차선 검출, 경로 계획, GUI 시각화, 그리고 LabVIEW 연동을 한 번에 처리하는 Python 애플리케이션입니다. Logitech BRIO급 USB 카메라에서 영상을 받아 OpenCV로 차선을 찾고, PID 기반 조향 각도와 각종 텔레메트리를 계산한 뒤 GUI/CSV/영상/파일 브릿지를 통해 외부 시스템으로 전달합니다.

## ✨ 주요 기능
- **고속 차선 검출 파이프라인**: 색상+에지 결합, 원근 변환(Remap 최적화), 슬라이딩 윈도우/이전 프레임 기반 탐색.
- **고급 ROI 및 필터링**: 4방향(상하좌우) ROI 크롭, 사다리꼴 마스크(Trapezoid Mask), Blob 필터(차량/노이즈 제거) 지원.
- **Kalman Filter & Joint Fitting**: 차선 위치 추적 및 양쪽 차선의 곡률을 평행하게 유지하는 Joint Fitting(Parallel Lock) 알고리즘 적용.
- **적응형 차선 폭 학습**: 주행 중 실제 차선 폭을 학습하여 한쪽 차선만 보일 때도 정확한 위치 복원.
- **정밀 제어 로직**: 튜닝된 PID 제어기(Kp=25.0)와 조향각 스무딩(LPF)을 통한 부드러운 주행.
- **동적 ROI 조정**: 초기 60프레임 동안은 근거리(ROI Top 0.6)를 집중 분석하여 안정성을 확보하고, 이후 원거리(0.3)로 자동 확장.
- **차선 보간 및 상태 추적**: 갭 마스크/플래그로 끊긴 구간을 시각화하고 LabVIEW에 신호 전달.
- **풍부한 시각화**: Tk GUI(슬라이더/체크박스로 실시간 튜닝), 오버레이 영상 저장.
- **데이터 로깅**: CSV·AVI 자동 저장, 프레임별 FPS/조향/오프셋 기록.
- **LabVIEW 브릿지**: JSON/이미지 파일을 통해 타 공정 장비와 느슨하게 결합.

## 🧱 기술 스택
- **언어**: Python 3.12
- **핵심 라이브러리**: OpenCV, NumPy, Pillow, Tkinter, csv
- **하드웨어**: Logitech BRIO VU0040 (848×480 @ 60fps 기준 튜닝)
- **운영체제**: Windows 10/11 (DirectShow 우선 백엔드)

## 📁 디렉터리 개요
```
.
├─ main.py                # 진입점/CLI
├─ lane_detector.py       # 차선 검출 + 보간
├─ path_planner.py        # PID/Pure Pursuit 기반 경로 계획
├─ gui_controller.py      # Tkinter GUI (옵션)
├─ data_logger.py         # CSV/비디오 기록
├─ labview_bridge.py      # 파일 기반 LabVIEW 연동
├─ config.py              # 모든 파라미터 중앙 관리
├─ calibration/           # 카메라 캘리브레이션 유틸
├─ labview_bridge/        # 상태 JSON과 오버레이 이미지가 저장되는 위치
└─ logs/, videos/         # 실행 결과물
```

## 🚀 실행 방법
### 1. 환경 준비
```cmd
cd /d C:\Users\wlgns\OneDrive\Desktop\lane_detection_system
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python numpy pillow
```
필요 시 pandas 등 추가 의존성은 상황에 맞춰 설치하세요.

### 2. 카메라 확인 (선택)
```cmd
.venv\Scripts\python.exe main.py --list-cameras --max-camera-index 5
```

### 3. 기본 실행
```cmd
.venv\Scripts\python.exe main.py --camera-index 1
```
- GUI는 기본 활성화, OpenCV 프리뷰는 `--preview`를 줘야 열립니다.
- 프로그램은 수동으로 중지(q 키, GUI 종료, Ctrl+C)할 때까지 계속 실행됩니다.

### 4. 주요 CLI 옵션
| 옵션 | 설명 |
|------|------|
| `--camera-index <n>` | 사용할 카메라 인덱스 지정 (기본값은 `config.py` 참고). |
| `--frame-width <px>` / `--frame-height <px>` | 런타임에 원하는 해상도로 오버라이드합니다 (예: 960×540). |
| `--no-gui` | Tk GUI 비활성화 (헤드리스 모드). |
| `--preview` | OpenCV 미리보기 창을 켭니다. 기본은 꺼져 있습니다. |
| `--enable-labview-bridge` | LabVIEW용 JSON/이미지 파일을 기록합니다. |
| `--labview-state-path path` | 상태 JSON 저장 경로 (기본 `labview_bridge/state.json`). |
| `--labview-frame-path path` | 오버레이 이미지를 저장할 경로. |
| `--labview-write-frame` | 브릿지가 오버레이 이미지를 JPEG로 씁니다. |
| `--list-cameras` | 사용 가능한 카메라 인덱스를 탐색하고 종료. |
| `--dev-max-frames <n>` | (개발자 전용) 지정 프레임만큼만 실행하고 종료. |

## 🔗 LabVIEW 연동 가이드
LabVIEW는 파일을 폴링하는 방식으로 Python 애플리케이션과 통신합니다.

### 1. Python 측 설정
```cmd
.venv\Scripts\python.exe main.py ^
  --camera-index 1 ^
  --enable-labview-bridge ^
  --labview-state-path labview_bridge\state.json ^
  --labview-frame-path labview_bridge\overlay.jpg ^
  --labview-write-frame
```
- `state.json`에는 최신 차선/경로 정보가 20~60Hz 수준으로 갱신됩니다.
- `overlay.jpg`는 Tk/GUI 없이도 LabVIEW에서 즉시 시각화할 수 있는 프레임입니다 (옵션).

### 2. JSON 구조
```json
{
  "timestamp": 1732398664.123,
  "fps": 58.7,
  "frame": 1520,
  "lane": {
    "detected": true,
    "gap_flags": {"left": false, "right": true},
    "had_gaps": true
  },
  "path": {
    "valid": true,
    "center_offset": -0.0453,
    "steering_angle": 3.8,
    "left_curvature": 42.1,
    "right_curvature": 39.7,
    "lane_departure_warning": false
  }
}
```
모든 값은 float/bool로 직렬화됐으며, NaN 이슈를 피하기 위해 numpy 스칼라는 표준 float로 변환됩니다.

### 3. LabVIEW 워크플로우
1. **파일 I/O 루프**: 10~30ms 주기로 `state.json`을 읽고 `timestamp`가 바뀌었는지 확인합니다.
2. **JSON 파싱**: `detected`, `gap_flags`, `center_offset`, `steering_angle` 등 필요한 항목만 Unbundle.
3. **제어 로직**: 경로 PID를 LabVIEW 제어기에 전달하거나, `lane_departure_warning`으로 알람 트리거.
4. **영상 피드 (옵션)**: `overlay.jpg`를 LabVIEW Picture Control/Image Display에 로드해 모니터링.
5. **에러 처리**: 파일이 잠시 비어도 1~2번 재시도 후 최신 데이터가 기록됩니다.

### 4. 팁
- JSON/이미지 파일은 항상 같은 경로에 덮어쓰기 때문에 LabVIEW 쪽에서는 링크 재설정이 필요 없습니다.
- 별도 실시간 소켓 없이도 LabVIEW 프로젝트에 쉽게 통합할 수 있으며, 네트워크 공유 폴더나 NI RT 타겟에도 동일한 방식으로 배포 가능합니다.

## 🛠️ 디버깅 & 참고
- 카메라가 열리지 않는다면 `--list-cameras`로 인덱스를 확인하고, Windows에서는 CAP_DSHOW 백엔드를 자동 사용합니다.
- GUI 없이 완전 헤드리스로 돌리고 싶다면 `--no-gui`와 `--preview` 생략 조합을 쓰면 됩니다.
- 로그/영상은 `logs/`와 `videos/` 폴더에 자동 저장되며, 파일명은 실행 시각을 포함합니다.
- 차량 보닛이 프레임 하단에 잡혀 차선이 왜곡될 경우 `lane_detection.enable_hood_mask`를 켜고 `hood_mask_polygon` (정규화 좌표)을 트랙 환경에 맞게 조정하세요.
- 흰-검-흰 패턴 테이프를 쓰는 경우 `lane_detection.enable_triplet_detection`과 관련 임계값(`triplet_gradient_threshold`, `triplet_morph_kernel`)을 조정하면 에지 기반 마스크가 생성되어 검출이 더 안정적입니다.
- 폴리곤을 정밀하게 따고 싶다면 `python tools\mask_calibrator.py`를 실행해 화면에서 클릭으로 꼭짓점을 지정하세요. 저장된 `calibration/hood_mask.json`은 다음 실행부터 자동으로 로드되며 정규화 좌표를 `config.py`에 수동으로 옮길 필요가 없습니다.
- 보닛을 가렸는데도 하단이 계속 잡히면 `lane_detection.bottom_trim_ratio`를 0.05~0.15처럼 올려 프레임 하단 일정 비율을 통째로 제거할 수 있습니다.
- 현재 시스템은 848×480 @ 60fps로 최적화되어 있습니다. 성능이 필요하면 `--frame-width`, `--frame-height`로 해상도를 조정할 수 있으며, 원근 변환 포인트는 자동으로 스케일됩니다.
- **튜닝 및 파라미터 설정**:
  - **GUI 튜닝**: 실행 중 GUI의 슬라이더와 체크박스를 통해 실시간으로 값을 조정할 수 있습니다.
    - **ROI Sliders**: 상하좌우(`Top`, `Bottom`, `Left`, `Right`) 슬라이더로 관심 영역 설정.
    - **Trapezoid Mask**: `Trap Top Width` 슬라이더로 ROI 상단을 사다리꼴로 좁힘 (1.0=비활성).
    - **Blob Filter**: `Blob Max Width` 슬라이더로 차량 등 큰 객체 필터링 (기본값 80).
    - **Parallel Lock**: `Joint Fitting` 체크박스로 양쪽 차선 평행 유지.
  - **영구 설정 변경**: `config.py` 파일을 열어 `LaneDetectionConfig` 등의 기본값을 수정하면 다음 실행 시부터 적용됩니다.

필요한 내용이 더 있으면 README를 자유롭게 확장해 주세요!
