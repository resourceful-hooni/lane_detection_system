# LabVIEW 연동 가이드 (LabVIEW Integration Guide)

이 문서는 Python 차선 검출 시스템(`lane_detection_system`)과 LabVIEW를 연동하여 자율주행 제어 시스템을 구축하는 방법을 상세히 설명합니다.

## 1. 연동 원리 (Architecture)

이 시스템은 **파일 기반 통신 (File-based Communication)** 방식을 사용합니다. 가장 단순하면서도 신뢰성이 높은 방식입니다.

1.  **Python (Sender)**: 차선을 검출하고 조향각(`steering_angle`)과 상태 정보를 계산하여 `state.json` 파일에 계속 덮어씁니다. (약 30~60Hz)
2.  **LabVIEW (Receiver)**: `state.json` 파일을 주기적으로 읽어(Polling), 데이터를 파싱하고 모터 제어 알고리즘에 입력으로 사용합니다.

---

## 2. Python 실행 준비

LabVIEW가 데이터를 읽을 수 있도록 Python 프로그램을 **브릿지 모드**로 실행해야 합니다.

### 실행 명령어
터미널(PowerShell 또는 CMD)에서 프로젝트 폴더로 이동 후 아래 명령어를 실행하세요.

```powershell
# 가상환경 활성화
.venv\Scripts\activate

# LabVIEW 연동 모드로 실행 (이미지 저장 포함)
python main.py --enable-labview-bridge --labview-write-frame
```

### 생성되는 파일 확인
실행 후 `labview_bridge/` 폴더에 다음 두 파일이 생성되고 계속 갱신되는지 확인하세요.
*   `state.json`: 텍스트 데이터 (조향각, 오프셋 등)
*   `overlay.jpg`: 현재 카메라 화면 (차선 오버레이 포함)

---

## 3. LabVIEW VI 작성 가이드

LabVIEW에서 새 VI를 만들고 아래 순서대로 블록다이어그램(Block Diagram)을 구성하세요.

### A. 필수 라이브러리 확인
LabVIEW 기본 함수 외에 특별한 툴킷은 필요 없으나, JSON 파싱을 위해 **"Connectivity"** 팔레트를 사용합니다.

### B. 데이터 구조 정의 (Type Definition)
JSON 데이터를 LabVIEW 데이터로 변환하기 위해 **Cluster(클러스터)** 상수를 만들어야 합니다.
`Unflatten From JSON` 함수의 `type` 입력에 연결할 클러스터입니다.

1.  **프런트패널**에 `Cluster`를 하나 만듭니다.
2.  클러스터 안에 다음 컨트롤들을 넣고 **라벨(Label) 이름을 정확히** 지정하세요. (대소문자 구분)

**Cluster 구조:**
*   `timestamp` (Numeric: DBL, 실수형)
*   `fps` (Numeric: DBL)
*   **Cluster** (이름: `path`)  <-- 클러스터 안에 또 다른 클러스터
    *   `valid` (Boolean)
    *   `steering_angle` (Numeric: DBL)  <-- **핵심 제어 값**
    *   `center_offset` (Numeric: DBL)
    *   `lane_departure_warning` (Boolean)
*   **Cluster** (이름: `lane`)
    *   `detected` (Boolean)

> **팁**: 이 클러스터 상수를 블록다이어그램으로 가져와서 `Unflatten From JSON`의 위쪽 입력(`type`)에 연결하면 됩니다.

### C. 블록다이어그램 로직 (Step-by-Step)

**1. While Loop 생성**
*   전체 로직을 감싸는 `While Loop`를 만듭니다.
*   루프 안에 `Wait (ms)` 함수를 넣고 **50** (50ms) 정도로 설정합니다. (너무 빠르면 파일 읽기 충돌 발생 가능)

**2. 파일 읽기 (Read Text File)**
*   `File I/O` -> `Read Text File` 함수를 배치합니다.
*   **file path** 입력에 `state.json`의 절대 경로를 연결합니다.
    *   예: `C:\Users\wlgns\OneDrive\Desktop\lane_detection_system\labview_bridge\state.json`

**3. 에러 처리 (중요!)**
*   Python이 파일을 쓰고 있는 순간에 LabVIEW가 읽으려 하면 **Error 1** 또는 **Error 5**가 발생할 수 있습니다.
*   `Read Text File`의 **error out** 단자를 **Case Structure**의 선택자(Selector)에 연결합니다.
*   **No Error 케이스**: 정상적으로 데이터를 처리합니다.
*   **Error 케이스**: `Clear Error` 함수를 사용하여 에러를 지워줍니다. (프로그램이 멈추지 않게 함)

**4. JSON 파싱 (Unflatten From JSON)**
*   **No Error 케이스** 내부에서 작업합니다.
*   `Connectivity` -> `JSON` -> `Unflatten From JSON` 함수를 배치합니다.
*   `Read Text File`의 출력(text)을 `JSON String`에 연결합니다.
*   위에서 만든 **Cluster 상수**를 `type`에 연결합니다.

**5. 데이터 추출 (Unbundle)**
*   `Unflatten From JSON`의 출력 데이터를 `Unbundle By Name` 함수에 연결합니다.
*   `path` 클러스터를 꺼내고, 다시 `Unbundle By Name`을 사용하여 `steering_angle`과 `center_offset`을 추출합니다.

**6. 제어 알고리즘 연결**
*   추출한 `steering_angle` 값을 LabVIEW의 모터 제어 VI나 PID 알고리즘의 입력으로 연결합니다.
*   `lane_departure_warning`이 True일 경우 정지하거나 경고음을 울리는 로직을 추가합니다.

---

## 4. 이미지 모니터링 (옵션)

LabVIEW 프런트패널에서 실시간 영상을 보고 싶다면 추가하세요.

1.  `Graphics & Sound` -> `Graphics Formats` -> `JPEG` -> `Read JPEG File` 함수 사용.
2.  경로는 `labview_bridge/overlay.jpg` 지정.
3.  출력된 이미지 데이터를 `Draw Flattened Pixmap` 함수에 연결.
4.  프런트패널에 **Picture Indicator**를 생성하여 연결.

---

## 5. 문제 해결 (Troubleshooting)

**Q. LabVIEW에서 에러가 계속 깜빡거려요.**
*   파일 I/O 충돌 때문입니다. `Read Text File` 뒤에 `Clear Error`를 반드시 붙여서 에러가 나도 루프가 멈추지 않게 하세요.

**Q. 데이터가 갱신되지 않아요.**
*   Python 터미널이 켜져 있고 `main.py`가 실행 중인지 확인하세요.
*   `state.json` 파일의 수정 날짜가 바뀌고 있는지 윈도우 탐색기에서 확인하세요.

**Q. 조향각이 반대인 것 같아요.**
*   Python 코드의 `steering_angle`은 일반적으로 **오른쪽이 양수(+)**, **왼쪽이 음수(-)**입니다.
*   LabVIEW 모터 방향에 맞춰 필요하다면 `-1`을 곱해서 사용하세요.
