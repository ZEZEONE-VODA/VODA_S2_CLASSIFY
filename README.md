# 이미지 라벨링 및 시각화 프로젝트

## 1. 프로젝트 목적

본 프로젝트는 이미지 내의 특정 객체나 영역을 식별하고 분류하기 위한 라벨링 작업을 수행하고, 그 결과를 시각적으로 확인하는 것을 목적으로 합니다. `label.py` 스크립트를 통해 라벨링을 진행하고, `quick_vis.py`를 통해 라벨링된 데이터를 빠르게 시각화할 수 있습니다.

## 2. 분류 기준

`label.py` 스크립트는 이미지 내에서 감지된 "spot"들의 공간적 분포 특성을 분석하여 이미지를 **A**와 **B** 두 가지 클래스로 분류합니다.

-   **Class A (정상/양호 추정):** 아래의 **세 가지 조건**을 **모두** 만족하는 경우입니다.
    1.  **겹침 없음 (No Overlap):** 감지된 spot들이 서로 겹치지 않아야 합니다.
    2.  **작은 군집 (Small Clusters):** Spot들이 큰 군집을 이루지 않아야 합니다. (가장 큰 군집의 크기가 6개 미만)
    3.  **높은 균일도 (High Uniformity):** Spot들이 이미지 전체에 걸쳐 균일하게 분포해야 합니다. (균일도 점수 >= 0.85)

-   **Class B (비정상/불량 추정):** 위의 'A' 조건 중 하나라도 만족하지 못하면 'B'로 분류됩니다. (spot이 겹치거나, 큰 군집을 이루거나, 분포가 불균일한 경우)

스크립트는 이 기준에 따라 원본 이미지를 `--dest_dir` 내의 `A` 또는 `B` 폴더로 복사하고, 시각화된 이미지를 `A_raw`, `B_raw` 폴더에 저장합니다.

## 3. 설치 및 실행

### 3.1. 가상환경 및 라이브러리 설치

```bash
# 1. 가상환경 생성
python -m venv .venv

# 2. 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 3. 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 3.2. 실행 명령어

라벨링 스크립트 실행
```bash
python label.py --img_dir "파일 경로" --dest_dir "출력 경로" --out_csv "출력 경로\result.csv"
```

시각화 스크립트 실행
```bash
python quick_vis.py
```

**참고:** 각 스크립트는 추가적인 인자(argument)를 필요로 할 수 있습니다. 자세한 사용법은 각 `.py` 파일의 코드를 확인해주세요. (예: `python label.py --input_dir ./images`)

## 4. Docker를 이용한 실행

이 프로젝트는 Docker를 사용하여 간편하게 실행할 수 있습니다. Dockerfile은 ARM 아키텍처를 포함한 다양한 환경에서 실행될 수 있도록 설정되어 있습니다.

### 4.1. Docker 이미지 빌드

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 Docker 이미지를 빌드합니다.

```bash
# 'label-app'이라는 이름으로 이미지를 빌드합니다.
docker build -t label-app .
```

### 4.2. Docker 컨테이너 실행

빌드된 이미지를 사용하여 컨테이너를 실행합니다. 로컬 컴퓨터의 이미지 폴더와 출력 폴더를 컨테이너 내부의 폴더와 연결(마운트)해야 합니다.

**Windows (CMD/PowerShell):**

```bash
docker run -v C:\path\to\your\images:/app/images -v C:\path\to\your\output:/app/output label-app --img_dir /app/images --dest_dir /app/output
```

**Linux/macOS 또는 Git Bash:**

```bash
docker run -v /path/to/your/images:/app/images -v /path/to/your/output:/app/output label-app --img_dir /app/images --dest_dir /app/output
```

**실행 예시 (Windows):**

```bash
docker run -v C:\Users\choho\Downloads\testimage\output2:/app/images -v C:\Users\choho\Downloads\testimage\ab_output:/app/output label-app --img_dir /app/images --dest_dir /app/output
```

**참고:**

*   `C:\path\to\your\images` 와 `/path/to/your/images` 부분을 실제 이미지 파일이 있는 로컬 폴더 경로로 변경해야 합니다.
*   `C:\path\to\your\output` 와 `/path/to/your/output` 부분을 결과물을 저장할 로컬 폴더 경로로 변경해야 합니다.
*   ARM 기반 기기에서 실행하려면, 해당 기기에서 위와 동일한 방법으로 Docker 이미지를 빌드하고 컨테이너를 실행하면 됩니다. Dockerfile은 ARM 아키텍처를 자동으로 감지하여 빌드합니다.

```