# ARM64 아키텍처를 지원하는 Python 3.10 이미지를 기반으로 합니다.
FROM python:3.10-slim

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 시스템 패키지를 업데이트하고 OpenCV 빌드에 필요한 라이브러리를 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 파일을 이미지로 복사합니다.
COPY requirements.txt .

# pip를 업그레이드하고 requirements.txt에 명시된 Python 패키지를 설치합니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드를 이미지로 복사합니다.
COPY . .

# 기본 실행 명령을 label.py로 설정합니다.
ENTRYPOINT ["python", "label.py"]
