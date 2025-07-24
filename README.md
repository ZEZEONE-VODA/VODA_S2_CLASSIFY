# Green Dot A/B Classification API

이 프로젝트는 이미지에 포함된 Green Dot의 분포를 분석하여 A/B 등급으로 분류하는 API를 제공합니다.

## 주요 기능

- 이미지 내 Green Dot 검출 및 위치 분석
- DBSCAN 클러스터링을 이용한 밀집도 분석
- 그리드 기반 분포 균일도 분석
- 분석 결과를 반영한 이미지 시각화 및 저장
- MongoDB를 이용한 분석 결과 데이터베이스 저장

## 기술 스택

- **Backend:** FastAPI, Python
- **Database:** MongoDB
- **Image Processing:** OpenCV, Scikit-learn, Numpy

## API Endpoints

- `POST /classify`: 이미지 파일을 업로드하여 A/B 분류를 수행합니다.
  - `return_type=json` (기본값): 분석 결과(JSON)와 Base64로 인코딩된 결과 이미지를 반환합니다.
  - `return_type=png`: 분석 결과 이미지를 PNG 형식으로 직접 반환합니다.
- `GET /annot/{fname}`: 저장된 분석 결과 이미지를 반환합니다.
- `GET /health`: 서버 상태를 확인합니다.

## 시작하기

1.  `.env` 파일을 생성하고 `MONGO_URI` 환경 변수를 설정합니다.

    ```
    MONGO_URI=mongodb://...
    ```

2.  필요한 라이브러리를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

3.  FastAPI 개발 서버를 실행합니다.

    ```bash
    uvicorn main:app --reload
    ```

4.  브라우저에서 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 로 접속하여 API 문서를 확인하고 테스트할 수 있습니다.