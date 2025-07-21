"""
/classify
  • return_type=json  →  JSON + annotated_png
  • return_type=png   →  image/png
  • 모든 하이퍼파라미터 쿼리 조정
  • MongoDB(MONGO_URI 환경변수) — annotated_png, overlap 저장 안 함
"""
import os, io, base64, cv2, numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from .analyser import analyse_bgr

# ───────── 환경변수 로드 ─────────
load_dotenv()                                     # .env → os.environ
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("환경변수 MONGO_URI 가 설정되지 않았습니다.")

mongo_col = MongoClient(MONGO_URI)["zezeone"]["results"]

# ───────── FastAPI 앱 ─────────
app = FastAPI(
    title="Spot Classifier API (env + Mongo)",
    description="하이퍼파라미터 전부 노출, MongoDB 자동 저장",
    version="1.2.0",
)

# ───────── 내부 헬퍼 ─────────
def _file_to_bgr(file: UploadFile):
    data = file.file.read()
    img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{file.filename}: 잘못된 이미지")
    return img

def _bgr_to_png(img: np.ndarray):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf).decode(), buf

# ───────── 엔드포인트 ─────────
@app.post(
    "/classify",
    responses={200: {"content": {"application/json": {}, "image/png": {}}}},
    summary="A/B 분석 + MongoDB 저장"
)
def classify(
    file: UploadFile = File(..., description="PNG/JPG"),
    return_type: str = Query("json", enum=["json", "png"]),

    # A/B 기준
    max_clu_thr: int = Query(15, ge=1),
    uni_thr: float   = Query(0.89, ge=0, le=1),

    # BlobDetector
    min_area: int = Query(50, ge=1),
    max_area: int = Query(2000, ge=1),
    min_threshold: int = Query(100, ge=0),
    max_threshold: int = Query(500, ge=0),

    # DBSCAN
    eps: float       = Query(30.0, ge=1),
    min_samples: int = Query(6,    ge=1),

    # 흰 영역 채우기
    big_area: int = Query(200, ge=1),
):
    # 1) 분석
    analysed = analyse_bgr(
        _file_to_bgr(file),
        max_clu_thr=max_clu_thr, uni_thr=uni_thr,
        min_area=min_area, max_area=max_area,
        min_threshold=min_threshold, max_threshold=max_threshold,
        eps=eps, min_samples=min_samples,
        big_area=big_area,
    )

    # 2) 이미지 변환
    b64_png, png_buf = _bgr_to_png(analysed.pop("annotated"))

    # 3) Mongo 저장 (annotated_png, overlap 제외)
    mongo_doc = {k: v for k, v in analysed.items() if k != "overlap"}
    inserted_id = mongo_col.insert_one(mongo_doc).inserted_id

    # 4) 응답
    if return_type == "png":
        return StreamingResponse(io.BytesIO(png_buf), media_type="image/png")

    analysed["_id"] = str(inserted_id)
    analysed["annotated_png"] = b64_png
    return JSONResponse(jsonable_encoder(analysed))
