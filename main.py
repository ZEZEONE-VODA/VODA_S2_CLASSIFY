"""
/classify
  • return_type=json → JSON + annotated_png(Base64)
  • return_type=png  → image/png
  • img_url          → GCS에 저장된 annotated 이미지 공개 URL
  • MongoDB          → annotated_png·overlap 제외하고 저장
  • date_time        → UTC 타임스탬프 (aware datetime)
  • ts_ms            → epoch milliseconds (정렬/쿼리용)
"""
import os
import io
import base64
from uuid import uuid4
from datetime import datetime, timezone

import cv2
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage

from analyser import analyse_bgr  # 사용자 보유 코드

# ───────── ENV ─────────
load_dotenv()
MONGO_URI   = os.getenv("MONGO_URI")
GCS_BUCKET  = os.getenv("GCS_BUCKET", "zezeone_images")
GCS_SUBDIR  = os.getenv("GCS_SUBDIR", "grade")

if not MONGO_URI:
    raise RuntimeError("환경변수 MONGO_URI 가 설정되지 않았습니다.")
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 가 설정되지 않았습니다.")

# ───────── Mongo ─────────
mongo_col = MongoClient(MONGO_URI)["zezeone"]["results"]

# ───────── GCS ─────────
gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(GCS_BUCKET)

# ───────── FastAPI ─────────
app = FastAPI(
    title="Spot Classifier API (Mongo + GCS)",
    description="하이퍼파라미터 노출, MongoDB 자동 저장, GCS 저장",
    version="1.3.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────── Helpers ─────────
def _file_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: 잘못된 이미지")
    return img

def _bgr_to_png(img: np.ndarray):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf).decode(), buf  # (b64 str, np.ndarray)

def upload_png_to_gcs(png_buf: np.ndarray) -> tuple[str, str]:
    object_name = f"{GCS_SUBDIR}/{uuid4()}.png"
    blob = gcs_bucket.blob(object_name)
    blob.upload_from_string(png_buf.tobytes(), content_type="image/png")
    public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{object_name}"
    return object_name, public_url

# ───────── Health ─────────
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}

# ───────── Main Endpoint ─────────
@app.post(
    "/classify",
    summary="A/B 분석 + MongoDB 저장 + GCS 업로드",
    responses={200: {"content": {"application/json": {}, "image/png": {}}}},
)
def classify(
    request: Request,
    file: UploadFile = File(..., description="PNG/JPG"),
    return_type: str = Query("json", enum=["json", "png"]),

    # A/B 기준
    max_clu_thr: int = Query(15, ge=1),
    uni_thr: float   = Query(0.89, ge=0, le=1),

    #원 반지름
    dot_radius: int = Query(5, ge=1, le=50, description="점 반경(px)"),

    # BlobDetector
    min_area: int = Query(130, ge=1),
    max_area: int = Query(400, ge=1),
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
        dot_radius=dot_radius
    )

    # 2) annotated PNG
    annotated = analysed.pop("annotated", None)
    if annotated is None:
        raise HTTPException(500, "annotated 이미지가 없습니다.")
    b64_png, png_buf = _bgr_to_png(annotated)

    # 3) GCS 업로드
    img_file_id, img_url = upload_png_to_gcs(png_buf)

    # 4) Mongo 저장 (overlap, annotated_png 제외)
    now_utc = datetime.now(timezone.utc)
    ts_ms   = int(now_utc.timestamp() * 1000)

    # ★ grade 아이디 생성 (count 기반)
    count = mongo_col.count_documents({})
    mongo_doc_id = f"grade_{count + 1}"

    mongo_doc = {
        "id":           mongo_doc_id,
        "label":        analysed.get("label"),
        "max_cluster":  analysed.get("max_cluster"),
        "uniformity":   analysed.get("uniformity"),
        "n_spots":      analysed.get("n_spots"),
        "min_nn_dist":  analysed.get("min_nn_dist"),
        "nn_cv":        analysed.get("nn_cv"),
        "n_clusters":   analysed.get("n_clusters"),
        "img_file_id":  img_file_id,
        "img_url":      img_url,
        "uploadDate":   now_utc,     # 기존 필드
        "date_time":    now_utc,     # 요청 필드
        "ts_ms":        ts_ms,       # 정수형 타임스탬프(옵션)
    }
    inserted_id = mongo_col.insert_one(mongo_doc).inserted_id

    # 5) 응답
    if return_type == "png":
        return StreamingResponse(io.BytesIO(png_buf), media_type="image/png")

    analysed_resp = analysed.copy()
    analysed_resp.update({
        "id":             mongo_doc_id,
        "annotated_png":  b64_png,
        "img_url":        img_url,
        "img_file_id":    img_file_id,
        "date_time":      now_utc.isoformat(),
        "ts_ms":          ts_ms,
    })

    return JSONResponse(jsonable_encoder(analysed_resp))
