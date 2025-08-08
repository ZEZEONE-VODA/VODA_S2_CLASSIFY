"""
/classify
  • return_type=json → JSON + annotated_png(Base64)
  • return_type=png  → image/png
  • img_url          → GCS에 저장된 annotated 이미지 공개 URL (라벨별 A/B/UNKNOWN 하위폴더)
  • MongoDB          → annotated_png·overlap 제외하고 저장
  • date_time        → UTC 타임스탬프 (aware datetime)
  • ts_ms            → epoch milliseconds (정렬/쿼리용)
"""

import os
import io
import base64
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Tuple, Optional

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
MONGO_URI  = os.getenv("MONGO_URI")
GCS_BUCKET = os.getenv("GCS_BUCKET", "zezeone_images")
GCS_SUBDIR = os.getenv("GCS_SUBDIR", "grade")

if not MONGO_URI:
    raise RuntimeError("환경변수 MONGO_URI 가 설정되지 않았습니다.")
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 가 설정되지 않았습니다.")

# ───────── Logger ─────────
logger = logging.getLogger("uvicorn.error")

# ───────── Mongo ─────────
mongo_col = MongoClient(MONGO_URI)["zezeone"]["results"]

# ───────── GCS ─────────
gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(GCS_BUCKET)

# ───────── FastAPI ─────────
app = FastAPI(
    title="Spot Classifier API (Mongo + GCS)",
    description="하이퍼파라미터 노출, MongoDB 자동 저장, GCS 저장",
    version="1.4.0",
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
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: 잘못된 이미지")
    return img

def _bgr_to_png(img: np.ndarray) -> Tuple[str, np.ndarray]:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf).decode(), buf  # (b64 str, np.ndarray)

def upload_png_to_gcs(png_buf: np.ndarray, *, label_folder: Optional[str] = None) -> tuple[str, str]:
    base = GCS_SUBDIR.strip("/")
    subdir = f"{base}/{label_folder}" if label_folder else base
    object_name = f"{subdir}/{uuid4()}.png"
    blob = gcs_bucket.blob(object_name)
    blob.upload_from_string(png_buf.tobytes(), content_type="image/png")
    public_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{object_name}"
    return object_name, public_url

# ───────── Health/Diag ─────────
@app.head("/health")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/_diag")
def diag():
    try:
        ping = mongo_col.database.command("ping")
    except Exception as e:
        ping = {"ok": 0, "err": str(e)}
    return {
        "gcp_project": gcs_client.project,
        "gcs_bucket": GCS_BUCKET,
        "gcs_subdir": GCS_SUBDIR,
        "mongo_db": mongo_col.database.name,
        "mongo_coll": mongo_col.name,
        "mongo_ping": ping,
    }

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

    # 원 반지름
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
    uploaded_object_name: Optional[str] = None  # GCS 롤백용
    inserted_id = None                           # Mongo 롤백용

    try:
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

        # 3) UTC 타임스탬프
        now_utc = datetime.now(timezone.utc)
        ts_ms   = int(now_utc.timestamp() * 1000)

        # 4) 라벨 정규화 → 라벨별(GCS_SUBDIR/A|B|UNKNOWN) 경로에 업로드
        raw_label = analysed.get("label")
        label = (str(raw_label).strip().upper() if raw_label is not None else "UNKNOWN")
        if label not in {"A", "B"}:
            label = "UNKNOWN"

        img_file_id, img_url = upload_png_to_gcs(png_buf, label_folder=label)
        uploaded_object_name = img_file_id  # 롤백용
        logger.info(f"[GCS UPLOAD] project={gcs_client.project} gs://{GCS_BUCKET}/{img_file_id} label={label}")

        # 5) Mongo 저장
        mongo_doc_id = f"grade_{uuid4().hex}"  # 동시성 안전
        mongo_doc = {
            "_id":          mongo_doc_id,
            "label":        label,
            "max_cluster":  analysed.get("max_cluster"),
            "uniformity":   analysed.get("uniformity"),
            "n_spots":      analysed.get("n_spots"),
            "min_nn_dist":  analysed.get("min_nn_dist"),
            "nn_cv":        analysed.get("nn_cv"),
            "n_clusters":   analysed.get("n_clusters"),
            "img_file_id":  img_file_id,
            "img_url":      img_url,
            "uploadDate":   now_utc,   # UTC aware
            "date_time":    now_utc,   # UTC aware
            "ts_ms":        ts_ms,     # epoch ms
        }
        inserted_id = mongo_col.insert_one(mongo_doc).inserted_id

        # 6) 응답
        if return_type == "png":
            return StreamingResponse(io.BytesIO(png_buf.tobytes()), media_type="image/png")

        analysed_resp = analysed.copy()
        analysed_resp.update({
            "_id":           mongo_doc_id,
            "annotated_png": b64_png,
            "img_url":       img_url,
            "img_file_id":   img_file_id,
            "date_time":     now_utc.isoformat(),
            "ts_ms":         ts_ms,
        })
        return JSONResponse(jsonable_encoder(analysed_resp))

    except HTTPException:
        # FastAPI 예외는 그대로 전달하되, 보상 삭제 시도
        if inserted_id is not None:
            try:
                mongo_col.delete_one({"_id": inserted_id})
            except Exception:
                pass
        if uploaded_object_name is not None:
            try:
                gcs_bucket.blob(uploaded_object_name).delete()
            except Exception:
                pass
        raise
    except Exception as e:
        # 일반 예외 → 롤백 후 500
        if inserted_id is not None:
            try:
                mongo_col.delete_one({"_id": inserted_id})
            except Exception:
                pass
        if uploaded_object_name is not None:
            try:
                gcs_bucket.blob(uploaded_object_name).delete()
            except Exception:
                pass
        logger.error(f"[INTERNAL ERROR] {e}")
        raise HTTPException(status_code=500, detail="internal error")

# ───────── Local run helper (optional) ─────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8100")))
