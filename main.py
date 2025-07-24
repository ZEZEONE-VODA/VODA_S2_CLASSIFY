"""
/classify
  • return_type=json → JSON + annotated_png(Base64)
  • return_type=png  → image/png
  • img_url          → 저장된 annotated 이미지 HTTP 링크
  • MongoDB          → annotated_png·overlap 제외하고 저장
"""
import os, io, base64, cv2, numpy as np
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.encoders import jsonable_encoder

from analyser import analyse_bgr  # 같은 폴더에 analyser.py 존재해야 함

# ───────── 환경 변수 로드 ─────────
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("환경변수 MONGO_URI 가 설정되지 않았습니다.")

# ───────── Mongo ─────────
client    = MongoClient(MONGO_URI)
mongo_col = client["zezeone"]["results"]

# ───────── 경로 ─────────
BASE_DIR  = Path(__file__).resolve().parent
ANNOT_DIR = (BASE_DIR / "static" / "annot").resolve()
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# ───────── FastAPI ─────────
app = FastAPI(
    title="Spot Classifier API (env + Mongo)",
    description="하이퍼파라미터 전부 노출, MongoDB 자동 저장",
    version="1.2.5",
)

# ───────── 헬퍼 ─────────
def _file_to_bgr(upload: UploadFile):
    data = upload.file.read()
    img  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: 잘못된 이미지")
    return img

def _bgr_to_png(img: np.ndarray):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 인코딩 실패")
    return base64.b64encode(buf).decode(), buf

# ───────── 이미지 제공 ─────────
@app.get("/annot/{fname}", summary="저장된 annotated PNG 반환")
def get_annot(fname: str):
    fpath = ANNOT_DIR / fname
    if not fpath.exists():
        raise HTTPException(404, "이미지 없음")
    return FileResponse(fpath, media_type="image/png")

# ───────── 헬스 체크 ─────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ───────── 메인 엔드포인트 ─────────
@app.post(
    "/classify",
    summary="A/B 분석 + MongoDB 저장",
    responses={200: {"content": {"application/json": {}, "image/png": {}}}},
)
def classify(
    request: Request,
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

    # 2) annotated PNG 처리
    annotated = analysed.pop("annotated", None)
    if annotated is None:
        raise HTTPException(500, "annotated 이미지가 없습니다.")
    b64_png, png_buf = _bgr_to_png(annotated)

    # 3) 파일 저장 → URL (request.url_for로 자동 생성)
    fname = f"{uuid4()}.png"
    (ANNOT_DIR / fname).write_bytes(png_buf.tobytes())
    img_url = str(request.url_for("get_annot", fname=fname))

    # 디버그 로그
    print(">>> BASE_DIR :", BASE_DIR)
    print(">>> ANNOT_DIR:", ANNOT_DIR)
    print(">>> IMG_PATH :", (ANNOT_DIR / fname).resolve())
    print(">>> IMG_URL  :", img_url)

    # 4) DB 저장 (overlap, annotated_png 제외)
    mongo_doc = {k: v for k, v in analysed.items() if k != "overlap"}
    mongo_doc["img_url"] = img_url
    inserted_id = mongo_col.insert_one(mongo_doc).inserted_id

    # 5) 응답
    if return_type == "png":
        return StreamingResponse(io.BytesIO(png_buf), media_type="image/png")

    analysed["_id"]           = str(inserted_id)
    analysed["annotated_png"] = b64_png
    analysed["img_url"]       = img_url
    return JSONResponse(jsonable_encoder(analysed))
