# pip install opencv-python numpy pandas scikit-learn scipy
import argparse, shutil
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

DOT_R = 5                     # fill_big_white 에서 쓰는 초록 점 반경(px)
MIN_GAP = DOT_R * 2 + 1       # 겹침 허용 안 함(≥ 11 px)

# ────────────── 1) spot 검출 ──────────────
def detect_spots(img_bgr, min_area=300, max_area=2000,
                 min_threshold=150, max_threshold=500):
    """SimpleBlobDetector 기반 spot 좌표·면적 반환"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea, p.minArea, p.maxArea = True, min_area, max_area
    p.minThreshold, p.maxThreshold = min_threshold, max_threshold
    p.filterByColor, p.blobColor = True, 255
    p.filterByCircularity = p.filterByInertia = p.filterByConvexity = False

    kps  = cv2.SimpleBlobDetector_create(p).detect(gray)
    pts  = np.array([kp.pt for kp in kps], dtype=np.float32)
    area = np.array([kp.size**2 * np.pi / 4 for kp in kps], dtype=np.float32)
    return pts, area, gray / 255.0

# ────────────── 2) green-circle(=raw) 판별 ──────────────
def has_green_circle(img_bgr, min_radius=30, max_radius=300, min_votes=30):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    blur = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=min_votes,
        minRadius=min_radius, maxRadius=max_radius
    )
    return circles is not None

# ────────────── 3) 큰 흰 덩어리 내부를 작은 초록 ●로 채우기 ──────────────
def fill_big_white(
        img_bgr,
        min_area=1_000,        # 최소 면적(px²)
        max_area=20_000,       # 최대 면적(px²) ← NEW
        thresh_val=180,        # 밝기 임계값 ↑ (100 → 180)
        dot_radius=5,
        dot_step=14,
        morph_ksize=9):        # 모폴로지 커널 크기 ↓
    """
    큰 흰 덩어리를 작은 초록 ●로 채움 (보수적 버전)
    - min_area, max_area : 면적 범위
    - thresh_val         : 이진화 임계값 (높일수록 덜 검출)
    - morph_ksize        : closing 커널 크기 (작을수록 덜 합쳐짐)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) 밝은 픽셀 이진화 (값 ↑)
    _, bin_ = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 2) 모폴로지 closing(작게) → noise 제거 후 팽창 적음
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)

    # 3) 컨투어 필터링
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue                       # 너무 작거나 너무 큰 덩어리 skip

        mask = np.zeros_like(gray, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        h, w = mask.shape
        for y in range(0, h, dot_step):
            for x in range(0, w, dot_step):
                if mask[y, x]:
                    cv2.circle(img_bgr, (x, y), dot_radius, (0, 255, 0), -1)


# ────────────── 4) 지표 계산 함수들 ──────────────
def nn_cv(pts):
    """최근접 거리 CV(표준편차/평균)"""
    if len(pts) < 2:
        return float("nan")
    dists = (
        NearestNeighbors(n_neighbors=2)
        .fit(pts)
        .kneighbors(pts)[0][:, 1]
    )
    return float(dists.std() / dists.mean())

# ────────────── 지표 함수 보강 ──────────────
# ────────────── 클러스터 최대 크기 ──────────────
def cluster_max(pts, eps=10, min_samples=2):
    if len(pts) < min_samples:
        return 0
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    if np.all(labels == -1):
        return 0
    sizes = np.bincount(labels[labels != -1])
    return int(sizes.max())

def grid_uniformity(pts, h, w, n=12):
    """엔트로피 기반 균일도 (0~1, 1이 이상적)"""
    if len(pts) == 0:
        return 0.0
    gx, gy = w / n, h / n
    hist = np.zeros(n * n, int)
    for x, y in pts:
        ix, iy = min(int(x // gx), n - 1), min(int(y // gy), n - 1)
        hist[iy * n + ix] += 1
    return entropy(hist, base=np.e) / np.log(n * n)

def has_overlap(pts, min_gap=MIN_GAP):
    """
    • pts : (N, 2) spot 중심좌표
    • min_gap : 두 점 중심 사이 최소 허용 거리(px)
    → True  : 겹치는 쌍이 하나라도 있음
      False : 전혀 안 겹침
    """
    if len(pts) < 2:
        return False
    nn = NearestNeighbors(n_neighbors=2).fit(pts)
    d = nn.kneighbors(pts, return_distance=True)[0][:, 1]  # 최근접 거리
    return np.any(d < min_gap)

# ────────────── analyse() 간단화 ──────────────
def analyse(path: Path, args):
    img = cv2.imread(str(path))
    pts, _, _ = detect_spots(img, args.min_area, args.max_area,
                             args.min_threshold, args.max_threshold)

    overlap    = has_overlap(pts)
    max_clu_sz = cluster_max(pts, eps=10, min_samples=2)
    uni_val    = grid_uniformity(pts, *img.shape[:2])      # ← 추가

    # ── A 조건: 겹침 X ∧ max_clu<6 ∧ uniformity≥uni_thr ──
    label = "A" if (not overlap and max_clu_sz < 6 and uni_val >= args.uni_thr) else "B"

    vis = img.copy()
    fill_big_white(vis, args.big_area)
    for x, y in pts.astype(int):
        cv2.circle(vis, (x, y), DOT_R, (0, 255, 0), -1)

    return dict(file=path.name,
                overlap=overlap,
                max_cluster=max_clu_sz,
                uniformity=uni_val,          # CSV에 포함
                label=label,
                vis=vis,
                path=path)


# ────────────── 6) 메인 루프 ──────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  required=True)
    ap.add_argument("--dest_dir", required=True)
    ap.add_argument("--out_csv",  default="result.csv")

    # 분포 기준
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--ratio",     type=float, default=0.40)
    ap.add_argument("--eps",       type=float, default=12)
    ap.add_argument("--min_samples", type=int, default=3)
    ap.add_argument("--uni_thr",   type=float, default=0.85)

    # spot & blob 파라미터
    ap.add_argument("--min_area",      type=int, default=16)
    ap.add_argument("--max_area",      type=int, default=400)
    ap.add_argument("--min_threshold", type=int, default=150)
    ap.add_argument("--max_threshold", type=int, default=500)
    ap.add_argument("--factor",        type=float, default=4.0)

    # 큰 덩어리 채우기
    ap.add_argument("--big_area", type=int, default=400)

    # 녹색 원(raw) 파라미터
    ap.add_argument("--min_radius", type=int, default=30)
    ap.add_argument("--max_radius", type=int, default=300)
    ap.add_argument("--min_votes",  type=int, default=30)
    args = ap.parse_args()


    imgs = sorted(Path(args.img_dir).glob("*.png")) + \
           sorted(Path(args.img_dir).glob("*.jpg"))
    if not imgs:
        raise RuntimeError("이미지가 없습니다.")

    for sub in ("A", "B", "A_raw", "B_raw"):
        d = Path(args.dest_dir) / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in imgs:
        metrics = analyse(p, args)
        metrics["path"] = p          # 원본 경로 보존
        rows.append(metrics)

    # 1) 클러스터 6↑ → 무조건 B 후보
    hard_B  = [r for r in rows if r["max_cluster"] >= 6]
    cands   = [r for r in rows if r["max_cluster"] <  6]

    # 3) 폴더 저장
# ────────────── main() 저장 루프 ──────────────
    for r in [analyse(p, args) for p in imgs]:
        # 1) 원본 → A/ 또는 B/
        shutil.copy2(r["path"], Path(args.dest_dir) / r["label"])

        # 2) annotated → A_raw/ or B_raw/
        raw_dir = Path(args.dest_dir) / (r["label"] + "_raw")
        raw_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(raw_dir / f"annotated_{r['file']}"), r["vis"])

        rows.append({k: r[k] for k in ("file", "overlap", "max_cluster", "label")})


    # 결과 CSV
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
